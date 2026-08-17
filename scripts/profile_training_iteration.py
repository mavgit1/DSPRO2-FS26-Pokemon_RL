"""Time one PPO training iteration: rollout, learn, self-play export, driver RPCs.

Does not log to MLflow. Starts Showdown on the requested ports if they are not already up.

Usage:
    .venv/bin/python scripts/profile_training_iteration.py --preset quick --iterations 3 --num-servers 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("ENABLE_DECISION_DIAGNOSTICS", "0")

import ray
import torch

from src.config.TM_optimal_config import get_config
from src.models.battle_transformer import configure_safe_sdp_backends
from src.training.env_bridge import (
    collect_env_memory_sentinels,
    collect_recent_episode_stats,
    collect_recent_outcomes,
    collect_selfplay_diagnostics,
)
from src.training.monitoring import SystemMetricsCollector
from src.training.trainer import PokemonTrainer


def _time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def _nvidia_snapshot() -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 3:
                out["gpu_util_pct"] = float(parts[0])
                out["gpu_mem_util_pct"] = float(parts[1])
                out["gpu_mem_used_mb"] = float(parts[2])
    except Exception:
        pass
    return out


def _classify_process(comm: str, args: str) -> Optional[str]:
    blob = f"{comm} {args}".lower()
    if "pokemon-showdown" in blob or "room-battle.js" in blob or "sockets.js" in blob:
        return "showdown"
    if "ray::" in blob or "raylet" in blob or "gcs_server" in blob:
        return "ray"
    if "python" in comm.lower() or "uv" in comm.lower():
        if "profile_training_iteration" in blob:
            return "driver"
        return "python"
    return None


def _cpu_snapshot() -> Dict[str, float]:
    """Sum %CPU by role from `ps`. Values can exceed 100 on multi-core."""
    buckets: Dict[str, float] = defaultdict(float)
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pcpu,comm,args", "--no-headers"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if proc.returncode != 0:
            return {}
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue
            try:
                cpu = float(parts[0])
            except ValueError:
                continue
            comm = parts[1]
            args = parts[2] if len(parts) > 2 else ""
            role = _classify_process(comm, args)
            if role:
                buckets[role] += cpu
    except Exception:
        pass
    return dict(buckets)


class Sampler:
    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if not self.samples:
            return {}
        keys = set()
        for s in self.samples:
            keys.update(s.keys())
        keys.discard("t")
        summary = {}
        for k in sorted(keys):
            vals = [float(s[k]) for s in self.samples if k in s]
            if vals:
                summary[f"{k}_mean"] = sum(vals) / len(vals)
                summary[f"{k}_max"] = max(vals)
        summary["n_samples"] = len(self.samples)
        return summary

    def _run(self) -> None:
        while not self._stop.is_set():
            row: Dict[str, Any] = {"t": time.time()}
            row.update({f"cpu_{k}": v for k, v in _cpu_snapshot().items()})
            row.update(_nvidia_snapshot())
            self.samples.append(row)
            self._stop.wait(self.interval_s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="quick")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--num-servers", type=int, default=None)
    p.add_argument("--start-port", type=int, default=8000)
    p.add_argument(
        "--random-only",
        action="store_true",
        help="Replace curriculum mix with 100% random (no self-play / heuristic).",
    )
    p.add_argument(
        "--output",
        default="logs/profile_training_iteration.json",
        help="Where to write the JSON report.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config(args.preset)
    config.validation.enabled = False
    config.total_timesteps = 1_000_000
    if args.random_only:
        for stage in config.curriculum.stages:
            stage.opponent_mix = {"random": 1.0}

    num_servers = args.num_servers if args.num_servers is not None else config.env.num_servers
    print("=" * 60)
    print("Training iteration profiler")
    print("=" * 60)
    print(f"preset={args.preset}  iterations={args.iterations}")
    print(f"workers={config.env.num_workers}  envs/worker={config.env.num_envs_per_worker}")
    print(f"batch={config.env.num_workers * config.env.num_envs_per_worker} parallel envs")
    print(f"train_batch_size={config.ppo.train_batch_size}  servers={num_servers}")
    print(f"cuda={torch.cuda.is_available()}  gpu_count={torch.cuda.device_count()}")
    if args.random_only:
        print("opponent mix override: random=1.0")
    print("=" * 60, flush=True)

    if torch.cuda.is_available():
        configure_safe_sdp_backends()

    t_ray = time.perf_counter()
    ray.init(
        ignore_reinit_error=True,
        num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
    )
    ray_init_s = time.perf_counter() - t_ray

    trainer = PokemonTrainer(
        config=config,
        preset=args.preset,
        num_servers=num_servers,
        start_port=args.start_port,
        mlflow_experiment_name="profile_local",
    )
    t0 = time.perf_counter()
    trainer._register_environments()
    register_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ppo_config = trainer._build_config()
    trainer.algo = ppo_config.build_algo()
    build_s = time.perf_counter() - t0
    print(f"ray.init={ray_init_s:.1f}s  register={register_s:.1f}s  build_algo={build_s:.1f}s", flush=True)

    if trainer.curriculum:
        trainer._apply_curriculum_stage(trainer.curriculum.current_stage)

    _, export0_s = _time_call(trainer._export_selfplay_weights)
    print(f"initial selfplay export={export0_s:.2f}s", flush=True)

    iterations: List[Dict[str, Any]] = []
    sys_metrics = SystemMetricsCollector()

    for i in range(args.iterations):
        print(f"\n--- iteration {i + 1}/{args.iterations} ---", flush=True)
        sampler = Sampler(interval_s=1.0)
        sampler.start()
        row: Dict[str, Any] = {"iteration": i + 1}

        _, row["export_selfplay_s"] = _time_call(trainer._export_selfplay_weights)
        result, row["algo_train_s"] = _time_call(trainer.train_step)
        env_stats = result.get("env_runners", {}) or {}
        row["env_steps"] = int(result.get("num_env_steps_sampled_lifetime", 0))
        row["episode_len_mean"] = float(env_stats.get("episode_len_mean") or 0.0)
        row["episode_return_mean"] = float(env_stats.get("episode_return_mean") or 0.0)

        _, row["collect_outcomes_s"] = _time_call(collect_recent_outcomes, trainer.algo)
        _, row["collect_episode_stats_s"] = _time_call(
            collect_recent_episode_stats, trainer.algo
        )
        _, row["collect_selfplay_diag_s"] = _time_call(
            collect_selfplay_diagnostics, trainer.algo
        )
        _, row["collect_memory_s"] = _time_call(collect_env_memory_sentinels, trainer.algo)
        _, row["system_metrics_s"] = _time_call(sys_metrics.collect)

        row["driver_rpc_s"] = (
            row["collect_outcomes_s"]
            + row["collect_episode_stats_s"]
            + row["collect_selfplay_diag_s"]
            + row["collect_memory_s"]
            + row["system_metrics_s"]
        )
        row["housekeeping_s"] = row["export_selfplay_s"] + row["driver_rpc_s"]
        row["iteration_total_s"] = row["algo_train_s"] + row["housekeeping_s"]
        if row["algo_train_s"] > 0 and i > 0:
            steps_this = row["env_steps"] - (
                iterations[-1]["env_steps"] if iterations else 0
            )
            row["steps_this_iter"] = steps_this
            row["env_steps_per_s"] = steps_this / row["algo_train_s"]
        else:
            row["steps_this_iter"] = None
            row["env_steps_per_s"] = None

        row["resource_samples"] = sampler.stop()
        iterations.append(row)

        sps = row["env_steps_per_s"]
        sps_txt = f"{sps:.1f}" if sps is not None else "n/a (warmup)"
        print(
            f"train={row['algo_train_s']:.1f}s  export={row['export_selfplay_s']:.2f}s  "
            f"driver_rpc={row['driver_rpc_s']:.2f}s  steps/s={sps_txt}",
            flush=True,
        )
        print(
            f"  outcomes={row['collect_outcomes_s']:.2f}s  "
            f"ep_stats={row['collect_episode_stats_s']:.2f}s  "
            f"selfplay_diag={row['collect_selfplay_diag_s']:.2f}s  "
            f"memory={row['collect_memory_s']:.2f}s  "
            f"sys={row['system_metrics_s']:.2f}s",
            flush=True,
        )
        rs = row["resource_samples"]
        if rs:
            print(
                f"  cpu showdown_mean={rs.get('cpu_showdown_mean', 0):.0f}%  "
                f"ray_mean={rs.get('cpu_ray_mean', 0):.0f}%  "
                f"python_mean={rs.get('cpu_python_mean', 0):.0f}%  "
                f"gpu_util_mean={rs.get('gpu_util_pct_mean', 0):.0f}%",
                flush=True,
            )

    report = {
        "preset": args.preset,
        "random_only": args.random_only,
        "num_workers": config.env.num_workers,
        "num_envs_per_worker": config.env.num_envs_per_worker,
        "num_servers": num_servers,
        "train_batch_size": config.ppo.train_batch_size,
        "cuda": torch.cuda.is_available(),
        "setup_s": {
            "ray_init": ray_init_s,
            "register": register_s,
            "build_algo": build_s,
            "initial_selfplay_export": export0_s,
        },
        "iterations": iterations,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)

    if trainer.algo is not None:
        trainer.algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
