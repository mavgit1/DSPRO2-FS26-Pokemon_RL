#!/usr/bin/env python3
"""Collect decision-diagnostics samples from real battles for a saved checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import ray
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.decision_diagnostics import run_full_analysis  # noqa: E402
from src.config.TM_optimal_config import get_config  # noqa: E402
from src.training.rllib_config_builder import (  # noqa: E402
    _custom_game_format,
    build_ppo_config,
    register_environments,
)
from src.validation.runner import (  # noqa: E402
    _build_validation_env,
    _compute_action,
    _flat_validation_obs,
    _get_module,
    _restore_checkpoint_for_validation,
    _to_batched_tensors,
)


def _diag_to_json(diag: dict) -> dict:
    out = {}
    for k, v in diag.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().tolist()
        elif isinstance(v, dict):
            out[k] = _diag_to_json(v)
        elif isinstance(v, list):
            out[k] = [
                _diag_to_json(x) if isinstance(x, dict) else x for x in v
            ]
        else:
            out[k] = v
    return out


def _batched_for_analyze(flat: dict, device: torch.device) -> dict:
    batch = _to_batched_tensors(flat, add_time_dim=False)
    return {k: v.to(device) for k, v in batch.items()}


def collect(
    checkpoint: Path,
    preset: str,
    opponents: list[str],
    battles_per_opponent: int,
    sample_every: int,
    max_steps_per_battle: int,
    start_port: int,
    num_servers: int,
) -> list[dict]:
    config = get_config(preset)
    player_team = None
    original_format = config.env.battle_format
    if config.env.player_team_path:
        player_team = (
            Path(config.env.player_team_path)
            .read_text(encoding="utf-8")
            .strip()
        )

    ray.init(ignore_reinit_error=True, num_gpus=1 if torch.cuda.is_available() else 0)
    algo = None
    samples: list[dict] = []
    sample_idx = 0

    try:
        register_environments(
            config=config,
            num_servers=num_servers,
            start_port=start_port,
            initial_stage=None,
        )
        if player_team and "randombattle" in original_format:
            config.env.battle_format = _custom_game_format(original_format)
        algo = build_ppo_config(
            config=config,
            start_port=start_port,
            num_servers=num_servers,
        ).build_algo()
        _restore_checkpoint_for_validation(algo, str(checkpoint.resolve()))

        module = _get_module(algo)
        analyze_fn = getattr(module, "analyze_observation", None)
        if not callable(analyze_fn):
            raise RuntimeError("RLModule has no analyze_observation")
        device = next(module.parameters()).device

        for opp in opponents:
            for battle_i in range(battles_per_opponent):
                port = start_port + (battle_i % max(1, num_servers))
                env = _build_validation_env(
                    config=config,
                    opponent_type=opp,
                    start_port=port,
                    player_team=player_team,
                )
                try:
                    obs, _info = env.reset()
                    steps = 0
                    terminated = truncated = False
                    recurrent_state = None
                    while (
                        not terminated
                        and not truncated
                        and steps < max_steps_per_battle
                    ):
                        flat = _flat_validation_obs(obs)
                        if steps % sample_every == 0:
                            batched = _batched_for_analyze(flat, device)
                            with torch.enable_grad():
                                diag = analyze_fn(
                                    batched, top_k=5, compute_saliency=True
                                )
                            samples.append(
                                {
                                    "iteration": sample_idx // 10,
                                    "total_steps": sample_idx,
                                    "opponent_type": opp,
                                    "battle": battle_i,
                                    "turn": steps,
                                    "obs": np.asarray(
                                        flat["obs"], dtype=np.float32
                                    ).tolist(),
                                    "diagnostics": _diag_to_json(diag),
                                }
                            )
                            sample_idx += 1

                        action, recurrent_state = _compute_action(
                            algo, obs, recurrent_state, explore=False
                        )
                        obs, _reward, terminated, truncated, _info = env.step(
                            int(action)
                        )
                        steps += 1
                    print(
                        f"  {opp} battle {battle_i + 1}/{battles_per_opponent} "
                        f"({steps} turns)",
                        flush=True,
                    )
                finally:
                    env.close()
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()

    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preset", default="pure_league_play")
    parser.add_argument("--opponents", default="random,heuristic")
    parser.add_argument("--battles-per-opponent", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--start-port", type=int, default=8000)
    parser.add_argument("--num-servers", type=int, default=6)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("logs/validation/decision_diagnostics_samples.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/validation/diagnostics_plots"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("logs/validation/diagnostics_report.json"),
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("logs/validation/diagnostics_report.md"),
    )
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if not args.analyze_only:
        opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
        print(
            f"Collecting from {args.checkpoint} | opponents={opponents}",
            flush=True,
        )
        samples = collect(
            checkpoint=args.checkpoint,
            preset=args.preset,
            opponents=opponents,
            battles_per_opponent=args.battles_per_opponent,
            sample_every=args.sample_every,
            max_steps_per_battle=args.max_steps,
            start_port=args.start_port,
            num_servers=args.num_servers,
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "checkpoint": str(args.checkpoint),
                "preset": args.preset,
                "opponents": opponents,
                "battles_per_opponent": args.battles_per_opponent,
                "sample_every": args.sample_every,
            },
            "samples": samples,
        }
        args.output_json.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f"Wrote {len(samples)} samples to {args.output_json}")

    run_full_analysis(
        args.output_json,
        args.output_dir,
        args.report_json,
        args.report_md,
    )
    print(f"Plots: {args.output_dir}")
    print(f"Report: {args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
