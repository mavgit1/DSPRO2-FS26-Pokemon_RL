import subprocess
from typing import Dict, List, Optional, Tuple

import torch


class SystemMetricsCollector:
    def __init__(self) -> None:
        self._last_cpu_times: Optional[Tuple[int, int]] = None

    def _read_cpu_percent_linux(self) -> Optional[float]:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f:
                first = f.readline().strip()
            if not first.startswith("cpu "):
                return None
            values = [int(x) for x in first.split()[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            total = sum(values)
            if self._last_cpu_times is None:
                self._last_cpu_times = (idle, total)
                return None
            last_idle, last_total = self._last_cpu_times
            self._last_cpu_times = (idle, total)
            delta_total = total - last_total
            delta_idle = idle - last_idle
            if delta_total <= 0:
                return None
            busy = max(0.0, float(delta_total - delta_idle))
            return float(100.0 * (busy / float(delta_total)))
        except OSError:
            return None

    @staticmethod
    def _read_mem_metrics_linux() -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            meminfo: Dict[str, int] = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) != 2:
                        continue
                    key = parts[0].strip()
                    value_part = parts[1].strip().split()[0]
                    meminfo[key] = int(value_part)
            mem_total = meminfo.get("MemTotal")
            mem_available = meminfo.get("MemAvailable")
            if mem_total and mem_available is not None:
                mem_used = max(0, mem_total - mem_available)
                out["sys/ram_used_gb"] = float(mem_used / 1024.0 / 1024.0)
                out["sys/ram_percent"] = float((mem_used / mem_total) * 100.0)
        except OSError:
            return out
        return out

    @staticmethod
    def _collect_gpu_metrics() -> Dict[str, float]:
        out: Dict[str, float] = {
            "sys/cuda_available": 1.0 if torch.cuda.is_available() else 0.0,
            "sys/gpu_count": float(torch.cuda.device_count() if torch.cuda.is_available() else 0),
        }
        if not torch.cuda.is_available():
            return out

        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode == 0 and proc.stdout.strip():
                util_vals: List[float] = []
                mem_pct_vals: List[float] = []
                mem_used_vals: List[float] = []
                for idx, row in enumerate(proc.stdout.strip().splitlines()):
                    parts = [p.strip() for p in row.split(",")]
                    if len(parts) != 3:
                        continue
                    util = float(parts[0])
                    mem_used = float(parts[1])
                    mem_total = float(parts[2])
                    mem_pct = (mem_used / mem_total) * 100.0 if mem_total > 0 else 0.0
                    out[f"sys/gpu{idx}_util_percent"] = util
                    out[f"sys/gpu{idx}_mem_used_mb"] = mem_used
                    out[f"sys/gpu{idx}_mem_percent"] = mem_pct
                    util_vals.append(util)
                    mem_used_vals.append(mem_used)
                    mem_pct_vals.append(mem_pct)
                if util_vals:
                    out["sys/gpu_util_percent"] = float(sum(util_vals) / len(util_vals))
                    out["sys/gpu_mem_used_mb"] = float(sum(mem_used_vals) / len(mem_used_vals))
                    out["sys/gpu_mem_percent"] = float(sum(mem_pct_vals) / len(mem_pct_vals))
        except (OSError, ValueError):
            pass

        for idx in range(torch.cuda.device_count()):
            out[f"sys/gpu{idx}_memory_allocated_mb"] = float(
                torch.cuda.memory_allocated(idx) / (1024.0 * 1024.0)
            )
            out[f"sys/gpu{idx}_memory_reserved_mb"] = float(
                torch.cuda.memory_reserved(idx) / (1024.0 * 1024.0)
            )
        return out

    def collect(self) -> Dict[str, float]:
        metrics = self._collect_gpu_metrics()
        cpu_pct = self._read_cpu_percent_linux()
        if cpu_pct is not None:
            metrics["sys/cpu_percent"] = float(cpu_pct)
        metrics.update(self._read_mem_metrics_linux())
        return metrics
