import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import mlflow

from src.validation.metrics import _canonical_opponent_type, wilson_score_interval


def write_validation_report(report: Dict[str, Any], output_path: Path) -> Path:
    """Write a validation report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def log_validation_to_mlflow(
    report: Dict[str, Any],
    metrics: Dict[str, float],
    experiment_name: str,
    run_name: str | None = None,
    run_id: str | None = None,
    step: int | None = None,
    metric_prefix: str | None = None,
) -> str | None:
    """Log validation metrics and report JSON to MLflow."""
    mlflow.set_experiment(experiment_name)

    start_kwargs: Dict[str, Any] = {"run_name": run_name}
    if run_id:
        start_kwargs = {"run_id": run_id}

    with mlflow.start_run(**start_kwargs) as run:
        if run_id:
            mlflow.set_tag(
                "last_validation_protocol", str(report["metadata"]["protocol"])
            )
            if step is not None:
                mlflow.set_tag("last_validation_step", str(step))
        else:
            mlflow.set_tag("run_type", "checkpoint_validation")
            for key, value in report.get("metadata", {}).items():
                if value is not None:
                    mlflow.set_tag(str(key), str(value))
        mlflow.log_metrics(
            _prefix_metrics(metrics, metric_prefix),
            step=step,
        )

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "validation_report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            artifact_path = _artifact_path(report, step)
            mlflow.log_artifact(str(report_path), artifact_path=artifact_path)

        return run.info.run_id


def _prefix_metrics(
    metrics: Dict[str, float], metric_prefix: str | None
) -> Dict[str, float]:
    if not metric_prefix:
        return metrics

    prefix = metric_prefix.strip("/")
    prefixed = {}
    for key, value in metrics.items():
        metric_key = str(key).strip("/")
        if metric_key.startswith("validation/"):
            metric_key = metric_key.removeprefix("validation/")
        prefixed[f"{prefix}/{metric_key}"] = float(value)
    return prefixed


def _artifact_path(report: Dict[str, Any], step: int | None) -> str:
    protocol = str(report.get("metadata", {}).get("protocol", "unknown"))
    if step is None:
        return f"validation/{protocol}"
    return f"validation/{protocol}/step_{step}"


def format_validation_summary(report: Dict[str, Any], step: int | None = None) -> str:
    """Format a validation report into a human-readable console summary.

    Includes per-opponent win rate bars, confidence intervals for benchmark
    protocol, and composite skill/consistency scores.
    """
    metrics = report.get("metrics", {})
    protocol = report.get("metadata", {}).get("protocol", "unknown")
    episodes = report.get("episodes", [])

    lines: list[str] = []
    lines.append("=" * 62)
    step_label = f" (step {step:,})" if step is not None else ""
    lines.append(f"  VALIDATION SUMMARY — {protocol}{step_label}")
    lines.append("=" * 62)

    # Group episodes by opponent type.
    by_opponent: Dict[str, list[Dict[str, Any]]] = {}
    for ep in episodes:
        opp = _canonical_opponent_type(ep.get("opponent_type"))
        if opp:
            by_opponent.setdefault(opp, []).append(ep)

    bar_width = 20
    for opp in sorted(by_opponent):
        group = by_opponent[opp]
        n = len(group)
        wins = sum(1 for ep in group if ep.get("outcome") == 1)
        wr = wins / n if n else 0.0

        lower, upper = wilson_score_interval(wins, n)
        filled = int(wr * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        label = f"vs {opp}"
        lines.append(
            f"  {label:<22s} WR={wr:6.1%} ({n:3d} eps) "
            f"[{bar}] [{lower:.0%}–{upper:.0%}]"
        )

    # Overall stats.
    total = len(episodes)
    total_wins = sum(1 for ep in episodes if ep.get("outcome") == 1)
    overall_wr = total_wins / total if total else 0.0
    total_steps = sum(ep.get("steps", 0) for ep in episodes)
    avg_len = total_steps / total if total else 0.0
    avg_reward = (
        sum(ep.get("total_reward", 0.0) for ep in episodes) / total if total else 0.0
    )

    lines.append("-" * 62)
    lines.append(f"  {'Overall WR':<22s} {overall_wr:6.1%} ({total} eps)")
    lines.append(f"  {'Avg battle length':<22s} {avg_len:.1f} steps")
    lines.append(f"  {'Avg reward':<22s} {avg_reward:.2f}")

    # Benchmark-specific composite scores.
    if protocol == "benchmark":
        skill = metrics.get("benchmark/skill_score", 0.0)
        consistency = metrics.get("benchmark/consistency", 0.0)
        lines.append(f"  {'Skill score':<22s} {skill:.3f}")
        lines.append(f"  {'Consistency':<22s} {consistency:.1%}")

    lines.append("=" * 62)
    return "\n".join(lines)
