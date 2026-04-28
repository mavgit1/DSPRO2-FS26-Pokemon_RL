import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import mlflow


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
            mlflow.set_tag("last_validation_protocol", str(report["metadata"]["protocol"]))
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


def _prefix_metrics(metrics: Dict[str, float], metric_prefix: str | None) -> Dict[str, float]:
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
