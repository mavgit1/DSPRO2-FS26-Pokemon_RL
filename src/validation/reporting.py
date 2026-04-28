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
) -> str | None:
    """Log validation metrics and report JSON to MLflow."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_type", "checkpoint_validation")
        for key, value in report.get("metadata", {}).items():
            if value is not None:
                mlflow.set_tag(str(key), str(value))
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "validation_report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(report_path), artifact_path="validation")

        return run.info.run_id
