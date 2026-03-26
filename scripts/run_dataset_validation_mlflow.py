#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
from dotenv import find_dotenv, load_dotenv

from scripts.validate_bdsp_trainers import load_json, validate_dataset


def main() -> int:
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(
        description="Validate BDSP trainer JSON and log the report to MLflow."
    )
    parser.add_argument(
        "--dataset",
        default="data/bdsp_trainers.json",
        help="Path to trainer dataset JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat empty moves as errors",
    )
    parser.add_argument(
        "--experiment-name",
        default="Pokemon_RL_DataValidation",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    data = load_json(dataset_path)
    report = validate_dataset(data, strict=args.strict)

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        mlflow.set_tag("run_type", "dataset_validation")
        mlflow.set_tag("dataset", str(dataset_path))
        mlflow.log_params(
            {
                "dataset": str(dataset_path),
                "strict": args.strict,
            }
        )

        summary = report.get("summary", {})
        mlflow.log_metrics(
            {
                "trainer_count": float(summary.get("trainer_count", 0)),
                "pokemon_count": float(summary.get("pokemon_count", 0)),
                "error_count": float(summary.get("error_count", 0)),
                "warning_count": float(summary.get("warning_count", 0)),
                "validation_ok": 1.0 if report.get("ok", False) else 0.0,
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            report_path = tmpdir / "validation_report.json"
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(report_path), artifact_path="dataset_validation")

        print("=" * 60)
        print("DATASET VALIDATION COMPLETE")
        print("=" * 60)
        print(json.dumps(report["summary"], indent=2))
        print("=" * 60)

        if report["errors"]:
            print("First errors:")
            for err in report["errors"][:10]:
                print("-", err)

        if report["warnings"]:
            print("First warnings:")
            for warn in report["warnings"][:10]:
                print("-", warn)

    return 0 if report.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
