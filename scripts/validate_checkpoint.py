#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.protocols import get_protocol  # noqa: E402
from src.validation.reporting import (  # noqa: E402
    log_validation_to_mlflow,
    write_validation_report,
)
from src.validation.runner import run_validation  # noqa: E402


def main() -> int:
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(
        description="Validate a Pokemon RL checkpoint with a fixed protocol."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path, or 'latest' to resolve from the configured checkpoint dir.",
    )
    parser.add_argument(
        "--protocol",
        choices=["smoke", "fixed_paired", "mirror", "gauntlet_first_loss"],
        default="smoke",
        help="Validation protocol to run.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of protocol episodes.",
    )
    parser.add_argument(
        "--preset",
        choices=["quick", "standard", "memory_safe", "optimal", "large"],
        default="quick",
        help="Training config preset used to rebuild the RLlib module.",
    )
    parser.add_argument(
        "--num-servers",
        type=int,
        default=1,
        help="Number of running Showdown servers available for validation.",
    )
    parser.add_argument(
        "--start-port",
        type=int,
        default=8000,
        help="First Showdown server port.",
    )
    parser.add_argument(
        "--max-steps-per-battle",
        type=int,
        default=500,
        help="Truncate validation battles after this many environment steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Validation RNG seed.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("logs/validation/checkpoint_validation_report.json"),
        help="Path for the local validation JSON report.",
    )
    parser.add_argument(
        "--team-manifest",
        type=Path,
        default=None,
        help="Validation team manifest for fixed-team protocols.",
    )
    parser.add_argument(
        "--battle-format",
        default=None,
        help=(
            "Override validation battle format. For fixed-paired manifests, "
            "the manifest execution_format is used by default."
        ),
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log validation metrics and report artifact to MLflow.",
    )
    parser.add_argument(
        "--mlflow-run-id",
        default=None,
        help="Existing MLflow run ID to log validation metrics into.",
    )
    parser.add_argument(
        "--mlflow-step",
        type=int,
        default=None,
        help="MLflow step for validation metrics.",
    )
    parser.add_argument(
        "--metric-prefix",
        default=None,
        help="Prefix applied to validation metric keys when logging to MLflow.",
    )
    parser.add_argument(
        "--experiment-name",
        default="Pokemon_RL_CheckpointValidation",
        help="MLflow experiment name for checkpoint validation.",
    )
    args = parser.parse_args()

    protocol = get_protocol(args.protocol, episodes=args.episodes)
    if protocol.requires_mlflow and not args.mlflow:
        print(
            f"Protocol '{protocol.name}' is intended to be logged to MLflow; "
            "continuing without MLflow because --mlflow was not passed."
        )

    try:
        report = run_validation(
            protocol=protocol,
            checkpoint=args.checkpoint,
            preset=args.preset,
            num_servers=args.num_servers,
            start_port=args.start_port,
            max_steps_per_battle=args.max_steps_per_battle,
            seed=args.seed,
            team_manifest=str(args.team_manifest) if args.team_manifest else None,
            battle_format=args.battle_format,
        )
    except (ConnectionError, FileNotFoundError, NotImplementedError, RuntimeError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    report_path = write_validation_report(report, args.output_json)

    run_id = None
    if args.mlflow:
        run_id = log_validation_to_mlflow(
            report=report,
            metrics=report["metrics"],
            experiment_name=args.experiment_name,
            run_name=f"{protocol.name}_validation",
            run_id=args.mlflow_run_id,
            step=args.mlflow_step,
            metric_prefix=args.metric_prefix,
        )

    print("=" * 60)
    print("CHECKPOINT VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Report: {report_path}")
    if run_id:
        print(f"MLflow run id: {run_id}")
    print(json.dumps(report["metrics"], indent=2))
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
