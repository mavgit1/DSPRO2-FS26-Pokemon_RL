#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.decision_diagnostics import (  # noqa: E402
    load_samples,
    run_full_analysis,
    run_smoke,
    try_checkpoint_fallback_sample,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze model decision diagnostics (plots + AI-readable reports)."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("logs/validation/decision_diagnostics_samples.json"),
        help="Input diagnostics samples JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/validation/diagnostics_plots"),
        help="Directory for generated PNG plots.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("logs/validation/diagnostics_report.json"),
        help="Structured diagnostics summary for tooling/LLMs.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("logs/validation/diagnostics_report.md"),
        help="Human-readable diagnostics narrative.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional RLlib checkpoint; used when input JSON is missing/empty.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run analyze_observation on random obs (NATIVE_ACTION_SPACE_N=22).",
    )
    args = parser.parse_args()

    if args.smoke:
        diag = run_smoke(args.checkpoint)
        print(json.dumps(diag, indent=2))
        conf = diag.get("decision_confidence", {})
        print(
            f"Smoke OK — top_prob={conf.get('top_prob_mean', 0):.4f} "
            f"margin={conf.get('margin_mean', 0):.4f} "
            f"entropy={conf.get('entropy_mean', 0):.4f}"
        )
        return 0

    file_meta, samples = load_samples(args.input_json)
    if not samples and args.checkpoint is not None:
        fallback = try_checkpoint_fallback_sample(args.checkpoint)
        if fallback is not None:
            samples = [fallback]
            file_meta = {"source": "checkpoint_fallback", "checkpoint": str(args.checkpoint)}

    if not samples:
        print(f"No samples in {args.input_json}; run with --smoke or provide --checkpoint.")
        return 1

    if not args.input_json.exists() and samples:
        args.input_json.parent.mkdir(parents=True, exist_ok=True)
        args.input_json.write_text(
            json.dumps({"meta": file_meta, "samples": samples}, indent=2),
            encoding="utf-8",
        )

    run_full_analysis(
        args.input_json,
        args.output_dir,
        args.report_json,
        args.report_md,
    )

    print(f"Analyzed {len(samples)} samples from {args.input_json}")
    print(f"Plots: {args.output_dir}")
    print(f"Report JSON: {args.report_json}")
    print(f"Report MD: {args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
