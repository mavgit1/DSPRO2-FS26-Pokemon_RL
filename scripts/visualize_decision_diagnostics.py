#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.decision_diagnostics import load_samples, run_plots  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize decision diagnostics samples (plot-only wrapper)."
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
        help="Directory for generated plots.",
    )
    args = parser.parse_args()

    _, samples = load_samples(args.input_json)
    if not samples:
        print(f"No samples in {args.input_json}")
        return 1

    run_plots(samples, args.output_dir)
    print(f"Plotted diagnostics from {args.input_json}")
    print(f"Output directory: {args.output_dir}")
    print("Files:")
    print(" - confidence_trends.png")
    print(" - component_importance_trends.png")
    print(" - token_saliency_heatmap.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
