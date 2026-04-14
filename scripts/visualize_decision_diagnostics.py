#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_samples(input_json: Path) -> list[dict]:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("Invalid diagnostics format: 'samples' must be a list.")
    return samples


def _group_by_iteration(samples: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for item in samples:
        it = item.get("iteration")
        if isinstance(it, int):
            grouped[it].append(item)
    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _plot_confidence_trends(grouped: dict[int, list[dict]], out_path: Path) -> None:
    iterations = []
    top_probs = []
    margins = []
    entropies = []

    for iteration, rows in grouped.items():
        p = []
        m = []
        e = []
        for row in rows:
            diag = row.get("diagnostics", {})
            conf = diag.get("decision_confidence", {})
            if "top_prob_mean" in conf:
                p.append(float(conf["top_prob_mean"]))
            if "margin_mean" in conf:
                m.append(float(conf["margin_mean"]))
            if "entropy_mean" in conf:
                e.append(float(conf["entropy_mean"]))
        if p or m or e:
            iterations.append(iteration)
            top_probs.append(_mean(p))
            margins.append(_mean(m))
            entropies.append(_mean(e))

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, top_probs, label="top_prob_mean")
    plt.plot(iterations, margins, label="margin_mean")
    plt.plot(iterations, entropies, label="entropy_mean")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Decision Confidence Trends")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_component_importance(grouped: dict[int, list[dict]], out_path: Path) -> None:
    keys = ["base_obs", "species", "item", "ability"]
    iterations = []
    series = {k: [] for k in keys}

    for iteration, rows in grouped.items():
        per_key = {k: [] for k in keys}
        for row in rows:
            comp = row.get("diagnostics", {}).get("component_importance", {})
            for key in keys:
                if key in comp:
                    per_key[key].append(float(comp[key]))
        if any(per_key[k] for k in keys):
            iterations.append(iteration)
            for key in keys:
                series[key].append(_mean(per_key[key]))

    plt.figure(figsize=(10, 5))
    for key in keys:
        plt.plot(iterations, series[key], label=key)
    plt.xlabel("Iteration")
    plt.ylabel("Importance (projected norm proxy)")
    plt.title("Input Component Importance Trends")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_token_saliency_heatmap(grouped: dict[int, list[dict]], out_path: Path) -> None:
    iterations = []
    rows_mean = []
    for iteration, rows in grouped.items():
        token_vectors = []
        for row in rows:
            token_importance = row.get("diagnostics", {}).get("token_importance", [])
            if token_importance and isinstance(token_importance, list):
                first = token_importance[0]
                if isinstance(first, list) and first:
                    token_vectors.append([float(v) for v in first])
        if token_vectors:
            iterations.append(iteration)
            rows_mean.append(np.mean(np.array(token_vectors, dtype=np.float32), axis=0))

    if not rows_mean:
        return
    matrix = np.array(rows_mean, dtype=np.float32)  # [iter, token]

    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Saliency")
    plt.xlabel("Token Index")
    plt.ylabel("Iteration Index")
    plt.title("Token Saliency Heatmap (Mean per Iteration)")
    y_ticks = np.linspace(0, len(iterations) - 1, min(8, len(iterations))).astype(int)
    plt.yticks(y_ticks, [str(iterations[i]) for i in y_ticks])
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize decision diagnostics samples.")
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

    samples = _load_samples(args.input_json)
    grouped = _group_by_iteration(samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _plot_confidence_trends(grouped, args.output_dir / "confidence_trends.png")
    _plot_component_importance(grouped, args.output_dir / "component_importance_trends.png")
    _plot_token_saliency_heatmap(grouped, args.output_dir / "token_saliency_heatmap.png")

    print(f"Plotted diagnostics from {args.input_json}")
    print(f"Output directory: {args.output_dir}")
    print("Files:")
    print(" - confidence_trends.png")
    print(" - component_importance_trends.png")
    print(" - token_saliency_heatmap.png")


if __name__ == "__main__":
    main()

