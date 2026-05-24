from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.action_space import NATIVE_ACTION_SPACE_N
from src.models.embedding import (
    GLOBAL_EXTRA_FEATURE_NAMES,
    GLOBAL_EXTRA_START_IDX,
    NUM_TOKENS,
)

TOKEN_NAMES = [
    "global",
    "our_active",
    "our_bench_1",
    "our_bench_2",
    "our_bench_3",
    "our_bench_4",
    "our_bench_5",
    "opp_active",
    "opp_bench_1",
    "opp_bench_2",
    "opp_bench_3",
    "opp_bench_4",
    "opp_bench_5",
]

OPPONENT_BUCKET_NAMES = [
    "random",
    "random_no_switch",
    "heuristic",
    "self",
    "historical",
    "other",
]


def load_samples(input_json: Path) -> Tuple[Dict[str, Any], List[dict]]:
    if not input_json.exists():
        return {}, []
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("Invalid diagnostics format: 'samples' must be a list.")
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    return meta, samples


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _group_by_iteration(samples: List[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for item in samples:
        it = item.get("iteration")
        if isinstance(it, int):
            grouped[it].append(item)
    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


def decode_opponent_from_obs(obs: np.ndarray) -> Optional[str]:
    """Decode opponent bucket from global token extras at GLOBAL_EXTRA_START_IDX."""
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 2:
        token0 = arr[0]
    else:
        token0 = arr
    start = GLOBAL_EXTRA_START_IDX
    for i, name in enumerate(OPPONENT_BUCKET_NAMES):
        idx = start + i
        if idx < len(token0) and token0[idx] > 0.5:
            return name
    return None


def resolve_opponent_type(row: dict) -> Optional[str]:
    opp = row.get("opponent_type")
    if opp is not None and str(opp).strip():
        return str(opp).strip().lower()
    obs = row.get("obs")
    if obs is not None:
        return decode_opponent_from_obs(np.asarray(obs))
    return None


def classify_top_action(action: int) -> str:
    action = int(action)
    if 0 <= action <= 5:
        return "switch"
    if 6 <= action <= 9:
        return "move"
    if 10 <= action <= 21:
        return "gimmick"
    return "other"


def _extract_confidence(row: dict) -> Dict[str, float]:
    conf = row.get("diagnostics", {}).get("decision_confidence", {})
    out: Dict[str, float] = {}
    for key in ("top_prob_mean", "margin_mean", "entropy_mean"):
        if key in conf:
            out[key] = float(conf[key])
    return out


def _extract_token_vector(row: dict) -> Optional[List[float]]:
    token_importance = row.get("diagnostics", {}).get("token_importance", [])
    if token_importance and isinstance(token_importance, list):
        first = token_importance[0]
        if isinstance(first, list) and first:
            return [float(v) for v in first]
    return None


def _trend_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"first": 0.0, "last": 0.0, "mean": 0.0}
    return {
        "first": float(values[0]),
        "last": float(values[-1]),
        "mean": _mean(values),
    }


def _action_breakdown(samples: List[dict]) -> Dict[str, float]:
    counts = {"switch": 0, "move": 0, "gimmick": 0, "other": 0, "missing": 0}
    for row in samples:
        top_actions = (
            row.get("diagnostics", {})
            .get("decision_confidence", {})
            .get("top_actions_batch0", [])
        )
        if not top_actions:
            counts["missing"] += 1
            continue
        kind = classify_top_action(top_actions[0].get("action", -1))
        counts[kind] = counts.get(kind, 0) + 1
    total = sum(counts.values()) - counts["missing"]
    if total <= 0:
        return {k: 0.0 for k in ("switch_pct", "move_pct", "gimmick_pct", "other_pct")}
    return {
        "switch_pct": 100.0 * counts["switch"] / total,
        "move_pct": 100.0 * counts["move"] / total,
        "gimmick_pct": 100.0 * counts["gimmick"] / total,
        "other_pct": 100.0 * counts["other"] / total,
        "missing_top_action": float(counts["missing"]),
    }


def build_report(
    input_json: Path,
    file_meta: Dict[str, Any],
    samples: List[dict],
) -> Dict[str, Any]:
    grouped = _group_by_iteration(samples)
    iterations = sorted(grouped.keys())
    steps = [
        int(s["total_steps"])
        for s in samples
        if isinstance(s.get("total_steps"), (int, float))
    ]

    conf_by_iter: Dict[str, List[float]] = {
        "top_prob_mean": [],
        "margin_mean": [],
        "entropy_mean": [],
    }
    comp_by_iter: Dict[str, List[float]] = {
        "base_obs": [],
        "species": [],
        "item": [],
        "ability": [],
    }
    token_by_iter: List[List[float]] = []

    for iteration in iterations:
        rows = grouped[iteration]
        conf_rows = [_extract_confidence(r) for r in rows]
        for key in conf_by_iter:
            vals = [c[key] for c in conf_rows if key in c]
            conf_by_iter[key].append(_mean(vals))
        for key in comp_by_iter:
            vals = [
                float(r.get("diagnostics", {}).get("component_importance", {}).get(key, 0))
                for r in rows
                if key in r.get("diagnostics", {}).get("component_importance", {})
            ]
            comp_by_iter[key].append(_mean(vals))
        token_vectors = [_extract_token_vector(r) for r in rows]
        token_vectors = [v for v in token_vectors if v]
        if token_vectors:
            token_by_iter.append(
                np.mean(np.array(token_vectors, dtype=np.float32), axis=0).tolist()
            )

    token_saliency_named: Dict[str, float] = {}
    if token_by_iter:
        mean_tokens = np.mean(np.array(token_by_iter, dtype=np.float32), axis=0)
        for i, name in enumerate(TOKEN_NAMES[: int(mean_tokens.shape[0])]):
            token_saliency_named[name] = float(mean_tokens[i])

    confidence_trends = {k: _trend_stats(v) for k, v in conf_by_iter.items()}
    component_trends = {k: _trend_stats(v) for k, v in comp_by_iter.items()}

    by_opponent: Dict[str, Dict[str, Any]] = {}
    opponent_groups: Dict[str, List[dict]] = defaultdict(list)
    for row in samples:
        opp = resolve_opponent_type(row)
        if opp:
            opponent_groups[opp].append(row)

    for opp, rows in sorted(opponent_groups.items()):
        conf_vals = []
        token_vecs = []
        for r in rows:
            c = _extract_confidence(r)
            if "top_prob_mean" in c:
                conf_vals.append(c["top_prob_mean"])
            tv = _extract_token_vector(r)
            if tv:
                token_vecs.append(tv)
        opp_entry: Dict[str, Any] = {
            "sample_count": len(rows),
            "confidence": {
                "top_prob_mean": _trend_stats(conf_vals),
            },
        }
        if token_vecs:
            mean_t = np.mean(np.array(token_vecs, dtype=np.float32), axis=0)
            opp_entry["token_saliency"] = {
                TOKEN_NAMES[i]: float(mean_t[i])
                for i in range(min(len(TOKEN_NAMES), len(mean_t)))
            }
        by_opponent[opp] = opp_entry

    return {
        "meta": {
            "source": str(input_json),
            "file_meta": file_meta,
            "sample_count": len(samples),
            "iteration_range": [min(iterations), max(iterations)] if iterations else None,
            "total_steps_range": [min(steps), max(steps)] if steps else None,
            "global_extra_start_idx": GLOBAL_EXTRA_START_IDX,
        },
        "confidence_trends": confidence_trends,
        "component_importance_trends": component_trends,
        "token_saliency": token_saliency_named,
        "action_breakdown": _action_breakdown(samples),
        "by_opponent": by_opponent,
        "interpretation": _build_interpretation(
            confidence_trends, token_saliency_named, component_trends
        ),
    }


def _build_interpretation(
    confidence: Dict[str, Dict[str, float]],
    token_saliency: Dict[str, float],
    component_trends: Dict[str, Dict[str, float]],
) -> List[str]:
    notes: List[str] = []
    top_prob = confidence.get("top_prob_mean", {})
    entropy = confidence.get("entropy_mean", {})
    if top_prob:
        if top_prob.get("last", 0) > top_prob.get("first", 0) + 0.05:
            notes.append(
                "Top-action probability rose over training — policy may be becoming more decisive."
            )
        elif top_prob.get("last", 0) < top_prob.get("first", 0) - 0.05:
            notes.append(
                "Top-action probability fell — policy may be exploring more or facing harder opponents."
            )
    if entropy:
        if entropy.get("last", 0) < entropy.get("first", 0) - 0.1:
            notes.append("Entropy decreased — fewer near-tie action distributions.")
        elif entropy.get("last", 0) > entropy.get("first", 0) + 0.1:
            notes.append("Entropy increased — broader action uncertainty at decision time.")

    global_sal = token_saliency.get("global", 0.0)
    opp_sal = token_saliency.get("opp_active", 0.0)
    our_sal = token_saliency.get("our_active", 0.0)
    if global_sal > max(our_sal, opp_sal) * 1.25:
        notes.append(
            "High global-token saliency — model may lean on cheat-sheet / field / opponent-type features."
        )
    if opp_sal > our_sal * 1.25:
        notes.append(
            "High opp_active saliency — decisions may be opponent-matchup driven."
        )
    elif our_sal > opp_sal * 1.25:
        notes.append(
            "High our_active saliency — decisions may prioritize own active Pokemon state."
        )

    base = component_trends.get("base_obs", {})
    species = component_trends.get("species", {})
    if species.get("mean", 0) > base.get("mean", 0) * 1.1:
        notes.append("Species embeddings dominate component importance vs raw obs.")
    if not notes:
        notes.append("No strong trend signals — collect more samples or train longer.")
    return notes


def build_report_markdown(report: Dict[str, Any]) -> str:
    meta = report.get("meta", {})
    lines = [
        "# Model Decision Diagnostics",
        "",
        "## Meta",
        f"- Source: `{meta.get('source', 'n/a')}`",
        f"- Samples: {meta.get('sample_count', 0)}",
        f"- Iteration range: {meta.get('iteration_range')}",
        f"- Step range: {meta.get('total_steps_range')}",
        f"- Global extra start index (token 0): {meta.get('global_extra_start_idx')}",
        "",
        "## Confidence trends",
    ]
    for key, stats in report.get("confidence_trends", {}).items():
        lines.append(
            f"- **{key}**: first={stats.get('first', 0):.4f}, "
            f"last={stats.get('last', 0):.4f}, mean={stats.get('mean', 0):.4f}"
        )
    lines.extend(["", "## Component importance trends"])
    for key, stats in report.get("component_importance_trends", {}).items():
        lines.append(
            f"- **{key}**: first={stats.get('first', 0):.4f}, "
            f"last={stats.get('last', 0):.4f}, mean={stats.get('mean', 0):.4f}"
        )
    lines.extend(["", "## Token saliency (named)"])
    for name, val in report.get("token_saliency", {}).items():
        lines.append(f"- {name}: {val:.4f}")
    ab = report.get("action_breakdown", {})
    lines.extend(
        [
            "",
            "## Top-action breakdown",
            f"- Switch (0-5): {ab.get('switch_pct', 0):.1f}%",
            f"- Move (6-9): {ab.get('move_pct', 0):.1f}%",
            f"- Gimmick (10-21): {ab.get('gimmick_pct', 0):.1f}%",
        ]
    )
    by_opp = report.get("by_opponent", {})
    if by_opp:
        lines.extend(["", "## By opponent"])
        for opp, data in by_opp.items():
            conf = data.get("confidence", {}).get("top_prob_mean", {})
            lines.append(
                f"- **{opp}** (n={data.get('sample_count', 0)}): "
                f"top_prob mean={conf.get('mean', 0):.4f}"
            )
    lines.extend(["", "## Interpretation"])
    for note in report.get("interpretation", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def plot_confidence_trends(grouped: Dict[int, List[dict]], out_path: Path) -> None:
    iterations = []
    top_probs = []
    margins = []
    entropies = []

    for iteration, rows in grouped.items():
        p, m, e = [], [], []
        for row in rows:
            conf = row.get("diagnostics", {}).get("decision_confidence", {})
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


def plot_component_importance(grouped: Dict[int, List[dict]], out_path: Path) -> None:
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


def plot_token_saliency_heatmap(grouped: Dict[int, List[dict]], out_path: Path) -> None:
    iterations = []
    rows_mean = []
    for iteration, rows in grouped.items():
        token_vectors = []
        for row in rows:
            tv = _extract_token_vector(row)
            if tv:
                token_vectors.append(tv)
        if token_vectors:
            iterations.append(iteration)
            rows_mean.append(np.mean(np.array(token_vectors, dtype=np.float32), axis=0))

    if not rows_mean:
        return
    matrix = np.array(rows_mean, dtype=np.float32)

    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Saliency")
    plt.xlabel("Token Index")
    plt.ylabel("Iteration Index")
    plt.title("Token Saliency Heatmap (Mean per Iteration)")
    plt.xticks(range(len(TOKEN_NAMES)), TOKEN_NAMES, rotation=45, ha="right")
    y_ticks = np.linspace(0, len(iterations) - 1, min(8, len(iterations))).astype(int)
    plt.yticks(y_ticks, [str(iterations[i]) for i in y_ticks])
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_plots(samples: List[dict], output_dir: Path) -> None:
    grouped = _group_by_iteration(samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confidence_trends(grouped, output_dir / "confidence_trends.png")
    plot_component_importance(grouped, output_dir / "component_importance_trends.png")
    plot_token_saliency_heatmap(grouped, output_dir / "token_saliency_heatmap.png")


def run_full_analysis(
    input_json: Path,
    plots_dir: Path,
    report_json: Path,
    report_md: Path,
) -> Dict[str, Any]:
    file_meta, samples = load_samples(input_json)
    if not samples:
        raise FileNotFoundError(f"No samples in {input_json}")

    report = build_report(input_json, file_meta, samples)
    run_plots(samples, plots_dir)

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md.write_text(build_report_markdown(report), encoding="utf-8")
    return report


def _load_checkpoint_model(
    checkpoint_dir: Path,
    preset: str = "pure_league_play",
):
    from src.config.TM_optimal_config import get_config
    from src.models.battle_transformer import PokemonTransformerModel

    ckpt = Path(checkpoint_dir)
    candidates = [
        ckpt
        / "learner_group"
        / "learner"
        / "rl_module"
        / "default_policy"
        / "module_state.pkl",
        ckpt / "module_state.pkl",
    ]
    state = None
    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                state = pickle.load(f)
            break
    if state is None:
        raise FileNotFoundError(f"No module_state.pkl under {checkpoint_dir}")

    cleaned: Dict[str, Any] = {}
    for key, val in state.items():
        new_k = key
        for prefix in ("_default_model.", "model.", "policy.model."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix) :]
                break
        cleaned[new_k] = val

    custom_cfg = get_config(preset).model.to_dict()
    model = PokemonTransformerModel(
        num_outputs=NATIVE_ACTION_SPACE_N,
        model_config={"custom_model_config": custom_cfg},
        name="pokemon_transformer_diag",
    )
    model_keys = set(model.state_dict().keys())
    matched = {k: v for k, v in cleaned.items() if k in model_keys}
    if not matched:
        raise RuntimeError("No matching keys between checkpoint and model")
    matched = {k: torch.as_tensor(v) for k, v in matched.items()}
    model.load_state_dict(matched, strict=False)
    model.eval()
    return model


def run_smoke(checkpoint: Optional[Path] = None) -> Dict[str, Any]:
    from src.models.battle_transformer import PokemonTransformerModel
    from src.models.vocab import vocab_sizes

    if checkpoint is not None:
        model = _load_checkpoint_model(checkpoint)
    else:
        model = PokemonTransformerModel(
            num_outputs=NATIVE_ACTION_SPACE_N,
            model_config={"custom_model_config": {}},
            name="pokemon_transformer_diag_smoke",
        )
        model.eval()

    sizes = vocab_sizes()
    obs = {
        "obs": torch.randn(1, NUM_TOKENS, 168),
        "species": torch.randint(0, sizes["species_vocab_size"], (1, NUM_TOKENS)),
        "items": torch.randint(0, sizes["item_vocab_size"], (1, NUM_TOKENS)),
        "abilities": torch.randint(0, sizes["ability_vocab_size"], (1, NUM_TOKENS)),
        "moves": torch.randint(0, sizes["move_vocab_size"], (1, NUM_TOKENS, 4)),
        "last_move": torch.randint(0, sizes["move_vocab_size"], (1, NUM_TOKENS)),
        "action_mask": torch.ones(1, NATIVE_ACTION_SPACE_N),
    }
    return model.analyze_observation(obs_dict=obs, top_k=5)


def try_checkpoint_fallback_sample(checkpoint: Path) -> Optional[dict]:
    try:
        from src.models.vocab import vocab_sizes

        model = _load_checkpoint_model(checkpoint)
        sizes = vocab_sizes()
        obs = {
            "obs": torch.randn(1, NUM_TOKENS, 168),
            "species": torch.randint(0, sizes["species_vocab_size"], (1, NUM_TOKENS)),
            "items": torch.randint(0, sizes["item_vocab_size"], (1, NUM_TOKENS)),
            "abilities": torch.randint(
                0, sizes["ability_vocab_size"], (1, NUM_TOKENS)
            ),
            "moves": torch.randint(0, sizes["move_vocab_size"], (1, NUM_TOKENS, 4)),
            "last_move": torch.randint(0, sizes["move_vocab_size"], (1, NUM_TOKENS)),
            "action_mask": torch.ones(1, NATIVE_ACTION_SPACE_N),
        }
        diag = model.analyze_observation(obs_dict=obs, top_k=5)
        return {
            "iteration": 0,
            "total_steps": 0,
            "diagnostics": diag,
            "source": "checkpoint_smoke",
        }
    except Exception:
        return None
