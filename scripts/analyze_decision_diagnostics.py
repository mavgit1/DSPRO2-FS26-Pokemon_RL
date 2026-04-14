#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.battle_transformer import PokemonTransformerModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test model confidence and input-importance diagnostics.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("logs/validation/decision_diagnostics_sample.json"),
        help="Output diagnostics JSON path.",
    )
    args = parser.parse_args()

    model = PokemonTransformerModel(
        num_outputs=22,
        model_config={"custom_model_config": {}},
        name="pokemon_transformer_diag_test",
    )
    model.eval()

    obs = {
        "obs": torch.randn(1, 13, 164),
        "species": torch.randint(0, 20000, (1, 13)),
        "items": torch.randint(0, 20000, (1, 13)),
        "abilities": torch.randint(0, 20000, (1, 13)),
        "action_mask": torch.ones(1, 22),
    }
    diag = model.analyze_observation(obs_dict=obs, top_k=5)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(f"Wrote diagnostics to {args.output_json}")
    print(f"Top prob mean: {diag['decision_confidence']['top_prob_mean']:.4f}")
    print(f"Margin mean: {diag['decision_confidence']['margin_mean']:.4f}")
    print(f"Entropy mean: {diag['decision_confidence']['entropy_mean']:.4f}")


if __name__ == "__main__":
    main()

