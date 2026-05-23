#!/usr/bin/env python3
"""Reproduce CUDA corruption from decision diagnostics on the live learner module."""

from __future__ import annotations

import os
import sys

import torch

# Repo root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.action_space import NATIVE_ACTION_SPACE_N
from src.models.battle_transformer import PokemonTransformerModel
from src.models.vocab import vocab_sizes


def _fake_obs(device: torch.device) -> dict:
    sizes = vocab_sizes()
    n_tok = 13
    return {
        "obs": torch.randn(4, n_tok, 164, device=device),
        "species": torch.randint(
            0, sizes["species_vocab_size"], (4, n_tok), device=device
        ),
        "items": torch.randint(
            0, sizes["item_vocab_size"], (4, n_tok), device=device
        ),
        "abilities": torch.randint(
            0, sizes["ability_vocab_size"], (4, n_tok), device=device
        ),
        "action_mask": torch.ones(4, NATIVE_ACTION_SPACE_N, device=device),
    }


def ppo_like_step(model: PokemonTransformerModel, obs: dict) -> None:
    """One forward+backward like PPO policy loss."""
    model.train()
    features, _, mask = model.compute_features(obs)
    logits, values = model.heads_from_features(features, mask)
    loss = logits.mean() + values.mean()
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available — skip")
        return

    device = torch.device("cuda")
    cfg = {"custom_model_config": {"hidden_dim": 256, "use_lstm": False}}
    model = PokemonTransformerModel(
        num_outputs=NATIVE_ACTION_SPACE_N, model_config=cfg, name="probe"
    ).to(device)

    obs = _fake_obs(device)

    print("=== Baseline: PPO-like step only ===")
    try:
        ppo_like_step(model, obs)
        torch.cuda.synchronize()
        print("OK")
    except Exception as exc:
        print("FAIL", repr(exc))
        return

    print("=== After saliency backward on same module (training diagnostics path) ===")
    model.train()
    try:
        model.analyze_observation(
            {k: v[:1] for k, v in obs.items()}, top_k=3, compute_saliency=True
        )
        torch.cuda.synchronize()
        print("saliency OK (sync)")
    except Exception as exc:
        print("saliency FAIL", repr(exc))
        return

    print("=== PPO-like step immediately after saliency (expect corruption) ===")
    try:
        ppo_like_step(model, obs)
        torch.cuda.synchronize()
        print("OK (unexpected — saliency did not corrupt)")
    except Exception as exc:
        print("FAIL as expected:", repr(exc))

    print("=== After saliency=False forward WITHOUT inference_mode (subtle path) ===")
    model2 = PokemonTransformerModel(
        num_outputs=NATIVE_ACTION_SPACE_N, model_config=cfg, name="probe2"
    ).to(device)
    obs2 = _fake_obs(device)
    ppo_like_step(model2, obs2)
    model2.train()
    model2.analyze_observation(
        {k: v[:1] for k, v in obs2.items()}, top_k=3, compute_saliency=False
    )
    try:
        ppo_like_step(model2, obs2)
        torch.cuda.synchronize()
        print("OK (no corruption from forward-only diag)")
    except Exception as exc:
        print("FAIL:", repr(exc))


if __name__ == "__main__":
    main()
