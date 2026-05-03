"""Extract and analyze attention patterns from a trained checkpoint.

Hooks into the MultiheadAttention sub-modules inside the TransformerEncoder
to capture per-head, per-layer attention weight matrices. Then runs a few
real battles to collect observations and prints summary statistics.

Usage:
    uv run scripts/debug/inspect_attention.py [--checkpoint PATH] [--num-battles 5]
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

TOKEN_NAMES = [
    "global",       # 0
    "our_active",   # 1
    "bench1",       # 2
    "bench2",       # 3
    "bench3",       # 4
    "bench4",       # 5
    "bench5",       # 6
    "opp_active",   # 7
    "opp_bench1",   # 8
    "opp_bench2",   # 9
    "opp_bench3",   # 10
    "opp_bench4",   # 11
    "opp_bench5",   # 12
]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_dir: str):
    """Load PokemonTransformerModel from a Ray RLlib checkpoint."""
    ckpt = Path(checkpoint_dir)

    # Try Ray's new API stack format:
    # learner_group/learner/rl_module/default_policy/module_state.pkl
    candidates = [
        ckpt / "learner_group" / "learner" / "rl_module" / "default_policy" / "module_state.pkl",
        ckpt / "module_state.pkl",
    ]

    state = None
    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                state = pickle.load(f)
            break

    if state is None:
        raise FileNotFoundError(f"No module_state.pkl in {checkpoint_dir}")

    # state is typically the raw state dict of the RLModule
    # Strip wrapper prefixes to get at the inner model's keys
    cleaned = {}
    for k, v in state.items():
        new_k = k
        for prefix in ("_default_model.", "model.", "policy.model."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        cleaned[new_k] = v

    from src.models.battle_transformer import PokemonTransformerModel

    # Infer number of layers from checkpoint keys
    max_layer = 0
    for k in cleaned.keys():
        import re
        m = re.search(r"transformer\.layers\.(\d+)\.", k)
        if m:
            max_layer = max(max_layer, int(m.group(1)))
    num_layers_inferred = max_layer + 1

    model = PokemonTransformerModel(
        num_outputs=14,
        model_config={"custom_model_config": {
            "embedding_dim": 32,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_transformer_layers": num_layers_inferred,
        }},
        name="pokemon_transformer",
    )

    model_keys = set(model.state_dict().keys())
    matched = {k: v for k, v in cleaned.items() if k in model_keys}

    if not matched:
        print("Keys in checkpoint:")
        for k in sorted(cleaned.keys())[:30]:
            print(f"  {k}: {cleaned[k].shape if hasattr(cleaned[k], 'shape') else type(cleaned[k])}")
        raise RuntimeError("No matching keys between checkpoint and model")

    # Ray serialises weights as numpy arrays — convert to torch tensors
    matched = {k: torch.as_tensor(v) for k, v in matched.items()}

    missing = model_keys - set(matched.keys())
    if missing:
        print(f"Warning: missing keys (random init): {sorted(missing)[:10]}")

    model.load_state_dict(matched, strict=False)
    model.eval()
    print(f"Loaded {len(matched)}/{len(model_keys)} parameters from checkpoint")
    return model


# ---------------------------------------------------------------------------
# Manual attention extraction
# ---------------------------------------------------------------------------

def extract_attention(model, obs_dict):
    """Forward pass with manual attention weight extraction.

    Returns dict[layer_name, np.ndarray] each of shape [H, T, T].
    """
    x = model._embed_obs(obs_dict)  # [1, T, D]

    layer_attns = {}
    for layer_i, layer in enumerate(model.transformer.layers):
        mha = layer.self_attn
        B, T, D = x.shape
        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        head_dim = mha.head_dim

        # Pre-norm (PyTorch TransformerEncoderLayer default is post-norm,
        # but let's check norm_first)
        if getattr(layer, "norm_first", False):
            x_norm = layer.norm1(x)
        else:
            x_norm = x

        # QKV projection
        if mha._qkv_same_embed_dim:
            qkv = torch.nn.functional.linear(x_norm, mha.in_proj_weight, mha.in_proj_bias)
            qkv = qkv.reshape(B, T, 3, embed_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        else:
            q = torch.nn.functional.linear(x_norm, mha.q_proj_weight, mha.bias_q)
            k = torch.nn.functional.linear(x_norm, mha.k_proj_weight, mha.bias_k)
            v = torch.nn.functional.linear(x_norm, mha.v_proj_weight, mha.bias_v)

        q = q.reshape(B, T, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(B, T, num_heads, head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)

        layer_attns[f"layer_{layer_i}"] = attn[0].detach().cpu().numpy()

        # Continue the actual forward pass so later layers get correct input
        v = v.reshape(B, T, num_heads, head_dim).transpose(1, 2) if v.dim() == 3 else v
        v_reshaped = v
        # Need to re-reshape v if we haven't already
        if hasattr(mha, "in_proj_weight") and mha._qkv_same_embed_dim:
            # v was already extracted above, reshape it
            pass
        v_rs = v.reshape(B, T, num_heads, head_dim).transpose(1, 2) \
            if v.dim() == 3 else v

        # Actually, let's just run the layer normally after capturing attention
        with torch.no_grad():
            x = layer(x)

    return layer_attns


def extract_attention_clean(model, obs_dict):
    """Extract per-layer attention weights including attn_bias."""
    x = model._embed_obs(obs_dict)

    has_bias = hasattr(model, "attn_bias")
    layer_attns = {}
    for layer_i, layer in enumerate(model.transformer.layers):
        mha = layer.self_attn
        B, T, D = x.shape
        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        head_dim = mha.head_dim

        norm_first = getattr(layer, "norm_first", False)
        x_norm = layer.norm1(x) if norm_first else x

        # QKV
        if mha._qkv_same_embed_dim:
            qkv = torch.nn.functional.linear(x_norm, mha.in_proj_weight, mha.in_proj_bias)
            qkv = qkv.reshape(B, T, 3, embed_dim)
            q = qkv[:, :, 0].reshape(B, T, num_heads, head_dim).transpose(1, 2)
            k = qkv[:, :, 1].reshape(B, T, num_heads, head_dim).transpose(1, 2)
        else:
            q = torch.nn.functional.linear(x_norm, mha.q_proj_weight, mha.bias_q)
            k = torch.nn.functional.linear(x_norm, mha.k_proj_weight, mha.bias_k)
            q = q.reshape(B, T, num_heads, head_dim).transpose(1, 2)
            k = k.reshape(B, T, num_heads, head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Add learnable cross-team bias
        if has_bias:
            bias = model.attn_bias[layer_i, :T, :T]  # [T, T]
            scores = scores + bias.unsqueeze(0).unsqueeze(0)  # broadcast to [B, H, T, T]

        attn = torch.softmax(scores, dim=-1)
        layer_attns[f"layer_{layer_i}"] = attn[0].detach().cpu().numpy()

        # Run the layer with bias to continue the forward pass correctly
        with torch.no_grad():
            if has_bias:
                x = layer(x, src_mask=model.attn_bias[layer_i, :T, :T])
            else:
                x = layer(x)

    return layer_attns


# ---------------------------------------------------------------------------
# Observation collection
# ---------------------------------------------------------------------------

async def collect_observations(num_battles=3, port=8000):
    """Run real battles against random opponent to collect observations."""
    from poke_env.player import RandomPlayer
    from poke_env.ps_client.account_configuration import AccountConfiguration
    from poke_env.ps_client.server_configuration import ServerConfiguration

    from src.models.embedding import embed_battle

    server_config = ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )

    observations = []

    class CollectPlayer(RandomPlayer):
        def choose_move(self, battle):
            obs = embed_battle(battle)
            observations.append(obs)
            return super().choose_move(battle)

    player = CollectPlayer(
        battle_format="gen5randombattle",
        account_configuration=AccountConfiguration(f"attn_{port}_{id(observations)%10000:04d}", None),
        server_configuration=server_config,
    )

    opponent = RandomPlayer(
        battle_format="gen5randombattle",
        account_configuration=AccountConfiguration(f"attn_opp_{port}_{id(observations)%10000:04d}", None),
        server_configuration=server_config,
    )

    await player.battle_against(opponent, n_battles=num_battles)
    return observations


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def print_attn_bar(pct, width=40):
    if pct != pct:  # NaN check
        return "!" * width
    filled = int(pct / 100 * width)
    return "#" * filled + "-" * (width - filled)


def analyze(obs_samples, all_layer_attns):
    """Print detailed attention analysis."""
    n = len(obs_samples)
    print(f"\n{'='*76}")
    print(f"ATTENTION ANALYSIS  ({n} samples)")
    print(f"{'='*76}")

    # Token presence
    print(f"\nToken presence:")
    for t in range(13):
        present = sum(1 for o in obs_samples if o["obs"][t, 0] > 0.5)
        fainted = sum(1 for o in obs_samples if o["obs"][t, 2] > 0.5)
        hp = [o["obs"][t, 3] for o in obs_samples if o["obs"][t, 0] > 0.5]
        hp_str = f" HP={np.mean(hp):.2f}" if hp else ""
        faint_str = f" (fainted={fainted})" if fainted else ""
        print(f"  {t:2d} {TOKEN_NAMES[t]:14s}: {present:4d}/{n}{faint_str}{hp_str}")

    # Per-layer analysis
    for layer_name in sorted(all_layer_attns.keys()):
        attn = all_layer_attns[layer_name]  # [N, H, T, T]
        N, H, T, _ = attn.shape

        print(f"\n{'─'*76}")
        print(f"  {layer_name}  (heads={H}, samples={N})")
        print(f"{'─'*76}")

        avg = attn.mean(axis=(0, 1))  # [T, T] mean over samples and heads
        avg = np.nan_to_num(avg, nan=0.0)  # handle NaN from numerical issues

        # CLS token attention
        print(f"\n  CLS token (0) attention distribution:")
        for t in range(T):
            pct = avg[0, t] * 100
            print(f"    {t:2d} {TOKEN_NAMES[t]:14s}: {pct:7.3f}%  {print_attn_bar(pct)}")

        # Our active token attention
        print(f"\n  Our active (1) attention distribution:")
        for t in range(T):
            pct = avg[1, t] * 100
            print(f"    {t:2d} {TOKEN_NAMES[t]:14s}: {pct:7.3f}%  {print_attn_bar(pct)}")

        # Opp active token attention
        print(f"\n  Opp active (7) attention distribution:")
        for t in range(T):
            pct = avg[7, t] * 100
            print(f"    {t:2d} {TOKEN_NAMES[t]:14s}: {pct:7.3f}%  {print_attn_bar(pct)}")

        # Per-head analysis for CLS token
        print(f"\n  Per-head CLS (token 0) attention to key tokens:")
        print(f"  {'Head':>6s}  {'our_active':>11s}  {'bench_avg':>9s}  {'opp_active':>10s}  {'opp_bench':>9s}  {'self':>6s}  {'global':>7s}")
        for h in range(H):
            h_avg = attn[:, h, 0, :].mean(axis=0)  # [T]
            vals = {
                "our_active": h_avg[1],
                "bench_avg": np.mean(h_avg[2:7]),
                "opp_active": h_avg[7],
                "opp_bench": np.mean(h_avg[8:13]),
                "self": h_avg[0],
                "global": h_avg[0],
            }
            parts = [f"{v*100:8.2f}%" for v in [
                h_avg[1], np.mean(h_avg[2:7]), h_avg[7], np.mean(h_avg[8:13]), h_avg[0], h_avg[0]
            ]]
            print(f"  {h:6d}  {'  '.join(parts)}")

        # Entropy
        print(f"\n  Attention entropy (lower = more focused):")
        max_ent = np.log(T)
        for q_tok, label in [(0, "CLS"), (1, "Our active"), (7, "Opp active")]:
            eps = 1e-10
            p = avg[q_tok] + eps
            ent = -np.sum(p * np.log(p))
            print(f"    {label:12s}: {ent:.3f} / {max_ent:.2f} ({ent/max_ent*100:.0f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-battles", type=int, default=3)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    # Load model
    if args.random_init or args.synthetic:
        print("Using randomly initialized model")
        from src.models.battle_transformer import PokemonTransformerModel
        model = PokemonTransformerModel(num_outputs=14, model_config={"custom_model_config": {
            "embedding_dim": 32, "hidden_dim": 512, "num_heads": 8,
            "num_transformer_layers": 2,
        }}, name="m")
        model.eval()
    else:
        ckpt = args.checkpoint
        if ckpt is None:
            ckpt_dir = ROOT / "checkpoints"
            ckpt = max(
                (d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step")),
                key=lambda d: d.stat().st_mtime,
                default=None,
            )
            if ckpt is None:
                print("No checkpoints found. Use --random-init or --synthetic")
                return
            ckpt = str(ckpt)
        print(f"Loading: {ckpt}")
        model = load_model(ckpt)

    # Collect observations
    if args.synthetic:
        print("Using synthetic observations")
        obs_samples = _synthetic_obs(10)
    else:
        print(f"Collecting {args.num_battles} battles on port {args.port}...")
        try:
            import asyncio
            obs_samples = asyncio.run(collect_observations(args.num_battles, args.port))
        except Exception as e:
            print(f"Failed: {e}")
            print("Falling back to synthetic observations.")
            obs_samples = _synthetic_obs(10)

    print(f"Got {len(obs_samples)} observations")

    # Extract attention
    all_layer_attns = defaultdict(list)
    for obs in obs_samples:
        obs_t = {
            "obs": torch.from_numpy(obs["obs"]).float().unsqueeze(0),
            "species": torch.from_numpy(obs["species"]).long().unsqueeze(0),
            "items": torch.from_numpy(obs["items"]).long().unsqueeze(0),
            "abilities": torch.from_numpy(obs["abilities"]).long().unsqueeze(0),
        }
        with torch.no_grad():
            layer_attns = extract_attention_clean(model, obs_t)
        for k, v in layer_attns.items():
            all_layer_attns[k].append(v[np.newaxis])

    merged = {k: np.concatenate(v, axis=0) for k, v in all_layer_attns.items()}
    analyze(obs_samples, merged)


def _synthetic_obs(n=10):
    from src.models.embedding import NUM_TOKENS, TOKEN_DIM
    from src.action_space import COMPRESSED_ACTION_SPACE_N

    obs_list = []
    for _ in range(n):
        obs = np.random.randn(NUM_TOKENS, TOKEN_DIM).astype(np.float32) * 0.1
        obs[0, :] = np.random.rand(TOKEN_DIM) * 0.3
        obs[1, 0] = 1.0; obs[1, 1] = 1.0; obs[1, 3] = np.random.rand()
        for t in range(2, 7):
            obs[t, 0] = 1.0; obs[t, 3] = np.random.rand()
            if np.random.rand() < 0.15:
                obs[t, 2] = 1.0; obs[t, 3] = 0.0
        obs[7, 0] = 1.0; obs[7, 1] = 1.0; obs[7, 3] = np.random.rand()
        for t in range(8, 11):
            obs[t, 0] = 1.0; obs[t, 3] = np.random.rand()

        mask = np.zeros(COMPRESSED_ACTION_SPACE_N, dtype=np.float32)
        mask[:4] = 1.0; mask[8:12] = 1.0
        obs_list.append({
            "obs": obs,
            "species": np.random.randint(1, 100, NUM_TOKENS).astype(np.int32),
            "items": np.random.randint(0, 20, NUM_TOKENS).astype(np.int32),
            "abilities": np.random.randint(0, 50, NUM_TOKENS).astype(np.int32),
            "action_mask": mask,
        })
    return obs_list


if __name__ == "__main__":
    main()
