"""Self-play opponent that uses a local copy of the training model for inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player import Player, RandomPlayer

from src.action_space import (
    COMPRESSED_ACTION_SPACE_N,
    COMPRESSED_MOVE_ACTIONS,
    COMPRESSED_SWITCH_ACTIONS,
    get_compressed_action_mask,
)
from src.models.battle_transformer import PokemonTransformerModel
from src.models.embedding import embed_battle


class SelfPlayPlayer(Player):
    """poke-env Player whose ``choose_move()`` runs the training model locally.

    At each checkpoint the trainer exports the raw model state dict to
    ``checkpoints/selfplay_latest.pt``.  This player loads it on first use and
    re-checks the file each turn so it picks up new weights automatically.
    """

    def __init__(
        self,
        model_config_dict: Dict[str, Any],
        weights_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Ensure the config has the nested ``custom_model_config`` key that
        # ``PokemonTransformerModel.__init__`` expects.
        if "custom_model_config" not in model_config_dict:
            model_config_dict = {"custom_model_config": model_config_dict}

        self.model = PokemonTransformerModel(
            num_outputs=COMPRESSED_ACTION_SPACE_N,
            model_config=model_config_dict,
            name="self_play",
        )
        self.model.eval()

        self._weights_path = weights_path
        self._last_mtime: float = 0.0
        self._load_count = 0

        # Per-battle LSTM state cache for cross-turn memory.
        # Keyed by battle_tag; values are {"h": Tensor, "c": Tensor}.
        self._lstm_states: Dict[str, Dict[str, torch.Tensor]] = {}

        if weights_path:
            self._try_load_weights()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _try_load_weights(self) -> None:
        if not self._weights_path:
            return
        path = Path(self._weights_path)
        if not path.exists():
            return
        try:
            mtime = path.stat().st_mtime
            if mtime == self._last_mtime:
                return
            state_dict = torch.load(
                path, map_location="cpu", weights_only=True
            )
            self.model.load_state_dict(state_dict, strict=True)
            self._last_mtime = mtime
            self._lstm_states.clear()  # stale state incompatible with new weights
            self._load_count += 1
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def choose_move(self, battle: AbstractBattle):
        self._try_load_weights()

        # Prune LSTM cache for finished battles
        if battle.won or battle.lost:
            self._lstm_states.pop(battle.battle_tag, None)

        try:
            return self._inference_move(battle)
        except Exception:
            return RandomPlayer.choose_random_singles_move(battle)

    def _inference_move(self, battle: AbstractBattle):
        # 1. Embed battle -> obs dict
        obs = embed_battle(battle, opponent_type=None)
        action_mask = get_compressed_action_mask(battle)
        obs["action_mask"] = action_mask

        # 2. Convert to tensors with batch dim [1, ...]
        obs_tensors = {
            "obs": torch.as_tensor(obs["obs"], dtype=torch.float32).unsqueeze(0),
            "species": torch.as_tensor(obs["species"], dtype=torch.long).unsqueeze(0),
            "items": torch.as_tensor(obs["items"], dtype=torch.long).unsqueeze(0),
            "abilities": torch.as_tensor(obs["abilities"], dtype=torch.long).unsqueeze(0),
            "action_mask": torch.as_tensor(action_mask, dtype=torch.float32).unsqueeze(0),
        }

        # 3. Forward pass through the model trunk + heads
        with torch.no_grad():
            if self.model.use_lstm:
                # Add time dim T=1 for LSTM path
                lstm_obs: Dict[str, Any] = {}
                for k, v in obs_tensors.items():
                    if k == "action_mask":
                        lstm_obs[k] = v
                    else:
                        lstm_obs[k] = v.unsqueeze(1)

                # Look up cached LSTM state for this battle, or use zeros
                tag = battle.battle_tag
                state = self._lstm_states.get(tag, None)
                if state is None:
                    state = {
                        "h": torch.zeros(1, self.model.lstm_hidden),
                        "c": torch.zeros(1, self.model.lstm_hidden),
                    }

                features, new_state, mask = self.model.compute_features(lstm_obs, state)
                # Cache updated state for next turn
                self._lstm_states[tag] = new_state
                features = features.squeeze(1)
            else:
                features, _, mask = self.model.compute_features(obs_tensors)

            logits, _ = self.model.heads_from_features(features, mask)

        # 4. Argmax -> compressed action
        action = int(logits.argmax(dim=-1).item())

        # 5. Convert to BattleOrder
        return self._action_to_order(action, battle)

    # ------------------------------------------------------------------
    # Action -> BattleOrder conversion
    # ------------------------------------------------------------------

    def _action_to_order(self, action: int, battle: AbstractBattle):
        active = battle.active_pokemon

        # Move actions (compressed 0-3)
        if action in COMPRESSED_MOVE_ACTIONS:
            if active and active.moves:
                known_moves = list(active.moves.values())
                slot = action  # 0, 1, 2, or 3
                if slot < len(known_moves):
                    return self.create_order(known_moves[slot])

        # Gimmick actions (compressed 4-7) - treat as regular move
        elif 4 <= action < 8:
            if active and active.moves:
                known_moves = list(active.moves.values())
                slot = action - 4
                if slot < len(known_moves):
                    return self.create_order(known_moves[slot])

        # Switch actions (compressed 8-13)
        elif action in COMPRESSED_SWITCH_ACTIONS:
            switch_idx = action - COMPRESSED_SWITCH_ACTIONS.start
            team_list = list(battle.team.values())
            bench = [mon for mon in team_list if mon is not active]
            if switch_idx < len(bench):
                return self.create_order(bench[switch_idx])

        # Fallback
        return RandomPlayer.choose_random_singles_move(battle)
