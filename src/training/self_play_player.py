"""Self-play opponent using poke-env action/mask APIs and a frozen policy snapshot per battle."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import Player

from src.action_space import NATIVE_ACTION_SPACE_N
from src.models.battle_transformer import PokemonTransformerModel
from src.models.embedding import embed_battle, get_action_mask


class SelfPlayPlayer(Player):
    """poke-env ``Player`` that mirrors the training policy for self-play.

    Weight sync:
        The trainer exports ``checkpoints/selfplay_latest.pt`` *before* each
        PPO iteration. ``begin_episode()`` loads that snapshot once and keeps
        it fixed for the whole battle so the live learner can be slightly newer.

    poke-env usage:
        - ``SinglesEnv.get_action_mask`` (via ``get_action_mask``)
        - ``SinglesEnv.action_to_order`` for move conversion
        - ``RandomPlayer`` is not used on mapping failures; poke-env handles that
          when ``strict=False``.
    """

    def __init__(
        self,
        model_config_dict: Dict[str, Any],
        weights_path: Optional[str] = None,
        *,
        deterministic: bool = True,
        freeze_weights_per_episode: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._deterministic = deterministic
        self._freeze_weights_per_episode = freeze_weights_per_episode
        self._episode_weights_frozen = False
        self._frozen_state_dict: Optional[Dict[str, torch.Tensor]] = None

        if "custom_model_config" not in model_config_dict:
            model_config_dict = {"custom_model_config": model_config_dict}

        self.model = PokemonTransformerModel(
            num_outputs=NATIVE_ACTION_SPACE_N,
            model_config=model_config_dict,
            name="self_play",
        )
        self.model.eval()

        self._weights_path = weights_path
        self._last_mtime: float = 0.0
        self._load_count = 0

        self._lstm_states: Dict[str, Dict[str, torch.Tensor]] = {}

        self._diag: Dict[str, Any] = {
            "weight_load_count": 0,
            "fallback_count": 0,
            "action_mapping_fallback_count": 0,
            "action_histogram": {},
            "top_prob_sum": 0.0,
            "top_prob_count": 0,
            "entropy_sum": 0.0,
            "entropy_count": 0,
            "valid_action_count_sum": 0,
            "valid_action_count_count": 0,
            "missing_weights_episodes": 0,
        }

        if weights_path:
            self._load_weights(force=True)

    def begin_episode(self) -> None:
        """Load the rollout snapshot once; opponent policy stays fixed for this battle."""
        self._episode_weights_frozen = self._freeze_weights_per_episode
        self._lstm_states.clear()
        if not self._load_weights(force=True):
            self._diag["missing_weights_episodes"] += 1
            self._frozen_state_dict = None
            return
        if self._episode_weights_frozen:
            self._frozen_state_dict = copy.deepcopy(self.model.state_dict())

    def _restore_frozen_weights(self) -> None:
        if self._frozen_state_dict is not None:
            self.model.load_state_dict(self._frozen_state_dict, strict=True)

    def _try_load_weights(self) -> None:
        if self._episode_weights_frozen:
            return
        self._load_weights(force=False)

    def _load_weights(self, *, force: bool) -> bool:
        if not self._weights_path:
            return False
        path = Path(self._weights_path)
        if not path.exists():
            return False
        try:
            mtime = path.stat().st_mtime
            if not force and mtime == self._last_mtime:
                return True
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=True)
            self._last_mtime = mtime
            self._load_count += 1
            self._diag["weight_load_count"] += 1
            return True
        except Exception as exc:
            print(f"[SelfPlayPlayer] FAILED to load weights from {path}: {exc!r}")
            return False

    def choose_move(self, battle: AbstractBattle):
        if self._episode_weights_frozen:
            self._restore_frozen_weights()
        else:
            self._try_load_weights()

        if battle.won or battle.lost:
            self._lstm_states.pop(battle.battle_tag, None)

        if self._frozen_state_dict is None and self._episode_weights_frozen:
            self._diag["fallback_count"] += 1
            return self.choose_random_move(battle)

        try:
            return self._inference_move(battle)
        except Exception as exc:
            self._diag["fallback_count"] += 1
            print(f"[SelfPlayPlayer] inference failed: {exc!r}")
            return self.choose_random_move(battle)

    def _inference_move(self, battle: AbstractBattle):
        obs = embed_battle(battle, opponent_type="self")
        action_mask = get_action_mask(battle)
        obs["action_mask"] = action_mask

        obs_tensors = {
            "obs": torch.as_tensor(obs["obs"], dtype=torch.float32).unsqueeze(0),
            "species": torch.as_tensor(obs["species"], dtype=torch.long).unsqueeze(0),
            "items": torch.as_tensor(obs["items"], dtype=torch.long).unsqueeze(0),
            "abilities": torch.as_tensor(obs["abilities"], dtype=torch.long).unsqueeze(
                0
            ),
            "action_mask": torch.as_tensor(action_mask, dtype=torch.float32).unsqueeze(
                0
            ),
        }

        with torch.no_grad():
            if self.model.use_lstm:
                lstm_obs: Dict[str, Any] = {}
                for key, value in obs_tensors.items():
                    if key == "action_mask":
                        lstm_obs[key] = value
                    else:
                        lstm_obs[key] = value.unsqueeze(1)

                tag = battle.battle_tag
                state = self._lstm_states.get(tag)
                if state is None:
                    state = {
                        "h": torch.zeros(1, self.model.lstm_hidden),
                        "c": torch.zeros(1, self.model.lstm_hidden),
                    }

                features, new_state, mask = self.model.compute_features(lstm_obs, state)
                self._lstm_states[tag] = new_state
                features = features.squeeze(1)
            else:
                features, _, mask = self.model.compute_features(obs_tensors)

            logits, _ = self.model.heads_from_features(features, mask)

        mask_np = np.asarray(action_mask, dtype=np.float32)
        action = self._pick_action(logits.squeeze(0), mask_np)

        valid_count = int(mask_np.sum())
        probs = torch.softmax(logits.squeeze(0), dim=-1)
        top_prob = float(probs.max().item())
        log_probs = torch.log_softmax(logits.squeeze(0), dim=-1)
        entropy = float(-(probs * log_probs).sum().item())

        self._diag["top_prob_sum"] += top_prob
        self._diag["top_prob_count"] += 1
        self._diag["entropy_sum"] += entropy
        self._diag["entropy_count"] += 1
        self._diag["valid_action_count_sum"] += valid_count
        self._diag["valid_action_count_count"] += 1
        self._diag["action_histogram"][action] = (
            self._diag["action_histogram"].get(action, 0) + 1
        )

        order = SinglesEnv.action_to_order(
            np.int64(action), battle, fake=False, strict=False
        )
        if str(order) not in [str(o) for o in battle.valid_orders]:
            self._diag["action_mapping_fallback_count"] += 1
        return order

    def _pick_action(self, logits: torch.Tensor, action_mask: np.ndarray) -> int:
        n = min(logits.shape[-1], len(action_mask), NATIVE_ACTION_SPACE_N)
        masked_logits = logits[:n].clone()
        mask_t = torch.as_tensor(action_mask[:n], dtype=torch.float32)
        masked_logits = masked_logits.masked_fill(mask_t <= 0, -1e8)
        if self._deterministic:
            return int(masked_logits.argmax(dim=-1).item())
        probs = torch.softmax(masked_logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def pop_diagnostics(self) -> Dict[str, Any]:
        diag = dict(self._diag)
        diag["action_histogram"] = dict(self._diag["action_histogram"])
        self._diag.update(
            {
                "weight_load_count": 0,
                "fallback_count": 0,
                "action_mapping_fallback_count": 0,
                "action_histogram": {},
                "top_prob_sum": 0.0,
                "top_prob_count": 0,
                "entropy_sum": 0.0,
                "entropy_count": 0,
                "valid_action_count_sum": 0,
                "valid_action_count_count": 0,
                "missing_weights_episodes": 0,
            }
        )
        return diag
