import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.columns import Columns
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from src.models.vocab import vocab_sizes

_VOCAB_SIZES = vocab_sizes()


# =============================================================================
# MODEL CONFIG, doesnt actually matter since overwritten by the config file
# =============================================================================

DEFAULT_MODEL_CONFIG = {
    "num_tokens": 13,
    "token_dim": 164,
    "species_vocab_size": _VOCAB_SIZES["species_vocab_size"],
    "item_vocab_size": _VOCAB_SIZES["item_vocab_size"],
    "ability_vocab_size": _VOCAB_SIZES["ability_vocab_size"],
    "embedding_dim": 16,
    "hidden_dim": 256,
    "num_heads": 4,
    "num_transformer_layers": 2,
    "dropout": 0.1,
    "use_position_embeddings": True,
    "use_role_embeddings": True,
    "lstm_hidden": 256,
    "use_lstm": False,
    "max_seq_len": 32,
}


# =============================================================================
# TRANSFORMER MODEL
# =============================================================================


class PokemonTransformerModel(nn.Module):
    """
    Transformer-based model for Pokemon battles with:
    - Categorical embeddings for species, items, abilities
    - Transformer encoder for token interactions
    - Optional LSTM head for cross-turn memory
    - Action masking for valid actions only

    Shape contract:
        Non-stateful (use_lstm=False):
            obs leaves are [B, ...]; logits are [B, A]; values are [B].
        Stateful (use_lstm=True):
            obs leaves are [B, T, ...]; logits are [B, T, A]; values are
            [B, T]; STATE_IN/STATE_OUT carry h/c of shape [B, H].
    """

    def __init__(
        self,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,
    ):
        nn.Module.__init__(self)

        cfg = {**DEFAULT_MODEL_CONFIG, **model_config.get("custom_model_config", {})}

        # Dimensions
        self.num_tokens = cfg["num_tokens"]
        self.token_dim = cfg["token_dim"]
        self.species_vocab_size = cfg["species_vocab_size"]
        self.item_vocab_size = cfg["item_vocab_size"]
        self.ability_vocab_size = cfg["ability_vocab_size"]
        self.embedding_dim = cfg["embedding_dim"]
        self.hidden_dim = cfg["hidden_dim"]
        self.num_heads = cfg["num_heads"]
        self.num_layers = cfg["num_transformer_layers"]
        self.use_position_embeddings = cfg["use_position_embeddings"]
        self.use_role_embeddings = cfg["use_role_embeddings"]
        self.lstm_hidden = cfg["lstm_hidden"]
        self.use_lstm = cfg["use_lstm"]
        self.max_seq_len = cfg["max_seq_len"]

        self.total_token_dim = self.token_dim + 3 * self.embedding_dim

        # -----------------------------------------------------------------
        # Categorical Embeddings
        # -----------------------------------------------------------------
        self.species_embed = nn.Embedding(
            self.species_vocab_size, self.embedding_dim, padding_idx=0
        )
        self.item_embed = nn.Embedding(
            self.item_vocab_size, self.embedding_dim, padding_idx=0
        )
        self.ability_embed = nn.Embedding(
            self.ability_vocab_size, self.embedding_dim, padding_idx=0
        )

        # -----------------------------------------------------------------
        # Input Projection
        # -----------------------------------------------------------------
        self.input_proj = nn.Linear(self.total_token_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)

        self.position_embed = (
            nn.Embedding(self.num_tokens, self.hidden_dim)
            if self.use_position_embeddings
            else None
        )
        self.role_embed = (
            nn.Embedding(5, self.hidden_dim) if self.use_role_embeddings else None
        )
        role_ids = self._build_role_ids(self.num_tokens)
        self.register_buffer("token_role_ids", role_ids, persistent=False)

        # -----------------------------------------------------------------
        # Transformer Encoder
        # -----------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=cfg["dropout"],
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )

        # -----------------------------------------------------------------
        # Cross-team attention bias
        # -----------------------------------------------------------------
        # Learnable bias [num_layers, T, T] added to self-attention scores
        # before softmax.  Initialized to encourage cross-team attention:
        #   CLS (0)       -> opp_active (7)  +2.0
        #   our_active (1)-> opp_active (7)  +2.0
        #   opp_active (7)-> our_active (1)  +1.0
        #   opp_active (7)-> bench (2-6)     +0.5  (switch awareness)
        #   CLS (0)       -> opp_bench (8-12)+0.5  (team awareness)
        # Everything else starts at 0 (learned freely).
        attn_bias = torch.zeros(self.num_layers, self.num_tokens, self.num_tokens)
        for _l in range(self.num_layers):
            b = attn_bias[_l]
            b[0, 7] = 2.0       # CLS -> opp_active
            b[1, 7] = 2.0       # our_active -> opp_active
            b[7, 1] = 1.0       # opp_active -> our_active
            b[7, 2:7] = 0.5     # opp_active -> our bench
            b[0, 8:13] = 0.5    # CLS -> opp bench
        self.attn_bias = nn.Parameter(attn_bias)

        # -----------------------------------------------------------------
        # LSTM (optional) for cross-turn memory.
        # -----------------------------------------------------------------
        if self.use_lstm:
            self.lstm = nn.LSTM(
                self.hidden_dim,
                self.lstm_hidden,
                num_layers=1,
                batch_first=True,
            )
            head_input_dim = self.lstm_hidden
        else:
            self.lstm = None
            head_input_dim = self.hidden_dim

        self.policy_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_outputs),
        )

        self.value_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    # ---------------------------------------------------------------------
    # Stateful API
    # ---------------------------------------------------------------------

    def get_initial_state(self) -> Dict[str, Any]:
        """RLlib expects an *unbatched* initial state ([H]) as numpy arrays.

        The default env-to-module/learner connectors take the dict returned
        here, run `convert_to_numpy` on it, then batch across envs/chunks so
        that STATE_IN reaches `_forward()` with shape [B, H] per leaf.
        """
        if self.use_lstm:
            return {
                "h": np.zeros((self.lstm_hidden,), dtype=np.float32),
                "c": np.zeros((self.lstm_hidden,), dtype=np.float32),
            }
        return {}

    # ---------------------------------------------------------------------
    # Embedding helpers
    # ---------------------------------------------------------------------

    def _embed_obs(self, obs_dict: Dict[str, TensorType]) -> TensorType:
        """Project a flat (B', tokens, ...) obs dict to token embeddings."""
        base_obs = obs_dict["obs"].float()
        species = obs_dict["species"].long()
        items = obs_dict["items"].long()
        abilities = obs_dict["abilities"].long()

        species_emb = self.species_embed(species)
        items_emb = self.item_embed(items)
        abilities_emb = self.ability_embed(abilities)

        x = torch.cat([base_obs, species_emb, items_emb, abilities_emb], dim=-1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self._add_token_structure_embeddings(x)
        return x

    @staticmethod
    def _build_role_ids(num_tokens: int) -> torch.Tensor:
        role_ids = torch.zeros(num_tokens, dtype=torch.long)
        if num_tokens > 1:
            role_ids[1] = 1  # our active
        if num_tokens > 2:
            role_ids[2 : min(num_tokens, 7)] = 2  # our bench
        if num_tokens > 7:
            role_ids[7] = 3  # opponent active
        if num_tokens > 8:
            role_ids[8 : min(num_tokens, 13)] = 4  # opponent bench
        return role_ids

    def _add_token_structure_embeddings(self, x: TensorType) -> TensorType:
        token_count = x.shape[1]
        if self.position_embed is not None:
            position_ids = torch.arange(token_count, device=x.device, dtype=torch.long)
            x = x + self.position_embed(position_ids).unsqueeze(0)
        if self.role_embed is not None:
            role_ids = self.token_role_ids[:token_count].to(device=x.device)
            x = x + self.role_embed(role_ids).unsqueeze(0)
        return x

    def _embed_obs_parts(
        self, obs_dict: Dict[str, TensorType]
    ) -> Dict[str, TensorType]:
        base_obs = obs_dict["obs"].float()
        species = obs_dict["species"].long()
        items = obs_dict["items"].long()
        abilities = obs_dict["abilities"].long()
        return {
            "base_obs": base_obs,
            "species_emb": self.species_embed(species),
            "item_emb": self.item_embed(items),
            "ability_emb": self.ability_embed(abilities),
        }

    def _transformer_forward(self, x: TensorType) -> TensorType:
        """Run transformer with per-layer learnable attention bias.

        Uses manual norm-first decomposition instead of
        ``layer(x, src_mask=bias)`` to avoid a PyTorch 2.10 NaN bug in the
        fused encoder-layer forward path when a float additive mask is passed.
        """
        T = x.shape[1]
        for i, layer in enumerate(self.transformer.layers):
            bias = self.attn_bias[i, :T, :T]  # [T, T]
            # norm-first: x = x + SA(norm1(x)); x = x + FF(norm2(x))
            x_norm = layer.norm1(x)
            sa_out, _ = layer.self_attn(
                x_norm, x_norm, x_norm, attn_mask=bias, need_weights=False
            )
            x = x + layer.dropout1(sa_out)
            x_norm2 = layer.norm2(x)
            ff_out = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(x_norm2)))
            )
            x = x + layer.dropout2(ff_out)
        return x

    @staticmethod
    def _get_cls_token(x: TensorType) -> TensorType:
        return x[:, 0, :]

    # ---------------------------------------------------------------------
    # Diagnostics (single-step, no LSTM)
    # ---------------------------------------------------------------------

    def analyze_observation(
        self,
        obs_dict: Dict[str, TensorType],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        action_mask = obs_dict.get("action_mask")
        parts = self._embed_obs_parts(obs_dict)
        x = torch.cat(
            [
                parts["base_obs"],
                parts["species_emb"],
                parts["item_emb"],
                parts["ability_emb"],
            ],
            dim=-1,
        )
        x = self.input_norm(self.input_proj(x))
        x = self._add_token_structure_embeddings(x)
        x.retain_grad()

        encoded = self._transformer_forward(x)
        cls_token = self._get_cls_token(encoded)

        # When LSTM is enabled, ``policy_head`` expects ``lstm_hidden`` inputs.
        # Run a single LSTM step with zero state so shapes match what the
        # heads were built for. This is a no-memory diagnostic path; the
        # rollout LSTM state lives in the env runner episodes, not here.
        if self.use_lstm:
            B = cls_token.shape[0]
            zeros = cls_token.new_zeros(1, B, self.lstm_hidden)
            lstm_step, _ = self.lstm(cls_token.unsqueeze(1), (zeros, zeros))
            cls_token = lstm_step.squeeze(1)

        logits = self.policy_head(cls_token)
        if action_mask is not None:
            mask = action_mask.clamp(min=1e-8)
            logits = logits - (1.0 - mask) * 1e8

        probs = torch.softmax(logits, dim=-1)
        top_vals, top_idxs = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
        top_prob = top_vals[:, 0]
        runner_up = (
            top_vals[:, 1] if top_vals.shape[-1] > 1 else torch.zeros_like(top_prob)
        )
        margin = top_prob - runner_up
        entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)

        selected_idx = top_idxs[:, 0]
        selected_logit = logits.gather(1, selected_idx.unsqueeze(1)).mean()

        self.zero_grad(set_to_none=True)
        selected_logit.backward()
        token_saliency = x.grad.norm(dim=-1)

        weight = self.input_proj.weight
        base_w = weight[:, : self.token_dim]
        species_w = weight[:, self.token_dim : self.token_dim + self.embedding_dim]
        item_w = weight[
            :,
            self.token_dim + self.embedding_dim : self.token_dim
            + 2 * self.embedding_dim,
        ]
        ability_w = weight[:, self.token_dim + 2 * self.embedding_dim :]

        base_proj = torch.einsum("btd,hd->bth", parts["base_obs"], base_w)
        species_proj = torch.einsum("btd,hd->bth", parts["species_emb"], species_w)
        item_proj = torch.einsum("btd,hd->bth", parts["item_emb"], item_w)
        ability_proj = torch.einsum("btd,hd->bth", parts["ability_emb"], ability_w)

        component_scores = {
            "base_obs": float(base_proj.norm(dim=-1).mean().detach().cpu()),
            "species": float(species_proj.norm(dim=-1).mean().detach().cpu()),
            "item": float(item_proj.norm(dim=-1).mean().detach().cpu()),
            "ability": float(ability_proj.norm(dim=-1).mean().detach().cpu()),
        }

        top_actions = [
            {"action": int(a), "prob": float(p)}
            for a, p in zip(
                top_idxs[0].detach().cpu().tolist(), top_vals[0].detach().cpu().tolist()
            )
        ]

        return {
            "decision_confidence": {
                "top_prob_mean": float(top_prob.mean().detach().cpu()),
                "margin_mean": float(margin.mean().detach().cpu()),
                "entropy_mean": float(entropy.mean().detach().cpu()),
                "top_actions_batch0": top_actions,
            },
            "token_importance": token_saliency.detach().cpu().tolist(),
            "component_importance": component_scores,
        }

    # ---------------------------------------------------------------------
    # Internal compute paths used by the RLModule wrapper
    # ---------------------------------------------------------------------

    def compute_features(
        self,
        obs_dict: Dict[str, TensorType],
        state: Optional[Dict[str, TensorType]] = None,
    ) -> Tuple[TensorType, Dict[str, TensorType], Optional[TensorType]]:
        """Run the (transformer + optional LSTM) trunk.

        Returns:
            features:
                - Stateful: [B, T, lstm_hidden]
                - Non-stateful: [B, hidden_dim]
            new_state: {"h": [B, H], "c": [B, H]} if stateful else {}
            action_mask:
                - Stateful: [B, T, A]
                - Non-stateful: [B, A]
                - None if not provided
        """
        action_mask = obs_dict.get("action_mask", None)

        if not self.use_lstm:
            x = self._embed_obs(obs_dict)
            x = self._transformer_forward(x)
            features = self._get_cls_token(x)
            return features, {}, action_mask

        base_obs = obs_dict["obs"]
        if base_obs.dim() != 4:
            raise ValueError(
                "Stateful PokemonTransformerModel expects obs with a time "
                f"axis [B, T, num_tokens, token_dim]; got shape {tuple(base_obs.shape)}."
            )
        B, T = base_obs.shape[0], base_obs.shape[1]

        flat_obs_dict: Dict[str, TensorType] = {}
        for key, value in obs_dict.items():
            if key == "action_mask":
                continue
            if (
                torch.is_tensor(value)
                and value.dim() >= 3
                and value.shape[:2] == (B, T)
            ):
                flat_obs_dict[key] = value.reshape(B * T, *value.shape[2:])
            else:
                flat_obs_dict[key] = value

        x = self._embed_obs(flat_obs_dict)
        x = self._transformer_forward(x)
        cls = self._get_cls_token(x)
        cls_seq = cls.reshape(B, T, cls.shape[-1])

        h, c = self._extract_state(state, batch_size=B, device=cls_seq.device)
        lstm_out, (new_h, new_c) = self.lstm(cls_seq, (h.unsqueeze(0), c.unsqueeze(0)))
        new_state = {"h": new_h.squeeze(0), "c": new_c.squeeze(0)}
        return lstm_out, new_state, action_mask

    def heads_from_features(
        self,
        features: TensorType,
        action_mask: Optional[TensorType],
    ) -> Tuple[TensorType, TensorType]:
        """Compute (logits, values) given trunk features.

        Stateful path: features [B, T, H_lstm], mask [B, T, A], logits
        [B, T, A], values [B, T].
        Non-stateful path: features [B, H], mask [B, A], logits [B, A],
        values [B].
        """
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        if action_mask is not None:
            mask = action_mask.clamp(min=1e-8)
            logits = logits - (1.0 - mask) * 1e8
        return logits, values

    def _extract_state(
        self,
        state: Optional[Dict[str, TensorType]],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[TensorType, TensorType]:
        """Pull h/c out of a STATE_IN dict and ensure they are on the right device.

        RLlib hands us STATE_IN as ``{"h": [B, H], "c": [B, H]}``. If the
        state is missing (e.g. very first call before the connector populated
        it), fall back to zeros so the module still runs end-to-end.
        """
        if state and "h" in state and "c" in state:
            h = state["h"].to(device=device, dtype=torch.float32)
            c = state["c"].to(device=device, dtype=torch.float32)
            if h.dim() != 2 or c.dim() != 2:
                raise ValueError(
                    "Expected STATE_IN h/c to be 2D [B, H]; got "
                    f"h.shape={tuple(h.shape)}, c.shape={tuple(c.shape)}."
                )
            return h, c
        # Missing state: return zeros instead of raising error
        h = torch.zeros(batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(batch_size, self.lstm_hidden, device=device)
        return h, c


# =============================================================================
# RLlib RLModule (New API stack)
# =============================================================================


class PokemonRLModule(TorchRLModule, ValueFunctionAPI):
    """RLlib RLModule wrapping :class:`PokemonTransformerModel`.

    Implements the recurrent shape contract documented in the model:
      - ``get_initial_state()`` returns an *unbatched* dict of numpy arrays.
      - Inference outputs include the time rank: ACTION_DIST_INPUTS is
        [B, T=1, A] so that RLlib's ``RemoveSingleTsTimeRankFromBatch``
        connector can squeeze it down to [A] per env.
      - Training outputs use [B, T, A] / [B, T] so PPO can apply the
        zero-padding LOSS_MASK and GAE can unpad the value predictions.
      - STATE_OUT carries [B, H] tensors (no time axis).
      - EMBEDDINGS are exposed in ``_forward_train`` so PPO's loss can call
        ``compute_values(batch, embeddings=...)`` without re-running the
        transformer + LSTM.
    """

    def __init__(
        self,
        observation_space=None,
        action_space=None,
        inference_only: bool = False,
        model_config: Optional[Dict[str, Any]] = None,
        catalog_class=None,
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )

    def setup(self):
        obs_space = getattr(self, "observation_space", None)
        action_space = getattr(self, "action_space", None)
        model_cfg = getattr(self, "model_config", None)

        if (obs_space is None or action_space is None or model_cfg is None) and hasattr(
            self, "config"
        ):
            obs_space = obs_space or self.config.observation_space
            action_space = action_space or self.config.action_space
            model_cfg = model_cfg or self.config.model_config_dict

        num_outputs = (
            action_space.n if hasattr(action_space, "n") else action_space.shape[0]
        )

        self.model = PokemonTransformerModel(
            num_outputs=num_outputs,
            model_config=model_cfg or {},
            name="pokemon_transformer",
        )

        # Surface max_seq_len at the top level of model_config so RLlib's
        # AddTimeDimToBatchAndZeroPad learner connector can read it. Without
        # this, the learner pipeline raises:
        #   "You are using a stateful RLModule and are not providing a
        #    'max_seq_len' key inside your `model_config`."
        if (
            isinstance(self.model_config, dict)
            and "max_seq_len" not in self.model_config
        ):
            self.model_config["max_seq_len"] = self.model.max_seq_len

    # ---- Stateful API ---------------------------------------------------

    def get_initial_state(self) -> Dict[str, Any]:
        return self.model.get_initial_state()

    # ---- Forward passes -------------------------------------------------

    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        obs_dict = batch[Columns.OBS]
        state = batch.get(Columns.STATE_IN, None)
        features, new_state, action_mask = self.model.compute_features(obs_dict, state)
        logits, _values = self.model.heads_from_features(features, action_mask)

        output: Dict[str, Any] = {Columns.ACTION_DIST_INPUTS: logits}
        if self.model.use_lstm:
            output[Columns.STATE_OUT] = new_state
        return output

    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        with torch.no_grad():
            return self._forward(batch, **kwargs)

    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        with torch.no_grad():
            return self._forward(batch, **kwargs)

    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        obs_dict = batch[Columns.OBS]
        state = batch.get(Columns.STATE_IN, None)
        features, new_state, action_mask = self.model.compute_features(obs_dict, state)
        logits, values = self.model.heads_from_features(features, action_mask)

        output: Dict[str, Any] = {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: values,
            # Stash the trunk output so PPO's loss can re-use it via
            # ``compute_values(batch, embeddings=...)`` without rerunning the
            # transformer + LSTM.
            Columns.EMBEDDINGS: features,
        }
        if self.model.use_lstm:
            output[Columns.STATE_OUT] = new_state
        return output

    # ---- ValueFunctionAPI ----------------------------------------------

    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[TensorType] = None,
    ) -> TensorType:
        """Recurrent-aware value head.

        Returns:
            Stateful: [B, T] (matches LOSS_MASK + GAE's unpad logic).
            Non-stateful: [B].
        """
        if embeddings is None:
            obs_dict = batch[Columns.OBS]
            state = batch.get(Columns.STATE_IN, None)
            embeddings, _, _ = self.model.compute_features(obs_dict, state)
        values = self.model.value_head(embeddings).squeeze(-1)
        return values

    # ---- Diagnostics ----------------------------------------------------

    def analyze_observation(self, obs_dict, top_k: int = 3):
        return self.model.analyze_observation(obs_dict=obs_dict, top_k=top_k)
