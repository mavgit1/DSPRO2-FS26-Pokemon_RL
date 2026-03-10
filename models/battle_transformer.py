import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.columns import Columns
from ray.rllib.utils.typing import ModelConfigDict, TensorType


# =============================================================================
# MODEL CONFIG
# =============================================================================

DEFAULT_MODEL_CONFIG = {
    "num_tokens": 13,
    "token_dim": 164,
    "max_id_val": 20001,
    "embedding_dim": 16,
    "hidden_dim": 256,
    "num_heads": 4,
    "num_transformer_layers": 2,
    "dropout": 0.1,
    "lstm_hidden": 256,
    "use_lstm": True,
}


# =============================================================================
# TRANSFORMER MODEL
# =============================================================================

class PokemonTransformerModel(nn.Module):
    """
    Transformer-based model for Pokemon battles with:
    - Categorical embeddings for species, items, abilities
    - Transformer encoder for token interactions
    - LSTM for memory across turns (Not sure if this is the best way to do it)
    - Action masking for valid actions only
    """
    
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        nn.Module.__init__(self)
        
        # Get config
        cfg = {**DEFAULT_MODEL_CONFIG, **model_config.get("custom_model_config", {})}
        
        # Dimensions
        self.num_tokens = cfg["num_tokens"]
        self.token_dim = cfg["token_dim"]
        # +1 because hashed IDs are in [1, max_id_val] and 0 is padding.
        self.max_id_val = cfg["max_id_val"] + 1
        self.embedding_dim = cfg["embedding_dim"]
        self.hidden_dim = cfg["hidden_dim"]
        self.num_heads = cfg["num_heads"]
        self.num_layers = cfg["num_transformer_layers"]
        self.lstm_hidden = cfg["lstm_hidden"]
        self.use_lstm = cfg["use_lstm"]
        
        # Total input dimension per token
        self.total_token_dim = self.token_dim + 3 * self.embedding_dim
        
        # -----------------------------------------------------------------
        # Categorical Embeddings (Please feedback on this)
        # -----------------------------------------------------------------
        self.species_embed = nn.Embedding(
            self.max_id_val, self.embedding_dim, padding_idx=0
        )
        self.item_embed = nn.Embedding(
            self.max_id_val, self.embedding_dim, padding_idx=0
        )
        self.ability_embed = nn.Embedding(
            self.max_id_val, self.embedding_dim, padding_idx=0
        )
        
        # -----------------------------------------------------------------
        # Input Projection
        # -----------------------------------------------------------------
        self.input_proj = nn.Linear(self.total_token_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        
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
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )
        
        # -----------------------------------------------------------------
        # LSTM for Memory (Optional)
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
        
        # -----------------------------------------------------------------
        # Policy Head
        # -----------------------------------------------------------------
        self.policy_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_outputs),
        )
        
        # -----------------------------------------------------------------
        # Value Head
        # -----------------------------------------------------------------
        self.value_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        
        # Store value for value_function()
        self._value_out = None
    
    def get_initial_state(self) -> Dict[str, TensorType]:
        """Get initial LSTM hidden state."""
        if self.use_lstm:
            return {
                "h": torch.zeros(1, self.lstm_hidden),
                "c": torch.zeros(1, self.lstm_hidden),
            }
        return {}
    
    def value_function(self) -> TensorType:
        """Return value function output."""
        return self._value_out.flatten()
    
    def _embed_obs(self, obs_dict: Dict[str, TensorType]) -> TensorType:
        """
        Embed observation dict into token embeddings.
        
        Args:
            obs_dict: Dict with 'obs', 'species', 'items', 'abilities'
        
        Returns:
            Tensor of shape (batch, num_tokens, hidden_dim)
        """
        # Extract components
        base_obs = obs_dict["obs"].float()  # (batch, tokens, token_dim)
        species = obs_dict["species"].long()
        items = obs_dict["items"].long()
        abilities = obs_dict["abilities"].long()
        
        # Get categorical embeddings
        species_emb = self.species_embed(species)  # (batch, tokens, embed_dim)
        items_emb = self.item_embed(items)
        abilities_emb = self.ability_embed(abilities)
        
        # Concatenate all features
        x = torch.cat([base_obs, species_emb, items_emb, abilities_emb], dim=-1)
        
        # Project to hidden dim
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        return x
    
    def _transformer_forward(self, x: TensorType) -> TensorType:
        """Apply transformer encoder."""
        return self.transformer(x)
    
    def _get_cls_token(self, x: TensorType) -> TensorType:
        """Extract CLS token (first token) for downstream tasks."""
        return x[:, 0, :]  # (batch, hidden_dim)
    
    def _apply_lstm(
        self, 
        x: TensorType, 
        state: Dict[str, TensorType]
    ) -> Tuple[TensorType, Dict[str, TensorType]]:
        """Apply LSTM with hidden state management."""
        if not self.use_lstm:
            return x, {}
        
        # x is (batch, hidden_dim), need to add sequence dimension
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # RLModule recurrent state is a dict with "h"/"c", each (batch, hidden).
        h, c = state["h"], state["c"]
        if h.dim() == 2:
            h = h.unsqueeze(0)  # (1, batch, hidden)
        if c.dim() == 2:
            c = c.unsqueeze(0)  # (1, batch, hidden)
        
        # Ensure batch dimension matches
        if h.shape[1] != x.shape[0]:
            h = h.transpose(0, 1)
            c = c.transpose(0, 1)
        
        lstm_out, (new_h, new_c) = self.lstm(x, (h, c))
        
        # Remove sequence dimension
        out = lstm_out.squeeze(1)  # (batch, lstm_hidden)
        
        new_state = {
            "h": new_h.squeeze(0),
            "c": new_c.squeeze(0),
        }
        
        return out, new_state
    
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: Optional[Dict[str, TensorType]],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, Dict[str, TensorType]]:
        """
        Forward pass through the model.
        
        Args:
            input_dict: Contains 'obs' dict with observation components
            state: LSTM hidden state (if using LSTM)
            seq_lens: Sequence lengths (not used for single-step)
        
        Returns:
            Tuple of (action_logits, new_state)
        """
        obs_dict = input_dict["obs"]
        action_mask = obs_dict.get("action_mask", None)
        base_obs = obs_dict["obs"]
        has_time_dim = base_obs.dim() == 4
        batch_size = None
        time_size = None

        # RLlib can provide recurrent batches as [B, T, ...]. The transformer
        # expects [batch_like, tokens, feat], so flatten B*T before encoding.
        if has_time_dim:
            batch_size, time_size = base_obs.shape[:2]
            flat_obs_dict = {}
            for key, value in obs_dict.items():
                if torch.is_tensor(value) and value.dim() >= 3 and value.shape[:2] == (batch_size, time_size):
                    flat_obs_dict[key] = value.reshape(batch_size * time_size, *value.shape[2:])
                else:
                    flat_obs_dict[key] = value
            obs_dict = flat_obs_dict

            if action_mask is not None and torch.is_tensor(action_mask) and action_mask.dim() == 3:
                action_mask = action_mask.reshape(batch_size * time_size, action_mask.shape[-1])
        
        # Embed observations
        x = self._embed_obs(obs_dict)
        
        # Transformer
        x = self._transformer_forward(x)
        
        # Get CLS token
        cls_token = self._get_cls_token(x)
        
        # LSTM (if enabled)
        if self.use_lstm and state and "h" in state and "c" in state:
            if has_time_dim and batch_size is not None and time_size is not None:
                cls_token_seq = cls_token.reshape(batch_size, time_size, cls_token.shape[-1])
                h, c = state["h"], state["c"]
                if h.dim() == 2:
                    h = h.unsqueeze(0)
                if c.dim() == 2:
                    c = c.unsqueeze(0)
                # RLlib may provide [B, 1, H]; PyTorch LSTM expects [1, B, H].
                if h.dim() == 3 and h.shape[0] != 1 and h.shape[1] == 1:
                    h = h.transpose(0, 1)
                if c.dim() == 3 and c.shape[0] != 1 and c.shape[1] == 1:
                    c = c.transpose(0, 1)
                # Last-resort alignment against the current batch dimension.
                if h.dim() == 3 and h.shape[1] != cls_token_seq.shape[0] and h.shape[0] == cls_token_seq.shape[0]:
                    h = h.transpose(0, 1)
                if c.dim() == 3 and c.shape[1] != cls_token_seq.shape[0] and c.shape[0] == cls_token_seq.shape[0]:
                    c = c.transpose(0, 1)
                lstm_out, (new_h, new_c) = self.lstm(cls_token_seq, (h, c))
                cls_token = lstm_out.reshape(batch_size * time_size, lstm_out.shape[-1])
                new_state = {"h": new_h.squeeze(0), "c": new_c.squeeze(0)}
            else:
                cls_token, new_state = self._apply_lstm(cls_token, state)
        else:
            new_state = state if state else self.get_initial_state()
        
        # Policy head
        logits = self.policy_head(cls_token)
        
        # Value head
        self._value_out = self.value_head(cls_token)
        
        # -----------------------------------------------------------------
        # ACTION MASKING
        # -----------------------------------------------------------------
        if action_mask is not None:
            # Mask invalid actions with large negative value
            # This ensures softmax will assign near-zero probability
            # Very important because there are a lot of invalid actions.
            mask = action_mask.clamp(min=1e-8)
            logits = logits - (1.0 - mask) * 1e8
        
        return logits, new_state


# =============================================================================
# RLlib RLModule (New API)
# =============================================================================

class PokemonRLModule(TorchRLModule, ValueFunctionAPI):
    """
    RLlib RLModule using the new API.
    Wraps PokemonTransformerModel for compatibility.
    """
    
    def setup(self):
        """Initialize the model."""
        obs_space = self.config.observation_space
        action_space = self.config.action_space
        
        num_outputs = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        self.model = PokemonTransformerModel(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=self.config.model_config_dict,
            name="pokemon_transformer",
        )
    
    def _forward(self, batch, **kwargs):
        """Shared forward pass."""
        obs_dict = batch[Columns.OBS]
        state = batch.get(Columns.STATE_IN, {})
        if isinstance(state, (list, tuple)) and len(state) == 2:
            # Backward compatibility for older list-based state paths.
            state = {"h": state[0], "c": state[1]}
        seq_lens = batch.get(Columns.SEQ_LENS, None)
        
        # Forward pass through base model
        logits, new_state = self.model(
            input_dict={"obs": obs_dict},
            state=state,
            seq_lens=seq_lens,
        )
        
        output = {Columns.ACTION_DIST_INPUTS: logits}
        
        if self.model.use_lstm:
            output[Columns.STATE_OUT] = new_state
            
        return output
    
    def _forward_train(self, batch, **kwargs):
        """Training forward pass."""
        obs_dict = batch[Columns.OBS]
        state = batch.get(Columns.STATE_IN, {})
        if isinstance(state, (list, tuple)) and len(state) == 2:
            state = {"h": state[0], "c": state[1]}
        seq_lens = batch.get(Columns.SEQ_LENS, None)
        
        logits, new_state = self.model(
            input_dict={"obs": obs_dict},
            state=state,
            seq_lens=seq_lens,
        )
        
        values = self.model.value_function()
        
        output = {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: values,
        }
        
        if self.model.use_lstm:
            output[Columns.STATE_OUT] = new_state
            
        return output
    
    def _forward_inference(self, batch, **kwargs):
        return self._forward(batch, **kwargs)
    
    def _forward_exploration(self, batch, **kwargs):
        return self._forward(batch, **kwargs)
    
    def compute_values(self, batch, embeddings=None):
        """Compute value function for GAE."""
        obs_dict = batch[Columns.OBS]
        base_obs = obs_dict["obs"]
        if base_obs.dim() == 4:
            b, t = base_obs.shape[:2]
            flat_obs_dict = {}
            for key, value in obs_dict.items():
                if torch.is_tensor(value) and value.dim() >= 3 and value.shape[:2] == (b, t):
                    flat_obs_dict[key] = value.reshape(b * t, *value.shape[2:])
                else:
                    flat_obs_dict[key] = value
            obs_dict = flat_obs_dict
        
        x = self.model._embed_obs(obs_dict)
        x = self.model._transformer_forward(x)
        cls_token = self.model._get_cls_token(x)
        
        return self.model.value_head(cls_token).squeeze(-1)
    
    def get_initial_state(self):
        return self.model.get_initial_state()