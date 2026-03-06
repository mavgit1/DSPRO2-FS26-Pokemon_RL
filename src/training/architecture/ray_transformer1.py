from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.columns import Columns
import torch.nn as nn
import torch


class RayTransformer1RLModule(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        # We expect a Dict observation space with obs, species, items, abilities
        # (ray_embedding.py)
        obs_space = self.config.observation_space

        # Action space for output dimension
        num_outputs = self.config.action_space.n if hasattr(self.config.action_space, 'n') else self.config.action_space.shape[0]

        # Dimensions based on RayEmbeddingEnv
        self.max_id_val = 20000 + 1  # +1 to handle 0 as padding/empty
        self.embedding_dim = 16

        # Extract shapes
        obs_feat_dim = obs_space["obs"].shape[1]
        self.num_tokens = obs_space["obs"].shape[0]

        # We have 3 categorical IDs per token (Species, Items, Abilities)
        self.total_token_dim = obs_feat_dim + (3 * self.embedding_dim)

        # Categorical Embeddings
        self.species_embed = nn.Embedding(self.max_id_val, self.embedding_dim, padding_idx=0)
        self.item_embed = nn.Embedding(self.max_id_val, self.embedding_dim, padding_idx=0)
        self.ability_embed = nn.Embedding(self.max_id_val, self.embedding_dim, padding_idx=0)

        # Transformer backbone
        self.transformer = nn.TransformerEncoderLayer(
            d_model=self.total_token_dim, nhead=4, batch_first=True
        )

        # Policy and Value heads
        flat_dim = self.num_tokens * self.total_token_dim
        self.policy_head = nn.Linear(flat_dim, num_outputs)
        self.value_head = nn.Linear(flat_dim, 1)

    def _encode(self, batch):
        """Shared backbone: embed obs dict -> transformer -> flat vector."""
        obs_dict = batch[Columns.OBS]

        base_obs = obs_dict["obs"].float()
        species = obs_dict["species"].long()
        items = obs_dict["items"].long()
        abilities = obs_dict["abilities"].long()

        spec_emb = self.species_embed(species)
        item_emb = self.item_embed(items)
        abil_emb = self.ability_embed(abilities)

        x = torch.cat([base_obs, spec_emb, item_emb, abil_emb], dim=-1)
        x = self.transformer(x)
        return x.reshape(x.size(0), -1)

    def _forward(self, batch, **kwargs):
        x_flat = self._encode(batch)
        logits = self.policy_head(x_flat)
        return {Columns.ACTION_DIST_INPUTS: logits}

    def _forward_train(self, batch, **kwargs):
        x_flat = self._encode(batch)
        logits = self.policy_head(x_flat)
        vf_preds = self.value_head(x_flat).squeeze(-1)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: vf_preds,
        }

    def _forward_inference(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    def _forward_exploration(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    def compute_values(self, batch, embeddings=None):
        """Required by ValueFunctionAPI for GAE computation in PPO."""
        x_flat = self._encode(batch)
        return self.value_head(x_flat).squeeze(-1)
