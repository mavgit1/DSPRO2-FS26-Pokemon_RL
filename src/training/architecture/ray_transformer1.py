from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import torch

# should use ray.rllib to train this

class RayTransformer1(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # We expect a Dict observation space with obs, species, items, abilities
        original_space = obs_space.original_space
        
        # Dimensions based on RayEmbeddingEnv
        self.max_id_val = 20000 + 1 # +1 to handle 0 as padding/empty
        self.embedding_dim = 16 # Dimensionality of categorical IDs
        
        # Extract shapes
        # original_space["obs"].shape is assumed to be (NUM_TOKENS, TOKEN_DIM) -> (13, 164)
        obs_feat_dim = original_space["obs"].shape[1] 
        self.num_tokens = original_space["obs"].shape[0]
        
        # We have 3 categorical IDs per token (Species, Items, Abilities)
        self.total_token_dim = obs_feat_dim + (3 * self.embedding_dim)

        # Categorical Embeddings
        self.species_embed = nn.Embedding(self.max_id_val, self.embedding_dim, padding_idx=0)
        self.item_embed = nn.Embedding(self.max_id_val, self.embedding_dim, padding_idx=0)
        self.ability_embed = nn.Embedding(self.max_id_val, self.embedding_dim, padding_idx=0)
        
        # Define Custom Transformer Setup
        self.encoder = nn.TransformerEncoderLayer(d_model=self.total_token_dim, nhead=4, batch_first=True)
        
        # The output of the transformer holds sequential token info. 
        # We flatten it to project into num_outputs (which usually matches action_space size)
        self.policy_head = nn.Linear(self.num_tokens * self.total_token_dim, num_outputs)
        self.value_head = nn.Linear(self.num_tokens * self.total_token_dim, 1)

    def forward(self, input_dict, state, seq_lens):
        # Ray packs dict spaces into input_dict["obs"]
        obs_dict = input_dict["obs"]
        
        # Base floats (Batch_size, Num_tokens, Obs_feat_dim)
        base_obs = obs_dict["obs"].float() 
        
        # Categorical IDs (Batch_size, Num_tokens)
        species = obs_dict["species"].long()
        items = obs_dict["items"].long()
        abilities = obs_dict["abilities"].long()

        # Pass through Embeddings -> (Batch_size, Num_tokens, Embedding_dim)
        spec_emb = self.species_embed(species)
        item_emb = self.item_embed(items)
        abil_emb = self.ability_embed(abilities)
        
        # Concatenate everything along the feature dimension -> (Batch, Num_tokens, Total_token_dim)
        x = torch.cat([base_obs, spec_emb, item_emb, abil_emb], dim=-1)
        
        # Pass through multi-headed self-attention layer
        x = self.encoder(x)
        
        # Flatten the tokens into one single vector to map to actions -> (Batch, Num_tokens * Total_token_dim)
        x_flat = x.reshape(x.size(0), -1)

        # Policy and Value Heads
        logits = self.policy_head(x_flat)
        self._value_out = self.value_head(x_flat)
        
        return logits, state

    def value_function(self):
        return torch.reshape(self._value_out, [-1])
        

# Register the model with Ray
ModelCatalog.register_custom_model("ray_transformer1", RayTransformer1)
