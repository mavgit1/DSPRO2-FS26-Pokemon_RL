from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn

# should use ray.rllib to train this

class RayTransformer1(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Define your custom Transformer layers here
        self.encoder = nn.TransformerEncoderLayer(d_model=obs_space.shape[0], nhead=4)
        self.policy_head = nn.Linear(obs_space.shape[0], num_outputs)
        self.value_head = nn.Linear(obs_space.shape[0], 1)

    def forward(self, input_dict, state, seq_lens):
        # input_dict["obs"] contains the current observations
        x = self.encoder(input_dict["obs"])
        logits = self.policy_head(x)
        self._value_out = self.value_head(x)
        return logits, state

    def value_function(self):
        return self._value_out
        

# Register the model with Ray
ModelCatalog.register_custom_model("ray_transformer1", RayTransformer1)
