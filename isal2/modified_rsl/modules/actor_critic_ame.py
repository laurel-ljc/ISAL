"""AME actor with the external RSL distribution, normalizers and MLP critic."""
import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP
from .terrain_attention import TerrainAttention


class ActorCriticAME(ActorCritic):
    def __init__(self, obs, obs_groups, num_actions, map_shape=(11, 17), map_resolution=0.1,
                 embedding_dim=32, cnn_channels=(16, 32, 32), query_hidden_dims=(128,), num_heads=4,
                 actor_hidden_dims=(512, 256, 128), activation="elu", state_dependent_std=False, **kwargs):
        if state_dependent_std:
            raise ValueError("AME currently supports only a state-independent action standard deviation")
        if obs_groups["policy"] != ["policy"] or obs_groups["critic"] != ["critic"]:
            raise ValueError("AME requires separate policy, critic and height_scan observations")
        super().__init__(obs, obs_groups, num_actions, actor_hidden_dims=actor_hidden_dims,
                         activation=activation, state_dependent_std=False, **kwargs)
        proprio_dim = obs["policy"].shape[-1]
        self.terrain_attention = TerrainAttention(proprio_dim, map_shape, map_resolution, embedding_dim,
                                                  cnn_channels, query_hidden_dims, num_heads)
        self.terrain_attention.scan_to_image(obs["height_scan"])
        self.actor = MLP(proprio_dim + embedding_dim, num_actions, list(actor_hidden_dims), activation)
        print(f"AME actor replaces the base MLP: {self.actor}\nTerrain encoder: {self.terrain_attention}")

    def _actor_features(self, obs):
        proprio = self.actor_obs_normalizer(self.get_actor_obs(obs))
        terrain = self.terrain_attention(proprio, obs["height_scan"])
        return torch.cat([proprio, terrain], dim=-1)

    def act(self, obs, **kwargs):
        self._update_distribution(self._actor_features(obs))
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.actor(self._actor_features(obs))

    def load_state_dict(self, state_dict, strict=True):
        key = "terrain_attention.position_encoding.coordinates"
        expected = self.terrain_attention.position_encoding.coordinates
        if key not in state_dict or state_dict[key].shape != expected.shape or not torch.equal(state_dict[key].to(expected), expected):
            raise RuntimeError("Incompatible AME checkpoint: terrain geometry differs or this is a Base checkpoint")
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as exc:
            raise RuntimeError(f"Incompatible AME checkpoint; resume requires the same network and observations: {exc}") from exc
