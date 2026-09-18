"""AME changes the actor architecture, retaining base PPO defaults."""
from dataclasses import dataclass
from isal2.tasks.common.base.agents.ppo_cfg import BaseAgentCfg


@dataclass
class AMEAgentCfg(BaseAgentCfg):
    experiment_name: str = "rpo_ame"

    def __post_init__(self):
        self.obs_groups["perception"] = ["height_scan"]
        self.policy.update(
            class_name="isal2.modified_rsl.modules:ActorCriticAME",
            map_shape=(11, 17), map_resolution=0.1, embedding_dim=32,
            cnn_channels=[16, 32, 32], query_hidden_dims=[128], num_heads=4)

    def configure_from_env(self, env_cfg):
        self.policy["map_shape"] = tuple(env_cfg.height_scan_shape)
        self.policy["map_resolution"] = env_cfg.scene_context.height_scanner.resolution
