"""ISAL humanoid rough-terrain baseline task."""

import gymnasium as gym

from . import agents, mdp
from .base_config import (
    BaseAgentCfg,
    BaseEnvCfg,
    CommandRangesCfg,
    CommandsCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    RewardCfg,
    RobotCfg,
    SceneContextCfg,
)
from .base_env import ISALHumanoidEnv
from .scene_cfg import SceneCfg
from .terrain_generator_cfg import GRAVEL_TERRAINS_CFG, ROUGH_HARD_TERRAINS_CFG, ROUGH_TERRAINS_CFG


gym.register(
    id="ISAL-Humanoid-Rough-v0",
    entry_point=f"{__name__}.base_env:ISALHumanoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isal_env_cfg:ISALHumanoidRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.isal_agent_cfg:ISALHumanoidRoughAgentCfg",
    },
)

__all__ = [
    "BaseAgentCfg",
    "BaseEnvCfg",
    "CommandRangesCfg",
    "CommandsCfg",
    "EventCfg",
    "GRAVEL_TERRAINS_CFG",
    "HeightScannerCfg",
    "ISALHumanoidEnv",
    "NoiseCfg",
    "NoiseScalesCfg",
    "NormalizationCfg",
    "ObsScalesCfg",
    "ROUGH_HARD_TERRAINS_CFG",
    "ROUGH_TERRAINS_CFG",
    "RewardCfg",
    "RobotCfg",
    "SceneCfg",
    "SceneContextCfg",
    "mdp",
]
