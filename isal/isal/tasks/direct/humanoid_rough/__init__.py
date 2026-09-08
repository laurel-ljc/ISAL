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
from .height_scan_env import ISALHumanoidHeightScanEnv
from .scene_cfg import SceneCfg
from .terrain_perception_cfg import TerrainPerceptionCfg
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

gym.register(
    id="ISAL-Humanoid-Rough-HeightScan-v0",
    entry_point=f"{__name__}.height_scan_env:ISALHumanoidHeightScanEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isal_env_cfg:ISALHumanoidRoughHeightScanEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.isal_agent_cfg:ISALHumanoidRoughHeightScanAgentCfg",
    },
)

gym.register(
    id="ISAL-Humanoid-Rough-Interaction-v0",
    entry_point=f"{__name__}.interaction_env:ISALHumanoidInteractionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isal_env_cfg:ISALHumanoidRoughInteractionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.isal_agent_cfg:ISALHumanoidRoughInteractionAgentCfg",
    },
)

for _task_suffix, _cfg_suffix in (("CNN", "CNN"), ("CNN-Aux", "CNNAux")):
    gym.register(
        id=f"ISAL-Humanoid-Rough-{_task_suffix}-v0",
        entry_point=f"{__name__}.interaction_env:ISALHumanoidInteractionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.isal_env_cfg:ISALHumanoidRough{_cfg_suffix}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.affordance_agent_cfg:ISALHumanoidRough{_cfg_suffix}AgentCfg",
        },
    )


for _variant in ("AffordanceObs", "AffordanceZero"):
    gym.register(
        id=f"ISAL-Humanoid-Rough-{_variant}-v0",
        entry_point=f"{__name__}.interaction_env:ISALHumanoidInteractionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.isal_env_cfg:ISALHumanoidRoughCNNAuxEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.affordance_agent_cfg:ISALHumanoidRough{_variant}AgentCfg",
        },
    )


for _suffix, _agent in (("CNN", "CNN"), ("CNN-Aux", "CNNAux"),
                         ("AffordanceObs", "AffordanceObs"), ("AffordanceZero", "AffordanceZero")):
    gym.register(
        id=f"ISAL-Humanoid-Rough-{_suffix}-Train-v0",
        entry_point=f"{__name__}.interaction_env:ISALHumanoidInteractionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.isal_env_cfg:ISALHumanoidRough{'CNN' if _suffix == 'CNN' else 'AuxTrain'}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.training_agent_cfg:{_agent}TrainAgentCfg",
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
    "ISALHumanoidHeightScanEnv",
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
    "TerrainPerceptionCfg",
    "mdp",
]
