"""Stage 4A/4B perceptive variants: ordinary PPO, with no auxiliary optimization."""

from dataclasses import asdict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from isal.learning.config import (
    AffordanceNetworkCfg, AffordanceObservationCfg, MODEL_CLASS, OBSERVATION_MODEL_CLASS,
)
from .isal_agent_cfg import ISALHumanoidRoughHeightScanAgentCfg


@configclass
class AffordancePolicyCfg(RslRlPpoActorCriticCfg):
    class_name: str = MODEL_CLASS
    observation_layout: dict | None = None
    scan_preprocessing: dict | None = None
    network: dict = asdict(AffordanceNetworkCfg())
    affordance_head_enabled: bool = False
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [256, 128]
    critic_hidden_dims: list[int] = [256, 128]
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    state_dependent_std: bool = False


@configclass
class ISALHumanoidRoughCNNAgentCfg(ISALHumanoidRoughHeightScanAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy = AffordancePolicyCfg()
        # Normalization is explicitly controlled by the policy, not this
        # deprecated runner-level fallback. Scan normalization is always fixed.
        self.empirical_normalization = None
        self.experiment_name = "isal_humanoid_rough_cnn"
        self.neptune_project = self.experiment_name
        self.wandb_project = self.experiment_name


@configclass
class ISALHumanoidRoughCNNAuxAgentCfg(ISALHumanoidRoughCNNAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.affordance_head_enabled = True
        self.experiment_name = "isal_humanoid_rough_cnn_aux"
        self.neptune_project = self.experiment_name
        self.wandb_project = self.experiment_name


@configclass
class AffordanceObservationPolicyCfg(AffordancePolicyCfg):
    class_name: str = OBSERVATION_MODEL_CLASS
    affordance_head_enabled: bool = True
    query_coordinates: list | None = None
    context_scales: dict | None = None
    affordance_observation: dict = asdict(AffordanceObservationCfg())


@configclass
class ISALHumanoidRoughAffordanceObsAgentCfg(ISALHumanoidRoughCNNAuxAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy = AffordanceObservationPolicyCfg()
        self.experiment_name = "isal_humanoid_rough_affordance_obs"
        self.neptune_project = self.experiment_name
        self.wandb_project = self.experiment_name


@configclass
class ISALHumanoidRoughAffordanceZeroAgentCfg(ISALHumanoidRoughAffordanceObsAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.affordance_observation["input_mode"] = "zero"
        self.experiment_name = "isal_humanoid_rough_affordance_zero"
        self.neptune_project = self.experiment_name
        self.wandb_project = self.experiment_name
