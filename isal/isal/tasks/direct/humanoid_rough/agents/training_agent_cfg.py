"""Separate Stage 5 tasks; preserve every Stage 4 configuration."""

from dataclasses import asdict
from copy import deepcopy
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
from isal.learning.training_config import (ALGORITHM_CLASS,RUNNER_CLASS,AuxiliaryLearningCfg,
                                          AffordanceGateScheduleCfg,DiagnosticsCfg)
from .affordance_agent_cfg import (ISALHumanoidRoughCNNAgentCfg,ISALHumanoidRoughCNNAuxAgentCfg,
                                   ISALHumanoidRoughAffordanceObsAgentCfg,ISALHumanoidRoughAffordanceZeroAgentCfg)


@configclass
class AffordanceAlgorithmCfg(RslRlPpoAlgorithmCfg):
    auxiliary_learning: dict = asdict(AuxiliaryLearningCfg())
    affordance_gate_schedule: dict = asdict(AffordanceGateScheduleCfg())
    diagnostics: dict = asdict(DiagnosticsCfg())


def configure_training(agent, auxiliary):
    old = agent.algorithm
    agent.algorithm = AffordanceAlgorithmCfg()
    for key in old.to_dict():
        setattr(agent.algorithm,key,deepcopy(getattr(old,key)))
    agent.algorithm.class_name = ALGORITHM_CLASS
    agent.algorithm.auxiliary_learning["enabled"] = auxiliary
    agent.algorithm.auxiliary_learning["sampling_seed"] = agent.seed
    agent.class_name = RUNNER_CLASS
    agent.experiment_name += "_train"
    agent.neptune_project = agent.wandb_project = agent.experiment_name


@configclass
class CNNTrainAgentCfg(ISALHumanoidRoughCNNAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_training(self,False)


@configclass
class CNNAuxTrainAgentCfg(ISALHumanoidRoughCNNAuxAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_training(self,True)


@configclass
class AffordanceObsTrainAgentCfg(ISALHumanoidRoughAffordanceObsAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_training(self,True)


@configclass
class AffordanceZeroTrainAgentCfg(ISALHumanoidRoughAffordanceZeroAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_training(self,True)
