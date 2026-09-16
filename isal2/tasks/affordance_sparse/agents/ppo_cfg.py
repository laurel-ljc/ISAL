from dataclasses import dataclass
from isal2.tasks.affordance.agents.ppo_cfg import AffordanceAgentCfg


@dataclass
class AffordanceSparseAgentCfg(AffordanceAgentCfg):
    experiment_name: str = 'rpo_affordance_sparse'

    def __post_init__(self):
        super().__post_init__()
        self.policy['init_noise_std'] = .30
