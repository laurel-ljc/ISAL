from dataclasses import dataclass
from isal2.tasks.ame.agents.ppo_cfg import AMEAgentCfg


@dataclass
class AMESparseAgentCfg(AMEAgentCfg):
    experiment_name: str = 'rpo_ame_sparse'

    def __post_init__(self):
        super().__post_init__()
        self.policy['init_noise_std'] = .30
