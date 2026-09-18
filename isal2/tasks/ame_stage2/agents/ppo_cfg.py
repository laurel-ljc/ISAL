from dataclasses import dataclass
from isal2.tasks.common.ame.agents.ppo_cfg import AMEAgentCfg


@dataclass
class AMEStage2AgentCfg(AMEAgentCfg):
    experiment_name: str = "rpo_ame_stage2"
