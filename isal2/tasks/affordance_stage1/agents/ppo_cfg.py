from dataclasses import dataclass
from isal2.tasks.common.affordance.agents.ppo_cfg import AffordanceAgentCfg


@dataclass
class AffordanceStage1AgentCfg(AffordanceAgentCfg):
    experiment_name: str = "rpo_affordance_stage1"
