from dataclasses import dataclass
from isal2.tasks.ame.agents.ppo_cfg import AMEAgentCfg


@dataclass
class AMEStage1AgentCfg(AMEAgentCfg):
    experiment_name: str = "rpo_ame_stage1"
