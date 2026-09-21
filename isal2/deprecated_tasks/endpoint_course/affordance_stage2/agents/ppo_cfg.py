from dataclasses import dataclass
from isal2.deprecated_tasks.endpoint_course.common.affordance.agents.ppo_cfg import AffordanceAgentCfg


@dataclass
class AffordanceStage2AgentCfg(AffordanceAgentCfg):
    experiment_name: str = "rpo_affordance_stage2"

    def __post_init__(self):
        super().__post_init__()
        self.affordance.update(gate_mode='immediate', warmup_iterations=0, ramp_iterations=0, gate_min_samples=0)
        self.policy['affordance_initial_alpha'] = 1.
