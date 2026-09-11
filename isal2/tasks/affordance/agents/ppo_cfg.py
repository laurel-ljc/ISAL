from dataclasses import dataclass, field
from isal2.tasks.ame.agents.ppo_cfg import AMEAgentCfg


@dataclass
class AffordanceAgentCfg(AMEAgentCfg):
    experiment_name: str = "rpo_affordance"
    runner_class: str = "isal2.modified_rsl.runners.affordance_runner:AffordanceRunner"
    affordance: dict = field(default_factory=lambda: dict(capacity=65536, max_age=32, batch_size=256,
        gradient_steps=8, min_samples=64, learning_rate=1e-4, max_grad_norm=1.,
        warmup_iterations=500, ramp_iterations=1000, gate_min_samples=256))

    def __post_init__(self):
        super().__post_init__()
        self.policy.update(class_name="isal2.modified_rsl.modules:ActorCriticAffordance", unet_channels=[16, 32, 64])
        self.algorithm["class_name"] = "isal2.modified_rsl.algorithms:AffordancePPO"
