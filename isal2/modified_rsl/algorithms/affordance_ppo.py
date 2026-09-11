import torch
from .ppo import PPO


class AffordancePPO(PPO):
    def __init__(self, policy, *args, **kwargs):
        super().__init__(policy, *args, **kwargs)
        self.optimizer = torch.optim.Adam(policy.ppo_parameters(), lr=self.learning_rate)

    def update(self):
        self.policy.affordance_net.zero_grad(set_to_none=True)
        return super().update()
