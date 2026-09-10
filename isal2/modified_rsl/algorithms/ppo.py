# SPDX-License-Identifier: BSD-3-Clause
"""Small extensions to external RSL-RL PPO for ISAL2 episode semantics."""
import torch
from rsl_rl.algorithms import PPO as RslPPO


class PPO(RslPPO):
    """Reuse upstream optimization/storage; customize terminal data and normalization."""

    def process_env_step(self, obs, rewards, dones, extras):
        # Keep policy normalizers fixed until the whole PPO update is complete.
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if self.rnd:
            self.rnd.update_normalization(obs)
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards
        timeouts = extras.get("time_outs")
        if timeouts is not None and timeouts.any():
            if "terminal_observation" not in extras:
                raise ValueError("Timeout bootstrap requires pre-reset terminal_observation")
            with torch.no_grad():
                terminal_values = self.policy.evaluate(extras["terminal_observation"].to(self.device)).squeeze(-1)
            self.transition.rewards += self.gamma * terminal_values * timeouts.to(self.device)
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        # Keep a view of observations across upstream storage.clear().
        observations = self.storage.observations.flatten(0, 1)
        losses = super().update()
        self.policy.update_normalization(observations)
        return losses
