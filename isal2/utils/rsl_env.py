"""ISAL2 environment adapter for the external RSL-RL runner."""
import torch
from tensordict import TensorDict


class RslEnvAdapter:
    def __init__(self, env):
        self.env = env
        self.unwrapped = env.unwrapped
        self.cfg = self.unwrapped.cfg
        self.num_envs = self.unwrapped.num_envs
        self.num_actions = self.unwrapped.num_actions
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self._obs, _ = env.reset()

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf[:] = value

    def get_observations(self):
        return TensorDict(self._obs, batch_size=[self.num_envs])

    def step(self, actions):
        self._obs, rewards, terminated, truncated, extras = self.env.step(actions)
        if "terminal_observation" in extras:
            extras["terminal_observation"] = TensorDict(extras["terminal_observation"], batch_size=[self.num_envs])
        return self.get_observations(), rewards, (terminated | truncated).long(), extras

    def close(self):
        self.env.close()
