"""History with independent episode boundaries for vectorized environments."""
import torch


class HistoryBuffer:
    def __init__(self, length, num_envs, dim, device):
        self.buffer = torch.zeros(num_envs, length, dim, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids):
        self.buffer[env_ids] = 0
        self.initialized[env_ids] = False

    def append(self, data, env_ids=None):
        ids = torch.arange(len(self.buffer), device=self.buffer.device) if env_ids is None else env_ids
        old = self.buffer[ids].clone()
        self.buffer[ids, :-1] = old[:, 1:]
        self.buffer[ids, -1] = data[ids]
        first = ids[~self.initialized[ids]]
        self.buffer[first] = data[first, None, :]
        self.initialized[ids] = True

    def flatten(self):
        return self.buffer.flatten(1).clone()
