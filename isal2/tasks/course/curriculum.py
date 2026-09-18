"""One independent ten-level curriculum per environment and terrain type."""
import torch


class CourseCurriculum:
    def __init__(self, num_envs, kinds, device='cpu'):
        self.kinds = tuple(kinds)
        self.levels = torch.zeros(num_envs, len(kinds), dtype=torch.long, device=device)
        self.assigned_type = torch.zeros(num_envs, dtype=torch.long, device=device)

    def sample(self, ids):
        self.assigned_type[ids] = torch.randint(len(self.kinds), (len(ids),), device=self.levels.device)
        return self.assigned_type[ids], self.levels[ids, self.assigned_type[ids]]

    def record(self, ids, success):
        kinds = self.assigned_type[ids]
        self.levels[ids, kinds] = (self.levels[ids, kinds] + 2*success.long()-1).clamp(0, 9)

    def state_dict(self):
        return dict(kinds=self.kinds, levels=self.levels.clone())

    def load_state_dict(self, state):
        levels = state['levels']
        if tuple(state['kinds']) != self.kinds or levels.shape != self.levels.shape:
            raise ValueError('Course resume requires matching terrain types and environment count')
        if levels.dtype != torch.long or not ((levels >= 0) & (levels <= 9)).all():
            raise ValueError('Invalid course levels')
        self.levels.copy_(levels)
