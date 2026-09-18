"""Per-environment competence bounded by independent per-type validation gates."""
import copy
import torch
from .terrain_cfg import TYPES, REVIEW


class SparseCurriculum:
    def __init__(self, kinds, max_level=9, replay=.2, device='cpu', beam_limit=6):
        self.kinds = list(kinds)
        self.max_level, self.replay = max_level, replay
        self.type_limits = {k: min(max_level, beam_limit) if k in ('single_beam', 'radial_beams') else max_level for k in TYPES}
        n = len(kinds)
        self.ability = torch.zeros(n, dtype=torch.long, device=device)
        self.assigned = self.ability.clone()
        self.success_streak = self.ability.clone()
        self.failure_streak = self.ability.clone()
        self.is_replay = torch.zeros(n, dtype=torch.bool, device=device)
        self.unlocked = {k: 0 for k in TYPES}
        self.validation = []
        self.gate_streak = {}
        self.last_evaluation = -1
        self.flat_baseline = None
        self.entry_report = None

    def record(self, ids, success):
        review = torch.tensor([self.kinds[i] in REVIEW for i in ids.tolist()], device=ids.device, dtype=torch.bool)
        eligible = ~self.is_replay[ids] & (self.assigned[ids] == self.ability[ids]) & ~review
        ids, success = ids[eligible], success[eligible]
        self.success_streak[ids] = torch.where(success, self.success_streak[ids]+1, 0)
        self.failure_streak[ids] = torch.where(success, 0, self.failure_streak[ids]+1)
        up, down = self.success_streak[ids] >= 2, self.failure_streak[ids] >= 2
        cap = torch.tensor([self.unlocked[self.kinds[i]] for i in ids.tolist()], device=ids.device, dtype=torch.long)
        self.ability[ids] = torch.minimum((self.ability[ids]+up.long()-down.long()).clamp(0, self.max_level), cap)
        self.success_streak[ids[up | down]] = 0
        self.failure_streak[ids[up | down]] = 0

    def sample(self, ids):
        self.is_replay[ids] = (torch.rand(len(ids), device=ids.device) < self.replay) & (self.ability[ids] > 0)
        easy = (torch.rand(len(ids), device=ids.device)*self.ability[ids]).long()
        self.assigned[ids] = torch.where(self.is_replay[ids], easy, self.ability[ids])
        return self.assigned[ids]

    def validate(self, report):
        iteration = report['iteration']
        if iteration <= self.last_evaluation:
            raise ValueError('Validation must come from a distinct newer policy update')
        self.last_evaluation = iteration
        self.validation.append(copy.deepcopy(report))
        flat = next((g for g in report['groups'] if g['kind'] == 'flat'), None)
        if self.flat_baseline is None and flat is not None:
            self.flat_baseline = flat['tracking_error']
        if self.entry_report is not None:
            reference = {(g['kind'], g['level']): g['success_rate'] for g in self.entry_report['groups']}
            if any(g['success_rate'] < reference.get((g['kind'], g['level']), 0.)-.10 for g in report['groups']):
                self.replay = .4
        for group in report['groups']:
            kind, level = group['kind'], group['level']
            if kind in REVIEW or level != self.unlocked[kind]:
                continue
            key = f'{kind}:{level}'
            passed = group['n'] >= 32 and group['success_rate'] >= .85 and group['fall_rate'] < .1
            self.gate_streak[key] = self.gate_streak.get(key, 0)+1 if passed else 0
            if self.gate_streak[key] >= 2:
                self.unlocked[kind] = min(level+1, self.type_limits[kind])

    def state_dict(self):
        return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in vars(self).items()}

    def load_state_dict(self, state):
        if state['kinds'] != self.kinds or state['max_level'] != self.max_level:
            raise ValueError('Sparse resume requires identical terrain assignment and environment count')
        for k, v in state.items():
            if torch.is_tensor(getattr(self, k, None)):
                getattr(self, k).copy_(v)
            else:
                setattr(self, k, copy.deepcopy(v))
