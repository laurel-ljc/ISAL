"""Capacity-bounded uniform GPU replay with iteration-based expiration."""
import torch


class AffordanceReplay:
    def __init__(self, scan_dim, capacity=65536, max_age=32, device="cpu"):
        if capacity < 1 or max_age < 1:
            raise ValueError("Replay capacity and max_age must be positive")
        self.capacity, self.max_age, self.device = capacity, max_age, torch.device(device)
        self.next = 0
        self.valid = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.data = {"height_scan": torch.zeros(capacity, scan_dim, device=device),
                     "query_xy": torch.zeros(capacity, 2, device=device),
                     "label": torch.zeros(capacity, device=device),
                     "metrics": torch.zeros(capacity, 3, device=device),
                     "foot_side": torch.zeros(capacity, dtype=torch.long, device=device),
                     "iteration": torch.zeros(capacity, dtype=torch.long, device=device)}

    def __len__(self):
        return int(self.valid.sum())

    @torch.no_grad()
    def add(self, sample):
        if sample is None or not len(sample["label"]):
            return
        n = min(len(sample["label"]), self.capacity)
        indices = (torch.arange(n, device=self.device) + self.next) % self.capacity
        for key in self.data:
            self.data[key][indices] = sample[key][-n:].to(self.device)
        self.valid[indices] = True
        self.next = (self.next + n) % self.capacity

    def expire(self, iteration):
        self.valid &= self.data["iteration"] > iteration - self.max_age

    def sample(self, batch_size):
        ids = self.valid.nonzero().flatten()
        if not ids.numel():
            raise ValueError("Cannot sample empty affordance replay")
        chosen = ids[torch.randint(len(ids), (batch_size,), device=self.device)]
        return {key: value[chosen] for key, value in self.data.items()}

    def state_dict(self):
        ids = self.valid.nonzero().flatten()
        return dict(capacity=self.capacity, max_age=self.max_age, next=self.next, ids=ids.cpu(),
                    data={key: value[ids].cpu() for key, value in self.data.items()})

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity or state["max_age"] != self.max_age:
            raise ValueError("Replay configuration differs from checkpoint")
        ids = state["ids"].to(self.device)
        self.valid.zero_()
        self.valid[ids] = True
        for key in self.data:
            self.data[key][ids] = state["data"][key].to(self.device)
        self.next = state["next"]
