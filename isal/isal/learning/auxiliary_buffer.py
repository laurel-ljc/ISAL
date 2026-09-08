"""Uniform bounded sampling of contacts newly finalized in the current rollout."""

import torch


class AuxiliaryRolloutBuffer:
    FIELDS = ("height_scan", "query_xy", "foot_side", "command", "base_ang_vel", "projected_gravity", "target")

    def __init__(self, capacity: int, device: str, seed: int):
        if capacity < 1:
            raise ValueError("Buffer capacity must be positive.")
        self.capacity, self.device = capacity, torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.clear()

    def clear(self):
        self.data = {}
        self.priorities = None
        self.received = 0
        self.cross_rollout = 0
        self.age_sum = 0
        self.age_max = 0

    def __len__(self):
        return 0 if not self.data else self.data["target"].shape[0]

    @torch.no_grad()
    def append(self, packet: dict, rollout_step: int):
        valid = packet["valid"]
        if valid.dtype != torch.bool or valid.ndim != 3 or valid.shape[1] != 2:
            raise ValueError("Expected valid[N,2,P] boolean mask.")
        if not valid.any():
            return
        if any(key not in packet for key in self.FIELDS):
            raise ValueError("Incomplete auxiliary packet.")
        # clone outside inference_mode: stored inputs must be usable by autograd later.
        with torch.inference_mode(False):
            data = {}
            for key, value in packet.items():
                if key == "valid":
                    continue
                if value.shape[:3] != valid.shape:
                    raise ValueError(f"Packet prefix mismatch: {key}.")
                data[key] = value[valid].detach().to(self.device).clone()
            for key in self.FIELDS:
                if not torch.isfinite(data[key]).all():
                    raise ValueError(f"Nonfinite auxiliary input: {key}.")
            if torch.any((data["target"] < 0) | (data["target"] > 1)):
                raise ValueError("Auxiliary targets must lie in [0,1].")
            if self.data and data.keys() != self.data.keys():
                raise ValueError("Auxiliary packet schema changed within a rollout.")
            count = data["target"].shape[0]
            if "snapshot_step" in data:
                age = data["finalized_step"] - data["snapshot_step"]
                if torch.any(age < 0):
                    raise ValueError("Sample age cannot be negative.")
                self.cross_rollout += int((age >= rollout_step).sum())
                self.age_sum += int(age.sum())
                self.age_max = max(self.age_max, int(age.max()))
            priority = torch.rand(count, device=self.device, generator=self.generator, dtype=torch.float64)
            if self.data:
                data = {key: torch.cat((self.data[key], value), 0) for key, value in data.items()}
                priority = torch.cat((self.priorities, priority))
            if priority.numel() > self.capacity:
                keep = torch.topk(priority, self.capacity, sorted=False).indices
                data = {key: value[keep] for key, value in data.items()}
                priority = priority[keep]
            self.data, self.priorities = data, priority
            self.received += count

    def sample(self, count: int):
        if not len(self):
            raise ValueError("Cannot sample an empty auxiliary buffer.")
        indices = torch.randint(len(self), (count,), device=self.device, generator=self.generator)
        return {key: value[indices] for key, value in self.data.items()}

    def statistics(self):
        return {"samples_received": self.received, "samples_retained": len(self),
                "samples_dropped": self.received-len(self), "cross_rollout_samples": self.cross_rollout,
                "sample_age_mean_steps": self.age_sum/max(self.received, 1), "sample_age_max_steps": self.age_max}
