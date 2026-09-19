"""Tensor-only contact pairing; independent of Isaac Sim for deterministic tests."""
from dataclasses import dataclass
import math
import torch


@dataclass
class ContactCollectionCfg:
    contact_on: float = 5.0
    contact_off: float = 2.0
    confirm_steps: int = 2
    evaluation_seconds: float = 0.25
    max_swing_seconds: float = 1.0
    pending_slots: int = 4
    sole_offset: tuple = (0.025, 0.0, -0.04)
    support_tolerance: float = 0.02
    slip_scale: float = 0.2
    label_weights: tuple = (0.4, 0.3, 0.3)


def world_to_map(foot, root, yaw):
    delta = foot[..., :2] - root[..., :2]
    c, s = yaw.cos(), yaw.sin()
    return torch.stack([c * delta[..., 0] + s * delta[..., 1],
                        -s * delta[..., 0] + c * delta[..., 1]], -1)


class ContactCollector:
    def __init__(self, num_envs, scan_dim, map_size, dt, cfg=None, device="cpu"):
        self.cfg = cfg or ContactCollectionCfg()
        c = self.cfg
        if not (c.contact_on > c.contact_off >= 0 and c.confirm_steps >= 1 and c.pending_slots >= 1
                and c.evaluation_seconds > 0 and c.max_swing_seconds > 0 and c.slip_scale > 0
                and len(c.label_weights) == 3 and min(c.label_weights) >= 0 and sum(c.label_weights) > 0):
            raise ValueError("Invalid contact collection configuration")
        self.num_envs, self.scan_dim, self.device = num_envs, scan_dim, torch.device(device)
        self.window = math.ceil(c.evaluation_seconds / dt)
        if self.window < c.confirm_steps:
            raise ValueError("Evaluation window must include contact confirmation")
        self.max_swing = math.ceil(c.max_swing_seconds / dt)
        self.extent = torch.tensor(map_size, device=device) / 2
        self.tick, self.iteration = 0, 0
        m, k = num_envs * 2, c.pending_slots
        self.contact = torch.zeros(m, dtype=torch.bool, device=device)
        self.on_count = torch.zeros(m, dtype=torch.long, device=device)
        self.off_count = self.on_count.clone()
        self.swing = self.contact.clone()
        self.lift_tick = self.on_count.clone()
        self.lift_iteration = self.on_count.clone()
        self.lift_map = torch.zeros(m, scan_dim, device=device)
        self.lift_root = torch.zeros(m, 3, device=device)
        self.lift_yaw = torch.zeros(m, device=device)
        self.td_point = torch.zeros(m, 3, device=device)
        self.td_metrics = torch.zeros(m, 3, device=device)
        self.active = torch.zeros(m, k, dtype=torch.bool, device=device)
        self.age = torch.zeros(m, k, dtype=torch.long, device=device)
        self.maps = torch.zeros(m, k, scan_dim, device=device)
        self.query = torch.zeros(m, k, 2, device=device)
        self.sums = torch.zeros(m, k, 3, device=device)
        self.sample_iteration = torch.zeros(m, k, dtype=torch.long, device=device)
        self.ready = []
        self.stats = {name: 0 for name in ("liftoff", "touchdown", "matured", "out_of_bounds",
            "unpaired", "swing_timeout", "overflow", "reset_discarded", "invalid", "failed")}

    @torch.no_grad()
    def update_course(self, height, root, yaw, feet, forces, speed, support, result):
        """Censor successful finishes like timeouts, while preserving mature windows."""
        self.update(height, root, yaw, feet, forces, speed, support,
                    result['failed'], result['success'] | result['timeout'])

    @torch.no_grad()
    def update(self, height, root, yaw, feet, forces, speed, support, terminated, truncated):
        """All inputs describe the same post-physics, pre-reset frame."""
        self.tick += 1
        f, v, q = forces.flatten(), speed.flatten(), support.flatten()
        touching = f > self.cfg.contact_off
        metrics = torch.stack([touching.float(), torch.where(touching, v, 0),
                               torch.where(touching, q, 0)], -1)
        self.age += self.active.long()
        self.sums += self.active[..., None] * metrics[:, None, :]
        expire = self.swing & (self.tick - self.lift_tick > self.max_swing)
        self.stats["swing_timeout"] += int(expire.sum())
        self.swing[expire] = False

        off = self.contact & (f < self.cfg.contact_off)
        first_off = off & (self.off_count == 0)
        ids = first_off.nonzero().flatten()
        envs = ids // 2
        self.lift_map[ids] = height[envs]
        self.lift_root[ids] = root[envs]
        self.lift_yaw[ids] = yaw[envs]
        self.lift_tick[ids] = self.tick
        self.lift_iteration[ids] = self.iteration
        self.off_count.copy_(torch.where(off, self.off_count + 1, 0))
        lifted = self.off_count >= self.cfg.confirm_steps
        self.contact[lifted] = False
        self.swing[lifted] = True
        self.off_count[lifted] = 0
        self.stats["liftoff"] += int(lifted.sum())

        on = ~self.contact & (f >= self.cfg.contact_on)
        first_on = on & (self.on_count == 0)
        self.td_point[first_on] = feet.reshape(-1, 3)[first_on]
        self.td_metrics[first_on] = 0
        self.td_metrics[on] += metrics[on]
        self.on_count.copy_(torch.where(on, self.on_count + 1, 0))
        landed = self.on_count >= self.cfg.confirm_steps
        self.contact[landed] = True
        self.stats["touchdown"] += int(landed.sum())
        self.stats["unpaired"] += int((landed & ~self.swing).sum())
        ids = (landed & self.swing).nonzero().flatten()
        xy = world_to_map(self.td_point[ids], self.lift_root[ids], self.lift_yaw[ids])
        finite = torch.isfinite(xy).all(-1) & torch.isfinite(self.lift_map[ids]).all(-1)
        inside = (xy.abs() <= self.extent).all(-1)
        self.stats["invalid"] += int((~finite).sum())
        self.stats["out_of_bounds"] += int((finite & ~inside).sum())
        xy, ids = xy[finite & inside], ids[finite & inside]
        if ids.numel():
            full = self.active[ids].all(-1)
            slots = (~self.active[ids]).long().argmax(-1)
            slots = torch.where(full, self.age[ids].argmax(-1), slots)
            self.stats["overflow"] += int(full.sum())
            self.active[ids, slots] = True
            self.age[ids, slots] = self.on_count[ids]
            self.maps[ids, slots] = self.lift_map[ids]
            self.query[ids, slots] = xy
            self.sums[ids, slots] = self.td_metrics[ids]
            self.sample_iteration[ids, slots] = self.lift_iteration[ids]
        self.swing[landed] = False
        self.on_count[landed] = 0

        failed = terminated.repeat_interleave(2)[:, None].expand_as(self.active)
        finished = self.active & ((self.age >= self.window) | failed)
        self._finalize(finished, failed)
        # Time truncation is censoring, not a failed contact. Full windows above survive.
        reset = (terminated | truncated).nonzero().flatten()
        if reset.numel():
            self.reset(reset)

    def _finalize(self, mask, failed):
        rows, slots = mask.nonzero(as_tuple=True)
        if not rows.numel():
            return
        sums = self.sums[rows, slots]
        s = sums[:, 1] / sums[:, 0].clamp_min(1)
        p = sums[:, 0] / self.age[rows, slots].clamp_min(1)
        q = sums[:, 2] / sums[:, 0].clamp_min(1)
        weights = torch.tensor(self.cfg.label_weights, device=self.device)
        label = (torch.stack([torch.exp(-s / self.cfg.slip_scale), p, q], -1) * weights).sum(-1) / weights.sum()
        label = torch.where(failed[rows, slots], 0, label).clamp(0, 1)
        samples = {"height_scan": self.maps[rows, slots].clone(), "query_xy": self.query[rows, slots].clone(),
                   "label": label, "foot_side": rows % 2, "metrics": torch.stack([s, p, q], -1),
                   "iteration": self.sample_iteration[rows, slots].clone()}
        valid = torch.isfinite(samples["metrics"]).all(-1) & torch.isfinite(label)
        self.stats["invalid"] += int((~valid).sum())
        self.stats["failed"] += int((failed[rows, slots] & valid).sum())
        self.stats["matured"] += int(valid.sum())
        if valid.any():
            self.ready.append({key: value[valid].detach() for key, value in samples.items()})
        self.active[rows, slots] = False
        self.sums[rows, slots] = 0

    def reset(self, env_ids):
        rows = (env_ids[:, None] * 2 + torch.arange(2, device=self.device)).flatten()
        self.stats["reset_discarded"] += int(self.active[rows].sum())
        self.active[rows] = False
        self.age[rows] = 0
        self.sums[rows] = 0
        self.contact[rows] = False
        self.swing[rows] = False
        self.on_count[rows] = 0
        self.off_count[rows] = 0

    def pop_samples(self):
        if not self.ready:
            return None
        result = {key: torch.cat([sample[key] for sample in self.ready]) for key in self.ready[0]}
        self.ready.clear()
        return result
