"""GPU-vectorized liftoff snapshots and overlapping contact outcome records.

Only tensors describing robot interaction enter this module. Pending records
track unfinished physical events; they are not a replay buffer. All hot-path
operations stay on the input device, without per-environment Python loops.
"""

from __future__ import annotations

import torch

from .config import SelfSupervisedCfg
from .scan_diagnostics import COUNT_NAMES, SCOPES, diagnostic_counts, diagnostic_fractions


def touchdown_query_xy(foot_xy: torch.Tensor, root_xy: torch.Tensor, root_yaw: torch.Tensor) -> torch.Tensor:
    """World foot xy -> liftoff root yaw frame, in metres (no scan-centre shift)."""
    delta = foot_xy - root_xy
    c, s = torch.cos(root_yaw), torch.sin(root_yaw)
    return torch.stack((c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1]), -1)


def _masked(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return torch.where(mask.reshape(mask.shape + (1,) * (value.ndim - mask.ndim)), value, torch.zeros_like(value))


class FootInteractionTracker:
    """One live swing per foot plus P independent pending outcomes per foot.

    ``update`` is called once after each control step, before automatic reset.
    Touchdown has age zero and is outcome observation one. Survival is known at
    age ``survival_steps``, or immediately on fall. Invalid/unused output slots
    are zero. Returned packets own their data and remain valid after later calls.
    """

    COUNTERS = (
        "liftoff", "touchdown", "samples", "out_of_bounds", "nonfinite",
        "overflow", "incomplete", "touchdown_without_snapshot", "partial_samples",
    )
    SNAPSHOT_DIMS = {"root_xy": (2,), "root_yaw": (), "command": (3,),
                     "base_ang_vel": (3,), "projected_gravity": (3,)}

    def __init__(self, num_envs: int, grid_shape: tuple[int, int], ray_starts_xy: torch.Tensor,
                 step_dt: float, cfg: SelfSupervisedCfg, device: str | torch.device):
        self.cfg, self.device = cfg, torch.device(device)
        self.outcome_steps, self.survival_steps, self.slots = cfg.window_steps(step_dt)
        self.num_envs, self.grid_shape = num_envs, grid_shape
        h, w = grid_shape
        if num_envs < 1 or h < 1 or w < 1:
            raise ValueError("Environment and grid dimensions must be positive.")
        if ray_starts_xy.shape != (h * w, 2) or not torch.isfinite(ray_starts_xy).all():
            raise ValueError("Finite ray origins matching the dynamic grid are required.")
        self.query_min = ray_starts_xy.amin(0).to(self.device)
        self.query_max = ray_starts_xy.amax(0).to(self.device)
        self.grid_x = torch.unique(ray_starts_xy[:, 0], sorted=True).to(self.device)
        self.grid_y = torch.unique(ray_starts_xy[:, 1], sorted=True).to(self.device)
        if self.grid_x.numel() != h or self.grid_y.numel() != w:
            raise ValueError("Diagnostic grid shape must match actual x/y ray coordinates.")
        self._grid_ix = torch.arange(h, device=self.device)
        self._grid_iy = torch.arange(w, device=self.device)
        self.prefix = (num_envs, 2, self.slots)
        self.active = self._zeros((), dtype=torch.bool)
        self.age = self._zeros((), dtype=torch.long)
        self.sample_clock = 0
        self.swing_snapshot_step = torch.zeros((num_envs, 2), dtype=torch.long, device=self.device)
        self.pending_snapshot_step = self._zeros((), dtype=torch.long)
        self.observed = self._zeros((), dtype=torch.long)
        self.contact_count = self._zeros((), dtype=torch.long)
        self.slip_sum = self._zeros(())
        self.tilt_max = self._zeros(())
        self.peak_force = self._zeros(())
        self.touchdown_tilt = self._zeros((2,))
        self.query = self._zeros((2,))
        self.initialized = torch.zeros((num_envs, 2), dtype=torch.bool, device=self.device)
        self.previous_contact = torch.zeros_like(self.initialized)
        self.swing_valid = torch.zeros_like(self.initialized)
        dims = {"height_scan": (1, h, w), **self.SNAPSHOT_DIMS}
        self.pending = {key: self._zeros(shape) for key, shape in dims.items()}
        self.swing = {key: torch.zeros((num_envs, 2, *shape), device=self.device) for key, shape in dims.items()}
        self.swing_scan_masks = torch.zeros((num_envs, 2, 3, h, w), dtype=torch.bool, device=self.device)
        self.swing_diagnostics_available = torch.zeros_like(self.swing_valid)
        self.pending_diagnostics_available = self._zeros((), torch.bool)
        self.pending_diagnostics = {f"{scope}_{name}": self._zeros((), torch.long)
                                    for scope in SCOPES for name in COUNT_NAMES}
        self.counters = {key: torch.zeros((), dtype=torch.long, device=self.device) for key in self.COUNTERS}
        self.peak_pending = torch.zeros((), dtype=torch.long, device=self.device)
        self._env_index = torch.arange(num_envs, device=self.device)[:, None].expand(-1, 2)
        self._foot_index = torch.arange(2, device=self.device)[None, :].expand(num_envs, -1)
        self.output = self._empty_packet()

    def _zeros(self, tail: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros((*self.prefix, *tail), dtype=dtype, device=self.device)

    def _empty_packet(self) -> dict[str, torch.Tensor]:
        packet = {"valid": self._zeros((), torch.bool), "height_scan": self._zeros((1, *self.grid_shape)),
                "query_xy": self._zeros((2,)), "foot_side": self._zeros((2,)),
                "command": self._zeros((3,)), "base_ang_vel": self._zeros((3,)),
                "projected_gravity": self._zeros((3,)), "target": self._zeros((1,))}
        if self.cfg.emit_sample_timing:
            packet.update(snapshot_step=self._zeros((), torch.long), finalized_step=self._zeros((), torch.long))
        return packet

    def _count(self, name: str, mask: torch.Tensor) -> None:
        self.counters[name].add_(mask.sum())

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor, *, clear_output: bool = True) -> None:
        self.counters["incomplete"].add_(self.active[env_ids].sum() + self.swing_valid[env_ids].sum())
        for value in (self.active, self.age, self.swing_snapshot_step, self.pending_snapshot_step,
                      self.observed, self.contact_count, self.slip_sum,
                      self.tilt_max, self.peak_force, self.touchdown_tilt, self.query,
                      self.initialized, self.previous_contact, self.swing_valid,
                      self.swing_scan_masks, self.swing_diagnostics_available,
                      self.pending_diagnostics_available, *self.pending_diagnostics.values(),
                      *self.pending.values(), *self.swing.values()):
            value[env_ids] = 0
        if clear_output:
            # Do not mutate packets already handed to callers.
            self.output = {key: value.clone() for key, value in self.output.items()}
            for value in self.output.values():
                value[env_ids] = 0

    @torch.no_grad()
    def update(self, *, height_scan: torch.Tensor, root_xy: torch.Tensor, root_yaw: torch.Tensor,
               command: torch.Tensor, base_ang_vel: torch.Tensor, projected_gravity: torch.Tensor,
               foot_pos_w: torch.Tensor, foot_vel_w: torch.Tensor, foot_force_w: torch.Tensor,
               base_roll_pitch: torch.Tensor, terminated: torch.Tensor,
               truncated: torch.Tensor, scan_diagnostic_masks: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Inputs have leading N (feet: N,2,3); all values are physical/unscaled except scan."""
        if scan_diagnostic_masks is not None and (
                scan_diagnostic_masks.shape != (self.num_envs, 3, *self.grid_shape)
                or scan_diagnostic_masks.dtype != torch.bool or scan_diagnostic_masks.device != self.swing_scan_masks.device):
            raise ValueError("Scan diagnostic masks must be boolean (N,3,H,W) on the tracker device.")
        self.age.add_(self.active.long())
        self.sample_clock += 1
        forces_finite = torch.isfinite(foot_force_w).all(-1)
        contact = (foot_force_w[..., 2].clamp_min(0) > self.cfg.contact_force_threshold) & forces_finite
        liftoff = self.initialized & self.previous_contact & ~contact & forces_finite
        touchdown = self.initialized & ~self.previous_contact & contact
        self._count("liftoff", liftoff)
        self._count("touchdown", touchdown)
        snapshot = {"height_scan": height_scan, "root_xy": root_xy, "root_yaw": root_yaw,
                    "command": command, "base_ang_vel": base_ang_vel, "projected_gravity": projected_gravity}
        snapshot_ok = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for value in snapshot.values():
            snapshot_ok &= torch.isfinite(value.reshape(self.num_envs, -1)).all(-1)
        valid_liftoff = liftoff & snapshot_ok[:, None]
        self.swing_snapshot_step = torch.where(valid_liftoff, self.sample_clock, self.swing_snapshot_step)
        self._count("nonfinite", liftoff & ~snapshot_ok[:, None])
        # Replace only the live swing, never an older pending outcome.
        for key, value in snapshot.items():
            expanded = value.detach()[:, None].expand_as(self.swing[key])
            mask = valid_liftoff.reshape(valid_liftoff.shape + (1,) * (expanded.ndim - 2))
            self.swing[key].copy_(torch.where(mask, expanded, self.swing[key]))
        self.swing_valid = torch.where(liftoff, valid_liftoff, self.swing_valid)
        self.swing_diagnostics_available = torch.where(
            liftoff, valid_liftoff & (scan_diagnostic_masks is not None), self.swing_diagnostics_available)
        if scan_diagnostic_masks is not None:
            self.swing_scan_masks.copy_(torch.where(
                valid_liftoff[..., None, None, None], scan_diagnostic_masks[:, None], self.swing_scan_masks))
        self._count("touchdown_without_snapshot", touchdown & ~self.swing_valid)
        candidate = touchdown & self.swing_valid
        query = touchdown_query_xy(foot_pos_w[..., :2], self.swing["root_xy"], self.swing["root_yaw"])
        finite_query = torch.isfinite(query).all(-1)
        eps = self.cfg.query_boundary_tolerance
        inside = ((query >= self.query_min - eps) & (query <= self.query_max + eps)).all(-1)
        self._count("nonfinite", candidate & ~finite_query)
        self._count("out_of_bounds", candidate & finite_query & ~inside)
        candidate &= finite_query & inside
        free = ~self.active
        has_free = free.any(-1)
        slot = free.long().argmax(-1)
        self._count("overflow", candidate & ~has_free)
        insert = candidate & has_free
        index = (self._env_index, self._foot_index, slot)
        self.pending_snapshot_step[index] = torch.where(insert, self.swing_snapshot_step, self.pending_snapshot_step[index])
        self._insert_diagnostics(index, insert, query)
        for key, value in self.swing.items():
            old = self.pending[key][index]
            mask = insert.reshape(insert.shape + (1,) * (old.ndim - 2))
            self.pending[key][index] = torch.where(mask, value, old)
        for dest, source in ((self.query, query), (self.touchdown_tilt, base_roll_pitch[:, None].expand(-1, 2, -1))):
            dest[index] = torch.where(insert[..., None], source, dest[index])
        for dest in (self.age, self.observed, self.contact_count, self.slip_sum, self.tilt_max, self.peak_force):
            dest[index] = torch.where(insert, 0, dest[index])
        self.active[index] |= insert
        self.swing_valid &= ~touchdown
        self.peak_pending = torch.maximum(self.peak_pending, self.active.sum(-1).amax())

        # Outcome statistics use only the first K frames, including touchdown.
        measuring = self.active & (self.observed < self.outcome_steps)
        feedback_ok = (forces_finite & torch.isfinite(foot_vel_w).all(-1)
                       & torch.isfinite(base_roll_pitch).all(-1)[:, None])
        invalid = measuring & ~feedback_ok[..., None]
        self._count("nonfinite", invalid)
        self.active &= ~invalid
        measuring &= self.active
        speed = torch.linalg.vector_norm(foot_vel_w[..., :2], dim=-1)
        contact_measuring = measuring & contact[..., None]
        self.slip_sum += _masked(contact_measuring, speed[..., None].expand_as(self.slip_sum))
        self.contact_count += contact_measuring.long()
        delta = base_roll_pitch[:, None, None, :] - self.touchdown_tilt
        delta = torch.atan2(torch.sin(delta), torch.cos(delta))
        tilt = torch.linalg.vector_norm(delta, dim=-1)
        self.tilt_max = torch.maximum(self.tilt_max, _masked(measuring, tilt))
        force_norm = torch.linalg.vector_norm(foot_force_w, dim=-1)[..., None].expand_as(self.peak_force)
        self.peak_force = torch.maximum(self.peak_force, _masked(measuring, force_norm))
        self.observed += measuring.long()

        fall = terminated[:, None, None]
        ended = (terminated | truncated)[:, None, None]
        mature = self.age >= self.survival_steps
        finalized = self.active & (mature | fall)
        self._count("incomplete", self.active & ended & ~finalized)
        self._count("samples", finalized)
        partial = self.observed < self.outcome_steps
        self._count("partial_samples", finalized & partial)
        slip = self.slip_sum / self.contact_count.clamp_min(1)
        slip_score = torch.exp(-slip / self.cfg.slip_scale)
        tilt_score = torch.exp(-self.tilt_max / self.cfg.tilt_scale)
        persistence = self.contact_count / self.outcome_steps
        survival = (~fall).expand_as(self.active).float()
        target = (self.cfg.slip_weight * slip_score + self.cfg.tilt_weight * tilt_score
                  + self.cfg.contact_weight * persistence + self.cfg.survival_weight * survival).clamp(0, 1)
        side = torch.eye(2, device=self.device)[None, :, None, :].expand(*self.prefix, 2)
        data = {key: self.pending[key] for key in ("height_scan", "command", "base_ang_vel", "projected_gravity")}
        data.update(query_xy=self.query, foot_side=side, target=target[..., None],
                    observed_frames=self.observed, partial_window=partial, slip_mean=slip,
                    tilt_change=self.tilt_max, persistence=persistence, survival=survival,
                    slip_score=slip_score, tilt_score=tilt_score, peak_force=self.peak_force)
        data["scan_diagnostics_available"] = self.pending_diagnostics_available
        if self.cfg.emit_sample_timing:
            data.update(snapshot_step=self.pending_snapshot_step,
                        finalized_step=torch.full_like(self.pending_snapshot_step, self.sample_clock))
        data.update(self.pending_diagnostics)
        for scope in SCOPES:
            counts = {name: self.pending_diagnostics[f"{scope}_{name}"] for name in COUNT_NAMES}
            data.update({f"{scope}_{name}": value for name, value in diagnostic_fractions(counts).items()})
        self.output = {"valid": finalized.clone(), **{key: _masked(finalized, value) for key, value in data.items()}}
        self.active &= ~(finalized | ended)
        # Count invalid live swings before terminal cleanup removes them.
        self._count("nonfinite", self.swing_valid & ~forces_finite)
        self.swing_valid &= forces_finite
        self._count("incomplete", self.swing_valid & (terminated | truncated)[:, None])
        self.swing_valid &= ~(terminated | truncated)[:, None]
        # A non-finite sensor frame invalidates the edge baseline and live swing.
        self.previous_contact.copy_(contact)
        self.initialized.copy_(forces_finite & ~(terminated | truncated)[:, None])
        return self.output

    def _insert_diagnostics(self, index, insert: torch.Tensor, query: torch.Tensor) -> None:
        """Reduce the saved liftoff masks at touchdown; pending stores only counts."""
        available = self.swing_diagnostics_available
        self.pending_diagnostics_available[index] = torch.where(
            insert, available, self.pending_diagnostics_available[index])
        # argmin breaks exact ties toward the smaller coordinate. The query may
        # be non-finite for a rejected candidate; insert masks prevent its use.
        ix = (query[..., 0, None] - self.grid_x).abs().argmin(-1)
        iy = (query[..., 1, None] - self.grid_y).abs().argmin(-1)
        neighborhood = ((self._grid_ix - ix[..., None]).abs() <= 1)[..., :, None] & (
            (self._grid_iy - iy[..., None]).abs() <= 1)[..., None, :]
        for scope, selection in (("scan", None), ("query", neighborhood)):
            for name, value in diagnostic_counts(self.swing_scan_masks, selection).items():
                dest = self.pending_diagnostics[f"{scope}_{name}"]
                dest[index] = torch.where(insert, torch.where(available, value, 0), dest[index])

    def statistics(self) -> dict[str, torch.Tensor]:
        """Cumulative device counters plus current occupancy; no host sync."""
        return {**{key: value.clone() for key, value in self.counters.items()},
                "pending": self.active.sum(), "peak_pending_per_foot": self.peak_pending.clone()}
