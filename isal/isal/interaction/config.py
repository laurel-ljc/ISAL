"""All Stage 3 sampling and label defaults, independent of Isaac Sim."""

from dataclasses import dataclass
import math


@dataclass
class SelfSupervisedCfg:
    enabled: bool = True
    emit_sample_timing: bool = False
    snapshot_mode: str = "liftoff"
    foot_body_names: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
    contact_force_threshold: float = 20.0
    outcome_window_s: float = 0.25
    survival_window_s: float = 0.50
    slip_scale: float = 0.15
    tilt_scale: float = 0.20
    slip_weight: float = 0.40
    tilt_weight: float = 0.25
    contact_weight: float = 0.20
    survival_weight: float = 0.15
    pending_slots: int | None = None
    query_boundary_tolerance: float = 1.0e-6
    bad_target_threshold: float = 0.35
    good_target_threshold: float = 0.70

    def window_steps(self, step_dt: float) -> tuple[int, int, int]:
        positive = (step_dt, self.outcome_window_s, self.survival_window_s, self.slip_scale, self.tilt_scale)
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("Time intervals and score scales must be finite and positive.")
        if self.snapshot_mode != "liftoff":
            raise ValueError("Stage 3 supports only liftoff snapshots.")
        if len(self.foot_body_names) != 2 or len(set(self.foot_body_names)) != 2:
            raise ValueError("Exactly two distinct foot names in left/right order are required.")
        weights = (self.slip_weight, self.tilt_weight, self.contact_weight, self.survival_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1.0):
            raise ValueError("Label weights must be finite, nonnegative and sum to one.")
        if not math.isfinite(self.contact_force_threshold) or self.contact_force_threshold < 0:
            raise ValueError("Contact threshold must be finite and nonnegative.")
        if not math.isfinite(self.query_boundary_tolerance) or self.query_boundary_tolerance < 0:
            raise ValueError("Query boundary tolerance must be finite and nonnegative.")
        if not 0 <= self.bad_target_threshold < self.good_target_threshold <= 1:
            raise ValueError("Target histogram thresholds must be ordered in [0,1].")
        outcome = round(self.outcome_window_s / step_dt)
        survival = round(self.survival_window_s / step_dt)
        if outcome < 1 or survival < outcome or self.survival_window_s < self.outcome_window_s:
            raise ValueError("Require 1 <= outcome_steps <= survival_steps.")
        slots = math.ceil(survival / 2) + 1 if self.pending_slots is None else self.pending_slots
        if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
            raise ValueError("pending_slots must be a positive integer or None.")
        return outcome, survival, slots
