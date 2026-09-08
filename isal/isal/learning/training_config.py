"""Stage 5 training settings, independent of the simulator."""

from dataclasses import dataclass
import math

ALGORITHM_CLASS = "isal.learning.ppo_affordance:PPOWithAffordance"
RUNNER_CLASS = "isal.learning.affordance_runner:AffordanceRunner"


@dataclass
class AuxiliaryLearningCfg:
    enabled: bool = True
    start_iteration: int = 100
    ramp_iterations: int = 200
    loss_coef: float = 0.05
    aux_batch_size: int = 2048
    min_samples_per_update: int = 256
    max_samples_per_rollout: int = 16384
    smooth_l1_beta: float = 0.1
    sampling_seed: int = 42


@dataclass
class AffordanceGateScheduleCfg:
    start_iteration: int = 300
    ramp_iterations: int = 100
    final: float = 1.0


@dataclass
class DiagnosticsCfg:
    enabled: bool = True
    probe_size: int = 256


def linear_schedule(k: int, start: int, ramp: int, final: float) -> float:
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (k, start, ramp)):
        raise ValueError("Iteration, start and ramp must be nonnegative integers.")
    if not math.isfinite(final) or final < 0:
        raise ValueError("Schedule final value must be finite and nonnegative.")
    return final * (float(k >= start) if ramp == 0 else min(1., max(0., (k-start)/ramp)))


def validate_training_config(aux, gate, diagnostics):
    linear_schedule(0, aux.start_iteration, aux.ramp_iterations, aux.loss_coef)
    linear_schedule(0, gate.start_iteration, gate.ramp_iterations, gate.final)
    if gate.final > 1:
        raise ValueError("Final input gate must lie in [0,1].")
    for value in (aux.aux_batch_size, aux.min_samples_per_update, aux.max_samples_per_rollout,
                  diagnostics.probe_size):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Buffer, batch and probe sizes must be positive integers.")
    if aux.min_samples_per_update > aux.max_samples_per_rollout:
        raise ValueError("Minimum samples cannot exceed buffer capacity.")
    if not math.isfinite(aux.smooth_l1_beta) or aux.smooth_l1_beta <= 0:
        raise ValueError("SmoothL1 beta must be finite and positive.")
