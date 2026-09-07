from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize(item) for item in value)
    if isinstance(value, Callable):
        return getattr(value, "__name__", type(value).__name__)
    return value


def test_height_scan_task_registration(isaac_app) -> None:
    import gymnasium as gym
    import isal.tasks  # noqa: F401

    spec = gym.spec("ISAL-Humanoid-Rough-HeightScan-v0")
    assert spec.entry_point.endswith(":ISALHumanoidHeightScanEnv")
    assert spec.kwargs["env_cfg_entry_point"].endswith(":ISALHumanoidRoughHeightScanEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":ISALHumanoidRoughHeightScanAgentCfg")


def test_height_scan_environment_only_changes_perception_contract(isaac_app) -> None:
    from isal.tasks.direct.humanoid_rough.isal_env_cfg import (
        ISALHumanoidRoughEnvCfg,
        ISALHumanoidRoughHeightScanEnvCfg,
    )
    from isal.tasks.direct.humanoid_rough.scene_cfg import SceneCfg

    baseline = ISALHumanoidRoughEnvCfg()
    candidate = ISALHumanoidRoughHeightScanEnvCfg()
    perception = candidate.terrain_perception

    baseline.terrain_perception = perception
    baseline.scene_context.height_scanner.enable_height_scan_actor = True
    baseline.scene_context.height_scanner.size = perception.size
    baseline.scene_context.height_scanner.resolution = perception.resolution
    baseline.scene_context.height_scanner.offset = (perception.offset_x, 0.0, 20.0)
    baseline.noise.noise_scales.height_scan = 0.0
    baseline.scene = SceneCfg(
        config=baseline.scene_context,
        physics_dt=baseline.sim.dt,
        step_dt=baseline.decimation * baseline.sim.dt,
    )

    assert _canonicalize(candidate.to_dict()) == _canonicalize(baseline.to_dict())
    assert perception.size == (1.6, 1.0)
    assert perception.resolution == 0.1
    assert perception.offset_x == 0.4
    assert (perception.min_height, perception.max_height, perception.height_scale) == (-1.5, 0.4, 0.5)
    assert (perception.noise_std, perception.dropout_prob) == (0.0, 0.0)
    assert candidate.action_space == 23
    assert candidate.observation_space == 78
    assert candidate.state_space == 326
    assert candidate.sim.dt == 0.005
    assert candidate.decimation == 4
    assert candidate.episode_length_s == 20.0


def test_height_scan_agent_only_changes_groups_and_names(isaac_app) -> None:
    from isal.tasks.direct.humanoid_rough.agents.isal_agent_cfg import (
        ISALHumanoidRoughAgentCfg,
        ISALHumanoidRoughHeightScanAgentCfg,
    )

    baseline = ISALHumanoidRoughAgentCfg()
    candidate = ISALHumanoidRoughHeightScanAgentCfg()

    baseline.policy.class_name = candidate.policy.class_name
    baseline.algorithm.class_name = candidate.algorithm.class_name
    baseline.obs_groups = candidate.obs_groups
    baseline.experiment_name = candidate.experiment_name
    baseline.neptune_project = candidate.neptune_project
    baseline.wandb_project = candidate.wandb_project
    assert _canonicalize(candidate.to_dict()) == _canonicalize(baseline.to_dict())
    assert candidate.obs_groups == {
        "policy": ["policy", "height_scan"],
        "critic": ["critic"],
    }
    assert candidate.policy.class_name == "rsl_rl.modules:ActorCritic"
    assert candidate.algorithm.class_name == "rsl_rl.algorithms:PPO"
    assert candidate.experiment_name == "isal_humanoid_rough_height_scan"
