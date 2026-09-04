from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _canonicalize(value: Any) -> Any:
    """Convert config dictionaries into namespace-independent comparable values."""
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize(item) for item in value)
    if isinstance(value, Callable):
        return getattr(value, "__name__", type(value).__name__)
    if isinstance(value, str):
        namespace_pairs = (
            (
                "isal.tasks.direct.humanoid_rough.mdp.rewards:",
                "robolab.tasks.direct.base.mdp.rewards:",
            ),
            (
                "isal.tasks.direct.humanoid_rough.agents.isal_agent_cfg:",
                "robolab.tasks.direct.base.agents.rpo_agent_cfg:",
            ),
        )
        for candidate_prefix, baseline_prefix in namespace_pairs:
            if value.startswith(candidate_prefix):
                return "baseline:" + value.removeprefix(candidate_prefix)
            if value.startswith(baseline_prefix):
                return "baseline:" + value.removeprefix(baseline_prefix)
    if hasattr(value, "pattern") and hasattr(value, "flags"):
        return (value.pattern, value.flags)
    return value


def _differences(left: Any, right: Any, path: str = "root") -> list[str]:
    """Return concise leaf-level differences for actionable parity failures."""
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(left.keys() | right.keys(), key=str):
            if key not in left or key not in right:
                differences.append(f"{path}.{key}: missing from one side")
            else:
                differences.extend(_differences(left[key], right[key], f"{path}.{key}"))
        return differences
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [f"{path}: lengths differ ({len(left)} != {len(right)})"]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(_differences(left_item, right_item, f"{path}[{index}]"))
        return differences
    if left != right:
        return [f"{path}: {left!r} != {right!r}"]
    return []


def test_task_registration(isaac_app) -> None:
    import gymnasium as gym
    import isal.tasks  # noqa: F401

    spec = gym.spec("ISAL-Humanoid-Rough-v0")
    assert spec.entry_point == "isal.tasks.direct.humanoid_rough.base_env:ISALHumanoidEnv"
    assert spec.kwargs["env_cfg_entry_point"].endswith(":ISALHumanoidRoughEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":ISALHumanoidRoughAgentCfg")


def test_environment_config_matches_effective_rpo_rough_baseline(isaac_app) -> None:
    import robolab.tasks  # noqa: F401
    import isal.tasks  # noqa: F401
    from isal.tasks.direct.humanoid_rough.isal_env_cfg import ISALHumanoidRoughEnvCfg
    from robolab.tasks.direct.base.rpo_env_cfg import RPORoughEnvCfg

    baseline = RPORoughEnvCfg()
    candidate = ISALHumanoidRoughEnvCfg()

    differences = _differences(_canonicalize(candidate.to_dict()), _canonicalize(baseline.to_dict()))
    assert not differences, "\n".join(differences[:20])
    assert candidate.scene_context.robot.init_state.pos == (0.0, 0.0, 0.75)


def test_agent_config_matches_rpo_rough_except_project_names(isaac_app) -> None:
    import robolab.tasks  # noqa: F401
    import isal.tasks  # noqa: F401
    from isal.tasks.direct.humanoid_rough.agents.isal_agent_cfg import ISALHumanoidRoughAgentCfg
    from robolab.tasks.direct.base.agents.rpo_agent_cfg import RPORoughAgentCfg

    baseline = RPORoughAgentCfg()
    candidate = ISALHumanoidRoughAgentCfg()

    baseline.experiment_name = candidate.experiment_name
    baseline.neptune_project = candidate.neptune_project
    baseline.wandb_project = candidate.wandb_project
    differences = _differences(_canonicalize(candidate.to_dict()), _canonicalize(baseline.to_dict()))
    assert not differences, "\n".join(differences[:20])


def test_stage_one_keeps_height_scan_out_of_actor(isaac_app) -> None:
    from isal.tasks.direct.humanoid_rough.isal_env_cfg import ISALHumanoidRoughEnvCfg

    cfg = ISALHumanoidRoughEnvCfg()
    assert cfg.scene_context.height_scanner.enable_height_scan is True
    assert cfg.scene_context.height_scanner.enable_height_scan_actor is False
    assert cfg.action_space == 23
    assert cfg.observation_space == 78
    assert cfg.state_space == 326
    assert cfg.sim.dt == 0.005
    assert cfg.decimation == 4
    assert cfg.episode_length_s == 20.0
