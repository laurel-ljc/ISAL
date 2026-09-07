"""Bounded real simulation plus explicitly synthetic feedback for reset ordering."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import subprocess

import pytest
import torch


TASK = "ISAL-Humanoid-Rough-Interaction-v0"


def canonical(value):
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(canonical(item) for item in value)
    if isinstance(value, Callable):
        return value.__name__
    return value


def test_registration_and_full_configuration_parity(isaac_app):
    import gymnasium as gym
    import isal.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    spec = gym.spec(TASK)
    assert spec.entry_point.endswith(":ISALHumanoidInteractionEnv")
    original = load_cfg_from_registry("ISAL-Humanoid-Rough-HeightScan-v0", "env_cfg_entry_point")
    new = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    new_dict = new.to_dict()
    assert new_dict.pop("self_supervised")["enabled"]
    assert canonical(new_dict) == canonical(original.to_dict())
    original = load_cfg_from_registry("ISAL-Humanoid-Rough-HeightScan-v0", "rsl_rl_cfg_entry_point")
    new = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    original_dict, new_dict = original.to_dict(), new.to_dict()
    for key in ("experiment_name", "neptune_project", "wandb_project"):
        original_dict.pop(key)
        new_dict.pop(key)
    assert canonical(original_dict) == canonical(new_dict)
    # Config instances own their nested dataclass state.
    env_a = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_b = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_a.self_supervised.enabled = False
    assert env_b.self_supervised.enabled


def test_explicit_body_mapping_and_noisy_scan_rejection(isaac_app):
    from isal.tasks.direct.humanoid_rough.interaction_env import ISALHumanoidInteractionEnv
    from isal.tasks.direct.humanoid_rough.isal_env_cfg import ISALHumanoidRoughInteractionEnvCfg
    names = ("left_ankle_roll_link", "right_ankle_roll_link")
    assert ISALHumanoidInteractionEnv._resolve_feet(["other", *names], names) == [1, 2]
    assert ISALHumanoidInteractionEnv._resolve_feet([names[1], "other", names[0]], names) == [2, 0]
    with pytest.raises(ValueError, match="exactly one"):
        ISALHumanoidInteractionEnv._resolve_feet([names[0]], names)
    cfg = ISALHumanoidRoughInteractionEnvCfg()
    cfg.terrain_perception.noise_std = 0.1
    with pytest.raises(ValueError, match="noiseless"):
        ISALHumanoidInteractionEnv(cfg)


@pytest.mark.parametrize("enabled,num_envs", [(False, 1), (True, 4)])
def test_fixed_action_wrapper_and_terminal_packet(request, monkeypatch, enabled, num_envs):
    # This Windows Isaac Sim build can stall recreating SimulationContext in one
    # Kit process; its RayCaster class also retains the previous scene's meshes.
    # Run each actual scene in a fresh process, without altering runtime libraries.
    if os.environ.get("ISAL_STAGE3_SIM_CHILD") != "1":
        root = Path(__file__).resolve().parents[3]
        node = f"{Path(__file__).as_posix()}::test_fixed_action_wrapper_and_terminal_packet[{enabled}-{num_envs}]"
        command = ["conda", "run", "--no-capture-output", "-n", "env_isaaclab", "python", "-u", "-m", "pytest",
                   node, "-q", "--tb=short", "-p", "no:cacheprovider"]
        result = subprocess.run(command, cwd=root, env={**os.environ, "ISAL_STAGE3_SIM_CHILD": "1"},
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        # Conda on this host can return zero even if its child failed; inspect
        # pytest's terminal report too, not just the process exit code.
        output = result.stdout + result.stderr
        assert result.returncode == 0 and "1 passed" in output and "FAILED" not in output, output[-14000:]
        return
    request.getfixturevalue("isaac_app")
    import gymnasium as gym
    import isal.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isal.tasks.direct.humanoid_rough.height_scan_env import ISALHumanoidHeightScanEnv

    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.seed = 42
    cfg.self_supervised.enabled = enabled
    cfg.scene.num_envs = num_envs
    cfg.scene.terrain.terrain_generator.num_rows = 1
    cfg.scene.terrain.terrain_generator.num_cols = 1
    cfg.scene.terrain.max_init_terrain_level = 0
    for name in ("height_scanner", "left_feet_scanner", "right_feet_scanner"):
        getattr(cfg.scene, name).debug_vis = False
    cfg.commands.debug_vis = False
    cfg.log_dir = None
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
    raw = env.unwrapped
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            actions = torch.zeros((num_envs, 23), device=env.device)
            for _ in range(5):
                obs, rewards, dones, extras = env.step(actions)
                assert torch.isfinite(rewards).all()
                assert obs["policy"].shape == (num_envs, 780)
                assert obs["height_scan"].shape == (num_envs, 187)
                assert obs["critic"].shape == (num_envs, 3260)
            if not enabled:
                assert not hasattr(raw, "interaction_tracker")
                assert "auxiliary" not in extras
                return
            tracker = raw.interaction_tracker
            assert extras["auxiliary"] is tracker.output
            assert extras["auxiliary"]["height_scan"].shape == (num_envs, 2, 14, 1, 17, 11)
            assert extras["auxiliary"]["valid"].device == actions.device
            before = tracker.statistics()
            env.get_observations()
            env.get_observations()
            assert all(torch.equal(value, tracker.statistics()[key]) for key, value in before.items())

            # Real simulator steps with synthetic robot feedback/termination:
            # prove the pre-reset hook and wrapper, not real walking quality.
            env.reset()
            phase = [0]
            captured_scan = []
            actual_update = tracker.update

            def injected_update(**data):
                data["foot_force_w"] = torch.zeros_like(data["foot_force_w"])
                data["foot_force_w"][..., 2] = 0 if phase[0] == 1 else 100
                data["foot_pos_w"] = torch.zeros_like(data["foot_pos_w"])
                data["foot_pos_w"][..., :2] = raw.robot.data.root_pos_w[:, None, :2]
                data["foot_vel_w"] = torch.zeros_like(data["foot_vel_w"])
                if phase[0] == 1:
                    captured_scan.append(data["height_scan"].clone())
                    torch.testing.assert_close(data["height_scan"], raw._compute_perceptive_height_scan()[1])
                return actual_update(**data)

            def controlled_dones(self):
                terminated = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
                terminated[0] = phase[0] == 3
                return terminated, torch.zeros_like(terminated)

            monkeypatch.setattr(tracker, "update", injected_update)
            monkeypatch.setattr(ISALHumanoidHeightScanEnv, "_get_dones", controlled_dones)
            for step in range(4):
                phase[0] = step
                _, _, dones, extras = env.step(actions)
            packet = extras["auxiliary"]
            assert dones[0]
            assert raw.episode_length_buf[0] == 0
            assert packet["valid"][0].sum() == 2
            assert not packet["valid"][1:].any()
            assert not packet["survival"].any()
            assert packet["partial_window"][0, :, 0].all()
            assert packet["scan_diagnostics_available"][packet["valid"]].all()
            assert (packet["scan_total_count"][packet["valid"]] == 187).all()
            torch.testing.assert_close(packet["height_scan"][0, :, 0], captured_scan[0][0].expand(2, -1, -1, -1))
            assert not tracker.active[0].any()
            assert tracker.active[1:].sum() == 6
            # Repeated observation/done reads outside step cannot duplicate events.
            count = tracker.counters["samples"].clone()
            raw._get_dones()
            env.get_observations()
            assert tracker.counters["samples"] == count
            env.reset()
            assert not tracker.active.any()
            assert not tracker.output["valid"].any()
            assert "auxiliary" not in raw.extras
            assert packet["valid"].sum() == 2
            assert not tracker.output["scan_diagnostics_available"].any()
            assert packet["scan_total_count"][packet["valid"]].sum() == 374
    finally:
        env.close()
