"""Each actual scene runs in a fresh Isaac Sim process; no learning calls."""

import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("variant", ["small_height", "large_interaction"])
def test_dynamic_scene_wrapper_forward_and_export(request, variant):
    root = Path(__file__).resolve().parents[3]
    if os.environ.get("ISAL_DYNAMIC_SIM_CHILD") != "1":
        node = f"{Path(__file__).as_posix()}::test_dynamic_scene_wrapper_forward_and_export[{variant}]"
        result = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", "env_isaaclab", "python", "-u", "-m", "pytest",
             node, "-q", "--tb=short", "-p", "no:cacheprovider"],
            cwd=root, env={**os.environ, "ISAL_DYNAMIC_SIM_CHILD": "1"},
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0 and "1 passed" in output and "FAILED" not in output, output[-14000:]
        return
    request.getfixturevalue("isaac_app")
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner
    from isal.interaction.recording import InteractionRecorder
    import isal.tasks  # noqa: F401

    interaction = variant == "large_interaction"
    task = "ISAL-Humanoid-Rough-Interaction-v0" if interaction else "ISAL-Humanoid-Rough-HeightScan-v0"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    # Modify only public perception fields AFTER __post_init__, like Hydra does.
    cfg.terrain_perception.size = (1.8, 1.) if interaction else (1.6, .8)
    cfg.scene.height_scanner.pattern_cfg.ordering = "yx" if interaction else "xy"
    cfg.robot.actor_obs_history_length = 3
    cfg.robot.critic_obs_history_length = 4
    cfg.scene.num_envs = 2
    cfg.scene.terrain.terrain_generator.num_rows = 1
    cfg.scene.terrain.terrain_generator.num_cols = 1
    cfg.scene.terrain.max_init_terrain_level = 0
    for name in ("height_scanner", "left_feet_scanner", "right_feet_scanner"):
        getattr(cfg.scene, name).debug_vis = False
    cfg.commands.debug_vis = False
    cfg.log_dir = None
    for key in ("optimizer", "share_cnn_encoders"):
        if hasattr(agent.algorithm, key):
            delattr(agent.algorithm, key)
    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
    try:
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        raw = env.unwrapped
        rays = 209 if interaction else 153
        assert cfg.state_space == 139 + rays
        assert raw.height_scanner.ray_starts.shape[1] == rays
        recorder = InteractionRecorder(1000)
        with torch.inference_mode():
            obs, _ = env.reset()
            actions = runner.get_inference_policy(device=agent.device)(obs.to(agent.device))
            assert actions.shape == (2, 23)
            assert obs["policy"].shape == (2, 234)
            assert obs["critic"].shape == (2, 4 * (139 + rays))
            assert obs["height_scan"].shape == (2, rays)
            augmented, _ = agent.algorithm.symmetry_cfg.data_augmentation_func(env, obs, actions)
            assert augmented["critic"].shape == (4, 4 * (139 + rays))
            for _ in range(40):
                obs, rewards, dones, extras = env.step(torch.zeros_like(actions))
                assert torch.isfinite(rewards).all()
                if interaction:
                    packet = extras["auxiliary"]
                    assert packet["height_scan"].shape[-2:] == (19, 11)
                    assert packet["scan_diagnostics_available"][packet["valid"]].all()
                    recorder.append(packet, extras["interaction_stats"])
            counts = raw.height_scan_diagnostics()
            assert counts["total_count"] == 2 * rays
            assert counts["finite_count"] + counts["invalid_count"] == counts["total_count"]
        if interaction:
            output = root / "outputs/perception_acceptance/large_interaction"
            summary = recorder.write(output, metadata={"synthetic": False, "training_started": False,
                                                      "task": task, "steps": 40, "num_envs": 2})
            assert summary["scan_diagnostics"]["missing_samples"] == 0
            assert (output / "samples.pt").is_file()
    finally:
        env.close()
