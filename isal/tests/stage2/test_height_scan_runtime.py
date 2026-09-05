from __future__ import annotations

import torch


def test_height_scan_environment_and_runner_smoke(isaac_app) -> None:
    import gymnasium as gym
    import isal.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    task = "ISAL-Humanoid-Rough-HeightScan-v0"
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.scene.terrain.terrain_generator.num_rows = 1
    env_cfg.scene.terrain.terrain_generator.num_cols = 1
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.scene.left_feet_scanner.debug_vis = False
    env_cfg.scene.right_feet_scanner.debug_vis = False
    env_cfg.scene.height_scanner.debug_vis = False
    env_cfg.log_dir = None
    for key in ("optimizer", "share_cnn_encoders"):
        if hasattr(agent_cfg.algorithm, key):
            delattr(agent_cfg.algorithm, key)

    env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    with torch.inference_mode():
        obs, _ = env.reset()
        actions = runner.get_inference_policy(device=agent_cfg.device)(obs.to(agent_cfg.device))
        next_obs, rewards, dones, extras = env.step(torch.zeros_like(actions, device=env.device))

    assert obs["policy"].shape == (1, 780)
    assert obs["height_scan"].shape == (1, 187)
    assert obs["critic"].shape == (1, 3260)
    assert env.unwrapped.height_scan_grid_shape == (17, 11)
    assert env.unwrapped.height_scan_grid.shape == (1, 1, 17, 11)
    assert runner.alg.policy.get_actor_obs(obs.to(agent_cfg.device)).shape == (1, 967)
    assert actions.shape == (1, 23)
    assert next_obs["height_scan"].shape == (1, 187)
    assert dones.shape == (1,)
    assert torch.isfinite(obs["height_scan"]).all()
    assert torch.isfinite(rewards).all()
    assert obs["height_scan"].min() >= -1.6
    assert obs["height_scan"].max() <= 0.8
    assert {"log", "time_outs"}.issubset(extras)

    symmetry = agent_cfg.algorithm.symmetry_cfg.data_augmentation_func
    mirrored_twice, _ = symmetry(env, obs, None)
    mirrored_once = mirrored_twice[1:]
    restored_twice, _ = symmetry(env, mirrored_once, None)
    torch.testing.assert_close(restored_twice[1:]["height_scan"], obs["height_scan"])
    env.close()
