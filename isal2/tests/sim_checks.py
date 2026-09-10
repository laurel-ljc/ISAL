"""Live simulator acceptance checks, invoked by train.py --smoke_steps."""
import torch
import sys


def smoke_check(env, steps, check_reset):
    # Isaac Lab and RSL-RL are allowed third-party dependencies.
    # Only the old application packages must stay absent.
    forbidden = [name for name in sys.modules if name.split(".")[0] in ("isal", "robolab")]
    assert not forbidden, f"Legacy packages imported: {forbidden}"
    raw = env.unwrapped
    obs = env.get_observations()
    assert obs["policy"].shape == (env.num_envs, raw.cfg.observation_space)
    assert obs["critic"].shape == (env.num_envs, raw.cfg.state_space)
    levels = raw.scene.terrain.terrain_levels.clone() if hasattr(raw.scene.terrain, "terrain_levels") else None
    raw.reset()
    if levels is not None:
        assert torch.equal(levels, raw.scene.terrain.terrain_levels), "Initial/explicit reset changed curriculum"
        if check_reset:
            terrain = raw.scene.terrain
            idx = torch.tensor([0], device=env.device)
            yes = torch.tensor([True], device=env.device)
            no = ~yes
            terrain.terrain_levels[0] = 0
            terrain.update_env_origins(idx, no, yes)
            assert terrain.terrain_levels[0] == 0, "Curriculum went below level zero"
            terrain.terrain_levels[0] = terrain.max_terrain_level - 1
            terrain.update_env_origins(idx, yes, no)
            assert 0 <= terrain.terrain_levels[0] < terrain.max_terrain_level
            assert torch.equal(terrain.terrain_levels[1:], levels[1:])
            terrain.terrain_levels[:] = levels
            all_ids = torch.arange(env.num_envs, device=env.device)
            hold = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            terrain.update_env_origins(all_ids, hold, hold)
            assert torch.equal(terrain.env_origins, terrain.terrain_origins[levels, terrain.terrain_types])
            raw.reset()
    resets = 0
    for step in range(steps):
        if check_reset and step == 5:
            # Force a timeout, independent of whether the random policy falls.
            raw.episode_length_buf[0] = raw.max_episode_length - 1
        obs, reward, done, extras = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        assert all(torch.isfinite(v).all() for v in obs.values()), "Non-finite observation"
        assert torch.isfinite(reward).all(), "Non-finite reward"
        assert torch.isfinite(raw.actions).all(), "Non-finite action target"
        resets += int(done.sum())
        if done.any():
            assert "terminal_observation" in extras
            for history in (raw.actor_obs_buffer, raw.critic_obs_buffer):
                reset_history = history.buffer[done.bool()]
                assert torch.equal(reset_history, reset_history[:, -1:].expand_as(reset_history))
            assert (raw.action_buffer.buffer[done.bool()] == 0).all()
        if check_reset and step == 10 and env.num_envs > 1:
            untouched = raw.actor_obs_buffer.buffer[1:].clone()
            raw._reset_idx(torch.tensor([0], device=env.device))
            raw.obs_buf = raw._get_observations(env_ids=torch.tensor([0], device=env.device))
            assert torch.equal(untouched, raw.actor_obs_buffer.buffer[1:]), "Partial reset changed other histories"
    return dict(terrain=raw.cfg.terrain_preset, num_envs=env.num_envs, steps=steps,
                resets=resets, actor_dim=obs["policy"].shape[-1], critic_dim=obs["critic"].shape[-1], finite=True,
                independent_imports=True)
