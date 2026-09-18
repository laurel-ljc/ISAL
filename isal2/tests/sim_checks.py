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
    ame = "height_scan" in obs
    sparse = hasattr(raw.cfg, 'sparse')
    course = hasattr(raw.cfg, 'course')
    drifting = sparse and raw.cfg.sparse.perturbations()['drift'] > 0
    if ame:
        from isaaclab.sensors.ray_caster.patterns import grid_pattern
        from isal2.modified_rsl.modules.terrain_attention import PositionEncoding2D
        assert raw.cfg.observation_space == 390 and raw.cfg.state_space == 1630
        assert obs["height_scan"].shape == (env.num_envs, 187)
        scan_cfg = raw.cfg.scene.height_scanner
        assert scan_cfg.ray_alignment == "yaw" and scan_cfg.pattern_cfg.ordering == "xy"
        assert scan_cfg.offset.pos == (0.0, 0.0, 20.0)
        assert raw.cfg.normalization.height_scan_offset == 0.75
        starts, _ = grid_pattern(scan_cfg.pattern_cfg, env.device)
        coords = PositionEncoding2D(32, 32, (11, 17), .1).coordinates.to(env.device)
        assert torch.allclose(starts[:, :2], coords[0].flatten(1).T, atol=1e-6), "Ray/token XY ordering mismatch"
    levels = raw.scene.terrain.terrain_levels.clone() if hasattr(raw.scene.terrain, "terrain_levels") and not course else None
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
    collected_samples = 0
    for step in range(steps):
        drift_before = raw.scene['actor_height_scanner'].ray_cast_drift.clone() if sparse else None
        if check_reset and step == 5:
            # Force a timeout, independent of whether the random policy falls.
            raw.episode_length_buf[0] = raw.max_episode_length - 1
        obs, reward, done, extras = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        if hasattr(raw, "pop_affordance_samples"):
            samples = raw.pop_affordance_samples()
            if samples is not None:
                assert all(torch.isfinite(value).all() for value in samples.values())
                assert ((samples["label"] >= 0) & (samples["label"] <= 1)).all()
                assert (samples["query_xy"].abs() <= torch.tensor([.8,.5],device=env.device)).all()
                collected_samples += len(samples["label"])
            if done.any():
                assert not raw.collector.active.reshape(env.num_envs,2,-1)[done.bool()].any()
                assert not raw.collector.swing.reshape(env.num_envs,2)[done.bool()].any()
        assert all(torch.isfinite(v).all() for v in obs.values()), "Non-finite observation"
        if ame:
            clean = raw._height_scan()
            critic_scan = obs["critic"].reshape(env.num_envs, 5, -1)[:, -1, -187:]
            assert torch.equal(clean, critic_scan), "Actor noise contaminated critic height"
            bound = raw.cfg.noise.noise_scales.height_scan * raw.obs_scales.height_scan if raw.add_noise else 0
            if drifting:
                sensor = raw.scene['actor_height_scanner']
                actor_clean = (sensor.data.pos_w[:, 2, None]-sensor.data.ray_hits_w[..., 2]-.75).clamp(-1, 1)
                actor_clean = torch.nan_to_num(actor_clean, nan=1, posinf=1, neginf=-1)*raw.obs_scales.height_scan
                assert (obs['height_scan']-actor_clean).abs().max() <= bound+1e-6
                assert torch.equal(drift_before[~done.bool()], sensor.ray_cast_drift[~done.bool()])
            else:
                assert (obs["height_scan"] - clean).abs().max() <= bound + 1e-6
            cached = {k: v.clone() for k, v in env.get_observations().items()}
            assert all(torch.equal(v, env.get_observations()[k]) for k, v in cached.items())
        assert torch.isfinite(reward).all(), "Non-finite reward"
        assert torch.isfinite(raw.actions).all(), "Non-finite action target"
        resets += int(done.sum())
        if done.any():
            assert "terminal_observation" in extras
            if ame:
                terminal = extras["terminal_observation"]
                assert terminal["height_scan"].shape == obs["height_scan"].shape
                terminal_clean = terminal["critic"].reshape(env.num_envs, 5, -1)[:, -1, -187:]
                if not drifting:
                    assert (terminal["height_scan"] - terminal_clean).abs().max() <= bound + 1e-6
                assert terminal["height_scan"].data_ptr() != obs["height_scan"].data_ptr()
            for history in (raw.actor_obs_buffer, raw.critic_obs_buffer):
                reset_history = history.buffer[done.bool()]
                assert torch.equal(reset_history, reset_history[:, -1:].expand_as(reset_history))
            assert (raw.action_buffer.buffer[done.bool()] == 0).all()
        if check_reset and step == 10 and env.num_envs > 1:
            untouched = raw.actor_obs_buffer.buffer[1:].clone()
            untouched_critic = raw.critic_obs_buffer.buffer[1:].clone()
            untouched_scan = raw.obs_buf["height_scan"][1:].clone() if ame else None
            raw._reset_idx(torch.tensor([0], device=env.device))
            raw.obs_buf = raw._get_observations(env_ids=torch.tensor([0], device=env.device))
            assert torch.equal(untouched, raw.actor_obs_buffer.buffer[1:]), "Partial reset changed other histories"
            assert torch.equal(untouched_critic, raw.critic_obs_buffer.buffer[1:])
            if ame:
                assert torch.equal(untouched_scan, raw.obs_buf["height_scan"][1:]), "Partial reset resampled other scans"
    reference_rays = 0
    course_checks = {}
    if course and check_reset:
        from isal2.tests.course_sim_checks import check_course
        course_checks = check_course(env)
    if sparse:
        from isal2.deprecated_tasks.sparse.geometry import verify_legacy_star
        from isal2.deprecated_tasks.base.terrain_generator_cfg import ROUGH_HARD_TERRAINS_CFG
        reference_rays = verify_legacy_star(ROUGH_HARD_TERRAINS_CFG.sub_terrains['star'])
    return dict(terrain=raw.cfg.terrain_preset, num_envs=env.num_envs, steps=steps,
                resets=resets, actor_dim=obs["policy"].shape[-1], critic_dim=obs["critic"].shape[-1], finite=True,
                independent_imports=True, height_scan_dim=obs["height_scan"].shape[-1] if ame else 0,
                collection=dict(raw.collector.stats) if hasattr(raw,"collector") else {}, collected_samples=collected_samples,
                sparse=sparse, actor_only_drift=drifting, legacy_star_reference_rays=reference_rays,
                course_checks=course_checks)
