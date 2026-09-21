"""Additional live course lifecycle checks, used by --smoke_steps --check_reset."""
import torch
from isal2.deprecated_tasks.endpoint_course.common.course.commands import push_base_horizontal


def check_course(env):
    raw = env.unwrapped
    n, device = env.num_envs, env.device
    affordance = hasattr(raw, 'collector')
    assert raw.cfg.noise.add_noise  # Preserve proprioception noise.
    assert raw.cfg.noise.noise_scales.height_scan == 0
    assert raw.cfg.scene.height_scanner.drift_range == (0., 0.)
    assert (raw.cfg.events.push_robot is None) == (raw.cfg.course.stage == 1)
    levels = raw.course_curriculum.levels.clone()
    env.reset()
    assert torch.equal(levels, raw.course_curriculum.levels)
    state = raw.course_state_dict()
    raw.course_curriculum.levels[:] = 8
    raw.load_course_state_dict(state)
    assert torch.equal(levels, raw.course_curriculum.levels)

    # Verify the actual articulation velocity write (not only the event config).
    ids = torch.tensor([0], device=device)
    if raw.cfg.course.stage == 2:
        before = raw.robot.data.root_vel_w.clone()
        push_base_horizontal(raw, ids)
        after = raw.robot.data.root_vel_w.clone()
        assert (after[ids, :2]-before[ids, :2]).abs().max() <= .10001
        assert torch.equal(after[ids, 2:], before[ids, 2:])
        assert torch.equal(after[1:], before[1:])
        env.reset()

    # Settle at the real finish platform. Success must end the episode without
    # timeout bootstrap or the failure termination reward.
    idx = 0
    kind = int(raw.course_curriculum.assigned_type[idx])
    old_level = int(raw.course_curriculum.levels[idx, kind])
    pose = raw.robot.data.default_root_state[ids, :7].clone()
    pose[:, :3] += raw.course_goal[ids]
    raw.robot.write_root_pose_to_sim(pose, ids)
    raw.robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), ids)
    raw.robot.write_joint_state_to_sim(raw.robot.data.default_joint_pos[ids],
                                     torch.zeros(1, raw.num_actions, device=device), env_ids=ids)
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    raw.scene.update(dt=0.)
    raw.command_generator._update_command()
    reached = False
    for _ in range(100):
        obs, reward, done, extras = env.step(torch.zeros(n, raw.num_actions, device=device))
        if done[idx]:
            assert raw.course_result['success'][idx], {k: bool(v[idx]) for k, v in raw.course_result.items()}
            assert not extras['time_outs'][idx]
            assert raw.course_curriculum.levels[idx, kind] == min(old_level+1, 9)
            assert not raw.course_result['failed'][idx]
            if affordance:
                assert not raw.collector.active[:2].any()
                assert not raw.collector.swing[:2].any()
            reached = True
            break
    assert reached, 'Robot never reached successful terminal state on finish platform'

    # A real timeout decreases only the current type, even though reset may
    # immediately select a different type/column and origin.
    env.reset()
    kind = int(raw.course_curriculum.assigned_type[0])
    raw.course_curriculum.levels[0, kind] = 5
    raw.episode_length_buf[0] = raw.max_episode_length-1
    _, _, done, extras = env.step(torch.zeros(n, raw.num_actions, device=device))
    assert done[0] and extras['time_outs'][0]
    assert raw.course_curriculum.levels[0, kind] == 4
    env.reset()
    return dict(success_terminal=True, timeout_bootstrap=True, independent_type_levels=True,
                clean_actor_and_critic=True, additive_push_checked=raw.cfg.course.stage == 2)


def check_affordance_config(cfg):
    """Compare resolved configs before scene creation mutates importer/asset settings."""
    from isal2.deprecated_tasks.endpoint_course.ame_stage1.env_cfg import RPOAMEStage1EnvCfg
    from isal2.deprecated_tasks.endpoint_course.ame_stage2.env_cfg import RPOAMEStage2EnvCfg
    reference = (RPOAMEStage1EnvCfg if cfg.course.stage == 1 else RPOAMEStage2EnvCfg)()
    reference.seed = cfg.seed
    reference.sim.device = cfg.sim.device
    reference.sim.log_dir = cfg.sim.log_dir
    reference.configure(num_envs=cfg.scene_context.num_envs, terrain_cols=cfg.scene_context.terrain_generator.num_cols)
    expected, actual = reference.to_dict(), cfg.to_dict()
    for key in ('robot', 'reward', 'sim', 'noise', 'normalization', 'commands', 'scene_context', 'scene', 'events', 'course'):
        assert expected[key] == actual[key], f'Affordance changed AME {key} configuration'
