"""Evaluate AME checkpoints on independent nominal flat-ground velocity trials."""
import argparse
import json
from pathlib import Path
from types import MethodType
from _bootstrap import bootstrap
ROOT = bootstrap()


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    parser.add_argument('--episodes_per_speed',type=int,default=64)
    parser.add_argument('--output',type=Path,required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.episodes_per_speed < 1:
        parser.error('episodes_per_speed must be positive')
    args.kit_args += f' --portable-root={ROOT.as_posix()}/.cache/evaluation'
    app = AppLauncher(args).app
    import torch
    import warp as wp
    from isal2.tasks.common.ame.ame_env import AMEEnv
    from isal2.tasks.common.ame.ame_env_cfg import RPOAMEEnvCfg
    from isal2.utils.rsl_env import RslEnvAdapter
    from isal2.deployment.export import load_actor
    wp.config.kernel_cache_dir = str(ROOT/'.cache/warp')
    cfg = RPOAMEEnvCfg()
    cfg.seed = 71
    cfg.sim.device = args.device
    cfg.sim.log_dir = str(ROOT/'outputs/sim_logs')
    cfg.ame_height_scan_noise = 0.
    cfg.noise.add_noise = False
    cfg.configure('flat',args.episodes_per_speed*4)
    # Nominal URDF masses, COM, gains, friction, armature; no impulses.
    for name,value in vars(cfg.events).items():
        if getattr(value,'mode',None) in ('startup','interval'):
            setattr(cfg.events,name,None)
    cfg.events.reset_base.params = dict(pose_range={k:(0.,0.) for k in ('x','y','yaw')},
        velocity_range={k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')})
    cfg.events.reset_robot_joints.params = dict(position_range=(0.,0.),velocity_range=(0.,0.))
    cfg.commands.heading_command = False
    cfg.commands.rel_standing_envs = cfg.commands.rel_heading_envs = 0.
    env = RslEnvAdapter(AMEEnv(cfg=cfg))
    raw = env.unwrapped
    speeds = torch.tensor([.3,.6,.9,1.2],device=env.device).repeat_interleave(args.episodes_per_speed)
    command = raw.command_generator

    def resample(term,ids):
        term.vel_command_b[ids,0] = speeds[ids]
        term.vel_command_b[ids,1:] = 0
        term.is_standing_env[ids] = False
        term.is_heading_env[ids] = False

    def update(term):
        resample(term,torch.arange(env.num_envs,device=env.device))

    command._resample_command = MethodType(resample,command)
    command._update_command = MethodType(update,command)
    reports = []
    args.output.mkdir(parents=True,exist_ok=True)
    for path in args.checkpoints:
        model,metadata = load_actor(path)
        if metadata['model_type'] != 'ame':
            raise ValueError('Reference walk evaluation requires an AME checkpoint')
        model = model.to(env.device).eval()
        raw.seed(71)
        env.reset()
        obs = env.get_observations()
        active = torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
        survived = torch.zeros_like(active)
        counts = torch.zeros(env.num_envs,device=env.device)
        vx_sum,err_sum,stagnant_sum = counts.clone(),counts.clone(),counts.clone()
        durations = counts.clone()
        # Capture velocities before auto-reset overwrites articulation data.
        original_hook = raw._after_physics_step
        latest = {}
        def capture():
            original_hook()
            latest['velocity'] = raw.robot.data.root_lin_vel_b[:,:2].clone()
        raw._after_physics_step = capture
        with torch.inference_mode():
            for step in range(raw.max_episode_length):
                obs,_,done,extras = env.step(model.act_inference(obs))
                velocity = latest['velocity']
                durations[active] += raw.step_dt
                if (step+1)*raw.step_dt > 2.:
                    counts[active] += 1
                    vx_sum[active] += velocity[active,0]
                    target = torch.stack((speeds,torch.zeros_like(speeds)),dim=1)
                    err_sum[active] += (velocity[active]-target[active]).norm(dim=-1)
                    stagnant_sum[active] += (velocity[active].norm(dim=-1)<.1).float()
                finished = active & done.bool()
                survived[finished] = extras['time_outs'][finished]
                active &= ~done.bool()
                if not active.any():
                    break
        raw._after_physics_step = original_hook
        assert not active.any(), 'Evaluation did not finish all 20 second episodes'
        rows = []
        for i,speed in enumerate((.3,.6,.9,1.2)):
            sl = slice(i*args.episodes_per_speed,(i+1)*args.episodes_per_speed)
            valid = counts[sl]>0
            vx = vx_sum[sl]/counts[sl].clamp_min(1)
            err = err_sum[sl]/counts[sl].clamp_min(1)
            # Early falls remain in survival statistics; tracking is undefined if no post-2s samples.
            mean_vx = float(vx[valid].mean()) if valid.any() else None
            mean_error = float(err[valid].mean()) if valid.any() else None
            survival = float(survived[sl].float().mean())
            row = dict(command_m_s=speed,episodes=args.episodes_per_speed,actual_vx_m_s=mean_vx,
                planar_error_m_s=mean_error,survival_rate=survival,
                stagnant_frame_fraction=float(stagnant_sum[sl].sum()/counts[sl].sum()) if valid.any() else None,
                episodes_without_post_2s_samples=int((~valid).sum()),
                mean_episode_seconds=float(durations[sl].mean()))
            row['pass'] = bool(valid.all() and survival>=.9 and .8*speed<=mean_vx<=1.2*speed
                and mean_error<=max(.1,.2*speed))
            rows.append(row)
        report = dict(checkpoint=str(path.resolve()),checkpoint_sha256=metadata['checkpoint_sha256'],
            nominal_dynamics=True,observation_noise=False,seed=71,episode_seconds=20,
            tracking_excludes_first_seconds=2,stagnation_definition='planar speed < 0.1 m/s',speeds=rows)
        reports.append(report)
        (args.output/f'{path.stem}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('WALK_EVALUATION '+json.dumps(report),flush=True)
    (args.output/'summary.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    env.close()
    print('WALK_EVALUATION_COMPLETE',flush=True)
    app.close(skip_cleanup=True)


if __name__ == '__main__':
    main()
