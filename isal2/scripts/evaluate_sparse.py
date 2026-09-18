"""Evaluate sparse checkpoints in an independent simulator with frozen normalization."""
import argparse
import json
from pathlib import Path
try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap
ROOT = bootstrap()


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-groups', type=int, help='Debug subset only; cannot satisfy full admission gates')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.kit_args += f' --portable-root={args.output.as_posix()}/kit --/log/file={args.output.as_posix()}/kit.log'
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        import torch
        import gymnasium as gym
        import isal2.deprecated_tasks
        from isal2.utils.task_config import load_task_configs
        from isal2.utils.rsl_env import RslEnvAdapter
        from isal2.deprecated_tasks.sparse.evaluation import manifest, write_report
        from isal2.modified_rsl.runners import OnPolicyRunner
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        source = checkpoint['sparse_state']
        signature = source['signature']
        scenes = manifest(source['curriculum']['unlocked'], signature['command_stage'], signature['eval_samples'])
        groups = list(dict.fromkeys((s['kind'], s['level'], s['legacy_difficulty']) for s in scenes))
        if args.max_groups:
            groups = groups[:args.max_groups]
            scenes = [s for s in scenes if (s['kind'], s['level'], s['legacy_difficulty']) in groups]
        cfg, agent = load_task_configs(args.task)
        if not hasattr(cfg, 'sparse'):
            raise ValueError('Evaluation requires a Sparse task')
        for key, value in signature.items():
            setattr(cfg.sparse, key, value)
        cfg.sparse.evaluation = True
        cfg.sparse.evaluation_tiles = tuple(groups)
        cfg.seed = 73000
        cfg.sim.device = args.device
        cfg.sim.log_dir = str(args.output/'sim_logs')
        cfg.configure(source['terrain'], len(scenes))
        agent.configure_from_env(cfg)
        env = RslEnvAdapter(gym.make(args.task, cfg=cfg))
        # Building the normal policy factory retains the saved architecture, but no learn/SL is called.
        runner = OnPolicyRunner(env, checkpoint['train_cfg'], device=args.device)
        runner.alg.policy.load_state_dict(checkpoint['model_state_dict'], strict=True)
        runner.eval_mode()
        raw = env.unwrapped
        initial_norm = {k: v.clone() for k, v in runner.alg.policy.state_dict().items() if 'normalizer' in k}
        conditions = ('stage', 'clean') if signature['phase'] == 'robust' else ('stage',)
        for condition in conditions:
            # Dynamics stay those of the stage. Clean control means clean perception;
            # the report explicitly records this, never claims nominal dynamics.
            if condition == 'clean':
                raw.add_noise = False
                sensor = raw.scene['actor_height_scanner']
                sensor.cfg.ray_cast_drift_range = dict(x=(0., 0.), y=(0., 0.), z=(0., 0.))
            records = []
            with torch.inference_mode():
                raw.sparse_evaluation_cases = scenes
                t = raw.scene.terrain
                t.terrain_types[:] = torch.tensor([groups.index((s['kind'], s['level'], s['legacy_difficulty'])) for s in scenes], device=raw.device)
                t.terrain_levels[:] = 0
                t.env_origins[:] = t.terrain_origins[0, t.terrain_types]
                raw.scene.env_origins[:] = t.env_origins
                raw.sparse_active[:] = torch.tensor([s['kind'] not in ('flat', 'stairs', 'rough') for s in scenes], device=raw.device)
                torch.manual_seed(73000)
                obs, _ = env.reset()
                raw.pop_sparse_records()
                finished = set()
                for step in range(raw.max_episode_length+1):
                    actions = runner.alg.policy.act_inference(obs)
                    obs, _, _, extras = env.step(actions)
                    if hasattr(raw, 'pop_affordance_samples'):
                        raw.pop_affordance_samples()  # Evaluation never accumulates training replay.
                    success = raw.sparse_result['success']
                    failure = raw.sparse_result['failed']
                    assert extras['time_outs'][success].all(), 'Success must bootstrap as truncation'
                    assert not extras['time_outs'][failure].any(), 'Failure must not bootstrap'
                    if success.any():
                        assert 'terminal_observation' in extras
                        assert extras['terminal_observation']['height_scan'].data_ptr() != obs['height_scan'].data_ptr()
                    for record in raw.pop_sparse_records():
                        idx = record['env_id']
                        if idx not in finished:
                            record['scene_id'] = scenes[idx]['id']
                            records.append(record)
                            finished.add(idx)
                    if step % 250 == 0:
                        print(f'EVALUATING {condition} step={step} completed={len(finished)}/{env.num_envs}', flush=True)
                    if len(finished) == env.num_envs:
                        break
                if len(finished) != env.num_envs:
                    raise RuntimeError('Evaluation failed to account for every attempt')
                print(f'EVALUATED {condition} attempts={len(finished)}', flush=True)
            report = write_report(args.output/condition, records, scenes, checkpoint, signature, condition)
            report['clean_scope'] = 'perception_only; stage dynamics retained' if condition == 'clean' else None
            (args.output/condition/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        assert all(torch.equal(v, runner.alg.policy.state_dict()[k]) for k, v in initial_norm.items())
        app.app.post_quit(0)
    except BaseException:
        import traceback
        traceback.print_exc()
        app.app.post_quit(1)
        raise
    finally:
        if env is not None:
            env.close()
        app.close(skip_cleanup=True)


if __name__ == '__main__':
    main()
