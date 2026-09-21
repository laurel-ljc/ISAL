"""Train a registered ISAL2 task using external RSL-RL and project extensions."""
import argparse
from datetime import datetime
from pathlib import Path
import json
import traceback
import sys
try:
    from ._bootstrap import bootstrap
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("_isal2_bootstrap", Path(__file__).with_name("_bootstrap.py"))
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    bootstrap = _module.bootstrap
ROOT = bootstrap()


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="ISAL2-RPO-AME-Stage1-v0")
    parser.add_argument("--terrain", default=None, help="Terrain preset validated by the selected task")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=12001, help="Additional updates, including on resume")
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument("--resume", type=Path, help="Checkpoint file to resume")
    load_group.add_argument("--warm-start", type=Path, help="Strict model-only initialization with fresh training state")
    load_group.add_argument("--advance-from", type=Path, help="Sparse checkpoint to advance one stage")
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--phase", choices=['acquire', 'robust'])
    parser.add_argument("--command-stage", choices=['C0', 'C1', 'C2'])
    parser.add_argument("--robust-step")
    parser.add_argument("--replay-probability", type=float)
    parser.add_argument("--skip-evaluation", action='store_true', help="Debug only: no validation or curriculum unlocking")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--save_interval", type=int, help="Checkpoint interval in completed PPO updates")
    parser.add_argument("--terrain_rows", type=int)
    parser.add_argument("--terrain_cols", type=int)
    parser.add_argument("--smoke_steps", type=int, default=0, help="Validate environment instead of training")
    parser.add_argument("--check_reset", action="store_true", help="Exercise partial reset/timeout during smoke test")
    parser.add_argument("--affordance_warmup", type=int, help="Override affordance gate warm-up iterations")
    parser.add_argument("--affordance_ramp", type=int, help="Override affordance gate ramp iterations")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.max_iterations < 1 or args.smoke_steps < 0:
        parser.error("iterations must be positive and smoke_steps nonnegative")
    cache = ROOT / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    kit_defaults = (
        f"--portable-root={cache.as_posix()}/kit "
        f"--/app/userConfigPath={cache.as_posix()}/user.config.json "
        f"--/log/file={cache.as_posix()}/kit.log")
    args.kit_args = kit_defaults + " " + args.kit_args
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        import torch
        from isal2.utils.checkpoint import load_checkpoint
        import warp as wp
        import gymnasium as gym
        from isaaclab.utils.io import dump_yaml
        import isal2.tasks
        from isal2.utils.task_config import load_task_configs
        from isal2.utils.rsl_env import RslEnvAdapter
        from isal2.modified_rsl.runners import OnPolicyRunner
        cfg, agent = load_task_configs(args.task)
        wp.config.kernel_cache_dir = str(ROOT / ".cache" / "warp")
        cfg.sim.log_dir = str(ROOT / "outputs" / "sim_logs")
        cfg.seed = args.seed
        cfg.sim.device = args.device
        sparse = hasattr(cfg, 'sparse')
        if any(x is not None for x in (args.phase, args.command_stage, args.robust_step, args.replay_probability)) and not sparse:
            parser.error('Sparse stage options require a Sparse task')
        if sparse:
            # Resume/advance inherit source settings unless explicitly overridden.
            if args.resume or args.advance_from:
                saved = load_checkpoint(args.resume or args.advance_from, map_location='cpu')
                if 'sparse_state' not in saved:
                    parser.error('Legacy checkpoints must use --warm-start')
                for key, value in saved['sparse_state']['signature'].items():
                    setattr(cfg.sparse, key, value)
            for key, value in (('phase', args.phase), ('command_stage', args.command_stage),
                               ('robust_step', args.robust_step), ('replay_probability', args.replay_probability)):
                if value is not None:
                    setattr(cfg.sparse, key, value)
            if not (args.resume or args.advance_from) and (cfg.sparse.phase != 'acquire' or cfg.sparse.command_stage != 'C0') and not args.smoke_steps:
                parser.error('Training later stages requires --advance-from and --validation-report')
        if args.advance_from and not (sparse and args.validation_report):
            parser.error('--advance-from requires a Sparse task and --validation-report')
        cfg.configure(args.terrain, args.num_envs, args.terrain_rows, args.terrain_cols)
        affordance_config_checked = args.smoke_steps > 0 and hasattr(cfg, 'course') and hasattr(cfg, 'collection')
        if affordance_config_checked:
            from isal2.tests.course_sim_checks import check_affordance_config
            check_affordance_config(cfg)
        agent.seed, agent.device, agent.max_iterations = args.seed, args.device, args.max_iterations
        if args.save_interval is not None:
            if args.save_interval < 1:
                parser.error('save_interval must be positive')
            agent.save_interval = args.save_interval
        if hasattr(agent, "configure_from_env"):
            agent.configure_from_env(cfg)
        for argument, key in ((args.affordance_warmup, "warmup_iterations"), (args.affordance_ramp, "ramp_iterations")):
            if argument is not None:
                if not hasattr(agent, "affordance"):
                    parser.error("Affordance overrides require the Affordance task")
                if agent.affordance.get('gate_mode') == 'immediate' and argument != 0:
                    parser.error('Affordance Stage2 enables predictions immediately; warm-up and ramp must be zero')
                agent.affordance[key] = argument
        run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        if Path(run_name).name != run_name or run_name in (".", ".."):
            parser.error("run_name must be a single directory name")
        log_dir = ROOT / "outputs" / agent.experiment_name / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(str(log_dir / "env.yaml"), cfg)
        dump_yaml(str(log_dir / "agent.yaml"), agent.to_dict())
        (log_dir / "arguments.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
        env = RslEnvAdapter(gym.make(args.task, cfg=cfg))
        joint_names = env.unwrapped.robot.joint_names
        (log_dir / "joint_names.json").write_text(json.dumps(joint_names, indent=2), encoding="utf-8")
        print(f"ISAL2 output: {log_dir}", flush=True)
        if hasattr(cfg, 'course'):
            (log_dir / 'terrain_atlas.json').write_text(
                json.dumps(env.unwrapped.scene.terrain.course_atlas, indent=2), encoding='utf-8')
        if hasattr(cfg, 'reference'):
            (log_dir / 'terrain_atlas.json').write_text(
                json.dumps(env.unwrapped.scene.terrain.reference_atlas, indent=2), encoding='utf-8')
        if args.smoke_steps:
            from isal2.tests.sim_checks import smoke_check
            result = smoke_check(env, args.smoke_steps, args.check_reset)
            if affordance_config_checked:
                result['affordance_matches_ame'] = True
        else:
            runner_cls = OnPolicyRunner
            if hasattr(agent, "runner_class"):
                from importlib import import_module
                module, name = agent.runner_class.split(":")
                runner_cls = getattr(import_module(module), name)
            runner = runner_cls(env, agent.to_dict(), str(log_dir), args.device)
            if args.resume:
                runner.load(str(args.resume.resolve()), map_location=args.device)
            elif args.warm_start:
                from isal2.modified_rsl.runners.checkpoint import warm_start
                warm_start(runner, str(args.warm_start.resolve()))
            elif args.advance_from:
                from isal2.modified_rsl.runners.checkpoint import advance
                advance(runner, str(args.advance_from.resolve()), json.loads(args.validation_report.read_text(encoding='utf-8')))
            if sparse:
                (log_dir / 'terrain_atlas.json').write_text(json.dumps(env.unwrapped.scene.terrain.sparse_atlas, indent=2), encoding='utf-8')
                if not args.skip_evaluation:
                    from isal2.deprecated_tasks.sparse.evaluation import validation_callback
                    runner.validation_callback = validation_callback(args.task, log_dir, args.device)
            before = {name: p.detach().clone() for name, p in runner.alg.policy.named_parameters()}
            start_iteration = runner.current_learning_iteration
            runner.learn(args.max_iterations)
            updated = {name: not torch.equal(before[name], p) for name, p in runner.alg.policy.named_parameters()}
            changed = any(updated.values())
            if not changed:
                raise AssertionError("PPO did not update any model parameter")
            result = dict(start_iteration=start_iteration, completed_iterations=runner.current_learning_iteration,
                parameters_updated=changed, losses=runner.last_loss_dict)
            if hasattr(runner.alg.policy, "terrain_attention"):
                groups = ["terrain_attention.policy_encoder", "terrain_attention.position_encoding",
                          "terrain_attention.query.", "terrain_attention.attention", "actor", "critic"]
                result["modules_updated"] = {group: any(v for k, v in updated.items() if k.startswith(group)) for group in groups}
                if not all(result["modules_updated"].values()):
                    raise AssertionError(f"AME modules did not update: {result['modules_updated']}")
            if hasattr(runner, "supervised_updates"):
                result["supervised_updates_total"] = runner.supervised_updates
                result["affordance_parameters_updated"] = any(v for k, v in updated.items() if k.startswith("affordance_net."))
                result["collection"] = dict(env.unwrapped.collector.stats)
            if runner.logger.writer:
                runner.logger.writer.flush()
                runner.logger.writer.close()
        (log_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("ISAL2_RESULT " + json.dumps(result), flush=True)
        app.app.post_quit(0)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        app.app.post_quit(1)
        raise
    finally:
        print("[ISAL2] Closing environment", flush=True)
        if env is not None:
            env.close()
        print("[ISAL2] Closing simulator", flush=True)
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
