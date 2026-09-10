"""Train the base task using external RSL-RL and ISAL2 extensions."""
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
    parser.add_argument("--task", default="ISAL2-RPO-Base-v0")
    parser.add_argument("--terrain", choices=["flat", "rough", "rough_hard"], default="rough")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=12001, help="Additional updates, including on resume")
    parser.add_argument("--resume", type=Path, help="Checkpoint file to resume")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--terrain_rows", type=int)
    parser.add_argument("--terrain_cols", type=int)
    parser.add_argument("--smoke_steps", type=int, default=0, help="Validate environment instead of training")
    parser.add_argument("--check_reset", action="store_true", help="Exercise partial reset/timeout during smoke test")
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
        import warp as wp
        import gymnasium as gym
        from isaaclab.utils.io import dump_yaml
        import isal2.tasks
        from isal2.tasks.base.base_env_cfg import RPOBaseEnvCfg
        from isal2.tasks.base.agents.ppo_cfg import BaseAgentCfg
        from isal2.utils.rsl_env import RslEnvAdapter
        from isal2.modified_rsl.runners import OnPolicyRunner
        cfg = RPOBaseEnvCfg()
        wp.config.kernel_cache_dir = str(ROOT / ".cache" / "warp")
        cfg.sim.log_dir = str(ROOT / "outputs" / "sim_logs")
        cfg.seed = args.seed
        cfg.sim.device = args.device
        cfg.configure(args.terrain, args.num_envs, args.terrain_rows, args.terrain_cols)
        agent = BaseAgentCfg(seed=args.seed, device=args.device, max_iterations=args.max_iterations)
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
        if args.smoke_steps:
            from isal2.tests.sim_checks import smoke_check
            result = smoke_check(env, args.smoke_steps, args.check_reset)
        else:
            runner = OnPolicyRunner(env, agent.to_dict(), str(log_dir), args.device)
            if args.resume:
                runner.load(str(args.resume.resolve()), map_location=args.device)
            before = [p.detach().clone() for p in runner.alg.policy.parameters()]
            start_iteration = runner.current_learning_iteration
            runner.learn(args.max_iterations)
            changed = any(not torch.equal(p, q) for p, q in zip(before, runner.alg.policy.parameters()))
            if not changed:
                raise AssertionError("PPO did not update any model parameter")
            result = dict(start_iteration=start_iteration, completed_iterations=runner.current_learning_iteration,
                parameters_updated=changed, losses=runner.last_loss_dict)
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
        app.close()


if __name__ == "__main__":
    main()
