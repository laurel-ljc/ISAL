"""Collect Stage 3 labels with bounded zero-action simulation; no runner exists."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="ISAL-Humanoid-Rough-Interaction-v0")
parser.add_argument("--num_envs", type=int, choices=range(1, 5), default=1)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--max_samples", type=int, default=1000)
parser.add_argument("--output", type=Path, default=PROJECT_DIR.parent / "outputs" / ("stage3_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or not 0 <= args.max_samples <= 1000:
    parser.error("--steps must be positive; --max_samples must be between 0 and 1000.")
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isal.tasks  # noqa: F401
from isal.interaction.recording import InteractionRecorder


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    if not hasattr(cfg, "self_supervised") or not cfg.self_supervised.enabled:
        raise ValueError("Select an interaction task with self_supervised.enabled=True.")
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.scene.terrain.terrain_generator.num_rows = 1
    cfg.scene.terrain.terrain_generator.num_cols = 1
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.height_scanner.debug_vis = False
    cfg.scene.left_feet_scanner.debug_vis = False
    cfg.scene.right_feet_scanner.debug_vis = False
    cfg.commands.debug_vis = False
    cfg.log_dir = None
    recorder = InteractionRecorder(args.max_samples, bad_threshold=cfg.self_supervised.bad_target_threshold,
                                   good_threshold=cfg.self_supervised.good_target_threshold)
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
    steps = 0
    try:
        env.reset()
        actions = torch.zeros((args.num_envs, env.unwrapped.num_actions), device=env.device)
        with torch.inference_mode():
            for _ in range(args.steps):
                if not app.is_running():
                    break
                _, _, _, extras = env.step(actions)
                recorder.append(extras["auxiliary"], extras["interaction_stats"])
                steps += 1
        summary = recorder.write(args.output, metadata={
            "source": "real_zero_action_simulation", "task": args.task, "num_envs": args.num_envs,
            "control_steps": steps, "environment_steps": steps * args.num_envs,
            "step_dt": env.unwrapped.step_dt, "grid_shape": env.unwrapped.height_scan_grid_shape,
            "height_preprocessing": {key: getattr(cfg.terrain_perception, key)
                                     for key in ("min_height", "max_height", "height_scale")},
            "scan_ordering": env.unwrapped.perceptive_observation_layout.ordering,
            "query_neighborhood": "nearest grid point plus one grid ring; border clipped",
            "seed": cfg.seed,
            "pending_slots": env.unwrapped.interaction_tracker.slots,
            "robot_foot_ids_left_right": env.unwrapped._interaction_robot_ids,
            "sensor_foot_ids_left_right": env.unwrapped._interaction_sensor_ids,
            "training_started": False, "synthetic": False,
        })
        print("[INTERACTION] " + json.dumps(summary, allow_nan=False))
        print("[INTERACTION] output=" + str(args.output.resolve()))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
