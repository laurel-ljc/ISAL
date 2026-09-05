"""Run a zero-action, non-training visualization of the Actor height scan."""

from __future__ import annotations

import argparse
import os
import sys


ISAL_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ISAL_PROJECT_DIR not in sys.path:
    sys.path.insert(0, ISAL_PROJECT_DIR)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visualize the ISAL RayCaster height-scan grid without training.")
parser.add_argument("--task", default="ISAL-Humanoid-Rough-HeightScan-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isal.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    if not env_cfg.scene_context.height_scanner.enable_height_scan_actor:
        raise ValueError(f"Task {args_cli.task!r} does not expose the Actor height scan.")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.terrain.terrain_generator.num_rows = 1
    env_cfg.scene.terrain.terrain_generator.num_cols = 1
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.scene.height_scanner.debug_vis = True
    env_cfg.scene.left_feet_scanner.debug_vis = False
    env_cfg.scene.right_feet_scanner.debug_vis = False
    env_cfg.commands.debug_vis = False
    env_cfg.log_dir = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    actions = torch.zeros(
        env.unwrapped.num_envs,
        env.unwrapped.num_actions,
        dtype=torch.float,
        device=env.unwrapped.device,
    )
    try:
        for _ in range(args_cli.steps):
            if not simulation_app.is_running():
                break
            env.step(actions)
        scan = env.unwrapped.height_scan_grid
        print("[HEIGHT_SCAN] grid_shape=", tuple(scan.shape))
        print("[HEIGHT_SCAN] value_range=", (float(scan.min()), float(scan.max())))
        print("[HEIGHT_SCAN] training_started=False")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
