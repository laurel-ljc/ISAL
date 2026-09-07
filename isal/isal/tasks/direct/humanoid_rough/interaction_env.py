"""Stage 3 height-scan task with pre-reset interaction sampling only."""

from __future__ import annotations

import torch
from isaaclab.utils.math import euler_xyz_from_quat

from isal.interaction import FootInteractionTracker
from .height_scan_env import ISALHumanoidHeightScanEnv
from .isal_env_cfg import ISALHumanoidRoughInteractionEnvCfg


class ISALHumanoidInteractionEnv(ISALHumanoidHeightScanEnv):
    cfg: ISALHumanoidRoughInteractionEnvCfg

    def __init__(self, cfg: ISALHumanoidRoughInteractionEnvCfg, **kwargs):
        # DirectRLEnv.__del__ calls close even when validation rejects construction.
        self._is_closed = True
        cfg.self_supervised.window_steps(cfg.sim.dt * cfg.decimation)
        if cfg.self_supervised.enabled and (cfg.terrain_perception.noise_std != 0 or cfg.terrain_perception.dropout_prob != 0):
            raise ValueError("Stage 3 requires the same noiseless scan for Actor and liftoff snapshots.")
        self._interaction_step_running = False
        self._interaction_last_step = -1
        super().__init__(cfg, **kwargs)
        if cfg.self_supervised.enabled:
            names = cfg.self_supervised.foot_body_names
            # Resolve each namespace separately: PhysX articulation and sensor
            # order are not assumed equal. Exact names also lock left/right order.
            self._interaction_robot_ids = self._resolve_feet(self.robot.body_names, names)
            self._interaction_sensor_ids = self._resolve_feet(self.contact_sensor.body_names, names)
            self.interaction_tracker = FootInteractionTracker(
                self.num_envs, self.height_scan_grid_shape, self.height_scanner.ray_starts[0, :, :2],
                self.step_dt, cfg.self_supervised, self.device,
            )

    @staticmethod
    def _resolve_feet(available, names):
        if any(available.count(name) != 1 for name in names):
            raise ValueError(f"Expected exactly one body for each of {names}; available={available}")
        return [available.index(name) for name in names]

    def step(self, actions):
        if not hasattr(self, "interaction_tracker"):
            return super().step(actions)
        self._interaction_step_running = True
        try:
            result = super().step(actions)
            # The parent reset replaces extras['log']; publish after it completes.
            self.extras["auxiliary"] = self.interaction_tracker.output
            self.extras["interaction_stats"] = self.interaction_tracker.statistics()
            return result
        finally:
            self._interaction_step_running = False

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        if (hasattr(self, "interaction_tracker") and self._interaction_step_running
                and self._interaction_last_step != self.common_step_counter):
            self._interaction_last_step = self.common_step_counter
            _, scan = self._compute_perceptive_height_scan()
            roll, pitch, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
            with torch.no_grad():
                self.interaction_tracker.update(
                    height_scan=scan, root_xy=self.robot.data.root_pos_w[:, :2], root_yaw=yaw,
                    scan_diagnostic_masks=self.height_scan_diagnostic_masks,
                    command=self.command_generator.command, base_ang_vel=self.robot.data.root_ang_vel_b,
                    projected_gravity=self.robot.data.projected_gravity_b,
                    foot_pos_w=self.robot.data.body_link_pos_w[:, self._interaction_robot_ids],
                    foot_vel_w=self.robot.data.body_link_lin_vel_w[:, self._interaction_robot_ids],
                    foot_force_w=self.contact_sensor.data.net_forces_w[:, self._interaction_sensor_ids],
                    base_roll_pitch=torch.stack((roll, pitch), -1), terminated=terminated, truncated=truncated,
                )
        return terminated, truncated

    def _reset_idx(self, env_ids):
        if hasattr(self, "interaction_tracker"):
            self.interaction_tracker.reset(env_ids, clear_output=not self._interaction_step_running)
            if not self._interaction_step_running:
                # Explicit reset must not leak the previous control-step packet.
                self.extras.pop("auxiliary", None)
                self.extras.pop("interaction_stats", None)
        super()._reset_idx(env_ids)
