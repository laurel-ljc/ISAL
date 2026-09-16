"""Route-aware commands and reset events; tensor helpers also run without Isaac."""
import math
import torch


def route_velocity(speed, route_yaw, robot_yaw):
    angle = route_yaw - robot_yaw
    return torch.stack((speed*angle.cos(), speed*angle.sin()), -1)


def reset_sparse_root(env, env_ids):
    from isaaclab.utils.math import quat_from_euler_xyz
    state = env.robot.data.default_root_state[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids] + env.sparse_spawn[env_ids]
    zero = torch.zeros(len(env_ids), device=env.device)
    state[:, 3:7] = quat_from_euler_xyz(zero, zero, env.sparse_spawn_yaw[env_ids])
    state[:, 7:] = 0
    env.robot.write_root_pose_to_sim(state[:, :7], env_ids)
    env.robot.write_root_velocity_to_sim(state[:, 7:], env_ids)


def create_command(env):
    from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg

    class SparseVelocityCommand(UniformVelocityCommand):
        def _resample_command(self, env_ids):
            super()._resample_command(env_ids)
            ids = env_ids[self._env.sparse_active[env_ids]]
            if len(ids):
                self.time_left[ids] = 1.e9
                self.is_standing_env[ids] = False
                self.is_heading_env[ids] = True
                self.heading_target[ids] = self._env.sparse_heading[ids]

        def _update_command(self):
            super()._update_command()
            raw = self._env
            ids = raw.sparse_active
            self.vel_command_b[ids, :2] = route_velocity(raw.sparse_speed[ids], raw.sparse_route_yaw[ids],
                                                       self.robot.data.heading_w[ids])
            # C2 platform turns finish before translation begins.
            error = torch.atan2(torch.sin(raw.sparse_heading-self.robot.data.heading_w),
                                torch.cos(raw.sparse_heading-self.robot.data.heading_w))
            turning = raw.sparse_turn & (error.abs() > math.radians(8))
            self.vel_command_b[turning, :2] = 0
            raw.sparse_turn &= turning

        def reset(self, env_ids=None):
            result = super().reset(env_ids)
            self._update_command()
            return result

    c = env.cfg.commands
    cfg = UniformVelocityCommandCfg(asset_name='robot', resampling_time_range=c.resampling_time_range,
        rel_standing_envs=c.rel_standing_envs, rel_heading_envs=c.rel_heading_envs,
        heading_command=True, heading_control_stiffness=c.heading_control_stiffness,
        debug_vis=c.debug_vis, ranges=c.ranges)
    return SparseVelocityCommand(cfg, env)
