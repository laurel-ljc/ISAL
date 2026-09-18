"""Endpoint velocity commands and course-specific reset/push events."""
import torch


def endpoint_velocity(position, goal, yaw, cruise_speed, heading_stiffness=.5):
    delta = goal[..., :2] - position[..., :2]
    distance = delta.norm(dim=-1)
    angle = torch.atan2(delta[..., 1], delta[..., 0]) - yaw
    angle = torch.atan2(angle.sin(), angle.cos())
    speed = cruise_speed * (distance/.6).clamp(max=1.)
    speed = torch.where(distance < .1, 0., speed)
    turn = (heading_stiffness*angle).clamp(-1.57, 1.57)
    turn = torch.where(distance < .1, 0., turn)
    return torch.stack((speed*angle.cos(), speed*angle.sin(), turn), -1)


def reset_course_root(env, env_ids):
    from isaaclab.utils.math import quat_from_euler_xyz
    state = env.robot.data.default_root_state[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids] + env.course_spawn[env_ids]
    zeros = torch.zeros(len(env_ids), device=env.device)
    state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, env.course_spawn_yaw[env_ids])
    state[:, 7:] = 0
    env.robot.write_root_pose_to_sim(state[:, :7], env_ids)
    env.robot.write_root_velocity_to_sim(state[:, 7:], env_ids)


def push_base_horizontal(env, env_ids, magnitude=.1):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    velocity = env.robot.data.root_vel_w[env_ids].clone()
    velocity[:, :2] += torch.empty(len(env_ids), 2, device=env.device).uniform_(-magnitude, magnitude)
    env.robot.write_root_velocity_to_sim(velocity, env_ids)


def create_command(env):
    from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg

    class EndpointVelocityCommand(UniformVelocityCommand):
        def _resample_command(self, env_ids):
            self.time_left[env_ids] = 1.e9
            self.is_standing_env[env_ids] = False
            self.is_heading_env[env_ids] = False

        def _update_command(self):
            raw = self._env
            self.vel_command_b[:] = endpoint_velocity(
                self.robot.data.root_pos_w, raw.course_goal, self.robot.data.heading_w,
                raw.course_speed, raw.cfg.commands.heading_control_stiffness)

        def reset(self, env_ids=None):
            result = super().reset(env_ids)
            self._update_command()
            return result

    c = env.cfg.commands
    cfg = UniformVelocityCommandCfg(asset_name='robot', resampling_time_range=c.resampling_time_range,
        rel_standing_envs=0., rel_heading_envs=0., heading_command=True,
        heading_control_stiffness=c.heading_control_stiffness, debug_vis=c.debug_vis, ranges=c.ranges)
    return EndpointVelocityCommand(cfg, env)
