"""The existing Stage2 push, independent of the archived endpoint task."""
import torch


def push_base_horizontal(env, env_ids, magnitude=.1):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    velocity = env.robot.data.root_vel_w[env_ids].clone()
    velocity[:, :2] += torch.empty(len(env_ids), 2, device=env.device).uniform_(-magnitude, magnitude)
    env.robot.write_root_velocity_to_sim(velocity, env_ids)
