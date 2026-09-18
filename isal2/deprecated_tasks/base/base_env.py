# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import RewardManager, SceneEntityCfg

from isal2.utils.history import HistoryBuffer
from .base_env_cfg import RPOBaseEnvCfg
from .mdp.curriculum import terrain_moves


class BaseEnv(DirectRLEnv):
    cfg: RPOBaseEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.contact_sensor = self.scene["contact_sensor"]
        self.height_scanner = self.scene["height_scanner"] if cfg.scene_context.height_scanner.enable_height_scan else None
        self.left_feet_scanner_cfg = SceneEntityCfg("left_feet_scanner")
        self.right_feet_scanner_cfg = SceneEntityCfg("right_feet_scanner")
        self.robot_cfg = SceneEntityCfg("robot")
        self.robot_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg("contact_sensor", body_names=cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)
        # Sensor and articulation body indices are resolved independently.
        self.feet_robot_cfg = SceneEntityCfg("robot", body_names=cfg.robot.feet_body_names)
        self.feet_robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg("contact_sensor", body_names=cfg.robot.terminate_contacts_body_names or [])
        self.termination_contact_cfg.resolve(self.scene)
        self.command_generator = self._create_command_generator()
        self.num_actions = self.robot.num_joints
        if self.num_actions != cfg.action_space:
            raise ValueError(f"Expected {cfg.action_space} joints, found {self.num_actions}")
        self._initialize_control(cfg)

    def _create_command_generator(self):
        cfg = self.cfg
        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot", resampling_time_range=cfg.commands.resampling_time_range,
            rel_standing_envs=cfg.commands.rel_standing_envs, rel_heading_envs=cfg.commands.rel_heading_envs,
            heading_command=cfg.commands.heading_command, heading_control_stiffness=cfg.commands.heading_control_stiffness,
            debug_vis=cfg.commands.debug_vis, ranges=cfg.commands.ranges)
        return UniformVelocityCommand(command_cfg, self)

    def _initialize_control(self, cfg):
        self.action_scale = cfg.robot.action_scale
        self.clip_actions = cfg.normalization.clip_actions
        self.clip_obs = cfg.normalization.clip_observations
        self.obs_scales = cfg.normalization.obs_scales
        self.add_noise = cfg.noise.add_noise
        self.action_buffer = HistoryBuffer(cfg.robot.action_history_length, self.num_envs, self.num_actions, self.device)
        self.actor_obs_buffer = HistoryBuffer(cfg.robot.actor_obs_history_length, self.num_envs, cfg.actor_frame_dim, self.device)
        self.critic_obs_buffer = HistoryBuffer(cfg.robot.critic_obs_history_length, self.num_envs, cfg.critic_frame_dim, self.device)
        self.actions = self.robot.data.default_joint_pos.clone()
        self.noise_scale_vec = torch.zeros(cfg.actor_frame_dim, device=self.device)
        n = self.num_actions
        noise = cfg.noise.noise_scales
        self.noise_scale_vec[:3] = noise.ang_vel * self.obs_scales.ang_vel
        self.noise_scale_vec[3:6] = noise.projected_gravity * self.obs_scales.projected_gravity
        self.noise_scale_vec[9:9+n] = noise.joint_pos * self.obs_scales.joint_pos
        self.noise_scale_vec[9+n:9+2*n] = noise.joint_vel * self.obs_scales.joint_vel
        self.reward_manager = RewardManager(cfg.reward, self)
        self._automatic_reset = False
        self.obs_buf = None

    def _setup_scene(self):
        self.robot = self.scene["robot"]
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def compute_current_observations(self):
        data = self.robot.data
        scale = self.obs_scales
        actor = torch.cat([
            data.root_ang_vel_b * scale.ang_vel, data.projected_gravity_b * scale.projected_gravity,
            self.command_generator.command * scale.commands,
            (data.joint_pos - data.default_joint_pos) * scale.joint_pos,
            (data.joint_vel - data.default_joint_vel) * scale.joint_vel,
            self.action_buffer.buffer[:, -1] * scale.actions], dim=-1)
        contact = self.contact_sensor.data
        feet_ids = self.feet_cfg.body_ids
        touching = contact.net_forces_w_history[:, :, feet_ids].norm(dim=-1).amax(dim=1) > 1.0
        heights = torch.stack([
            self.scene[name].data.pos_w[:, 2] - self.scene[name].data.ray_hits_w[..., 2].mean(dim=-1)
            for name in ("left_feet_scanner", "right_feet_scanner")], dim=-1)
        heights = torch.nan_to_num((heights - 0.04).clamp(0, 1), nan=1, posinf=1, neginf=0)
        critic = torch.cat([
            actor, data.root_lin_vel_b * scale.lin_vel, touching.float(),
            contact.net_forces_w[:, feet_ids].flatten(1), contact.current_air_time[:, feet_ids],
            heights, data.joint_acc, data.applied_torque], dim=-1)
        return actor, critic

    def _height_scan(self):
        # RayCaster pos_w is the attached body position; ray offset sets the ray origin only.
        data = self.height_scanner.data
        height = data.pos_w[:, 2, None] - data.ray_hits_w[..., 2]
        height = (height - self.cfg.normalization.height_scan_offset).clamp(-1, 1)
        return torch.nan_to_num(height, nan=1, posinf=1, neginf=-1) * self.obs_scales.height_scan

    def _get_observations(self, env_ids=None):
        actor, critic = self.compute_current_observations()
        if self.height_scanner is not None:
            critic = torch.cat([critic, self._height_scan()], dim=-1)
        if actor.shape[-1] != self.cfg.actor_frame_dim or critic.shape[-1] != self.cfg.critic_frame_dim:
            raise RuntimeError(f"Observation mismatch: actor={actor.shape}, critic={critic.shape}")
        if self.add_noise:
            actor = actor + (2 * torch.rand_like(actor) - 1) * self.noise_scale_vec
        self.actor_obs_buffer.append(actor, env_ids)
        self.critic_obs_buffer.append(critic, env_ids)
        return {"policy": self.actor_obs_buffer.flatten().clamp(-self.clip_obs, self.clip_obs),
                "critic": self.critic_obs_buffer.flatten().clamp(-self.clip_obs, self.clip_obs)}

    def reset(self, *, seed=None, options=None):
        obs, extras = super().reset(seed=seed, options=options)
        self.obs_buf = obs
        return obs, extras

    def _pre_physics_step(self, actions):
        actions = actions.to(self.device).clamp(-self.clip_actions, self.clip_actions)
        self.action_buffer.append(actions)
        self.actions.copy_(self.robot.data.default_joint_pos + self.action_scale * actions)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.actions)

    def _after_physics_step(self):
        """Extension hook: sensors updated, commands/rewards/reset not yet processed."""

    def _before_reset(self, env_ids):
        """Extension hook: terminal contact/pose and done flags are still available."""

    def _after_termination_check(self):
        """Post-physics frame with current done flags, before reward and reset."""

    def step(self, actions):
        self.extras = {"log": {}}
        self._pre_physics_step(actions)
        rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if rendering and self._sim_step_counter % self.cfg.sim.render_interval == 0:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._after_physics_step()
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self._after_termination_check()
        self.reward_buf = self._get_rewards()
        # Reward and curriculum use the command that produced this transition.
        ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._automatic_reset = True
        if ids.numel():
            self._before_reset(ids)
            self._update_curriculum(ids)
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self._get_observations()
        self.extras["time_outs"] = (self.reset_time_outs & ~self.reset_terminated).clone()
        if ids.numel():
            # Full terminal observation includes the final history frame, before reset.
            self.extras["terminal_observation"] = {k: v.clone() for k, v in self.obs_buf.items()}
            self._reset_idx(ids)
            self.obs_buf = self._get_observations(env_ids=ids)
            if self.sim.has_rtx_sensors():
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
        self._automatic_reset = False
        return self.obs_buf, self.reward_buf, self.reset_terminated.clone(), self.reset_time_outs.clone(), self.extras

    def _get_rewards(self):
        return self.reward_manager.compute(dt=self.step_dt)

    def _get_dones(self):
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.robot.terminate_contacts_body_names:
            forces = self.contact_sensor.data.net_forces_w_history[:, :, self.termination_contact_cfg.body_ids]
            terminated |= (forces.norm(dim=-1).amax(dim=1) > 1.0).any(dim=1)
        if self.cfg.robot.terminate_base_orientation is not None:
            terminated |= torch.acos((-self.robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)) > self.cfg.robot.terminate_base_orientation
        if self.cfg.robot.terminate_base_height is not None:
            terminated |= self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2] < self.cfg.robot.terminate_base_height
        return terminated, self.episode_length_buf >= self.max_episode_length

    def _update_curriculum(self, env_ids):
        gen = self.cfg.scene_context.terrain_generator
        if gen is None or not gen.curriculum or not self._automatic_reset:
            return
        distance = (self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2]).norm(dim=-1)
        up, down = terrain_moves(distance, self.command_generator.command[env_ids, :2],
            self.max_episode_length_s, gen.size[0], self.episode_length_buf[env_ids],
            self.cfg.curriculum_up_fraction, self.cfg.curriculum_down_fraction,
            self.cfg.curriculum_standing_threshold)
        self.scene.terrain.update_env_origins(env_ids, up, down)
        self.extras["log"]["Curriculum/terrain_levels"] = self.scene.terrain.terrain_levels.float().mean()

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        # Explicit resets do not count as curriculum outcomes.
        self.extras.setdefault("log", {}).update(self.reward_manager.reset(env_ids))
        self.scene.reset(env_ids)
        self._prepare_reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(mode="reset", env_ids=env_ids, dt=self.step_dt,
                global_env_step_count=self._sim_step_counter // self.cfg.decimation)
        self.command_generator.reset(env_ids)
        self.event_manager.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.actions[env_ids] = self.robot.data.default_joint_pos[env_ids]
        self.episode_length_buf[env_ids] = 0
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)

    def _prepare_reset(self, env_ids):
        """Select reset metadata before reset events and command resampling."""

