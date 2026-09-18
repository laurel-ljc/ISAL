"""Shared lifecycle hooks for stage environments, without duplicating the RL loop."""
import torch
from .geometry import STAGES
from .curriculum import CourseCurriculum
from .commands import create_command
from .outcomes import OutcomeTracker, polygon_planes


class CourseTaskMixin:
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        atlas = self.scene.terrain.course_atlas
        cols = len(atlas[0])
        count = max(len(t['supports']) for row in atlas for t in row)
        self.course_planes = torch.zeros(10, cols, count, 5, 3, device=self.device)
        self.course_tops = torch.full((10, cols, count), float('inf'), device=self.device)
        self.course_metadata = torch.empty(10, cols, 7, device=self.device)
        for r, row in enumerate(atlas):
            for c, tile in enumerate(row):
                planes, tops = polygon_planes(tile['supports'], self.device)
                self.course_planes[r, c, :len(tops)] = planes
                self.course_tops[r, c, :len(tops)] = tops
                self.course_metadata[r, c] = torch.tensor(
                    [*tile['spawn'], *tile['goal'], tile['route_half_width']], device=self.device)
        self.course_curriculum = CourseCurriculum(self.num_envs, STAGES[cfg.course.stage], self.device)
        self.course_tracker = OutcomeTracker(self.num_envs, self.step_dt, cfg.course.success_seconds, self.device)
        self.course_spawn = torch.zeros(self.num_envs, 3, device=self.device)
        self.course_goal = torch.zeros_like(self.course_spawn)
        self.course_spawn_yaw = torch.zeros(self.num_envs, device=self.device)
        self.course_speed = torch.zeros_like(self.course_spawn_yaw)
        self.course_width = torch.zeros_like(self.course_spawn_yaw)
        self.course_result = {}
        self.course_foot_ids = self.robot.find_bodies(
            ['left_ankle_roll_link', 'right_ankle_roll_link'], preserve_order=True)[0]
        self.course_force_ids = self.contact_sensor.find_bodies(
            ['left_ankle_roll_link', 'right_ankle_roll_link'], preserve_order=True)[0]

    def _create_command_generator(self):
        return create_command(self)

    def _read_actor_height(self):
        return self._height_scan().clone()

    def _prepare_reset(self, env_ids):
        super()._prepare_reset(env_ids)
        c = self.course_curriculum
        kinds, levels = c.sample(env_ids)
        terrain = self.scene.terrain
        copies = terrain.terrain_origins.shape[1] // len(c.kinds)
        columns = kinds + len(c.kinds)*torch.randint(copies, (len(env_ids),), device=self.device)
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids] = levels, columns
        terrain.env_origins[env_ids] = terrain.terrain_origins[levels, columns]
        self.scene.env_origins[env_ids] = terrain.env_origins[env_ids]
        data = self.course_metadata[levels, columns]
        self.course_spawn[env_ids] = data[:, :3]
        self.course_spawn[env_ids, :2] += torch.empty(len(env_ids), 2, device=self.device).uniform_(
            -self.cfg.course.spawn_jitter, self.cfg.course.spawn_jitter)
        self.course_goal[env_ids] = data[:, 3:6] + self.scene.env_origins[env_ids]
        delta = data[:, 3:5] - self.course_spawn[env_ids, :2]
        self.course_spawn_yaw[env_ids] = torch.atan2(delta[:, 1], delta[:, 0])
        self.course_speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.course.speed_range)
        self.course_width[env_ids] = data[:, 6]
        self.course_tracker.reset(env_ids)

    def _get_dones(self):
        from isaaclab.utils.math import quat_apply
        failed, timeout = super()._get_dones()
        data, terrain = self.robot.data, self.scene.terrain
        offset = torch.tensor((.025, 0., -.04), device=self.device).expand(self.num_envs, 2, 3)
        feet = data.body_pos_w[:, self.course_foot_ids] + quat_apply(data.body_quat_w[:, self.course_foot_ids], offset)
        origin = self.scene.env_origins
        rows, cols = terrain.terrain_levels, terrain.terrain_types
        self.course_result = self.course_tracker.update(
            data.root_pos_w-origin, feet-origin[:, None],
            self.contact_sensor.data.net_forces_w[:, self.course_force_ids].norm(dim=-1),
            self.course_planes[rows, cols], self.course_tops[rows, cols], self.course_goal-origin,
            self.course_width, failed, timeout, self.cfg.course.success_radius)
        result = self.course_result
        return result['failed'] | result['success'], result['timeout']

    def _before_reset(self, env_ids):
        super()._before_reset(env_ids)
        result, curriculum = self.course_result, self.course_curriculum
        for k, kind in enumerate(curriculum.kinds):
            ids = env_ids[curriculum.assigned_type[env_ids] == k]
            self.extras['log'][f'Course/{kind}/level'] = curriculum.levels[:, k].float().mean()+1
            if not len(ids):
                continue
            for flag, values in result.items():
                self.extras['log'][f'Course/{kind}/{flag}'] = values[ids].float().mean()
            self.extras['log'][f'Course/{kind}/episode_seconds'] = self.episode_length_buf[ids].float().mean()*self.step_dt
            success_ids = ids[result['success'][ids]]
            if len(success_ids):
                self.extras['log'][f'Course/{kind}/completion_seconds'] = self.episode_length_buf[success_ids].float().mean()*self.step_dt

    def _update_curriculum(self, env_ids):
        if self._automatic_reset:
            self.course_curriculum.record(env_ids, self.course_result['success'][env_ids])

    def course_state_dict(self):
        return dict(version=1, signature=self.cfg.course.signature(), seed=self.cfg.seed,
            columns=self.cfg.scene_context.terrain_generator.num_cols,
            curriculum=self.course_curriculum.state_dict())

    def validate_course_state(self, state):
        expected = self.course_state_dict()
        for key in ('version', 'signature', 'seed', 'columns'):
            if state.get(key) != expected[key]:
                raise ValueError(f'Incompatible course resume: {key}; use --warm-start for stage transfer')
        levels = state['curriculum']['levels']
        if (tuple(state['curriculum']['kinds']) != self.course_curriculum.kinds or
                levels.shape != self.course_curriculum.levels.shape or levels.dtype != torch.long or
                not ((levels >= 0) & (levels <= 9)).all()):
            raise ValueError('Incompatible course curriculum state')

    def load_course_state_dict(self, state):
        self.validate_course_state(state)
        self.course_curriculum.load_state_dict(state['curriculum'])
