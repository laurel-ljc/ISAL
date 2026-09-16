"""Cooperative task mixin: no physics loop, reward, network or PPO duplication."""
import math
import torch
from .terrain_cfg import REVIEW
from .commands import create_command
from .outcomes import OutcomeTracker
from .curriculum import SparseCurriculum
from . import perception


class SparseTaskMixin:
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        atlas = self.scene.terrain.sparse_atlas
        rows, cols = len(atlas), len(atlas[0])
        max_supports = max(len(t['supports']) for row in atlas for t in row)
        self.support_atlas = torch.zeros(rows, cols, max_supports, 6, device=self.device)
        self.support_atlas[..., 2:4] = -10
        self.route_atlas = torch.zeros(rows, cols, 24, 13, device=self.device)
        self.route_counts = torch.zeros(rows, cols, dtype=torch.long, device=self.device)
        for r, row in enumerate(atlas):
            for c, tile in enumerate(row):
                self.support_atlas[r, c, :len(tile['supports'])] = torch.tensor(tile['supports'], device=self.device)
                self.route_counts[r, c] = len(tile['routes'])
                for j, route in enumerate(tile['routes']):
                    self.route_atlas[r, c, j] = torch.tensor([*route['spawn'], route['yaw'], route['entry'],
                        route['exit'], route['half_width'], *route['exit_region']], device=self.device)
        terrain = self.scene.terrain
        self.sparse_kinds = [atlas[0][c]['kind'] for c in terrain.terrain_types.tolist()]
        self.sparse_active = torch.tensor([k not in REVIEW for k in self.sparse_kinds], device=self.device)
        self.sparse_curriculum = SparseCurriculum(self.sparse_kinds, rows-1,
                                                 cfg.sparse.replay_probability, self.device, cfg.sparse.target_level)
        self.sparse_tracker = OutcomeTracker(self.num_envs, self.step_dt, cfg.sparse, self.device)
        self.sparse_routes = torch.zeros(self.num_envs, 13, device=self.device)
        self.sparse_spawn = torch.zeros(self.num_envs, 3, device=self.device)
        self.sparse_spawn_yaw = torch.zeros(self.num_envs, device=self.device)
        self.sparse_route_yaw = torch.zeros_like(self.sparse_spawn_yaw)
        self.sparse_heading = torch.zeros_like(self.sparse_spawn_yaw)
        self.sparse_speed = torch.zeros_like(self.sparse_spawn_yaw)
        self.sparse_turn = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.sparse_tracking_error = torch.zeros_like(self.sparse_spawn_yaw)
        self.sparse_initial_progress = torch.zeros_like(self.sparse_spawn_yaw)
        self.sparse_records = []
        self.sparse_phase_updates = 0
        self._sparse_scan = None
        self._sparse_scan_tick = -1
        self.sparse_evaluation_cases = None
        self.sparse_foot_ids = self.robot.find_bodies(['left_ankle_roll_link', 'right_ankle_roll_link'], preserve_order=True)[0]
        self.sparse_force_ids = self.contact_sensor.find_bodies(['left_ankle_roll_link', 'right_ankle_roll_link'], preserve_order=True)[0]

    def _create_command_generator(self):
        return create_command(self)

    def _prepare_reset(self, env_ids):
        super()._prepare_reset(env_ids)
        terrain = self.scene.terrain
        levels, cols = terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
        routes = (torch.rand(len(env_ids), device=self.device)*self.route_counts[levels, cols]).long()
        self.sparse_routes[env_ids] = self.route_atlas[levels, cols, routes]
        data = self.sparse_routes[env_ids]
        self.sparse_spawn[env_ids] = data[:, :3]
        self.sparse_route_yaw[env_ids] = data[:, 3]
        stage = self.cfg.sparse.command_stage
        use_c0 = torch.rand(len(env_ids), device=self.device) < .2
        advanced = ~use_c0 if stage != 'C0' else torch.zeros_like(use_c0)
        half_angle = torch.where(advanced, math.radians(15), math.radians(5))
        jitter = (torch.rand(len(env_ids), device=self.device)*2-1)*half_angle
        self.sparse_heading[env_ids] = data[:, 3]
        self.sparse_spawn_yaw[env_ids] = data[:, 3]+jitter
        self.sparse_turn[env_ids] = False
        if stage == 'C2':
            sideways = advanced & (torch.rand(len(env_ids), device=self.device) < .5)
            self.sparse_heading[env_ids] += sideways*math.pi/2
            self.sparse_spawn_yaw[env_ids] += advanced*math.pi/2
            self.sparse_turn[env_ids] = advanced & ~sideways & self.sparse_active[env_ids]
        jitter_xy = (torch.rand(len(env_ids), 2, device=self.device)*2-1)*torch.where(advanced, .06, .03)[:, None]
        self.sparse_spawn[env_ids, :2] += jitter_xy
        self.sparse_speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(.35, .65)
        if self.sparse_evaluation_cases is not None:
            for idx in env_ids.tolist():
                case = self.sparse_evaluation_cases[idx]
                route = self.route_atlas[terrain.terrain_levels[idx], terrain.terrain_types[idx], case['route']]
                self.sparse_routes[idx] = route
                self.sparse_spawn[idx] = route[:3]
                self.sparse_spawn[idx, :2] += torch.tensor(case['offset'], device=self.device)
                self.sparse_route_yaw[idx] = route[3]
                self.sparse_heading[idx] = route[3]+case.get('heading_offset', 0.)
                self.sparse_spawn_yaw[idx] = self.sparse_heading[idx]+case['yaw_offset']
                self.sparse_speed[idx] = case['speed']
                self.sparse_turn[idx] = False
        self.sparse_tracking_error[env_ids] = 0
        progress = (self.sparse_spawn[env_ids, :2] * torch.stack((self.sparse_route_yaw[env_ids].cos(),
                    self.sparse_route_yaw[env_ids].sin()), -1)).sum(-1)
        self.sparse_initial_progress[env_ids] = progress
        self.sparse_tracker.reset(env_ids, progress)
        self._sparse_scan_tick = -1

    def _read_actor_height(self):
        if self._sparse_scan_tick != self.common_step_counter:
            self._sparse_scan = perception.actor_scan(self)
            self._sparse_scan_tick = self.common_step_counter
        return self._sparse_scan.clone()

    def _actor_map_root(self):
        return perception.actor_map_root(self)

    def _get_dones(self):
        from isaaclab.utils.math import quat_apply
        terminated, truncated = super()._get_dones()
        data = self.robot.data
        offset = torch.tensor((.025, 0, -.04), device=self.device).expand(self.num_envs, 2, 3)
        feet = data.body_pos_w[:, self.sparse_foot_ids] + quat_apply(data.body_quat_w[:, self.sparse_foot_ids], offset)
        origin = self.scene.env_origins
        terrain = self.scene.terrain
        r = self.sparse_routes
        self.sparse_result = self.sparse_tracker.update(data.root_pos_w-origin, feet-origin[:, None],
            self.contact_sensor.data.net_forces_w[:, self.sparse_force_ids].norm(dim=-1),
            self.support_atlas[terrain.terrain_levels, terrain.terrain_types], r[:, 7:], r[:, 3], r[:, 4], r[:, 5],
            r[:, 6], self.sparse_active, self.episode_length_buf, self.command_generator.command[:, :2].norm(dim=-1),
            terminated, central_platform=r[:, 4] > 0)
        self.sparse_tracking_error += (data.root_lin_vel_b[:, :2]-self.command_generator.command[:, :2]).norm(dim=-1)
        return self.sparse_result['failed'], (truncated | self.sparse_result['success']) & ~self.sparse_result['failed']

    def _before_reset(self, env_ids):
        super()._before_reset(env_ids)
        terrain = self.scene.terrain
        result = self.sparse_result
        # One transfer per vector avoids per-scalar GPU synchronization on reset.
        indices = env_ids.tolist()
        levels = terrain.terrain_levels[env_ids].tolist()
        columns = terrain.terrain_types[env_ids].tolist()
        flags = {k: v[env_ids].tolist() for k, v in result.items()}
        flags['progress'] = (result['progress'][env_ids]-self.sparse_initial_progress[env_ids]).tolist()
        timeout = self.reset_time_outs[env_ids].tolist()
        seconds = (self.episode_length_buf[env_ids]*self.step_dt).tolist()
        tracking = (self.sparse_tracking_error[env_ids]/self.episode_length_buf[env_ids].clamp_min(1)).tolist()
        for j, i in enumerate(indices):
            tile = terrain.sparse_atlas[levels[j]][columns[j]]
            self.sparse_records.append(dict(env_id=i, kind=tile['kind'], level=tile['level'],
                geometry=tile['params'],
                success=flags['success'][j], fall=flags['fall'][j], bypass=flags['bypass'][j],
                stuck=flags['stuck'][j], timeout=timeout[j] and not flags['success'][j],
                seconds=seconds[j], progress=flags['progress'][j], tracking_error=tracking[j]))
        for name in ('success', 'fall', 'bypass', 'stuck'):
            self.extras['log']['Sparse/'+name] = self.sparse_result[name][env_ids].float().mean()
            for kind in {terrain.sparse_atlas[r][c]['kind'] for r, c in zip(levels, columns)}:
                selected = [j for j, (r, c) in enumerate(zip(levels, columns)) if terrain.sparse_atlas[r][c]['kind'] == kind]
                self.extras['log'][f'Sparse/{kind}/{name}'] = sum(flags[name][j] for j in selected)/len(selected)

    def _update_curriculum(self, env_ids):
        if self.cfg.sparse.evaluation or not self._automatic_reset:
            return
        c = self.sparse_curriculum
        c.record(env_ids, self.sparse_result['success'][env_ids])
        levels = c.sample(env_ids)
        t = self.scene.terrain
        t.terrain_levels[env_ids] = levels
        t.env_origins[env_ids] = t.terrain_origins[levels, t.terrain_types[env_ids]]
        self.scene.env_origins[env_ids] = t.env_origins[env_ids]
        self.extras['log']['Sparse/ability'] = c.ability.float().mean()

    def pop_sparse_records(self):
        records, self.sparse_records = self.sparse_records, []
        return records

    def sparse_state_dict(self):
        return dict(version=1, signature=self.cfg.sparse.signature(), terrain=self.cfg.terrain_preset,
                    curriculum=self.sparse_curriculum.state_dict(), phase_updates=self.sparse_phase_updates,
                    seed=self.cfg.seed, rows=self.cfg.scene_context.terrain_generator.num_rows,
                    cols=self.cfg.scene_context.terrain_generator.num_cols)

    def load_sparse_state_dict(self, state):
        expected = self.sparse_state_dict()
        for k in ('version', 'signature', 'terrain', 'seed', 'rows', 'cols'):
            if state[k] != expected[k]:
                raise ValueError(f'Incompatible sparse resume: {k}')
        self.sparse_curriculum.load_state_dict(state['curriculum'])
        self.sparse_phase_updates = state['phase_updates']
        t = self.scene.terrain
        t.terrain_levels[:] = self.sparse_curriculum.assigned
        t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
        self.scene.env_origins[:] = t.env_origins
