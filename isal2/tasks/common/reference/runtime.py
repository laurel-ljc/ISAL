"""Distance curriculum with independent reset, logging and checkpoint state."""
import torch
from .curriculum import distance_moves


class ReferenceTaskMixin:
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # Base reset resamples the raw yaw value. Resolve heading immediately
        # after the new pose is available, before the first policy observation.
        self.command_generator._update_command()

    def _read_actor_height(self):
        return self._height_scan().clone()

    def _update_curriculum(self, env_ids):
        if not self._automatic_reset:
            return
        terrain = self.scene.terrain
        distance = (self.robot.data.root_pos_w[env_ids,:2]-self.scene.env_origins[env_ids,:2]).norm(dim=-1)
        up, down = distance_moves(distance, self.command_generator.command[env_ids,:2],
            self.max_episode_length_s, terrain.cfg.terrain_generator.size[0])
        terrain.update_env_origins(env_ids, up, down)
        self.extras['log']['Curriculum/terrain_levels'] = terrain.terrain_levels.float().mean()
        self.extras['log']['Reference/move_up'] = up.float().mean()
        self.extras['log']['Reference/move_down'] = down.float().mean()
        self.extras['log']['Reference/distance'] = distance.mean()

    def _before_reset(self, env_ids):
        super()._before_reset(env_ids)
        t = self.scene.terrain
        atlas = t.reference_atlas
        columns = {x['column']:x['kind'] for x in atlas}
        for kind in dict.fromkeys(columns.values()):
            match = torch.zeros(len(env_ids),dtype=torch.bool,device=self.device)
            for column, name in columns.items():
                if name == kind:
                    match |= t.terrain_types[env_ids] == column
            ids = env_ids[match]
            if not len(ids):
                continue
            prefix = f'Reference/{kind}/'
            self.extras['log'][prefix+'level'] = t.terrain_levels[ids].float().mean()+1
            self.extras['log'][prefix+'failure'] = self.reset_terminated[ids].float().mean()
            self.extras['log'][prefix+'timeout'] = (self.reset_time_outs[ids]&~self.reset_terminated[ids]).float().mean()
            self.extras['log'][prefix+'distance'] = (self.robot.data.root_pos_w[ids,:2]-self.scene.env_origins[ids,:2]).norm(dim=-1).mean()
            self.extras['log'][prefix+'episode_seconds'] = self.episode_length_buf[ids].float().mean()*self.step_dt

    def _after_physics_step(self):
        super()._after_physics_step()
        cmd = self.command_generator.command
        vel = self.robot.data.root_lin_vel_b
        self.extras['log'].update({
            'Reference/command_vx':cmd[:,0].mean(), 'Reference/command_yaw_abs':cmd[:,2].abs().mean(),
            'Reference/actual_vx':vel[:,0].mean(),
            'Reference/stagnant':((cmd[:,0]>.3)&(vel[:,0]<.1)).float().mean()})

    def reference_state_dict(self):
        t = self.scene.terrain
        return dict(version=1, signature=self.cfg.reference_signature(), num_envs=self.num_envs,
            levels=t.terrain_levels.clone(), columns=t.terrain_types.clone())

    def validate_reference_state(self, state):
        if state.get('version') != 1 or state.get('signature') != self.cfg.reference_signature() or state.get('num_envs') != self.num_envs:
            raise ValueError('Incompatible reference curriculum resume; use --warm-start for model-only initialization')
        t = self.scene.terrain
        for name, limit in (('levels',t.terrain_origins.shape[0]),('columns',t.terrain_origins.shape[1])):
            value = state.get(name)
            if value is None or value.dtype != torch.long or value.shape != (self.num_envs,) or not ((value>=0)&(value<limit)).all():
                raise ValueError(f'Invalid reference curriculum {name}')

    def load_reference_state_dict(self, state):
        self.validate_reference_state(state)
        t = self.scene.terrain
        t.terrain_levels.copy_(state['levels'])
        t.terrain_types.copy_(state['columns'])
        self._sync_reference_origins()

    def _sync_reference_origins(self):
        t = self.scene.terrain
        t.env_origins[:] = t.terrain_origins[t.terrain_levels,t.terrain_types]
        self.scene.env_origins[:] = t.env_origins

    def reset_reference_curriculum(self):
        t = self.scene.terrain
        t.terrain_levels.random_(0, min(5,t.terrain_origins.shape[0]-1)+1)
        t.terrain_types[:] = torch.div(torch.arange(self.num_envs,device=self.device),
            self.num_envs/t.terrain_origins.shape[1],rounding_mode='floor').long()
        self._sync_reference_origins()
