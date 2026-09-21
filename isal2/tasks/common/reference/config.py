"""Only terrain, root reset and command settings replace the endpoint recipe."""
import math
from dataclasses import dataclass
from isaaclab.managers import EventTermCfg
from isal2.tasks.common.base import mdp
from .terrain import terrain_config, ReferenceTerrainImporter, plain
from .commands import push_base_horizontal
from . import SOURCE_SHA, RECIPE_VERSION


@dataclass
class ReferenceCfg:
    stage: int = 1


class ReferenceConfigMixin:
    def configure(self, terrain=None, num_envs=None, terrain_rows=None, terrain_cols=None):
        if terrain is not None and terrain != self.terrain_preset:
            raise ValueError(f'This stage requires {self.terrain_preset}')
        return super().configure(terrain, num_envs, terrain_rows, terrain_cols)

    def _configure_custom_terrain(self, rows, cols):
        stage = self.reference.stage
        if rows not in (None, 10):
            raise ValueError('Reference courses have 10 rows')
        self.scene_context.terrain_type = 'generator'
        self.scene_context.terrain_generator = terrain_config(stage, self.seed, 20 if cols is None else cols)
        self.scene_context.max_init_terrain_level = 5
        self.scene_context.height_scanner.drift_range = (0.,0.)
        self.ame_height_scan_noise = 0.
        self.reward.ang_vel_xy_l2.weight = -.05
        self.reward.lin_vel_z_l2.weight = -.05
        self.reward.termination_penalty.func = mdp.is_terminated
        self.commands.resampling_time_range = (10.,10.)
        self.commands.rel_standing_envs = 0.
        self.commands.rel_heading_envs = 1.
        self.commands.heading_command = True
        self.commands.heading_control_stiffness = .5
        self.commands.ranges.lin_vel_x = (0.,1.5)
        self.commands.ranges.lin_vel_y = (0.,0.)
        self.commands.ranges.ang_vel_z = (-1.,1.)
        self.commands.ranges.heading = (-math.pi,math.pi) if stage == 1 else (0.,0.)
        self.events.reset_base.func = mdp.reset_root_state_uniform
        self.events.reset_base.params = dict(
            pose_range={'x':(-.5,.5), 'y':(-.5,.5), 'yaw':(-3.14,3.14)} if stage == 1 else
                       {'x':(0.,0.), 'y':(0.,0.), 'yaw':(0.,0.)},
            velocity_range={k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')})
        self.events.push_robot = None if stage == 1 else EventTermCfg(func=push_base_horizontal,
            mode='interval', interval_range_s=(5.,8.), params={'magnitude':.1})
        return True

    def _configure_scene(self):
        self.scene.terrain.class_type = ReferenceTerrainImporter

    def reference_signature(self):
        terrain = plain(self.scene_context.terrain_generator.to_dict())
        terrain.pop('cache_dir', None)
        return dict(version=RECIPE_VERSION, source_sha=SOURCE_SHA, stage=self.reference.stage,
            terrain=terrain, commands=plain(self.commands.to_dict()),
            reset=plain(self.events.reset_base.to_dict()), episode_seconds=self.episode_length_s,
            max_init_level=self.scene_context.max_init_terrain_level)
