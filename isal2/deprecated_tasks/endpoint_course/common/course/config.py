"""Shared stage configuration; changes only course behavior and height-map noise."""
from dataclasses import dataclass, asdict
from isaaclab.managers import EventTermCfg
from .geometry import STAGES, GEOMETRY_VERSION
from .commands import reset_course_root, push_base_horizontal
from .outcomes import failure_penalty
from .terrain import CourseGeneratorCfg, CourseTerrainImporter


@dataclass
class CourseCfg:
    stage: int = 1
    success_radius: float = .3
    success_seconds: float = .2
    speed_range: tuple = (.3, .6)
    spawn_jitter: float = .05

    def signature(self):
        return dict(**asdict(self), geometry_version=GEOMETRY_VERSION)


class CourseConfigMixin:
    course_family: str = 'ame'

    def configure(self, terrain=None, num_envs=None, terrain_rows=None, terrain_cols=None):
        if terrain is not None and terrain != self.terrain_preset:
            raise ValueError(f'This stage requires terrain {self.terrain_preset}')
        return super().configure(terrain, num_envs, terrain_rows, terrain_cols)

    def _configure_custom_terrain(self, rows, cols):
        kinds = STAGES[self.course.stage]
        if rows is not None and rows != 10:
            raise ValueError('Stage courses always have exactly 10 difficulty levels')
        cols = len(kinds)*2 if cols is None else cols
        if cols < len(kinds) or cols % len(kinds):
            raise ValueError(f'terrain_cols must be a positive multiple of {len(kinds)}')
        if self.course_family not in ('ame', 'affordance') or self.terrain_preset != f'{self.course_family}_stage{self.course.stage}':
            raise ValueError('Task and terrain stage do not match')
        self.scene_context.terrain_type = 'generator'
        self.scene_context.terrain_generator = CourseGeneratorCfg(stage=self.course.stage,
            seed=self.seed, num_rows=10, num_cols=cols,
            scan_resolution=self.scene_context.height_scanner.resolution,
            scan_size=tuple(self.scene_context.height_scanner.size))
        self.scene_context.max_init_terrain_level = 0
        self.scene_context.height_scanner.drift_range = (0., 0.)
        self.ame_height_scan_noise = 0.
        self.reward.ang_vel_xy_l2.weight = -.05
        self.reward.lin_vel_z_l2.weight = -.05
        self.reward.termination_penalty.func = failure_penalty
        self.events.reset_base.func = reset_course_root
        self.events.reset_base.params = {}
        self.events.push_robot = None if self.course.stage == 1 else EventTermCfg(
            func=push_base_horizontal, mode='interval', interval_range_s=(5., 8.), params={'magnitude': .1})
        return True

    def _configure_scene(self):
        self.scene.terrain.class_type = CourseTerrainImporter
