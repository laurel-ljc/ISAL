"""Serializable phase settings plus the small Isaac configuration adapter."""
from dataclasses import dataclass, asdict
from copy import deepcopy

ROBUST_STEPS = ('clean', 'height_005', 'height_010', 'height_025', 'proprio_25', 'proprio_50',
                'proprio_100', 'mass', 'com', 'material', 'gains', 'delay',
                'push_25', 'push_50', 'push_100', 'drift_01', 'drift_02', 'topology')


@dataclass
class SparseCfg:
    phase: str = 'acquire'
    command_stage: str = 'C0'
    robust_step: str = 'clean'
    replay_probability: float = .2
    evaluation: bool = False
    evaluation_tiles: tuple = ()
    clean_evaluation: bool = False
    success_seconds: float = .3
    warmup_seconds: float = 1.
    stuck_seconds: float = 3.
    stuck_distance: float = .1
    eval_interval: int = 250
    eval_samples: int = 32
    target_level: int = 6

    def validate(self):
        if self.phase not in ('acquire', 'robust') or self.command_stage not in ('C0', 'C1', 'C2'):
            raise ValueError('Unknown sparse phase or command stage')
        if self.robust_step not in ROBUST_STEPS or (self.phase == 'acquire' and self.robust_step != 'clean'):
            raise ValueError('robust_step requires robust phase')
        if not 0 <= self.replay_probability < 1 or self.eval_interval < 1 or self.eval_samples < 32 or not 0 <= self.target_level <= 9:
            raise ValueError('Invalid replay/evaluation settings')

    def signature(self):
        return {k: v for k, v in asdict(self).items() if k not in
                ('evaluation', 'evaluation_tiles', 'clean_evaluation')}

    def perturbations(self):
        self.validate()
        i = 0 if self.phase == 'acquire' or self.clean_evaluation else ROBUST_STEPS.index(self.robust_step)
        return dict(height=(0., .005, .010, .025)[min(i, 3)],
                    proprio=(0., .25, .5, 1.)[max(0, min(i-3, 3))],
                    mass=i >= 7, com=i >= 8, material=i >= 9, gains=i >= 10, delay=i >= 11,
                    push=(0., .25, .5, 1.)[max(0, min(i-11, 3))],
                    drift=0. if i < 15 else (.01 if i == 15 else .02), topology=i >= 17)


class SparseConfigMixin:
    """Used before the existing AME/Affordance config in the MRO."""
    def _configure_custom_terrain(self, rows, cols):
        from .terrain import SparseGeneratorCfg
        if self.terrain_preset not in ('sparse', 'sparse_rescue'):
            return False
        self.sparse.validate()
        if not hasattr(self, '_sparse_nominal_events'):
            self._sparse_nominal_events = deepcopy(self.events)
            self._sparse_nominal_noise = deepcopy(self.noise)
        self.events = deepcopy(self._sparse_nominal_events)
        self.noise = deepcopy(self._sparse_nominal_noise)
        perturb = self.sparse.perturbations()
        generator = SparseGeneratorCfg(seed=self.seed, num_rows=rows or 10, num_cols=cols or 20,
            topology=perturb['topology'], rescue=self.terrain_preset == 'sparse_rescue',
            evaluation_tiles=self.sparse.evaluation_tiles)
        if generator.rescue and rows is None:
            generator.num_rows = 3
        self.scene_context.terrain_type = 'generator'
        self.scene_context.terrain_generator = generator
        self.scene_context.max_init_terrain_level = 0
        self.scene_context.height_scanner.enable_height_scan = True
        self.scene_context.height_scanner.drift_range = (0., 0.)
        self.reward.ang_vel_xy_l2.weight = -.05
        self.reward.lin_vel_z_l2.weight = -.05
        groups = {'add_base_mass': 'mass', 'scale_link_mass': 'mass', 'randomize_rigid_body_com': 'com',
                  'physics_material': 'material', 'scale_actuator_gains': 'gains', 'scale_joint_parameters': 'gains'}
        for name, group in groups.items():
            if not perturb[group]:
                setattr(self.events, name, None)
        if not perturb['push']:
            self.events.push_robot = None
        else:
            ranges = self.events.push_robot.params['velocity_range']
            self.events.push_robot.params['velocity_range'] = {
                k: tuple(x*perturb['push'] for x in v) for k, v in ranges.items()}
        for actuator in self.scene_context.robot.actuators.values():
            actuator.min_delay, actuator.max_delay = 0, 2 if perturb['delay'] else 0
        from .commands import reset_sparse_root
        self.events.reset_base.func = reset_sparse_root
        self.events.reset_base.params = {}
        self.events.reset_robot_joints.params['position_range'] = (-.03, .03)
        self.events.reset_robot_joints.params['velocity_range'] = (0., 0.)
        self.noise.add_noise = bool(perturb['height'] or perturb['proprio'])
        for name in ('ang_vel', 'projected_gravity', 'joint_pos', 'joint_vel'):
            setattr(self.noise.noise_scales, name, getattr(self.noise.noise_scales, name)*perturb['proprio'])
        self.ame_height_scan_noise = perturb['height']
        return True

    def configure(self, terrain=None, num_envs=None, terrain_rows=None, terrain_cols=None):
        if terrain is not None and terrain not in ('sparse', 'sparse_rescue'):
            raise ValueError('Sparse tasks require sparse or sparse_rescue terrain')
        return super().configure(terrain, num_envs, terrain_rows, terrain_cols)

    def _configure_scene(self):
        from .terrain import SparseTerrainImporter
        self.scene.terrain.class_type = SparseTerrainImporter
        self.scene.actor_height_scanner = self.scene.height_scanner.copy()
        d = self.sparse.perturbations()['drift']
        self.scene.actor_height_scanner.ray_cast_drift_range = {'x': (-d, d), 'y': (-d, d), 'z': (0., 0.)}
