"""Reference configs with an auditable, seeded Isaac Lab terrain generator."""
import numpy as np
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg, TerrainImporter
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from .loco_hf_terrains_cfg import (HfConcentricGapTerrainCfg, HfDoubleColumnStakesTerrainCfg,
    HfAlternateColumnStakesTerrainCfg, HfStonesBridgeTerrainCfg)
from .seeding import terrain_seed


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if callable(value):
        return f'{value.__module__}:{value.__name__}'
    if isinstance(value, np.generic):
        return value.item()
    return value


class ReferenceTerrainGenerator(TerrainGenerator):
    def __init__(self, cfg, device='cpu'):
        self.reference_atlas = []
        super().__init__(cfg, device)
        for item in self.reference_atlas:
            item['origin'] = self.terrain_origins[item['row'], item['column']].tolist()
        self.terrain_mesh.metadata['reference_atlas'] = self.reference_atlas

    def _get_terrain_mesh(self, difficulty, cfg):
        self._last_difficulty = float(difficulty)
        with terrain_seed(self.cfg.seed, difficulty, cfg.function.__name__):
            return super()._get_terrain_mesh(difficulty, cfg)

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        kind = next(k for k, c in self.cfg.sub_terrains.items() if c is sub_terrain_cfg)
        self.reference_atlas.append(dict(row=row, column=col, kind=kind,
            difficulty=self._last_difficulty, parameters=plain(sub_terrain_cfg.to_dict())))
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)


class ReferenceTerrainImporter(TerrainImporter):
    def import_mesh(self, name, mesh):
        self.reference_atlas = mesh.metadata['reference_atlas']
        super().import_mesh(name, mesh)


def terrain_config(stage, seed=42, columns=20):
    if stage not in (1, 2):
        raise ValueError('Reference terrain stage must be 1 or 2')
    if columns < 10 or columns % 10:
        raise ValueError('Reference terrain columns must be a positive multiple of 10')
    stairs = dict(proportion=.1, step_height_range=(.05, .2 if stage == 1 else .25),
                  step_width=.3, platform_width=3., border_width=1., holes=False)
    sub = dict(pyramid_stairs=terrain_gen.MeshPyramidStairsTerrainCfg(**stairs),
        pyramid_stairs_inv=terrain_gen.MeshInvertedPyramidStairsTerrainCfg(**stairs))
    if stage == 1:
        sub.update(
            boxes=terrain_gen.MeshRandomGridTerrainCfg(proportion=.1, grid_width=.45,
                grid_height_range=(.05,.2), platform_width=2.),
            random_rough=terrain_gen.HfRandomUniformTerrainCfg(proportion=.1,
                noise_range=(.02,.10), noise_step=.02, downsampled_scale=.1, border_width=.25),
            hf_pyramid_slope=terrain_gen.HfPyramidSlopedTerrainCfg(proportion=.1,
                slope_range=(0.,.4), platform_width=2., border_width=.25),
            hf_pyramid_slope_inv=terrain_gen.HfInvertedPyramidSlopedTerrainCfg(proportion=.1,
                slope_range=(0.,.4), platform_width=2., border_width=.25),
            hf_steppingstones=terrain_gen.HfSteppingStonesTerrainCfg(proportion=.2,
                stone_height_max=.05, stone_width_range=(.25,.5), stone_distance_range=(.05,.25),
                platform_width=2., holes_depth=-2., border_width=.25),
            hf_gaps=HfConcentricGapTerrainCfg(proportion=.2, gap_width_range=(.1,.5),
                platform_width=2., border_width=.25, gap_depth=-2.,
                ground_width_range=(.5,.5), ground_height_max=.025))
    else:
        sub.update(
            stakes1=HfDoubleColumnStakesTerrainCfg(proportion=.1, stake_height_max=.03,
                stake_side_range=(.2,.4), stake_gap_range=(.1,.3), column_gap_range=(.1,.1),
                column_jitter=0., holes_depth=-2., platform_width=2., border_width=.25),
            stakes2=HfAlternateColumnStakesTerrainCfg(proportion=.2, stake_height_max=.03,
                stake_side_range=(.2,.4), stake_gap_range=(.05,.15), column_gap_range=(0.,.2),
                column_jitter=0., holes_depth=-2., platform_width=2., border_width=.25),
            stakes3=HfAlternateColumnStakesTerrainCfg(proportion=.2, stake_height_max=.03,
                stake_side_range=(.2,.4), stake_gap_range=(.05,.25), column_gap_range=(.3,.2),
                column_jitter=0., holes_depth=-2., platform_width=2., border_width=.25),
            hf_gaps=HfConcentricGapTerrainCfg(proportion=.1, gap_width_range=(.2,.6),
                platform_width=2., border_width=.25, gap_depth=-2.,
                ground_width_range=(.5,.5), ground_height_max=.03),
            stonebridge=HfStonesBridgeTerrainCfg(proportion=.1, platform_width=2., border_width=.25,
                holes_depth=-2., stone_height_max=.03, stone_width_range=(.25,.35),
                stone_distance_range=(.3,.5), stone_length_range=(.6,1.), stone_lateral_distance_range=(0.,0.)),
            rails=terrain_gen.MeshRailsTerrainCfg(proportion=.1, rail_height_range=(.25,.05),
                rail_thickness_range=(.1,.3), platform_width=2.))
    return TerrainGeneratorCfg(class_type=ReferenceTerrainGenerator, seed=seed, size=(8.,8.),
        border_width=50., num_rows=10, num_cols=columns, horizontal_scale=.05,
        vertical_scale=.005, slope_threshold=.75, use_cache=False, curriculum=True, sub_terrains=sub)
