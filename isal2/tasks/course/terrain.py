"""Isaac terrain atlas adapter with exact discrete levels and equal type columns."""
import numpy as np
import trimesh
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter
from .geometry import STAGES, build_tile, validate_perception


class CourseTerrainGenerator:
    def __init__(self, cfg, device='cpu'):
        kinds = STAGES[cfg.stage]
        if cfg.num_rows != 10 or cfg.num_cols < len(kinds) or cfg.num_cols % len(kinds):
            raise ValueError('Courses require 10 rows and a positive multiple of the terrain type count columns')
        self.terrain_origins = np.zeros((10, cfg.num_cols, 3))
        self.flat_patches = {}
        atlas, meshes = [], []
        for level in range(10):
            row = []
            for col in range(cfg.num_cols):
                tile = build_tile(kinds[col % len(kinds)], level, cfg.seed+col*9176)
                validate_perception(tile, cfg.scan_resolution, cfg.scan_size)
                origin = np.array(((level-4.5)*8., (col-(cfg.num_cols-1)/2)*8., 0.))
                self.terrain_origins[level, col] = origin
                for mesh in tile.meshes():
                    mesh.apply_translation(origin)
                    meshes.append(mesh)
                record = tile.metadata()
                record['origin'] = origin.tolist()
                row.append(record)
            atlas.append(row)
        self.terrain_mesh = trimesh.util.concatenate(meshes)
        self.terrain_mesh.metadata['course_atlas'] = atlas


@configclass
class CourseGeneratorCfg(TerrainGeneratorCfg):
    class_type: type = CourseTerrainGenerator
    size: tuple = (8., 8.)
    num_rows: int = 10
    num_cols: int = 14
    curriculum: bool = True
    use_cache: bool = False
    stage: int = 1
    scan_resolution: float = .1
    scan_size: tuple = (1.6, 1.)
    sub_terrains: dict = {}


class CourseTerrainImporter(TerrainImporter):
    def import_mesh(self, name, mesh):
        self.course_atlas = mesh.metadata['course_atlas']
        super().import_mesh(name, mesh)
