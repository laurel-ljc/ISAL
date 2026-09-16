"""Isaac terrain adapter. Mesh metadata carries the exact generated atlas to the importer."""
import numpy as np
import trimesh
import hashlib
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter
from .geometry import build_tile
from .terrain_cfg import columns


class SparseTerrainGenerator:
    def __init__(self, cfg, device='cpu'):
        if not 1 <= cfg.num_rows <= 10:
            raise ValueError('Sparse rows must be 1..10')
        names = columns(cfg.topology, cfg.num_cols) if not cfg.evaluation_tiles else [x[0] for x in cfg.evaluation_tiles]
        rows, cols = (1, len(names)) if cfg.evaluation_tiles else (cfg.num_rows, len(names))
        self.terrain_origins = np.zeros((rows, cols, 3))
        self.flat_patches = {}
        atlas, meshes = [], []
        for row in range(rows):
            atlas_row = []
            for col, kind in enumerate(names):
                level = cfg.evaluation_tiles[col][1] if cfg.evaluation_tiles else row
                seed = (cfg.seed + int.from_bytes(hashlib.sha256(f'{kind}:{level}'.encode()).digest()[:4], 'little')) % (2**32) \
                    if cfg.evaluation_tiles else cfg.seed + row*1009 + col*9176
                legacy = cfg.evaluation_tiles[col][2] if cfg.evaluation_tiles and len(cfg.evaluation_tiles[col]) > 2 else None
                tile = build_tile(kind, level, seed, cfg.rescue, legacy)
                origin = np.array(((row-(rows-1)/2)*8., (col-(cols-1)/2)*8., 0.))
                self.terrain_origins[row, col] = origin
                for mesh in tile.meshes:
                    mesh.apply_translation(origin)
                    meshes.append(mesh)
                record = tile.metadata()
                record['origin'] = origin.tolist()
                atlas_row.append(record)
            atlas.append(atlas_row)
        self.terrain_mesh = trimesh.util.concatenate(meshes)
        self.terrain_mesh.metadata['sparse_atlas'] = atlas


@configclass
class SparseGeneratorCfg(TerrainGeneratorCfg):
    class_type: type = SparseTerrainGenerator
    size: tuple = (8., 8.)
    num_rows: int = 10
    num_cols: int = 20
    curriculum: bool = True
    use_cache: bool = False
    topology: bool = False
    rescue: bool = False
    evaluation_tiles: tuple = ()
    sub_terrains: dict = {}


class SparseTerrainImporter(TerrainImporter):
    def import_mesh(self, name, mesh):
        self.sparse_atlas = mesh.metadata['sparse_atlas']
        super().import_mesh(name, mesh)
