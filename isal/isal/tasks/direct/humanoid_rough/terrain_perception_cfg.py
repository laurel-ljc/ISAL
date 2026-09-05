"""Terrain-perception configuration for the Stage 2 height-scan task."""

from isaaclab.utils import configclass


@configclass
class TerrainPerceptionCfg:
    size: tuple[float, float] = (1.6, 1.0)
    resolution: float = 0.1
    offset_x: float = 0.4
    min_height: float = -0.8
    max_height: float = 0.4
    height_scale: float = 0.5
    noise_std: float = 0.0
    dropout_prob: float = 0.0

    def __post_init__(self):
        if self.resolution <= 0.0:
            raise ValueError("Terrain perception resolution must be positive.")
        if self.min_height >= self.max_height:
            raise ValueError("Terrain perception min_height must be smaller than max_height.")
        if self.height_scale <= 0.0:
            raise ValueError("Terrain perception height_scale must be positive.")
        if self.noise_std < 0.0:
            raise ValueError("Terrain perception noise_std must be non-negative.")
        if not 0.0 <= self.dropout_prob < 1.0:
            raise ValueError("Terrain perception dropout_prob must be in [0, 1).")
