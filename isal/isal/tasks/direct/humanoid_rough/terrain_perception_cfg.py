"""Terrain-perception configuration for the Stage 2 height-scan task."""

import math

from isaaclab.utils import configclass


@configclass
class TerrainPerceptionCfg:
    size: tuple[float, float] = (1.6, 1.0)
    resolution: float = 0.1
    offset_x: float = 0.4
    min_height: float = -1.5
    max_height: float = 0.4
    height_scale: float = 0.5
    noise_std: float = 0.0
    dropout_prob: float = 0.0

    def __post_init__(self):
        self.validate_values()

    def validate_values(self):
        """Also called after CLI overrides, before constructing the sensor."""
        if len(self.size) != 2 or any(not math.isfinite(v) or v <= 0 for v in self.size):
            raise ValueError("Terrain perception size must contain two finite positive lengths.")
        if not all(math.isfinite(v) for v in (self.resolution, self.offset_x, self.min_height,
                                              self.max_height, self.height_scale, self.noise_std,
                                              self.dropout_prob)):
            raise ValueError("Terrain perception parameters must be finite.")
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
