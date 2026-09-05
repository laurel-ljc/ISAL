"""Actor height-scan variant of the ISAL humanoid rough-terrain environment."""

from __future__ import annotations

import torch

from .base_env import ISALHumanoidEnv
from .height_scan import (
    flat_ray_order_to_xy_grid,
    infer_height_scan_grid_shape,
    preprocess_root_relative_height_scan,
)
from .isal_env_cfg import ISALHumanoidRoughHeightScanEnvCfg


class ISALHumanoidHeightScanEnv(ISALHumanoidEnv):
    """Expose a current root-relative terrain scan separately from proprio history."""

    cfg: ISALHumanoidRoughHeightScanEnvCfg

    @property
    def height_scan_grid_shape(self) -> tuple[int, int]:
        if not hasattr(self, "_height_scan_grid_shape"):
            ray_starts_xy = self.height_scanner.ray_starts[0, :, :2]
            self._height_scan_grid_shape = infer_height_scan_grid_shape(ray_starts_xy)
        return self._height_scan_grid_shape

    def _compute_perceptive_height_scan(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return clean scan in native ray order and canonical ``(N,1,x,y)`` order."""
        perception_cfg = self.cfg.terrain_perception
        clean_native = preprocess_root_relative_height_scan(
            self.height_scanner.data.ray_hits_w[..., 2],
            self.robot.data.root_pos_w[:, 2],
            min_height=perception_cfg.min_height,
            max_height=perception_cfg.max_height,
            height_scale=perception_cfg.height_scale,
        )
        clean_grid = flat_ray_order_to_xy_grid(
            clean_native,
            self.height_scan_grid_shape,
            self.height_scanner.cfg.pattern_cfg.ordering,
        )
        return clean_native, clean_grid

    def _actor_height_scan(self, clean_grid: torch.Tensor) -> torch.Tensor:
        """Apply Actor-only perception noise and return the canonical flattened view."""
        perception_cfg = self.cfg.terrain_perception
        actor_grid = clean_grid
        if perception_cfg.noise_std > 0.0:
            actor_grid = actor_grid + torch.randn_like(actor_grid) * perception_cfg.noise_std
        if perception_cfg.dropout_prob > 0.0:
            keep = torch.rand_like(actor_grid) >= perception_cfg.dropout_prob
            actor_grid = torch.where(keep, actor_grid, torch.zeros_like(actor_grid))
        self.height_scan_grid = actor_grid
        return actor_grid.flatten(start_dim=1)

    def _get_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        clean_native, clean_grid = self._compute_perceptive_height_scan()
        current_critic_obs = torch.cat([current_critic_obs, clean_native], dim=-1)
        actor_height_scan = self._actor_height_scan(clean_grid)

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        return {
            "policy": torch.clip(actor_obs, -self.clip_obs, self.clip_obs),
            "height_scan": torch.clip(actor_height_scan, -self.clip_obs, self.clip_obs),
            "critic": torch.clip(critic_obs, -self.clip_obs, self.clip_obs),
        }
