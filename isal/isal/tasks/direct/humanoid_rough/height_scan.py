"""Pure tensor utilities for the Stage 2 terrain height scan."""

from __future__ import annotations

import torch


def infer_height_scan_grid_shape(ray_starts_xy: torch.Tensor) -> tuple[int, int]:
    """Infer the canonical ``(x, y)`` grid shape from actual ray origins."""
    if ray_starts_xy.ndim != 2 or ray_starts_xy.shape[-1] != 2:
        raise ValueError(f"Expected ray starts shaped (num_rays, 2), got {tuple(ray_starts_xy.shape)}.")
    height = int(torch.unique(ray_starts_xy[:, 0]).numel())
    width = int(torch.unique(ray_starts_xy[:, 1]).numel())
    if height * width != ray_starts_xy.shape[0]:
        raise ValueError(
            "Ray origins do not form a complete Cartesian grid: "
            f"{height} x {width} != {ray_starts_xy.shape[0]}."
        )
    return height, width


def preprocess_root_relative_height_scan(
    ray_hit_z: torch.Tensor,
    root_z: torch.Tensor,
    *,
    min_height: float,
    max_height: float,
    height_scale: float,
) -> torch.Tensor:
    """Convert world-space ray hits to clipped and scaled root-relative heights."""
    if ray_hit_z.ndim != 2:
        raise ValueError(f"Expected ray hits shaped (num_envs, num_rays), got {tuple(ray_hit_z.shape)}.")
    if root_z.ndim == 1:
        root_z = root_z.unsqueeze(-1)
    if root_z.shape != (ray_hit_z.shape[0], 1):
        raise ValueError(f"Expected root z shaped ({ray_hit_z.shape[0]}, 1), got {tuple(root_z.shape)}.")
    if min_height >= max_height:
        raise ValueError("min_height must be smaller than max_height.")
    if height_scale <= 0.0:
        raise ValueError("height_scale must be positive.")

    relative_height = ray_hit_z - root_z
    relative_height = torch.where(
        torch.isfinite(relative_height),
        relative_height,
        torch.full_like(relative_height, min_height),
    )
    return torch.clamp(relative_height, min=min_height, max=max_height) / height_scale


def flat_ray_order_to_xy_grid(
    values: torch.Tensor,
    grid_shape: tuple[int, int],
    ordering: str,
) -> torch.Tensor:
    """Convert RayCaster flat order to canonical ``(N, 1, x, y)`` layout."""
    if values.ndim != 2:
        raise ValueError(f"Expected values shaped (num_envs, num_rays), got {tuple(values.shape)}.")
    height, width = grid_shape
    if values.shape[-1] != height * width:
        raise ValueError(f"Expected {height * width} rays, got {values.shape[-1]}.")
    if ordering == "xy":
        grid = values.reshape(values.shape[0], width, height).transpose(-2, -1)
    elif ordering == "yx":
        grid = values.reshape(values.shape[0], height, width)
    else:
        raise ValueError(f"Unsupported GridPattern ordering: {ordering!r}.")
    return grid.unsqueeze(1).contiguous()


def mirror_flat_height_scan(values: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    """Mirror a canonical flattened height grid across its lateral (y) axis."""
    if values.ndim != 2:
        raise ValueError(f"Expected flattened scan shaped (batch, num_rays), got {tuple(values.shape)}.")
    height, width = grid_shape
    if values.shape[-1] != height * width:
        raise ValueError(f"Expected {height * width} rays, got {values.shape[-1]}.")
    return values.reshape(values.shape[0], height, width).flip(-1).reshape(values.shape[0], -1)
