from __future__ import annotations

import torch


def test_root_relative_preprocessing_clip_scale_and_nonfinite(isaac_app) -> None:
    from isal.tasks.direct.humanoid_rough.height_scan import preprocess_root_relative_height_scan

    ray_hit_z = torch.tensor([[0.0, 1.2, 2.0, float("nan"), float("inf")]])
    root_z = torch.tensor([1.0])
    result = preprocess_root_relative_height_scan(
        ray_hit_z,
        root_z,
        min_height=-0.8,
        max_height=0.4,
        height_scale=0.5,
    )
    torch.testing.assert_close(result, torch.tensor([[-1.6, 0.4, 0.8, -1.6, -1.6]]))


def test_dynamic_grid_shape_and_xy_layout(isaac_app) -> None:
    from isal.tasks.direct.humanoid_rough.height_scan import (
        flat_ray_order_to_xy_grid,
        infer_height_scan_grid_shape,
    )

    x = torch.arange(17, dtype=torch.float)
    y = torch.arange(11, dtype=torch.float)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    ray_starts_xy = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)

    grid_shape = infer_height_scan_grid_shape(ray_starts_xy)
    assert grid_shape == (17, 11)

    values = (10.0 * grid_x + grid_y).flatten().unsqueeze(0)
    canonical = flat_ray_order_to_xy_grid(values, grid_shape, ordering="xy")
    assert canonical.shape == (1, 1, 17, 11)
    assert canonical[0, 0, 3, 7] == 37.0


def test_height_scan_lateral_mirror_is_an_involution(isaac_app) -> None:
    from isal.tasks.direct.humanoid_rough.height_scan import mirror_flat_height_scan

    original = torch.arange(17 * 11, dtype=torch.float).reshape(1, -1)
    mirrored = mirror_flat_height_scan(original, (17, 11))
    restored = mirror_flat_height_scan(mirrored, (17, 11))

    torch.testing.assert_close(restored, original)
    original_grid = original.reshape(1, 17, 11)
    torch.testing.assert_close(mirrored.reshape(1, 17, 11), original_grid.flip(-1))
