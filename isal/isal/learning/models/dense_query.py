"""Pure tensor coordinate/context adapters and chunked full-grid query evaluation."""

import torch
from torch import nn


def canonical_query_coordinates(ray_origins: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    """Read an actual Cartesian ray grid, retaining its sensor offset in metres."""
    if ray_origins.shape != (grid_shape[0] * grid_shape[1], 2) or not torch.isfinite(ray_origins).all():
        raise ValueError("Ray origins must be finite (H*W,2) coordinates.")
    x, y = torch.unique(ray_origins[:, 0], sorted=True), torch.unique(ray_origins[:, 1], sorted=True)
    if (x.numel(), y.numel()) != tuple(grid_shape):
        raise ValueError("Ray coordinate axes do not match the resolved H/W.")
    if torch.unique(ray_origins, dim=0).shape[0] != ray_origins.shape[0]:
        raise ValueError("Ray origins must contain each Cartesian grid point exactly once.")
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), -1)


def current_query_context(policy: torch.Tensor, frame_dim: int, scales: torch.Tensor) -> torch.Tensor:
    """Use the last *unnormalized* noisy Actor frame; preserve noise and clipping.

    Input frame starts with angular velocity, gravity and command. Head context
    is ordered command, angular velocity, gravity, all in their physical units.
    """
    current = policy[:, -frame_dim:]
    return torch.cat((current[:, 6:9] / scales[2], current[:, :3] / scales[0],
                      current[:, 3:6] / scales[1]), dim=-1)


class DenseQueryHead(nn.Sequential):
    """Same scalar head layers/parameter names as 4A, with a scripted grid method."""

    @torch.jit.export
    def grid(self, z: torch.Tensor, context: torch.Tensor, coordinates: torch.Tensor,
             foot_side: torch.Tensor, chunk_size: int) -> torch.Tensor:
        queries = coordinates.repeat(2, 1)  # all left points, then all right points
        outputs = torch.jit.annotate(list[torch.Tensor], [])
        batch = z.shape[0]
        for start in range(0, queries.shape[0], chunk_size):
            q = queries[start:start + chunk_size]
            side = foot_side[start:start + chunk_size]
            width = q.shape[0]
            values = torch.cat((z.unsqueeze(1).expand(-1, width, -1),
                                q.unsqueeze(0).expand(batch, -1, -1),
                                side.unsqueeze(0).expand(batch, -1, -1),
                                context.unsqueeze(1).expand(-1, width, -1)), dim=-1)
            outputs.append(self.forward(values.reshape(batch * width, -1)).reshape(batch, width))
        return torch.cat(outputs, dim=-1)
