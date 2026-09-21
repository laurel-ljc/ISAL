"""Exact distance decisions used by Isaac Lab's terrain_levels_vel curriculum."""
import torch


def distance_moves(distance, command, episode_seconds=20., tile_length=8.):
    up = distance > tile_length/2
    down = (distance < command.norm(dim=-1)*episode_seconds*.5) & ~up
    return up, down


def update_levels(levels, up, down, max_level=10):
    result = levels + up.long() - down.long()
    # Upstream samples a complete tensor, including positions below the limit.
    return torch.where(result >= max_level, torch.randint_like(result, max_level), result.clamp_min(0))
