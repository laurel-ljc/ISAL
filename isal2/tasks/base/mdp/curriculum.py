"""Pure tensor curriculum decisions; terrain importer applies level bounds."""
import torch


def terrain_moves(distance, command_xy, episode_seconds, tile_length,
                  completed_steps, up_fraction=0.5, down_fraction=0.5,
                  standing_threshold=0.05):
    speed = torch.linalg.vector_norm(command_xy, dim=-1)
    eligible = (completed_steps > 0) & (speed > standing_threshold)
    up = eligible & (distance > tile_length * up_fraction)
    down = eligible & ~up & (distance < speed * episode_seconds * down_fraction)
    return up, down
