"""Small pure-PyTorch fixtures; Isaac Sim is only requested by runtime tests."""

from copy import deepcopy

import pytest
import torch
from tensordict import TensorDict


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        assert torch.cuda.is_available(), "Stage 4A acceptance requires the local CUDA device."
    return request.param


@pytest.fixture(autouse=True)
def small_tensor_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def model_inputs(grid=(17, 11), ordering="xy", histories=(10, 10), batch=3, device="cpu"):
    from isal.learning.models import AffordanceActorCritic

    rays = grid[0] * grid[1]
    obs = TensorDict({
        "policy": torch.randn(batch, 78 * histories[0], device=device),
        "height_scan": torch.randn(batch, rays, device=device).clamp(-3, .8),
        "critic": torch.randn(batch, histories[1] * (139 + rays), device=device),
    }, batch_size=[batch])
    kwargs = dict(
        obs_groups={"policy": ["policy", "height_scan"], "critic": ["critic"]}, num_actions=23,
        observation_layout=dict(grid_shape=list(grid), ordering=ordering,
                                actor_history_length=histories[0], critic_history_length=histories[1],
                                actor_frame_dim=78, critic_state_dim=139),
        scan_preprocessing=dict(definition="terrain_z_minus_root_z", min_height=-1.5, max_height=.4,
                                height_scale=.5, offset_x=.4, size=[(grid[0]-1)*.1, (grid[1]-1)*.1],
                                resolution=.1, noise_std=0., dropout_prob=0.),
    )
    return AffordanceActorCritic, obs, kwargs


def auxiliary_input(obs, grid):
    b = obs.shape[0]
    dev = obs.device or obs["policy"].device
    return {
        "height_scan": obs["height_scan"].reshape(b, 1, *grid).clone(),
        "query_xy": torch.zeros(b, 2, device=dev),
        "foot_side": torch.tensor([[1., 0.]], device=dev).expand(b, -1),
        "command": torch.randn(b, 3, device=dev),
        "base_ang_vel": torch.randn(b, 3, device=dev),
        "projected_gravity": torch.tensor([[0., 0., -1.]], device=dev).expand(b, -1),
        "target": torch.full((b, 1), .2, device=dev),
    }


def model_state(model):
    return deepcopy(model.state_dict())
