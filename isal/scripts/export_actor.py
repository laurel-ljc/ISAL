"""Export a Stage 4A/4B checkpoint without launching a simulator or updating weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

project = str(Path(__file__).resolve().parents[1])
if project not in sys.path:
    sys.path.insert(0, project)

import torch
from tensordict import TensorDict

from isal.learning.models import AffordanceActorCritic, AffordanceObservationActorCritic


def load_checkpoint_model(checkpoint: str):
    """Reconstruct a metadata-bound model on CPU without launching Isaac Sim."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = saved.get("model_state_dict", saved)
    metadata = state.get("_extra_state")
    if not isinstance(metadata, dict) or metadata.get("stage") not in ("4A", "4B"):
        raise ValueError("Expected a Stage 4A/4B checkpoint with model metadata; old MLP models are not migrated.")
    layout = metadata["layout"]
    rays = layout["grid_shape"][0] * layout["grid_shape"][1]
    obs = TensorDict({
        "policy": torch.zeros(1, layout["actor_history_length"] * layout["actor_frame_dim"]),
        "height_scan": torch.zeros(1, rays),
        "critic": torch.zeros(1, layout["critic_history_length"] * (layout["critic_state_dim"] + rays)),
    }, batch_size=[1])
    kwargs = {key: metadata[key] for key in (
        "obs_groups", "num_actions", "network", "activation", "actor_hidden_dims", "critic_hidden_dims",
        "actor_obs_normalization", "critic_obs_normalization", "affordance_head_enabled", "scan_preprocessing",
    )}
    model_class = AffordanceActorCritic
    if metadata["stage"] == "4B":
        model_class = AffordanceObservationActorCritic
        kwargs.update({key: metadata[key] for key in
                       ("affordance_observation", "query_coordinates", "context_scales")})
    model = model_class(obs, observation_layout=layout, **kwargs)
    model.load_state_dict(state)
    model.eval()
    return model


def export_checkpoint(checkpoint: str, output: str):
    return load_checkpoint_model(checkpoint).export_actor(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"Actor-only TorchScript/ONNX exported to {export_checkpoint(args.checkpoint, args.output)}")
