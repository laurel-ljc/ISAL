"""Stage 4B: detached full-grid predictions augment the Actor's first hidden layer."""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch
from torch import nn

from ..config import AffordanceObservationCfg
from .affordance_actor_critic import ActorExport, AffordanceActorCritic
from .dense_query import DenseQueryHead, canonical_query_coordinates, current_query_context


class AffordanceObservationExport(nn.Module):
    """Complete predicted-input policy, including its head even when gate is zero."""

    def __init__(self, model):
        super().__init__()
        self.height, self.width = model.grid_shape
        self.frame_dim = model.actor_frame_dim
        self.query_chunk_size = model.query_chunk_size
        self.normalizer = deepcopy(model.actor_obs_normalizer)
        self.proprio_encoder = deepcopy(model.actor_proprio_encoder)
        self.terrain_encoder = deepcopy(model.actor_terrain_encoder)
        self.affordance_head = deepcopy(model.affordance_head)
        self.affordance_projection = deepcopy(model.affordance_projection)
        self.actor_first = deepcopy(model.actor[0])
        self.actor_tail = nn.Sequential(*[deepcopy(layer) for layer in list(model.actor.children())[1:]])
        for name in ("query_xy", "query_foot_side", "context_scales", "input_gate"):
            self.register_buffer(name, getattr(model, name).detach().clone())

    def forward(self, policy: torch.Tensor, height_scan: torch.Tensor) -> torch.Tensor:
        scan = height_scan.reshape(height_scan.shape[0], 1, self.height, self.width)
        z_prop = self.proprio_encoder(self.normalizer(policy))
        z_terrain = self.terrain_encoder(scan)
        with torch.no_grad():
            context = current_query_context(policy, self.frame_dim, self.context_scales)
            scores = self.affordance_head.grid(z_terrain.detach(), context, self.query_xy,
                                               self.query_foot_side, self.query_chunk_size).detach()
        features = self.input_gate * (2.0 * scores - 1.0)
        first = self.actor_first(torch.cat((z_prop, z_terrain), -1)) + self.affordance_projection(features)
        return self.actor_tail(first)


class AffordanceObservationActorCritic(AffordanceActorCritic):
    def __init__(self, obs, obs_groups, num_actions, *, affordance_observation=None,
                 query_coordinates=None, context_scales=None, **kwargs):
        cfg = AffordanceObservationCfg(**(affordance_observation or {}))
        if cfg.input_mode not in ("predicted", "zero"):
            raise ValueError("affordance input_mode must be 'predicted' or 'zero'.")
        if isinstance(cfg.query_chunk_size, bool) or not isinstance(cfg.query_chunk_size, int) or cfg.query_chunk_size < 1:
            raise ValueError("query_chunk_size must be a positive integer.")
        self._validate_gate(cfg.input_gate)
        if query_coordinates is None or context_scales is None:
            raise ValueError("Bind actual query_coordinates and context_scales before building the 4B runner.")
        if not kwargs.get("affordance_head_enabled", True):
            raise ValueError("Both 4B variants require an auxiliary head.")
        kwargs["affordance_head_enabled"] = True
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        if self.actor_frame_dim != 78:
            raise ValueError("The 4B context adapter requires the established 78-dimensional Actor frame.")
        coordinates = torch.as_tensor(query_coordinates, dtype=torch.float32)
        canonical = canonical_query_coordinates(coordinates, self.grid_shape)
        if not torch.equal(coordinates, canonical):
            raise ValueError("query_coordinates must be in canonical x/y order.")
        y = coordinates[:, 1].reshape(*self.grid_shape)[0]
        if not torch.allclose(y, -y.flip(0), rtol=0, atol=1e-6):
            raise ValueError("The current mirror augmentation requires a laterally symmetric query grid.")
        scale_values = [context_scales[key] for key in ("ang_vel", "projected_gravity", "commands")]
        if any(not math.isfinite(value) or value <= 0 for value in scale_values):
            raise ValueError("Context observation scales must be finite and positive.")
        self.input_mode = cfg.input_mode
        self.query_chunk_size = cfg.query_chunk_size
        self.register_buffer("query_xy", coordinates.clone())
        self.register_buffer("context_scales", torch.tensor(scale_values, dtype=torch.float32))
        sides = torch.eye(2, dtype=torch.float32).repeat_interleave(self.num_rays, dim=0)
        self.register_buffer("query_foot_side", sides)
        self.register_buffer("input_gate", torch.tensor(float(cfg.input_gate), dtype=torch.float32))
        # Preserve existing 4A head keys and layers; this only adds a grid method.
        self.affordance_head = DenseQueryHead(*list(self.affordance_head.children()))
        with torch.random.fork_rng():
            self.affordance_projection = nn.Linear(2 * self.num_rays, self.actor[0].out_features, bias=False)
        observation_metadata = asdict(cfg)
        observation_metadata.pop("input_gate")  # mutable state, not structural compatibility
        self._metadata.update({
            "schema_version": 2, "stage": "4B", "affordance_observation": observation_metadata,
            "query_coordinates": coordinates.tolist(),
            "context_scales": dict(zip(("ang_vel", "projected_gravity", "commands"), scale_values)),
            "control_context": "last_noisy_actor_frame_unscaled;current_root_yaw_xy_m",
            "affordance_projection": {"in_features": 2 * self.num_rays,
                                      "out_features": self.actor[0].out_features, "bias": False},
        })

    @staticmethod
    def _validate_gate(value):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("input_gate must be finite and in [0,1].")

    @torch.no_grad()
    def set_affordance_input_gate(self, value: float):
        self._validate_gate(value)
        self.input_gate.fill_(float(value))

    def extract_query_context(self, policy):
        self._shape(policy, (self.actor_state_size,), "policy")
        return current_query_context(policy, self.actor_frame_dim, self.context_scales)

    @torch.no_grad()
    def _grid_from_latent(self, z, state):
        return self.affordance_head.grid(z.detach(), self.extract_query_context(state),
                                         self.query_xy, self.query_foot_side, self.query_chunk_size).detach()

    @torch.no_grad()
    def predict_affordance_grid(self, obs):
        state, scan = self.get_actor_obs(obs)
        z = self.actor_terrain_encoder(scan)
        return self._grid_from_latent(z, state).reshape(state.shape[0], 2, *self.grid_shape)

    def act_inference(self, obs):
        state, scan = self.get_actor_obs(obs)
        z_prop = self.actor_proprio_encoder(self.actor_obs_normalizer(state))
        z_terrain = self.actor_terrain_encoder(scan)
        if self.input_mode == "predicted":
            scores = self._grid_from_latent(z_terrain, state)
            features = self.input_gate * (2.0 * scores - 1.0)
        else:
            features = z_terrain.new_zeros((state.shape[0], 2 * self.num_rays))
        first = self.actor[0](torch.cat((z_prop, z_terrain), -1)) + self.affordance_projection(features)
        for layer in list(self.actor.children())[1:]:
            first = layer(first)
        return first

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Check structural metadata and fixed geometry before any weight mutation.
        if state_dict.get("_extra_state") != self._metadata:
            raise ValueError("Incompatible 4B checkpoint metadata; 4A migration and cross-mode loading are not supported.")
        for name in ("query_xy", "query_foot_side", "context_scales"):
            value = state_dict.get(name)
            expected = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.shape != expected.shape or not torch.equal(
                value.to(device=expected.device, dtype=expected.dtype), expected
            ):
                raise ValueError(f"Checkpoint {name} conflicts with the bound input definition.")
        gate = state_dict.get("input_gate")
        if not isinstance(gate, torch.Tensor) or gate.shape != torch.Size([]):
            raise ValueError("Checkpoint input_gate must be a scalar tensor.")
        self._validate_gate(gate.item())
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def make_actor_export(self):
        if self.input_mode == "zero":
            return ActorExport(self).eval()
        return AffordanceObservationExport(self).eval()

    def export_actor(self, filedir):
        directory = Path(filedir)
        directory.mkdir(parents=True, exist_ok=True)
        actor = self.make_actor_export().cpu()
        inputs = (torch.zeros(1, self.actor_state_size), torch.zeros(1, self.num_rays))
        with torch.inference_mode():
            torch.jit.script(actor).save(str(directory / "actor.pt"))
            torch.onnx.export(
                actor, inputs, str(directory / "actor.onnx"), opset_version=17, dynamo=False,
                do_constant_folding=False, input_names=["policy", "height_scan"], output_names=["action_mean"],
                dynamic_axes={"policy": {0: "batch"}, "height_scan": {0: "batch"}, "action_mean": {0: "batch"}},
            )
        metadata = self.get_extra_state()
        metadata.update(export="deterministic_actor_only", includes_affordance_head=self.input_mode == "predicted",
                        input_gate=float(self.input_gate.item()),
                        scan_input="canonical_xy_flattened;already_clipped_and_scaled")
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return directory
