"""Stage 4A perceptive Actor/Critic and training-only query predictor.

This module deliberately does not import task packages or Isaac Sim. It implements
the local RSL-RL 3.3 policy protocol, including its boolean load_state_dict result.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.networks import EmpiricalNormalization

from ..config import AffordanceNetworkCfg


def _activation(name: str) -> nn.Module:
    choices = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}
    if name not in choices:
        raise ValueError(f"Unsupported activation: {name!r}; choose {tuple(choices)}.")
    return choices[name]()


def _positive_dimensions(values, name):
    if not values or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
        raise ValueError(f"{name} must contain positive integer dimensions.")


def _mlp(input_dim, hidden_dims, output_dim=None, activation="elu"):
    _positive_dimensions(hidden_dims, "MLP hidden dimensions")
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), _activation(activation)))
        input_dim = width
    if output_dim is not None:
        layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class TerrainEncoder(nn.Module):
    def __init__(self, grid_shape, cfg: AffordanceNetworkCfg, activation="elu"):
        super().__init__()
        lengths = [len(getattr(cfg, key)) for key in
                   ("terrain_channels", "terrain_kernels", "terrain_strides", "terrain_paddings")]
        if len(set(lengths)) != 1:
            raise ValueError("Terrain convolution settings must have equal lengths.")
        for key in ("terrain_channels", "terrain_kernels", "terrain_strides"):
            _positive_dimensions(getattr(cfg, key), key)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in cfg.terrain_paddings):
            raise ValueError("Terrain paddings must be nonnegative integers.")
        _positive_dimensions([cfg.terrain_fc_dim, cfg.terrain_latent_dim], "Terrain FC dimensions")
        layers = []
        in_channels = 1
        for out_channels, kernel, stride, padding in zip(
            cfg.terrain_channels, cfg.terrain_kernels, cfg.terrain_strides, cfg.terrain_paddings
        ):
            layers.extend((nn.Conv2d(in_channels, out_channels, kernel, stride, padding), _activation(activation)))
            in_channels = out_channels
        self.convolutions = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *grid_shape, device=next(self.parameters()).device)
            self.flat_dim = self.convolutions(dummy).flatten(1).shape[1]
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(self.flat_dim, cfg.terrain_fc_dim),
            _activation(activation), nn.Linear(cfg.terrain_fc_dim, cfg.terrain_latent_dim),
        )

    def forward(self, scan: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convolutions(scan))


class ActorExport(nn.Module):
    """Self-contained deterministic actor; no head, critic, distribution or simulator."""

    def __init__(self, model):
        super().__init__()
        self.height, self.width = model.grid_shape
        self.normalizer = deepcopy(model.actor_obs_normalizer)
        self.proprio_encoder = deepcopy(model.actor_proprio_encoder)
        self.terrain_encoder = deepcopy(model.actor_terrain_encoder)
        self.actor = deepcopy(model.actor)

    def forward(self, policy: torch.Tensor, height_scan: torch.Tensor) -> torch.Tensor:
        scan = height_scan.reshape(height_scan.shape[0], 1, self.height, self.width)
        z_prop = self.proprio_encoder(self.normalizer(policy))
        z_terrain = self.terrain_encoder(scan)
        return self.actor(torch.cat((z_prop, z_terrain), dim=-1))


class AffordanceActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self, obs, obs_groups, num_actions, *, observation_layout=None, scan_preprocessing=None,
        network=None, affordance_head_enabled=False, actor_obs_normalization=True,
        critic_obs_normalization=True, actor_hidden_dims=(256, 128), critic_hidden_dims=(256, 128),
        activation="elu", init_noise_std=1.0, noise_std_type="scalar", state_dependent_std=False,
    ):
        super().__init__()
        if observation_layout is None or scan_preprocessing is None:
            raise ValueError("Bind observation_layout and scan_preprocessing from the actual environment before runner creation.")
        if noise_std_type != "scalar" or state_dependent_std:
            raise ValueError("Stage 4A supports the existing state-independent scalar action std only.")
        if not math.isfinite(init_noise_std) or init_noise_std <= 0:
            raise ValueError("init_noise_std must be finite and positive.")
        if obs_groups.get("policy") != ["policy", "height_scan"] or obs_groups.get("critic") != ["critic"]:
            raise ValueError("Expected policy groups ['policy', 'height_scan'] and critic group ['critic'].")
        self.obs_groups = deepcopy(obs_groups)
        self.layout = deepcopy(observation_layout)
        self.grid_shape = tuple(self.layout["grid_shape"])
        self.layout["grid_shape"] = list(self.grid_shape)
        if len(self.grid_shape) != 2:
            raise ValueError("Expected a two-dimensional scan grid.")
        self.actor_history = self.layout["actor_history_length"]
        self.critic_history = self.layout["critic_history_length"]
        self.actor_frame_dim = self.layout["actor_frame_dim"]
        self.critic_state_dim = self.layout["critic_state_dim"]
        _positive_dimensions([*self.grid_shape, self.actor_history, self.critic_history,
                              self.actor_frame_dim, self.critic_state_dim, num_actions], "Observation layout")
        self.ordering = self.layout["ordering"]
        if self.ordering not in ("xy", "yx"):
            raise ValueError(f"Unsupported native scan ordering: {self.ordering!r}.")
        self.num_rays = math.prod(self.grid_shape)
        self.actor_state_size = self.actor_history * self.actor_frame_dim
        self.critic_state_size = self.critic_history * self.critic_state_dim
        self.critic_frame_size = self.critic_state_dim + self.num_rays
        self.num_actions = num_actions
        self.get_actor_obs(obs)
        self.get_critic_obs(obs)
        cfg = AffordanceNetworkCfg(**(network or {}))
        self.affordance_head_enabled = bool(affordance_head_enabled)
        self.actor_obs_normalization = bool(actor_obs_normalization)
        self.critic_obs_normalization = bool(critic_obs_normalization)
        self.noise_std_type = noise_std_type
        self.state_dependent_std = False
        self._metadata = {
            "schema_version": 1, "stage": "4A", "layout": self.layout,
            "scan_preprocessing": deepcopy(scan_preprocessing), "obs_groups": self.obs_groups,
            "num_actions": num_actions, "network": asdict(cfg), "activation": activation,
            "actor_hidden_dims": list(actor_hidden_dims), "critic_hidden_dims": list(critic_hidden_dims),
            "actor_obs_normalization": self.actor_obs_normalization,
            "critic_obs_normalization": self.critic_obs_normalization,
            "affordance_head_enabled": self.affordance_head_enabled,
            "query_context": "liftoff_yaw_xy_m;side_onehot;command_m_s_rad_s;ang_vel_body_rad_s;gravity_body",
        }
        self.actor_obs_normalizer = (EmpiricalNormalization(self.actor_state_size)
                                     if self.actor_obs_normalization else nn.Identity())
        self.critic_obs_normalizer = (EmpiricalNormalization(self.critic_state_size)
                                      if self.critic_obs_normalization else nn.Identity())
        self.actor_terrain_encoder = TerrainEncoder(self.grid_shape, cfg, activation)
        self.actor_proprio_encoder = _mlp(self.actor_state_size, cfg.proprio_hidden_dims, activation=activation)
        self.actor = _mlp(cfg.proprio_hidden_dims[-1] + cfg.terrain_latent_dim,
                          actor_hidden_dims, num_actions, activation)
        self.critic_terrain_encoder = TerrainEncoder(self.grid_shape, cfg, activation)
        self.critic_state_encoder = _mlp(self.critic_state_size, cfg.critic_state_hidden_dims, activation=activation)
        self.critic = _mlp(cfg.critic_state_hidden_dims[-1] + self.critic_history * cfg.terrain_latent_dim,
                           critic_hidden_dims, 1, activation)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        if self.affordance_head_enabled:
            # Restore CPU/CUDA RNG states: enabling the head must not perturb
            # common initialization or subsequent environment/action randomness.
            with torch.random.fork_rng():
                self.affordance_head = nn.Sequential(
                    _mlp(cfg.terrain_latent_dim + 13, cfg.affordance_hidden_dims, 1, activation),
                    nn.Sigmoid(),
                )

    @staticmethod
    def _shape(value, tail, name):
        if value.ndim != len(tail) + 1 or tuple(value.shape[1:]) != tuple(tail):
            raise ValueError(f"{name}: expected (batch, {', '.join(map(str, tail))}), got {tuple(value.shape)}.")

    def get_actor_obs(self, obs):
        state, scan = obs["policy"], obs["height_scan"]
        self._shape(state, (self.actor_state_size,), "policy")
        self._shape(scan, (self.num_rays,), "height_scan")
        if state.shape[0] != scan.shape[0]:
            raise ValueError("Policy and height-scan batch sizes differ.")
        return state, scan.reshape(scan.shape[0], 1, *self.grid_shape)

    def get_critic_obs(self, obs):
        values = obs["critic"]
        self._shape(values, (self.critic_history * self.critic_frame_size,), "critic")
        history = values.reshape(values.shape[0], self.critic_history, self.critic_frame_size)
        state = history[..., :self.critic_state_dim].flatten(1)
        native = history[..., self.critic_state_dim:]
        h, w = self.grid_shape
        if self.ordering == "xy":
            scan = native.reshape(values.shape[0] * self.critic_history, w, h).transpose(-2, -1)
        else:
            scan = native.reshape(values.shape[0] * self.critic_history, h, w)
        return state, scan.unsqueeze(1).contiguous()

    def act_inference(self, obs):
        state, scan = self.get_actor_obs(obs)
        z_prop = self.actor_proprio_encoder(self.actor_obs_normalizer(state))
        z_terrain = self.actor_terrain_encoder(scan)
        return self.actor(torch.cat((z_prop, z_terrain), dim=-1))

    def act(self, obs, **kwargs):
        mean = self.act_inference(obs)
        self.distribution = Normal(mean, self.std.expand_as(mean), validate_args=False)
        return self.distribution.sample()

    def evaluate(self, obs, **kwargs):
        state, scan = self.get_critic_obs(obs)
        z_state = self.critic_state_encoder(self.critic_obs_normalizer(state))
        z_scan = self.critic_terrain_encoder(scan).reshape(state.shape[0], -1)
        return self.critic(torch.cat((z_state, z_scan), dim=-1))

    def forward(self, obs):
        return self.act_inference(obs)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(-1)

    def reset(self, dones=None):
        pass

    @torch.no_grad()
    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs)[0])
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs)[0])

    def predict_affordance(self, aux_batch):
        if not hasattr(self, "affordance_head"):
            raise RuntimeError("This model has no affordance head (Baseline or actor-only export).")
        scan = aux_batch["height_scan"]
        self._shape(scan, (1, *self.grid_shape), "auxiliary height_scan")
        context = []
        for key, size in (("query_xy", 2), ("foot_side", 2), ("command", 3),
                          ("base_ang_vel", 3), ("projected_gravity", 3)):
            value = aux_batch[key]
            self._shape(value, (size,), f"auxiliary {key}")
            if value.shape[0] != scan.shape[0]:
                raise ValueError(f"Auxiliary {key} batch size differs from scan.")
            context.append(value)
        z = self.actor_terrain_encoder(scan)
        return self.affordance_head(torch.cat((z, *context), dim=-1))

    def get_extra_state(self):
        return deepcopy(self._metadata)

    def set_extra_state(self, state):
        if state != self._metadata:
            raise ValueError("Checkpoint layout, preprocessing or model configuration does not match this model.")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Check metadata before mutating any weights, even when strict=False.
        if state_dict.get("_extra_state") != self._metadata:
            raise ValueError("Incompatible checkpoint metadata; old MLP checkpoints are not migrated.")
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        return True  # RSL-RL runner uses this to restore optimizer and iteration.

    def make_actor_export(self):
        return ActorExport(self).eval()

    def export_actor(self, filedir):
        """Write CPU TorchScript/ONNX and input metadata without changing this model."""
        import json

        directory = Path(filedir)
        directory.mkdir(parents=True, exist_ok=True)
        actor = self.make_actor_export().cpu()
        inputs = (torch.zeros(1, self.actor_state_size), torch.zeros(1, self.num_rays))
        with torch.inference_mode():
            torch.jit.script(actor).save(str(directory / "actor.pt"))
            torch.onnx.export(
                actor, inputs, str(directory / "actor.onnx"), opset_version=17, dynamo=False,
                input_names=["policy", "height_scan"], output_names=["action_mean"],
                dynamic_axes={"policy": {0: "batch"}, "height_scan": {0: "batch"}, "action_mean": {0: "batch"}},
            )
        metadata = self.get_extra_state()
        metadata.update(export="deterministic_actor_only", includes_affordance_head=False,
                        scan_input="canonical_xy_flattened;already_clipped_and_scaled")
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return directory
