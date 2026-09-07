"""Pure Python model settings and the bridge from a resolved environment layout."""

from dataclasses import asdict, dataclass, field


MODEL_CLASS = "isal.learning.models.affordance_actor_critic:AffordanceActorCritic"


@dataclass
class AffordanceNetworkCfg:
    terrain_channels: list[int] = field(default_factory=lambda: [16, 32, 32])
    terrain_kernels: list[int] = field(default_factory=lambda: [3, 3, 3])
    terrain_strides: list[int] = field(default_factory=lambda: [1, 2, 1])
    terrain_paddings: list[int] = field(default_factory=lambda: [1, 1, 1])
    terrain_fc_dim: int = 128
    terrain_latent_dim: int = 64
    proprio_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    critic_state_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    affordance_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])


def bind_perceptive_model_config(env, agent_cfg) -> bool:
    """Bind final runtime geometry after Gym construction and before runner creation.

    Only the Stage 4A model is affected. The serializable fields also accompany
    saved agent YAML; no guessed square grid or module-level environment is used.
    """
    if agent_cfg.policy.class_name != MODEL_CLASS:
        return False
    raw = getattr(env, "unwrapped", env)
    layout = asdict(raw.perceptive_observation_layout)
    layout["grid_shape"] = list(layout["grid_shape"])
    agent_cfg.policy.observation_layout = layout
    p = raw.cfg.terrain_perception
    agent_cfg.policy.scan_preprocessing = {
        "definition": "terrain_z_minus_root_z",
        "min_height": p.min_height, "max_height": p.max_height,
        "height_scale": p.height_scale, "offset_x": p.offset_x,
        "size": list(p.size), "resolution": p.resolution,
        "noise_std": p.noise_std, "dropout_prob": p.dropout_prob,
    }
    return True
