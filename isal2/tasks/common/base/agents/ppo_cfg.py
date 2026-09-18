"""Base PPO defaults: standard external MLP with project-specific PPO hooks."""
from dataclasses import dataclass, field, asdict


@dataclass
class BaseAgentCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 12001
    save_interval: int = 500
    logger: str = "tensorboard"
    experiment_name: str = "rpo_base"
    obs_groups: dict = field(default_factory=lambda: {"policy": ["policy"], "critic": ["critic"]})
    policy: dict = field(default_factory=lambda: dict(
        class_name="rsl_rl.modules:ActorCritic",
        init_noise_std=1.0, noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        activation="elu", actor_obs_normalization=True, critic_obs_normalization=True))
    algorithm: dict = field(default_factory=lambda: dict(
        class_name="isal2.modified_rsl.algorithms:PPO", value_loss_coef=1.0,
        use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=1e-4,
        schedule="adaptive", gamma=0.994, lam=0.9, desired_kl=0.01,
        max_grad_norm=1.0, normalize_advantage_per_mini_batch=False,
        rnd_cfg=None, symmetry_cfg=None))

    def to_dict(self):
        return asdict(self)
