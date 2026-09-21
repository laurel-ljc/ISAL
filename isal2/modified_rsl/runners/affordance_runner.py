"""Fixed rollout/PPO/SL phases with disjoint optimizers and recent contact replay."""
import torch
from isal2.utils.checkpoint import load_checkpoint
from torch.nn import functional as F
from .on_policy_runner import OnPolicyRunner
from ..algorithms.affordance_replay import AffordanceReplay


class AffordanceRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        if self.is_distributed:
            raise ValueError("Affordance replay currently supports single-process training only")
        self.aux_cfg = self.cfg["affordance"]
        c = self.aux_cfg
        self.gate_mode = c.get('gate_mode', 'scheduled')
        if self.gate_mode not in ('scheduled', 'immediate'):
            raise ValueError('Unknown affordance gate mode')
        if min(c["batch_size"], c["gradient_steps"], c["min_samples"]) < 1:
            raise ValueError("Invalid supervised batch/update/gate configuration")
        if self.gate_mode == 'scheduled':
            if c['ramp_iterations'] < 1 or min(c['warmup_iterations'], c['gate_min_samples']) < 0:
                raise ValueError('Invalid scheduled affordance gate configuration')
        elif any(c[key] != 0 for key in ('warmup_iterations', 'ramp_iterations', 'gate_min_samples')):
            raise ValueError('Immediate affordance mode does not allow warm-up, ramp or sample gates')
        self.replay = AffordanceReplay(env.get_observations()["height_scan"].shape[-1], c["capacity"], c["max_age"], device)
        self.supervised_optimizer = torch.optim.Adam(self.alg.policy.affordance_net.parameters(), lr=c["learning_rate"])
        orange = {id(p) for g in self.alg.optimizer.param_groups for p in g["params"]}
        blue = {id(p) for g in self.supervised_optimizer.param_groups for p in g["params"]}
        if orange & blue:
            raise RuntimeError("PPO and supervised optimizers must be disjoint")
        self.total_samples, self.supervised_updates = 0, 0
        self.iteration = 0
        self.reset_affordance_gate()

    def reset_affordance_gate(self):
        """Initialize the target task gate, also after a model-only warm start."""
        self.alg.policy.affordance_alpha.fill_(1. if self.gate_mode == 'immediate' else 0.)

    def _start_iteration(self, iteration):
        self.iteration = iteration
        self.env.unwrapped.set_collection_iteration(iteration)
        self.replay.expire(iteration)
        c = self.aux_cfg
        if self.gate_mode == 'immediate':
            alpha = 1.
        else:
            alpha = min(1., max(0., (iteration - c["warmup_iterations"]) / c["ramp_iterations"]))
            if self.total_samples < c["gate_min_samples"]:
                alpha = 0.
        self.alg.policy.affordance_alpha.fill_(alpha)

    def _after_env_step(self):
        sample = self.env.unwrapped.pop_affordance_samples()
        if sample is not None:
            self.total_samples += len(sample["label"])
            self.replay.add(sample)

    @torch.no_grad()
    def _diagnostic(self, obs):
        model = self.alg.policy
        mean = model.act_inference(obs).clone()
        std = (model.std if model.noise_std_type == "scalar" else model.log_std.exp()).clone()
        quality = model.affordance_map(obs["height_scan"]).clone()
        return mean, std, quality

    def _update_algorithm(self):
        # Storage is cleared by PPO; keep independent observations for drift measurement.
        batch = self.alg.storage.observations.flatten(0, 1)
        indices = torch.randperm(len(batch), device=self.device)[:256]
        diagnostic_obs = batch[indices].clone()
        losses = self.alg.update()
        losses.update(self.supervised_update(diagnostic_obs))
        return losses

    def supervised_update(self, diagnostic_obs):
        model, c = self.alg.policy, self.aux_cfg
        self.replay.expire(self.iteration)
        before = self._diagnostic(diagnostic_obs)
        model.zero_grad(set_to_none=True)
        total_loss, updates = 0., 0
        orange = model.ppo_parameters()
        requires = [p.requires_grad for p in orange]
        for p in orange:
            p.requires_grad_(False)
        try:
            if len(self.replay) >= c["min_samples"]:
                rows, cols = model.terrain_attention.map_shape
                resolution = float(self.cfg["policy"]["map_resolution"])
                extent = torch.tensor([(cols - 1) * resolution / 2, (rows - 1) * resolution / 2], device=self.device)
                for _ in range(c["gradient_steps"]):
                    sample = self.replay.sample(c["batch_size"])
                    quality = model.affordance_map(sample["height_scan"])
                    grid = (sample["query_xy"] / extent).reshape(-1, 1, 1, 2)
                    prediction = F.grid_sample(quality, grid, align_corners=True).flatten().clamp(1e-6, 1-1e-6)
                    loss = F.binary_cross_entropy(prediction, sample["label"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite affordance supervised loss")
                    self.supervised_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.affordance_net.parameters(), c["max_grad_norm"])
                    self.supervised_optimizer.step()
                    total_loss += float(loss.detach())
                    updates += 1
        finally:
            model.affordance_net.zero_grad(set_to_none=True)
            for p, flag in zip(orange, requires):
                p.requires_grad_(flag)
        self.supervised_updates += updates
        after = self._diagnostic(diagnostic_obs)
        kl = (torch.log(after[1]/before[1]) + (before[1].square() + (before[0]-after[0]).square()) /
              (2*after[1].square()) - .5).sum(-1).mean().clamp_min(0)
        stats = self.env.unwrapped.collector.stats
        return {"affordance_loss": total_loss/max(updates, 1), "affordance_updates": float(updates),
                "affordance_replay_size": float(len(self.replay)), "affordance_samples_total": float(self.total_samples),
                "affordance_alpha": float(model.affordance_alpha), "affordance_policy_kl": float(kl),
                "affordance_map_change": float((before[2]-after[2]).abs().mean()),
                **{"contact_"+k: float(v) for k, v in stats.items()}}

    def _extra_checkpoint(self):
        return {"affordance_state": dict(optimizer=self.supervised_optimizer.state_dict(), replay=self.replay.state_dict(),
                total_samples=self.total_samples, supervised_updates=self.supervised_updates,
                collection_stats=dict(self.env.unwrapped.collector.stats), config=self.aux_cfg)}

    def load(self, path, load_optimizer=True, map_location=None):
        checkpoint = load_checkpoint(path, map_location=map_location or self.device)
        state = checkpoint.get("affordance_state")
        if state is None or state["config"] != self.aux_cfg:
            raise ValueError("Affordance resume requires matching supervised/replay/gate configuration")
        if self.gate_mode == 'immediate' and float(checkpoint['model_state_dict']['affordance_alpha']) != 1.:
            raise ValueError('Immediate Affordance checkpoint must have alpha=1')
        infos = super().load(path, load_optimizer, map_location)
        if load_optimizer:
            self.supervised_optimizer.load_state_dict(state["optimizer"])
        self.replay.load_state_dict(state["replay"])
        self.total_samples, self.supervised_updates = state["total_samples"], state["supervised_updates"]
        self.iteration = self.current_learning_iteration
        collector = self.env.unwrapped.collector
        collector.reset(torch.arange(self.env.num_envs, device=self.env.device))
        collector.ready.clear()
        collector.stats.update(state["collection_stats"])
        return infos
