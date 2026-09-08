# PPO loss/update adapted from rsl_rl/rsl_rl/algorithms/ppo.py, BSD-3-Clause.
# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION. All rights reserved.
# Audited upstream commit: 6986d4d1b9fbab96fb61d51f07de419bc95432f1 (local 3.3.0).
"""Ordinary PPO plus current-rollout interaction supervision, on one device."""

from dataclasses import asdict
import torch
from torch import nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO

from .auxiliary_buffer import AuxiliaryRolloutBuffer
from .training_config import AuxiliaryLearningCfg, AffordanceGateScheduleCfg, DiagnosticsCfg, linear_schedule, validate_training_config
from .training_diagnostics import auxiliary_metrics, capture_probe, probe_changes, contact_label_metrics


class PPOWithAffordance(PPO):
    def __init__(self, policy, storage, *, auxiliary_learning=None, affordance_gate_schedule=None,
                 diagnostics=None, **kwargs):
        if kwargs.get("multi_gpu_cfg") is not None:
            raise ValueError("Stage 5 supports single-device execution only.")
        if kwargs.get("rnd_cfg") or kwargs.get("enable_aux_loss") or kwargs.get("aux_loss_coef", 0):
            raise ValueError("Stage 5 does not support RND or the upstream auxiliary-loss hook.")
        if policy.is_recurrent:
            raise ValueError("Stage 5 requires the feed-forward 4A/4B policy.")
        self.aux_cfg = AuxiliaryLearningCfg(**(auxiliary_learning or {}))
        self.gate_cfg = AffordanceGateScheduleCfg(**(affordance_gate_schedule or {}))
        self.diagnostics_cfg = DiagnosticsCfg(**(diagnostics or {}))
        validate_training_config(self.aux_cfg, self.gate_cfg, self.diagnostics_cfg)
        if self.aux_cfg.enabled and not hasattr(policy, "affordance_head"):
            raise ValueError("Auxiliary learning requires an affordance head.")
        super().__init__(policy, storage, **kwargs)
        if (self.num_learning_epochs < 1 or self.num_mini_batches < 1
                or self.num_mini_batches > storage.num_envs*storage.num_transitions_per_env):
            raise ValueError("Require positive epochs/minibatches and nonempty PPO minibatches.")
        self.aux_buffer = (AuxiliaryRolloutBuffer(self.aux_cfg.max_samples_per_rollout, self.device,
                                               self.aux_cfg.sampling_seed) if self.aux_cfg.enabled else None)
        self.coefficient, self.completed_updates, self.environment_steps = 0., 0, 0
        self.phase = "boundary"
        self.rollout_steps = 0
        self.contact_stats = {}
        self.last_metrics = {}

    def training_definition(self):
        return {"auxiliary_learning": asdict(self.aux_cfg), "affordance_gate_schedule": asdict(self.gate_cfg),
                "diagnostics": asdict(self.diagnostics_cfg)}

    def begin_rollout(self):
        if self.phase != "boundary" or (self.aux_buffer is not None and len(self.aux_buffer)):
            raise RuntimeError("Begin rollout requires a completed update boundary and empty buffer.")
        c = self.aux_cfg
        self.coefficient = linear_schedule(self.completed_updates, c.start_iteration, c.ramp_iterations, c.loss_coef) if c.enabled else 0.
        if hasattr(self.policy, "set_affordance_input_gate"):
            g = self.gate_cfg
            gate = linear_schedule(self.completed_updates, g.start_iteration, g.ramp_iterations, g.final)
            self.policy.set_affordance_input_gate(gate if self.policy.input_mode == "predicted" else 0.)
        self.active_gate = float(getattr(self.policy, "input_gate", 0.))
        self.phase, self.rollout_steps = "rollout", 0

    def _check_gate(self):
        if float(getattr(self.policy, "input_gate", 0.)) != self.active_gate:
            raise RuntimeError("Input gate changed inside a rollout/update.")

    def collect_auxiliary(self, extras):
        """Called once after each physical env step; also usable by validate-only."""
        if self.phase != "rollout":
            raise RuntimeError("Contact collection requires an active rollout.")
        self._check_gate()
        self.rollout_steps += 1
        self.environment_steps += self.storage.num_envs
        if self.aux_buffer is not None:
            if "auxiliary" not in extras:
                raise ValueError("Auxiliary training requires the tracker packet.")
            self.aux_buffer.append(extras["auxiliary"], self.rollout_steps)
            self.contact_stats = {key: value.detach().clone() for key, value in extras.get("interaction_stats", {}).items()}

    def process_env_step(self, obs, rewards, dones, extras):
        super().process_env_step(obs, rewards, dones, extras)
        self.collect_auxiliary(extras)

    def auxiliary_loss(self):
        zero = self.policy.std.new_zeros(())
        if self.aux_buffer is None:
            return zero, "disabled"
        if self.coefficient == 0:
            return zero, "zero_coefficient"
        if len(self.aux_buffer) < self.aux_cfg.min_samples_per_update:
            return zero, "insufficient_samples"
        batch = self.aux_buffer.sample(self.aux_cfg.aux_batch_size)
        return F.smooth_l1_loss(self.policy.predict_affordance(batch), batch["target"],
                                beta=self.aux_cfg.smooth_l1_beta), "active"

    def compute_minibatch_loss(self, batch, *, adapt_learning_rate=True):
        """Production loss path, independently testable with backward but no optimizer step."""
        (obs, actions, values, advantages, returns, old_log, old_mu, old_sigma, hidden, masks) = batch
        original = obs.batch_size[0]
        if self.normalize_advantage_per_mini_batch:
            advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
        augmentation = self.symmetry["data_augmentation_func"] if self.symmetry else None
        if self.symmetry and self.symmetry["use_data_augmentation"]:
            obs, actions = augmentation(obs=obs, actions=actions, env=self.symmetry["_env"])
            count = obs.batch_size[0]//original
            old_log, values, advantages, returns = (v.repeat(count, 1) for v in (old_log, values, advantages, returns))
        self.policy.act(obs, masks=masks, hidden_state=hidden[0])
        log_prob = self.policy.get_actions_log_prob(actions)
        value = self.policy.evaluate(obs, masks=masks, hidden_state=hidden[1])
        mu, sigma, entropy = self.policy.action_mean[:original], self.policy.action_std[:original], self.policy.entropy[:original]
        with torch.no_grad():
            # Preserve the existing PPO epsilon and adaptive-LR thresholds exactly.
            kl = (torch.log(sigma/old_sigma+1e-5)+(old_sigma.square()+(old_mu-mu).square())/(2*sigma.square())-.5).sum(-1).mean()
            if adapt_learning_rate and self.desired_kl is not None and self.schedule == "adaptive":
                if kl > self.desired_kl*2:
                    self.learning_rate = max(1e-5, self.learning_rate/1.5)
                elif 0 < kl < self.desired_kl/2:
                    self.learning_rate = min(1e-2, self.learning_rate*1.5)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate
        ratio = torch.exp(log_prob-old_log.squeeze(-1))
        surrogate = torch.maximum(-advantages.squeeze(-1)*ratio,
                                   -advantages.squeeze(-1)*ratio.clamp(1-self.clip_param,1+self.clip_param)).mean()
        if self.use_clipped_value_loss:
            clipped = values+(value-values).clamp(-self.clip_param,self.clip_param)
            value_loss = torch.maximum((value-returns).square(),(clipped-returns).square()).mean()
        else:
            value_loss = (returns-value).square().mean()
        ppo_loss = surrogate+self.value_loss_coef*value_loss-self.entropy_coef*entropy.mean()
        symmetry_loss = value_loss.new_zeros(())
        if self.symmetry:
            if not self.symmetry["use_data_augmentation"]:
                obs, _ = augmentation(obs=obs, actions=None, env=self.symmetry["_env"])
            mean = self.policy.act_inference(obs.detach().clone())
            _, target = augmentation(obs=None, actions=mean[:original], env=self.symmetry["_env"])
            symmetry_loss = F.mse_loss(mean[original:], target.detach()[original:])
            if self.symmetry["use_mirror_loss"]:
                ppo_loss = ppo_loss+self.symmetry["mirror_loss_coeff"]*symmetry_loss
        aux_loss, reason = self.auxiliary_loss()
        return {"total": ppo_loss+self.coefficient*aux_loss, "ppo": ppo_loss, "auxiliary": aux_loss,
                "value": value_loss, "surrogate": surrogate, "entropy": entropy.mean(),
                "symmetry": symmetry_loss, "pre_update_kl": kl, "auxiliary_reason": reason}

    def finish_update(self):
        """Commit an already completed update; never clears tracker pending events."""
        if self.phase != "update":
            raise RuntimeError("finish_update requires the update phase.")
        self.storage.clear()
        if self.aux_buffer is not None:
            self.aux_buffer.clear()
        self.completed_updates += 1
        self.phase = "boundary"

    def update(self):
        if self.phase != "rollout" or self.rollout_steps != self.storage.num_transitions_per_env:
            raise RuntimeError("Update requires exactly one complete rollout.")
        self._check_gate()
        self.phase = "update"
        d = self.diagnostics_cfg
        probe_obs = self.storage.observations.flatten(0, 1)[:d.probe_size].detach().clone()
        aux_probe = ({k: v[:d.probe_size] for k,v in self.aux_buffer.data.items()}
                     if self.aux_buffer is not None and len(self.aux_buffer) else None)
        before = capture_probe(self.policy, probe_obs, aux_probe) if d.enabled else None
        metrics = {"coefficient": self.coefficient, "input_gate": self.active_gate,
                   "contact": {key:float(value) for key,value in self.contact_stats.items()},
                   "auxiliary_updates": 0, "skip_reasons": {}, "gradient": {}}
        if self.aux_buffer is not None:
            metrics.update(self.aux_buffer.statistics())
            metrics["retained_labels"] = contact_label_metrics(self.aux_buffer.data)
        sums, count, norms = {}, 0, []
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            self._check_gate()
            losses = self.compute_minibatch_loss(batch)
            reason = losses["auxiliary_reason"]
            metrics["skip_reasons"][reason] = metrics["skip_reasons"].get(reason,0)+1
            metrics["auxiliary_updates"] += int(reason == "active")
            if count == 0 and d.enabled and reason == "active":
                for name in ("actor_terrain_encoder", "affordance_head"):
                    params = tuple(getattr(self.policy,name).parameters())
                    gradients = torch.autograd.grad(losses["auxiliary"],params,retain_graph=True)
                    metrics["gradient"][name+"_aux_norm"] = float(torch.stack([g.norm().square() for g in gradients]).sum().sqrt())
            self.optimizer.zero_grad()
            losses["total"].backward()
            norms.append(float(nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm)))
            self.optimizer.step()
            for key in ("value","surrogate","entropy","symmetry","auxiliary","pre_update_kl"):
                sums[key] = sums.get(key,0.)+float(losses[key].detach())
            count += 1
        metrics["gradient"]["joint_pre_clip_norm_mean"] = sum(norms)/max(count,1)
        if d.enabled:
            metrics.update(probe_changes(before,capture_probe(self.policy,probe_obs,aux_probe)))
            metrics["auxiliary_quality"] = auxiliary_metrics(self.policy,aux_probe,self.aux_cfg.smooth_l1_beta)
        self.last_metrics = metrics
        self.finish_update()
        return {key:value/count for key,value in sums.items()}
