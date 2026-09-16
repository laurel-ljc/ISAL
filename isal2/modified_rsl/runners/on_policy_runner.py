# SPDX-License-Identifier: BSD-3-Clause
"""ISAL2 runner: external RSL construction/utilities with explicit update counts."""
import copy
import math
from pathlib import Path
import time

import torch
from rsl_rl.runners import OnPolicyRunner as RslOnPolicyRunner
from rsl_rl.utils.logger import Logger


class _ProjectLogger(Logger):
    def _store_code_state(self):
        # Resolved configs are saved by train.py. Do not traverse sibling projects.
        self.git_status_repos = []


class OnPolicyRunner(RslOnPolicyRunner):
    """Reuse upstream factories, distributed setup and inference methods.

    The customized loop records completed update counts, checks finite losses and
    leaves a clear update boundary for the later affordance supervised phase.
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        config = copy.deepcopy(train_cfg)
        # Upstream construction pops class names; retain a clean config snapshot.
        super().__init__(env, copy.deepcopy(train_cfg), log_dir=None, device=device)
        self.cfg = config
        self.alg.policy.update_normalization(env.get_observations())
        self.logger = _ProjectLogger(
            log_dir=log_dir, cfg=config, env_cfg=env.cfg, num_envs=env.num_envs,
            is_distributed=self.is_distributed, gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank, device=device)
        self.last_loss_dict = {}
        self.validation_callback = None

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.validation_callback is not None:
            self.validation_callback(self)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations().to(self.device)
        self.train_mode()
        if self.is_distributed:
            self.alg.broadcast_parameters()
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            self._start_iteration(it)
            start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self._after_env_step()
                    intrinsic = self.alg.intrinsic_rewards if self.alg.rnd else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic)
                self.alg.compute_returns(obs)
            collected = time.perf_counter()
            losses = self._update_algorithm()
            if not all(math.isfinite(v) for v in losses.values()):
                raise FloatingPointError(f"Non-finite training loss: {losses}")
            self.last_loss_dict = losses
            self.current_learning_iteration = it + 1
            raw = getattr(self.env, 'unwrapped', None)
            if raw is not None and hasattr(raw, 'sparse_phase_updates'):
                raw.sparse_phase_updates += 1
                raw.pop_sparse_records()
            if self.validation_callback is not None:
                self.validation_callback(self)
            self.logger.log(
                it=it, start_it=start_it, total_it=total_it,
                collect_time=collected-start, learn_time=time.perf_counter()-collected,
                loss_dict=losses, learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std,
                rnd_weight=self.alg.rnd.weight if self.alg.rnd else None)
            if self.current_learning_iteration % self.cfg["save_interval"] == 0:
                self._save_current()
        self._save_current()

    def _start_iteration(self, iteration):
        pass

    def _after_env_step(self):
        pass

    def _update_algorithm(self):
        return self.alg.update()

    def _extra_checkpoint(self):
        return {}

    def _save_current(self):
        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(str(Path(self.logger.log_dir) / f"model_{self.current_learning_iteration}.pt"))

    def save(self, path, infos=None):
        data = dict(model_state_dict=self.alg.policy.state_dict(),
            optimizer_state_dict=self.alg.optimizer.state_dict(), iter=self.current_learning_iteration,
            infos=infos, learning_rate=self.alg.learning_rate, train_cfg=self.cfg,
            last_loss_dict=self.last_loss_dict)
        if self.alg.rnd:
            data["rnd_state_dict"] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer:
                data["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        data.update(self._extra_checkpoint())
        from .checkpoint import extra_state
        data.update(extra_state(self))
        torch.save(data, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True, map_location=None):
        # Preserve compatibility with existing ISAL2 state-dict checkpoints.
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        raw = getattr(self.env, 'unwrapped', None)
        if raw is not None and hasattr(raw, 'sparse_state_dict') and 'sparse_state' not in checkpoint:
            raise ValueError('Old checkpoints require --warm-start for a sparse task')
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        if load_optimizer:
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
        from .checkpoint import restore_sparse
        restore_sparse(self, checkpoint)
        return infos
