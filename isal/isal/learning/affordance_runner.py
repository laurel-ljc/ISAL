# Runner loop adapted from RSL-RL OnPolicyRunner, BSD-3-Clause.
# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION. All rights reserved.
"""Single-device Stage 5 runner with explicit completed-update checkpoint semantics."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner


def canonical_config(value):
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        return {k:canonical_config(v) for k,v in value.items() if k not in ("log_dir", "_env")}
    if isinstance(value, (list,tuple)):
        return [canonical_config(v) for v in value]
    if isinstance(value, slice):
        return {"slice": [value.start,value.stop,value.step]}
    if isinstance(value, (torch.Tensor,np.ndarray)):
        return {"array":value.tolist(),"dtype":str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if isinstance(value, (str,int,float,bool)) or value is None:
        return value
    raise TypeError(f"Unsupported checkpoint configuration value: {type(value)}")


class AffordanceRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Stage 5 supports single-device execution only.")
        # Capture before upstream pops FQNs and injects live env into symmetry cfg.
        self.training_definition = {
            "policy": canonical_config(train_cfg["policy"]), "algorithm": canonical_config(train_cfg["algorithm"]),
            "obs_groups": canonical_config(train_cfg["obs_groups"]), "num_steps_per_env": train_cfg["num_steps_per_env"],
            "seed": train_cfg.get("seed"), "environment": canonical_config(env.cfg),
        }
        super().__init__(env, deepcopy(train_cfg), log_dir, device)
        self.elapsed_seconds = 0.

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """External-machine entrypoint. Local acceptance never calls this method."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations().to(self.device)
        self.train_mode()
        start_it = self.alg.completed_updates
        total_it = start_it+num_learning_iterations
        for _ in range(num_learning_iterations):
            start = time.perf_counter()
            self.alg.begin_rollout()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs,rewards,dones,extras)
                    self.logger.process_env_step(rewards,dones,extras,None)
                self.alg.compute_returns(obs)
            collected = time.perf_counter()
            losses = self.alg.update()
            ended = time.perf_counter()
            self.elapsed_seconds += ended-start
            self.current_learning_iteration = self.alg.completed_updates
            self.logger.log(it=self.current_learning_iteration-1,start_it=start_it,total_it=total_it,
                            collect_time=collected-start,learn_time=ended-collected,loss_dict=losses,
                            learning_rate=self.alg.learning_rate,action_std=self.alg.policy.std,rnd_weight=None)
            self.write_metrics()
            if self.logger.log_dir and self.current_learning_iteration % self.cfg["save_interval"] == 0:
                self.save(str(Path(self.logger.log_dir)/f"model_{self.current_learning_iteration}.pt"))
        if self.logger.log_dir:
            self.save(str(Path(self.logger.log_dir)/f"model_{self.current_learning_iteration}.pt"))

    def write_metrics(self):
        metrics = dict(self.alg.last_metrics, completed_updates=self.alg.completed_updates,
                       environment_steps=self.alg.environment_steps, elapsed_seconds=self.elapsed_seconds)
        print("[STAGE5] "+json.dumps(metrics,allow_nan=False))
        if self.logger.log_dir:
            with (Path(self.logger.log_dir)/"stage5_metrics.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(metrics,allow_nan=False)+"\n")
        if self.logger.writer:
            def scalars(values, prefix="Stage5"):
                for key,value in values.items():
                    if isinstance(value,dict):
                        scalars(value,f"{prefix}/{key}")
                    elif isinstance(value,(int,float,bool)):
                        self.logger.writer.add_scalar(f"{prefix}/{key}",value,self.alg.environment_steps)
            scalars(metrics)

    def save(self, path, infos=None):
        if self.alg.phase != "boundary" or self.alg.storage.step != 0:
            raise RuntimeError("Stage 5 checkpoints require a complete update boundary.")
        if self.current_learning_iteration != self.alg.completed_updates:
            raise RuntimeError("Runner and algorithm completed-update counters differ.")
        numpy_state = np.random.get_state()
        saved = {
            "model_state_dict": self.alg.policy.state_dict(), "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.alg.completed_updates, "infos": infos,
            "stage5": {"schema_version":1, "definition":self.training_definition,
                       "completed_updates":self.alg.completed_updates, "environment_steps":self.alg.environment_steps,
                       "elapsed_seconds":self.elapsed_seconds, "coefficient":self.alg.coefficient,
                       "input_gate":float(getattr(self.alg.policy,"input_gate",0.)),
                       "learning_rate":self.alg.learning_rate,
                       "aux_generator":self.alg.aux_buffer.generator.get_state() if self.alg.aux_buffer is not None else None,
                       "torch_rng":torch.get_rng_state(),
                       "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                       "python_rng":random.getstate(),
                       "numpy_rng":(numpy_state[0],numpy_state[1].tolist(),*numpy_state[2:])},
        }
        torch.save(saved,path)
        self.logger.save_model(path,self.alg.completed_updates)

    def load(self,path,load_optimizer=True,map_location=None):
        if not load_optimizer:
            raise ValueError("Stage 5 resume requires optimizer state; use export_actor for inference-only loading.")
        saved = torch.load(path,weights_only=True,map_location=map_location or self.device)
        state = saved.get("stage5",{})
        if state.get("schema_version") != 1 or state.get("definition") != self.training_definition:
            raise ValueError("Incompatible Stage 5 training checkpoint/configuration; migration is not supported.")
        for key in ("completed_updates","environment_steps"):
            if not isinstance(state[key],int) or state[key] < 0:
                raise ValueError(f"Invalid checkpoint {key}.")
        for key in ("elapsed_seconds","coefficient","input_gate","learning_rate"):
            if not math.isfinite(state[key]) or state[key] < 0:
                raise ValueError(f"Invalid checkpoint {key}.")
        if saved["iter"] != state["completed_updates"] or state["input_gate"] > 1:
            raise ValueError("Inconsistent checkpoint iteration/gate.")
        model_gate = saved["model_state_dict"].get("input_gate",torch.tensor(0.))
        if float(model_gate) != state["input_gate"]:
            raise ValueError("Checkpoint model and training gates disagree.")
        self.alg.policy.load_state_dict(saved["model_state_dict"])
        self.alg.optimizer.load_state_dict(saved["optimizer_state_dict"])
        self.alg.completed_updates = self.current_learning_iteration = state["completed_updates"]
        self.alg.environment_steps = state["environment_steps"]
        self.elapsed_seconds = state["elapsed_seconds"]
        self.alg.coefficient, self.alg.learning_rate = state["coefficient"], state["learning_rate"]
        self.alg.storage.clear()
        self.alg.transition.clear()
        self.alg.phase, self.alg.rollout_steps = "boundary",0
        if self.alg.aux_buffer is not None:
            self.alg.aux_buffer.clear()
            self.alg.aux_buffer.generator.set_state(state["aux_generator"].cpu())
        # Restart physical state, including live swings and pending contacts. Reset
        # may draw random numbers; restore training RNG only after that reset.
        self.env.reset()
        self.logger.tot_timesteps = self.alg.environment_steps
        self.logger.tot_time = self.elapsed_seconds
        for name in ("cur_reward_sum","cur_episode_length"):
            getattr(self.logger,name).zero_()
        self.logger.rewbuffer.clear(); self.logger.lenbuffer.clear(); self.logger.ep_extras.clear()
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        random.setstate(state["python_rng"])
        n = state["numpy_rng"]
        np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),*n[2:]))
        return saved["infos"]
