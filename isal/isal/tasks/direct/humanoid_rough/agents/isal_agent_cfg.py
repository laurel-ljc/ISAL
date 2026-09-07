# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)
import torch
from tensordict import TensorDict
from functools import lru_cache

from isal.tasks.direct.humanoid_rough.base_config import BaseAgentCfg
from isal.tasks.direct.humanoid_rough.height_scan import (
    mirror_flat_height_scan, mirror_native_height_scan, restore_observation_history,
)


def generate_height_scan_mirror(start_idx=140, rows=11, cols=17):
    mirror_indices = []
    for row in range(rows):
        mirror_row = rows - 1 - row
        for col in range(cols):
            mirror_idx = start_idx + col + mirror_row * cols
            mirror_indices.append(mirror_idx)
    mirror_signs = [1] * (rows * cols)
    return mirror_indices, mirror_signs

def generate_joint_mirror(start_idx):
    mirror_indices = []
    mirror_indices.extend([start_idx + 1, start_idx])    
    mirror_indices.append(start_idx + 2)
    for i in range(start_idx + 3, start_idx + 23, 2):
        mirror_indices.extend([i + 1, i])
    mirror_signs = [-1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
    return mirror_indices, mirror_signs

joint_pos_mirror_indices, joint_pos_mirror_signs = generate_joint_mirror(9)
joint_vel_mirror_indices, joint_vel_mirror_signs = generate_joint_mirror(32)
action_mirror_indices, action_mirror_signs = generate_joint_mirror(55)
policy_obs_mirror_indices = [0, 1, 2,\
                             3, 4, 5,\
                             6, 7, 8]\
                            + joint_pos_mirror_indices + joint_vel_mirror_indices + action_mirror_indices
policy_obs_mirror_signs = [-1, 1, -1,\
                           1, -1, 1,\
                           1, -1, -1] + joint_pos_mirror_signs + joint_vel_mirror_signs + action_mirror_signs
joint_acc_mirror_indices, joint_acc_mirror_signs = generate_joint_mirror(93)
joint_torques_mirror_indices, joint_torques_mirror_signs = generate_joint_mirror(116)
critic_obs_mirror_indices = policy_obs_mirror_indices +\
                            [78, 79, 80,\
                             82, 81,\
                             86, 87, 88, 83, 84, 85,\
                             90, 89,\
                             92, 91]\
                            + joint_acc_mirror_indices + joint_torques_mirror_indices
height_scan_mirror_indices, height_scan_mirror_signs = generate_height_scan_mirror(139, 11, 17)
critic_obs_mirror_indices += height_scan_mirror_indices
critic_obs_mirror_signs = policy_obs_mirror_signs +\
                           [1, -1, 1,\
                            1, 1,\
                            1, -1, 1, 1, -1, 1,\
                            1, 1,\
                            1, 1]\
                            + joint_acc_mirror_signs + joint_torques_mirror_signs
critic_obs_mirror_signs += height_scan_mirror_signs
act_mirror_indices = [1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21]
act_mirror_signs = [-1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
policy_obs_mirror_indices_expanded = []
for i in range(10):
    offset = i * 78
    for idx in policy_obs_mirror_indices:
        policy_obs_mirror_indices_expanded.append(idx + offset)
policy_obs_mirror_signs_expanded = policy_obs_mirror_signs * 10

critic_obs_mirror_indices_expanded = []
for i in range(10):
    offset = i * 326
    for idx in critic_obs_mirror_indices:
        critic_obs_mirror_indices_expanded.append(idx + offset)
critic_obs_mirror_signs_expanded = critic_obs_mirror_signs * 10

@lru_cache(maxsize=None)
def get_policy_obs_mirror_signs_tensor(device, dtype):
    return torch.tensor(policy_obs_mirror_signs_expanded, device=device, dtype=dtype)

def mirror_policy_observation(policy_obs):
    mirrored_policy_obs = policy_obs[..., policy_obs_mirror_indices_expanded]
    signs = get_policy_obs_mirror_signs_tensor(device=policy_obs.device, dtype=policy_obs.dtype)
    mirrored_policy_obs = mirrored_policy_obs * signs
    return mirrored_policy_obs

@lru_cache(maxsize=None)
def get_critic_obs_mirror_signs_tensor(device, dtype):
    return torch.tensor(critic_obs_mirror_signs_expanded, device=device, dtype=dtype)

def mirror_critic_observation(critic_obs):
    mirrored_critic_obs = critic_obs[..., critic_obs_mirror_indices_expanded]
    signs = get_critic_obs_mirror_signs_tensor(device=critic_obs.device, dtype=critic_obs.dtype)
    mirrored_critic_obs = mirrored_critic_obs * signs
    return mirrored_critic_obs

@lru_cache(maxsize=None)
def get_act_mirror_signs_tensor(device, dtype):
    return torch.tensor(act_mirror_signs, device=device, dtype=dtype)

def mirror_actions(actions):
    mirrored_actions = actions[..., act_mirror_indices]
    signs = get_act_mirror_signs_tensor(device=actions.device, dtype=actions.dtype)
    mirrored_actions = mirrored_actions * signs
    return mirrored_actions


def mirror_height_scan_observation(env, height_scan):
    """Mirror the Stage 2 height-scan group using the environment's resolved grid shape."""
    base_env = getattr(env, "unwrapped", env)
    return mirror_flat_height_scan(height_scan, base_env.height_scan_grid_shape)


@lru_cache(maxsize=None)
def get_state_mirror_tensors(device, dtype):
    """Single-frame maps only; history length and terrain size are resolved at runtime."""
    return (
        torch.tensor(policy_obs_mirror_indices, device=device),
        torch.tensor(policy_obs_mirror_signs, device=device, dtype=dtype),
        torch.tensor(critic_obs_mirror_indices[:139], device=device),
        torch.tensor(critic_obs_mirror_signs[:139], device=device, dtype=dtype),
    )


def mirror_perceptive_observations(env, obs):
    base_env = getattr(env, "unwrapped", env)
    layout = base_env.perceptive_observation_layout
    result = obs.clone()
    values = obs["policy"]
    actor_indices, actor_signs, _, _ = get_state_mirror_tensors(values.device, values.dtype)
    history = restore_observation_history(values, layout.actor_history_length, layout.actor_frame_dim, "Actor")
    result["policy"] = (history[..., actor_indices] * actor_signs).flatten(start_dim=1)
    if "height_scan" in obs.keys():
        result["height_scan"] = mirror_flat_height_scan(obs["height_scan"], layout.grid_shape)
    if "critic" in obs.keys():
        values = obs["critic"]
        _, _, critic_indices, critic_signs = get_state_mirror_tensors(values.device, values.dtype)
        history = restore_observation_history(values, layout.critic_history_length, layout.critic_frame_dim, "Critic")
        state = history[..., :layout.critic_state_dim]
        scan = history[..., layout.critic_state_dim:]
        scan = mirror_native_height_scan(scan.reshape(-1, layout.num_rays), layout.grid_shape, layout.ordering)
        mirrored = torch.cat((state[..., critic_indices] * critic_signs, scan.reshape_as(history[..., layout.critic_state_dim:])), -1)
        result["critic"] = mirrored.flatten(start_dim=1)
    return result

def data_augmentation_func(env, obs, actions):
    if obs is None:
        obs_aug = None
    else:
        if hasattr(getattr(env, "unwrapped", env), "perceptive_observation_layout"):
            obs_mirror = mirror_perceptive_observations(env, obs)
        else:
            # Preserve the Stage 1 baseline path and its observation contract.
            obs_mirror = obs.clone()
            obs_mirror["policy"] = mirror_policy_observation(obs["policy"])
            if "height_scan" in obs.keys():
                obs_mirror["height_scan"] = mirror_height_scan_observation(env, obs["height_scan"])
            if "critic" in obs.keys():
                obs_mirror["critic"] = mirror_critic_observation(obs["critic"])
        obs_aug = torch.cat([obs, obs_mirror], dim=0)
    if actions is None:
        actions_aug = None
    else:
        actions_aug = torch.cat((actions, mirror_actions(actions)), dim=0)
    return obs_aug, actions_aug


@configclass
class ISALHumanoidFlatAgentCfg(BaseAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name: str = "isal_humanoid_flat"
        self.wandb_project: str = "isal_humanoid_flat"
        self.seed = 42
        self.num_steps_per_env = 24
        self.max_iterations = 9001
        self.save_interval = 1000
        self.actor_obs_normalization = True
        self.critic_obs_normalization = True
        self.algorithm = RslRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=None,
            rnd_cfg=None,  # RslRlRndCfg()
        )
        self.clip_actions = 100.0


@configclass
class ISALHumanoidRoughAgentCfg(ISALHumanoidFlatAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name: str = "isal_humanoid_rough"
        self.neptune_project: str = "isal_humanoid_rough"
        self.wandb_project: str = "isal_humanoid_rough"
        self.algorithm = RslRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True, 
                use_mirror_loss=True,
                mirror_loss_coeff=0.2, 
                data_augmentation_func=data_augmentation_func
            ),
            rnd_cfg=None,  # RslRlRndCfg()
        )


@configclass
class ISALHumanoidRoughHeightScanAgentCfg(ISALHumanoidRoughAgentCfg):
    """Ordinary PPO baseline that concatenates proprio history and the current scan."""

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "rsl_rl.modules:ActorCritic"
        self.algorithm.class_name = "rsl_rl.algorithms:PPO"
        self.obs_groups = {
            "policy": ["policy", "height_scan"],
            "critic": ["critic"],
        }
        self.experiment_name = "isal_humanoid_rough_height_scan"
        self.neptune_project = "isal_humanoid_rough_height_scan"
        self.wandb_project = "isal_humanoid_rough_height_scan"


@configclass
class ISALHumanoidRoughInteractionAgentCfg(ISALHumanoidRoughHeightScanAgentCfg):
    """Ordinary PPO; Stage 3 collects labels but never optimizes an auxiliary loss."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "isal_humanoid_rough_interaction"
        self.neptune_project = "isal_humanoid_rough_interaction"
        self.wandb_project = "isal_humanoid_rough_interaction"
