"""AME adds a cached actor scan while retaining base observation/reset semantics."""
import torch
from isal2.tasks.base.base_env import BaseEnv


class AMEEnv(BaseEnv):
    def _read_actor_height(self):
        height = self._height_scan().clone()
        if self.add_noise:
            height += (2 * torch.rand_like(height) - 1) * (
                self.cfg.noise.noise_scales.height_scan * self.obs_scales.height_scan)
        return height

    def _actor_map_root(self):
        return self.robot.data.root_pos_w

    def _get_observations(self, env_ids=None):
        observations = super()._get_observations(env_ids)
        if not hasattr(self, "_actor_height_scan"):
            self._actor_height_scan = torch.empty(
                self.num_envs, self.cfg.height_scan_shape[0] * self.cfg.height_scan_shape[1],
                device=self.device)
            env_ids = None
        ids = slice(None) if env_ids is None else env_ids
        height = self._read_actor_height()[ids]
        self._actor_height_scan[ids] = height
        # A snapshot keeps rollout/terminal observations independent of later resets.
        observations["height_scan"] = self._actor_height_scan.clone()
        return observations
