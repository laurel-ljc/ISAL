"""Collect real contact outcomes after done evaluation and before automatic reset."""
import torch
from isaaclab.utils.math import quat_apply, euler_xyz_from_quat
from isal2.tasks.ame.ame_env import AMEEnv
from .collection import ContactCollector


class AffordanceEnv(AMEEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.collector = ContactCollector(self.num_envs, self.cfg.height_scan_shape[0] * self.cfg.height_scan_shape[1],
            cfg.scene_context.height_scanner.size, self.step_dt, cfg.collection, self.device)
        # Resolve explicitly: contact and articulation order are separate contracts.
        names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self._aff_body_ids = self.robot.find_bodies(names, preserve_order=True)[0]
        self._aff_sensor_ids = self.contact_sensor.find_bodies(names, preserve_order=True)[0]

    def _after_termination_check(self):
        data, c = self.robot.data, self.cfg.collection
        ids = self._aff_body_ids
        quat = data.body_quat_w[:, ids]
        offset = torch.tensor(c.sole_offset, device=self.device).expand(self.num_envs, 2, 3)
        feet = data.body_pos_w[:, ids] + quat_apply(quat, offset)
        forces = self.contact_sensor.data.net_forces_w[:, self._aff_sensor_ids].norm(dim=-1)
        speed = data.body_lin_vel_w[:, ids, :2].norm(dim=-1)
        support = []
        for i, name in enumerate(("left_feet_scanner", "right_feet_scanner")):
            hits = self.scene[name].data.ray_hits_w[..., 2]
            support.append((torch.isfinite(hits) & ((hits - feet[:, i, 2, None]).abs() <= c.support_tolerance)).float().mean(-1))
        height = self._height_scan().clone()
        if self.add_noise:
            height += (2 * torch.rand_like(height) - 1) * self.cfg.noise.noise_scales.height_scan * self.obs_scales.height_scan
        yaw = euler_xyz_from_quat(data.root_quat_w)[2]
        self.collector.update(height, data.root_pos_w, yaw, feet, forces, speed, torch.stack(support, -1),
                              self.reset_terminated, self.reset_time_outs)

    def _reset_idx(self, env_ids):
        if hasattr(self, "collector"):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            self.collector.reset(env_ids)
        super()._reset_idx(env_ids)

    def pop_affordance_samples(self):
        return self.collector.pop_samples()

    def set_collection_iteration(self, iteration):
        self.collector.iteration = iteration
