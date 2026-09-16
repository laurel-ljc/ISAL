"""Actor-only scan and its actual yaw-aligned map origin."""
import torch


def actor_scan(env):
    data = env.scene['actor_height_scanner'].data
    height = (data.pos_w[:, 2, None] - data.ray_hits_w[..., 2] - env.cfg.normalization.height_scan_offset).clamp(-1, 1)
    height = torch.nan_to_num(height, nan=1, posinf=1, neginf=-1) * env.obs_scales.height_scan
    if env.add_noise:
        height += (2*torch.rand_like(height)-1)*env.cfg.noise.noise_scales.height_scan*env.obs_scales.height_scan
    return height


def actor_map_root(env):
    # Ray hit XY already includes yaw rotation and the episode-fixed local drift.
    # Its grid centroid is the true sampling origin, including at missing Z hits.
    from isaaclab.utils.math import quat_apply_yaw
    sensor = env.scene['actor_height_scanner']
    root = env.robot.data.root_pos_w.clone()
    drift = sensor.ray_cast_drift.clone()
    root[:, :2] += quat_apply_yaw(env.robot.data.root_quat_w, drift)[:, :2]
    return root
