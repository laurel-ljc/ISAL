import torch
from tensordict import TensorDict


def inputs(grid=(17, 11), ordering="xy", histories=(10, 10), batch=2, device="cpu"):
    h, w = grid
    obs = TensorDict({"policy": torch.randn(batch, histories[0]*78, device=device),
                      "height_scan": torch.randn(batch, h*w, device=device).clamp(-3, .8),
                      "critic": torch.randn(batch, histories[1]*(139+h*w), device=device)}, batch_size=[batch])
    x, y = torch.meshgrid(torch.linspace(-(h-1)*.05+.4, (h-1)*.05+.4, h),
                          torch.linspace(-(w-1)*.05, (w-1)*.05, w), indexing="ij")
    common = dict(obs_groups={"policy": ["policy", "height_scan"], "critic": ["critic"]}, num_actions=23,
                  affordance_head_enabled=True,
                  observation_layout=dict(grid_shape=list(grid), ordering=ordering, actor_history_length=histories[0],
                                          critic_history_length=histories[1], actor_frame_dim=78, critic_state_dim=139),
                  scan_preprocessing=dict(definition="terrain_z_minus_root_z", min_height=-1.5, max_height=.4,
                                          height_scale=.5, offset_x=.4, size=[(h-1)*.1, (w-1)*.1],
                                          resolution=.1, noise_std=0., dropout_prob=0.))
    extra = dict(query_coordinates=torch.stack((x.flatten(), y.flatten()), -1).tolist(),
                 context_scales=dict(ang_vel=2., projected_gravity=3., commands=4.))
    return obs, common, extra


def aux_sample(obs, grid):
    b, dev = obs.shape[0], obs["policy"].device
    return dict(height_scan=obs["height_scan"].reshape(b, 1, *grid).clone(),
                query_xy=torch.zeros(b, 2, device=dev), foot_side=torch.tensor([[1., 0.]], device=dev).expand(b, -1),
                command=torch.ones(b, 3, device=dev), base_ang_vel=torch.zeros(b, 3, device=dev),
                projected_gravity=torch.tensor([[0., 0., -1.]], device=dev).expand(b, -1),
                target=torch.full((b, 1), .1, device=dev))
