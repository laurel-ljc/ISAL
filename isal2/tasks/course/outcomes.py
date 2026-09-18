"""Tensor-only geometry queries and episode outcome classification."""
import math
import torch


def polygon_planes(supports, device='cpu'):
    """Convex inward unit normals plus offsets. Unused edges impose no constraint."""
    planes = torch.zeros(len(supports), 5, 3, device=device)
    planes[..., 2] = 1.
    tops = torch.empty(len(supports), device=device)
    for i, support in enumerate(supports):
        vertices = torch.tensor(support['polygon'], device=device)
        edge = vertices.roll(-1, 0)-vertices
        normals = torch.stack((-edge[:, 1], edge[:, 0]), -1)
        normals /= normals.norm(dim=-1, keepdim=True)
        planes[i, :len(vertices), :2] = normals
        planes[i, :len(vertices), 2] = -(normals*vertices).sum(-1)
        tops[i] = support['top']
    return planes, tops


def supported(feet, planes, tops):
    # [N,F,3], [N,S,5,3], [N,S]. Padded supports have infinite height.
    signed = torch.einsum('nfd,nskd->nfsk', feet[..., :2], planes[..., :2]) + planes[:, None, :, :, 2]
    inside = (signed >= -1.e-5).all(-1)
    height = (feet[..., 2, None]-tops[:, None]).abs() <= .07
    return (inside & height).any(-1)


def failure_penalty(env):
    """Keep the old penalty weight while excluding successful terminal episodes."""
    return env.course_result['failed'].float()


class OutcomeTracker:
    def __init__(self, n, dt, success_seconds=.2, device='cpu'):
        self.required_steps = math.ceil(success_seconds/dt)
        self.stable = torch.zeros(n, dtype=torch.long, device=device)

    def reset(self, ids):
        self.stable[ids] = 0

    def update(self, root, feet, forces, planes, tops, goal, half_width, base_failure, timeout, radius=.3):
        contact = forces > 5.
        valid = supported(feet, planes, tops)
        invalid = (contact & ~valid).any(-1)
        # Wider start/finish platforms; the obstacle corridor has no side detour.
        width = torch.where(root[:, 0].abs() >= 2., 1.2, half_width)
        bypass = (root[:, 0].abs() > 3.95) | (root[:, 1].abs() > width+.05)
        falling = root[:, 2] < -.35
        failed = base_failure | invalid | bypass | falling
        on_finish = ((feet[..., 0] >= 2.) & (feet[..., 0] <= 4.) &
                     (feet[..., 1].abs() <= 1.2) & (feet[..., 2].abs() <= .07) & contact).any(-1)
        reached = ((root[:, :2]-goal[:, :2]).norm(dim=-1) <= radius) & on_finish & ~failed
        self.stable.copy_(torch.where(reached, self.stable+1, 0))
        success = (self.stable >= self.required_steps) & ~failed
        timed_out = timeout & ~success & ~failed
        return dict(success=success, failed=failed, timeout=timed_out,
                    base_failure=base_failure, invalid_support=invalid, bypass=bypass, fall=falling)
