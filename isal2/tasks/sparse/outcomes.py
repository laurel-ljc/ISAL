"""Vectorized support and episode results, independent of the simulator."""
import math
import torch


def inside_rectangles(points, rectangles, tolerance=0.):
    """points [N,F,3], rectangles [N,M,6] -> [N,F,M]. Padding has negative half sizes."""
    delta = points[:, :, None, :2] - rectangles[:, None, :, :2]
    yaw = rectangles[:, None, :, 4]
    x = delta[..., 0]*yaw.cos() + delta[..., 1]*yaw.sin()
    y = -delta[..., 0]*yaw.sin() + delta[..., 1]*yaw.cos()
    return ((x.abs() <= rectangles[:, None, :, 2]+tolerance) &
            (y.abs() <= rectangles[:, None, :, 3]+tolerance) &
            ((points[:, :, None, 2]-rectangles[:, None, :, 5]).abs() < .07))


class OutcomeTracker:
    def __init__(self, n, dt, cfg, device='cpu'):
        self.dt, self.cfg = dt, cfg
        self.window = math.ceil(cfg.stuck_seconds/dt)
        self.progress_history = torch.zeros(n, self.window+1, device=device)
        self.tick = 0
        self.contact_count = torch.zeros(n, 2, dtype=torch.long, device=device)
        self.contact = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.stable = torch.zeros(n, dtype=torch.long, device=device)
        self.stuck = torch.zeros(n, dtype=torch.bool, device=device)

    def reset(self, ids, progress):
        self.progress_history[ids] = progress[:, None]
        self.contact_count[ids] = 0
        self.contact[ids] = False
        self.stable[ids] = 0
        self.stuck[ids] = False

    def update(self, root, feet, forces, supports, exit_regions, yaw, entry, exit_s, width,
               active, steps, speed, base_failure, central_platform=None):
        self.contact_count.copy_(torch.where(forces >= 5., self.contact_count+1, 0))
        self.contact.copy_((self.contact | (self.contact_count >= 2)) & (forces > 2.))
        valid = inside_rectangles(feet, supports, .01).any(-1)
        if central_platform is not None:
            angles = (torch.arange(24, device=feet.device)+.5)*(2*math.pi/24)
            normals = torch.stack((angles.cos(), angles.sin()), -1)
            in_polygon = (feet[..., :2] @ normals.T).amax(-1) <= math.cos(math.pi/24)+.01
            valid |= central_platform[:, None] & in_polygon & (feet[..., 2].abs() < .07)
        s = root[:, 0]*yaw.cos()+root[:, 1]*yaw.sin()
        fs = feet[..., 0]*yaw[:, None].cos()+feet[..., 1]*yaw[:, None].sin()
        fl = -feet[..., 0]*yaw[:, None].sin()+feet[..., 1]*yaw[:, None].cos()
        start_platform = (fs < entry[:, None]) & torch.where(entry[:, None] > 0,
                           feet[..., :2].norm(dim=-1) < 1., feet[..., 0] >= -4.)
        on_route = start_platform | (fs >= exit_s[:, None]-.3) | (fl.abs() <= width[:, None]+.08)
        invalid = (self.contact & ~valid).any(-1) & active
        bypass = ((self.contact & valid & ~on_route).any(-1) | (root[:, :2].abs() > 4.).any(-1)) & active
        falling = (root[:, 2] < -.35) & active
        failed = base_failure | invalid | bypass | falling
        exit_support = (inside_rectangles(feet, exit_regions[:, None], .01).squeeze(-1) & self.contact).any(-1)
        stable = (s >= exit_s) & exit_support & ~failed & active
        self.stable.copy_(torch.where(stable, self.stable+1, 0))
        success = (self.stable*self.dt >= self.cfg.success_seconds) & ~failed
        slot = self.tick % (self.window+1)
        old = self.progress_history[:, (self.tick-self.window) % (self.window+1)].clone()
        self.progress_history[:, slot] = s
        self.tick += 1
        warmed = steps*self.dt >= self.cfg.warmup_seconds+self.cfg.stuck_seconds
        self.stuck |= warmed & (speed > .05) & ((s-old) < self.cfg.stuck_distance) & active
        return dict(success=success, failed=failed, fall=(invalid | falling | base_failure),
                    bypass=bypass, stuck=self.stuck.clone(), progress=s)
