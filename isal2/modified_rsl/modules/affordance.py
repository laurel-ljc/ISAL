"""Independent U-Net and AME actor fusion. Blue parameters never receive PPO gradients."""
import torch
from torch import nn
from torch.nn import functional as F
from .actor_critic_ame import ActorCriticAME
from .terrain_attention import PositionEncoding2D


def conv_block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, padding_mode="replicate"), nn.ELU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1, padding_mode="replicate"), nn.ELU())


class AffordanceUNet(nn.Module):
    def __init__(self, channels=(16, 32, 64)):
        super().__init__()
        a, b, c = channels
        self.enc1, self.enc2, self.bottom = conv_block(1, a), conv_block(a, b), conv_block(b, c)
        self.dec2, self.dec1 = conv_block(c + b, b), conv_block(b + a, a)
        self.output = nn.Conv2d(a, 1, 1)

    def forward(self, image):
        one = self.enc1(image)
        two = self.enc2(F.max_pool2d(one, 2))
        bottom = self.bottom(F.max_pool2d(two, 2))
        two_up = self.dec2(torch.cat([F.interpolate(bottom, size=two.shape[-2:], mode="bilinear", align_corners=True), two], 1))
        one_up = self.dec1(torch.cat([F.interpolate(two_up, size=one.shape[-2:], mode="bilinear", align_corners=True), one], 1))
        return self.output(one_up)


class ActorCriticAffordance(ActorCriticAME):
    def __init__(self, *args, unet_channels=(16, 32, 64), affordance_initial_alpha=0., **kwargs):
        if not 0. <= affordance_initial_alpha <= 1.:
            raise ValueError('affordance_initial_alpha must be between zero and one')
        super().__init__(*args, **kwargs)
        t = self.terrain_attention
        if min(t.map_shape) < 4:
            raise ValueError("U-Net requires both map dimensions >= 4")
        old = t.position_encoding
        t.position_encoding = PositionEncoding2D(old.projection.in_channels - 2 + 1,
            old.projection.out_channels, t.map_shape, kwargs.get("map_resolution", .1))
        self.affordance_net = AffordanceUNet(unet_channels)
        self.register_buffer("affordance_alpha", torch.tensor(float(affordance_initial_alpha)))

    def affordance_map(self, scan):
        return self.affordance_net(self.terrain_attention.scan_to_image(scan)).sigmoid()

    def _actor_features(self, obs):
        proprio = self.actor_obs_normalizer(self.get_actor_obs(obs))
        t = self.terrain_attention
        features = t.encode_features(obs["height_scan"])
        with torch.no_grad():
            quality = self.affordance_map(obs["height_scan"])
            quality = .5 + self.affordance_alpha * (quality - .5)
            quality = F.interpolate(quality, size=features.shape[-2:], mode="bilinear", align_corners=True)
        terrain = t.attend(proprio, torch.cat([features, quality], 1))
        return torch.cat([proprio, terrain], -1)

    def ppo_parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith("affordance_net.")]

    def load_state_dict(self, state_dict, strict=True):
        if "affordance_alpha" not in state_dict or not any(k.startswith("affordance_net.") for k in state_dict):
            raise RuntimeError("Affordance resume requires an Affordance checkpoint; AME/Base migration is unsupported")
        return super().load_state_dict(state_dict, strict)
