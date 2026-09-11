"""Base locomotion with current actor height scan and a privileged MLP critic."""
from isaaclab.utils import configclass
from isal2.tasks.base.base_env_cfg import RPOBaseEnvCfg


@configclass
class RPOAMEEnvCfg(RPOBaseEnvCfg):
    ame_height_scan_offset: float = 0.75
    ame_height_scan_noise: float = 0.025

    def _configure_observation_sensors(self):
        scan = self.scene_context.height_scanner
        scan.enable_height_scan = True
        scan.enable_height_scan_actor = True
        self.normalization.height_scan_offset = self.ame_height_scan_offset
        self.noise.noise_scales.height_scan = self.ame_height_scan_noise
        self.height_scan_shape = (round(scan.size[1] / scan.resolution) + 1,
                                  round(scan.size[0] / scan.resolution) + 1)
