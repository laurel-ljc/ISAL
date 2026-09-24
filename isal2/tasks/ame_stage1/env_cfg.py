from isaaclab.utils import configclass
from isal2.tasks.common.ame.ame_env_cfg import RPOAMEEnvCfg
from isal2.tasks.common.reference.config import ReferenceCfg, ReferenceConfigMixin


@configclass
class RPOAMEStage1EnvCfg(ReferenceConfigMixin, RPOAMEEnvCfg):
    terrain_preset: str = "ame_stage1"
    reference: ReferenceCfg = ReferenceCfg(stage=1)
    stand_still_full_iterations: int = 500
    stand_still_end_iterations: int = 1500
