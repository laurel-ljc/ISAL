from isaaclab.utils import configclass
from isal2.tasks.common.ame.ame_env_cfg import RPOAMEEnvCfg
from isal2.tasks.common.reference.config import ReferenceCfg, ReferenceConfigMixin


@configclass
class RPOAMEStage2EnvCfg(ReferenceConfigMixin, RPOAMEEnvCfg):
    terrain_preset: str = "ame_stage2"
    reference: ReferenceCfg = ReferenceCfg(stage=2)
