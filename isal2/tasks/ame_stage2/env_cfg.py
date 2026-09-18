from isaaclab.utils import configclass
from isal2.tasks.ame.ame_env_cfg import RPOAMEEnvCfg
from isal2.tasks.course.config import CourseCfg, CourseConfigMixin


@configclass
class RPOAMEStage2EnvCfg(CourseConfigMixin, RPOAMEEnvCfg):
    terrain_preset: str = "ame_stage2"
    course: CourseCfg = CourseCfg(stage=2)
