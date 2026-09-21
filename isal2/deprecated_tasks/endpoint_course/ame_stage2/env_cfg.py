from isaaclab.utils import configclass
from isal2.tasks.common.ame.ame_env_cfg import RPOAMEEnvCfg
from isal2.deprecated_tasks.endpoint_course.common.course.config import CourseCfg, CourseConfigMixin


@configclass
class RPOAMEStage2EnvCfg(CourseConfigMixin, RPOAMEEnvCfg):
    terrain_preset: str = "ame_stage2"
    course: CourseCfg = CourseCfg(stage=2)
