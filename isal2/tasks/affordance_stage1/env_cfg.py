from isaaclab.utils import configclass
from isal2.tasks.common.affordance.affordance_env_cfg import RPOAffordanceEnvCfg
from isal2.tasks.common.course.config import CourseCfg, CourseConfigMixin


@configclass
class RPOAffordanceStage1EnvCfg(CourseConfigMixin, RPOAffordanceEnvCfg):
    terrain_preset: str = "affordance_stage1"
    course_family: str = "affordance"
    course: CourseCfg = CourseCfg(stage=1)
