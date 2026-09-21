from isaaclab.utils import configclass
from isal2.deprecated_tasks.endpoint_course.common.affordance.affordance_env_cfg import RPOAffordanceEnvCfg
from isal2.deprecated_tasks.endpoint_course.common.course.config import CourseCfg, CourseConfigMixin


@configclass
class RPOAffordanceStage2EnvCfg(CourseConfigMixin, RPOAffordanceEnvCfg):
    terrain_preset: str = "affordance_stage2"
    course_family: str = "affordance"
    course: CourseCfg = CourseCfg(stage=2)
