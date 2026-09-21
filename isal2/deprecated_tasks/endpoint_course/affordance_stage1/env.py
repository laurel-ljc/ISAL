from isal2.deprecated_tasks.endpoint_course.common.affordance.affordance_env import AffordanceEnv
from isal2.deprecated_tasks.endpoint_course.common.course.runtime import CourseTaskMixin


class AffordanceStage1Env(CourseTaskMixin, AffordanceEnv):
    """Contact-supervised Affordance policy on stage 1 courses."""
