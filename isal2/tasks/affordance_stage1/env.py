from isal2.tasks.common.affordance.affordance_env import AffordanceEnv
from isal2.tasks.common.course.runtime import CourseTaskMixin


class AffordanceStage1Env(CourseTaskMixin, AffordanceEnv):
    """Contact-supervised Affordance policy on stage 1 courses."""
