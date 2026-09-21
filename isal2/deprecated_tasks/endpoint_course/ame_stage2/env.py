from isal2.tasks.common.ame.ame_env import AMEEnv
from isal2.deprecated_tasks.endpoint_course.common.course.runtime import CourseTaskMixin


class AMEStage2Env(CourseTaskMixin, AMEEnv):
    """AME with stage 2 courses and clean height observations."""
