from isal2.tasks.ame.ame_env import AMEEnv
from isal2.tasks.course.runtime import CourseTaskMixin


class AMEStage2Env(CourseTaskMixin, AMEEnv):
    """AME with stage 2 courses and clean height observations."""
