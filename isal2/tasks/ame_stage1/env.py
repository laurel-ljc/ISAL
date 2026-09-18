from isal2.tasks.ame.ame_env import AMEEnv
from isal2.tasks.course.runtime import CourseTaskMixin


class AMEStage1Env(CourseTaskMixin, AMEEnv):
    """AME with stage 1 courses and clean height observations."""
