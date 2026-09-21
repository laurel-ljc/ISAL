from isal2.tasks.common.ame.ame_env import AMEEnv
from isal2.tasks.common.reference.runtime import ReferenceTaskMixin


class AMEStage1Env(ReferenceTaskMixin, AMEEnv):
    """AME on the pinned reference stage 1 distance curriculum."""
