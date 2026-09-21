from isal2.tasks.common.ame.ame_env import AMEEnv
from isal2.tasks.common.reference.runtime import ReferenceTaskMixin


class AMEStage2Env(ReferenceTaskMixin, AMEEnv):
    """AME on the pinned reference stage 2 distance curriculum."""
