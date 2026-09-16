from isal2.tasks.ame.ame_env import AMEEnv
from isal2.tasks.sparse.runtime import SparseTaskMixin


class AMESparseEnv(SparseTaskMixin, AMEEnv):
    """Existing AME with shared sparse course behavior."""
