from isal2.deprecated_tasks.ame.ame_env import AMEEnv
from isal2.deprecated_tasks.sparse.runtime import SparseTaskMixin


class AMESparseEnv(SparseTaskMixin, AMEEnv):
    """Existing AME with shared sparse course behavior."""
