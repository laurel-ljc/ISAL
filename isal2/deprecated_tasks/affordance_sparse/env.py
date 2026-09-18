from isal2.deprecated_tasks.affordance.affordance_env import AffordanceEnv
from isal2.deprecated_tasks.sparse.runtime import SparseTaskMixin


class AffordanceSparseEnv(SparseTaskMixin, AffordanceEnv):
    """Existing Affordance with shared sparse course behavior."""
