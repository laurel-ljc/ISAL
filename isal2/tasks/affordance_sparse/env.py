from isal2.tasks.affordance.affordance_env import AffordanceEnv
from isal2.tasks.sparse.runtime import SparseTaskMixin


class AffordanceSparseEnv(SparseTaskMixin, AffordanceEnv):
    """Existing Affordance with shared sparse course behavior."""
