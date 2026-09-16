from isaaclab.utils import configclass
from isal2.tasks.affordance.affordance_env_cfg import RPOAffordanceEnvCfg
from isal2.tasks.sparse.config import SparseCfg, SparseConfigMixin


@configclass
class RPOAffordanceSparseEnvCfg(SparseConfigMixin, RPOAffordanceEnvCfg):
    terrain_preset: str = 'sparse'
    sparse: SparseCfg = SparseCfg()
