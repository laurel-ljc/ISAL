from isaaclab.utils import configclass
from isal2.tasks.ame.ame_env_cfg import RPOAMEEnvCfg
from isal2.tasks.sparse.config import SparseCfg, SparseConfigMixin


@configclass
class RPOAMESparseEnvCfg(SparseConfigMixin, RPOAMEEnvCfg):
    terrain_preset: str = 'sparse'
    sparse: SparseCfg = SparseCfg()
