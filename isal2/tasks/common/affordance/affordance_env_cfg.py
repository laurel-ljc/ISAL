from isaaclab.utils import configclass
from isal2.tasks.common.ame.ame_env_cfg import RPOAMEEnvCfg
from .collection import ContactCollectionCfg


@configclass
class RPOAffordanceEnvCfg(RPOAMEEnvCfg):
    collection = ContactCollectionCfg()
