"""Lazy exports keep agent configuration usable without starting Isaac Sim."""
from importlib import import_module


def __getattr__(name):
    if name == "BaseAgentCfg":
        module = ".agents.ppo_cfg"
    elif name == "SceneCfg":
        module = ".scene_cfg"
    elif name.endswith("TERRAINS_CFG"):
        module = ".terrain_generator_cfg"
    else:
        module = ".base_config"
    return getattr(import_module(module, __name__), name)
