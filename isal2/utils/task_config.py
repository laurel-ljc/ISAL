"""Resolve a registered task's configuration without hard-coding its class."""
from importlib import import_module
import gymnasium as gym


def load_task_configs(task_id):
    if not task_id.startswith("ISAL2-"):
        raise ValueError(f"Expected a registered ISAL2 task, got {task_id}")
    spec = gym.spec(task_id)
    configs = []
    for key in ("env_cfg_entry_point", "rsl_rl_cfg_entry_point"):
        module, name = spec.kwargs[key].split(":")
        configs.append(getattr(import_module(module), name)())
    return tuple(configs)
