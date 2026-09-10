"""List registered ISAL2 tasks without launching the simulator."""
try:
    from ._bootstrap import bootstrap
except ImportError:
    import importlib.util
    from pathlib import Path
    _spec = importlib.util.spec_from_file_location("_isal2_bootstrap", Path(__file__).with_name("_bootstrap.py"))
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    bootstrap = _module.bootstrap
bootstrap()

import gymnasium as gym
import isal2.tasks


def main():
    for name, spec in sorted(gym.registry.items()):
        if name.startswith("ISAL2-"):
            print(f"{name}\n  env: {spec.entry_point}\n  config: {spec.kwargs['env_cfg_entry_point']}\n  agent: {spec.kwargs['rsl_rl_cfg_entry_point']}")


if __name__ == "__main__":
    main()
