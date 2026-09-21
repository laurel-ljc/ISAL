"""List registered ISAL2 tasks without launching the simulator."""
import argparse

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--including_deprecated",
        type=str.lower,
        choices=("true", "false"),
        nargs="?",
        const="true",
        default="false",
        help="Include deprecated tasks (default: false).",
    )
    args = parser.parse_args()

    import isal2.deprecated_tasks
    for name, spec in sorted(gym.registry.items()):
        if name.startswith("ISAL2-"):
            status = 'deprecated' if '.deprecated_tasks.' in spec.entry_point else 'active'
            if args.including_deprecated == "false" and (status == "deprecated" or "[deprecated]" in name):
                continue
            print(f"[{status}] {name}\n  env: {spec.entry_point}\n  config: {spec.kwargs['env_cfg_entry_point']}\n  agent: {spec.kwargs['rsl_rl_cfg_entry_point']}")


if __name__ == "__main__":
    main()
