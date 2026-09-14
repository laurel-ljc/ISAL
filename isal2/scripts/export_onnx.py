"""Export a trusted local ISAL2 checkpoint without starting Isaac Sim."""
import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", help="Output directory (default: checkpoint directory/export)")
    parser.add_argument("--env-config")
    parser.add_argument("--joint-names")
    parser.add_argument("--agent-config")
    args = parser.parse_args()
    from isal2.deployment.export import export_checkpoint
    metadata = export_checkpoint(args.checkpoint, args.output, args.env_config, args.joint_names, args.agent_config)
    print(json.dumps({"model_type": metadata["model_type"], "alpha": metadata["affordance_alpha"],
                      "validation": metadata["validation"]}, indent=2))


if __name__ == "__main__":
    main()
