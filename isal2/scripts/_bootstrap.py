"""Allow direct scripts without adding any sibling project package to sys.path."""
from pathlib import Path
import importlib.util
import sys


def bootstrap():
    root = Path(__file__).resolve().parents[1]
    if "isal2" not in sys.modules:
        spec = importlib.util.spec_from_file_location("isal2", root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["isal2"] = module
        spec.loader.exec_module(module)
    return root
