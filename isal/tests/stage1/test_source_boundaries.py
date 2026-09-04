from __future__ import annotations

import ast
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "isal"
PACKAGE_ROOT = PROJECT_ROOT / "isal"


def test_upstream_submodules_are_clean() -> None:
    for name in ("robolab", "rsl_rl"):
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT / name), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout == "", f"{name} contains local changes:\n{result.stdout}"


def test_isal_task_code_does_not_import_robolab_tasks() -> None:
    occurrences: list[tuple[Path, str]] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "robolab" in line:
                occurrences.append((path.relative_to(PROJECT_ROOT), line.strip()))

    assert occurrences == [
        (
            Path("isal/tasks/direct/humanoid_rough/isal_env_cfg.py"),
            "from robolab.assets.robots import RPO_CFG",
        )
    ]


def test_training_entrypoint_owns_registration_and_validation_exit() -> None:
    path = PROJECT_ROOT / "scripts" / "rsl_rl" / "train.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "import isal.tasks" in source
    assert "import robolab.tasks" not in source
    assert '"--validate-only"' in source
    assert "runner.learn(" in source

    validation_returns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "args_cli.validate_only" in ast.unparse(node.test)
        and any(isinstance(child, ast.Return) for child in ast.walk(node))
    ]
    assert validation_returns, "--validate-only must return before the training call"


def test_provenance_records_pinned_robolab_commit() -> None:
    provenance = (PROJECT_ROOT / "BASELINE_PROVENANCE.md").read_text(encoding="utf-8")
    assert "6b1c3d9988497c8961dcba77892de32edc1770e1" in provenance
