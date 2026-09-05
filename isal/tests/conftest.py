from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def isaac_app():
    """Launch one shared Isaac Sim instance for all integration tests."""
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    yield app
    # On Windows, explicit close can terminate pytest before its final report.
