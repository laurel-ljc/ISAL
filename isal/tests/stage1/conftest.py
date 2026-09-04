from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def isaac_app():
    """Launch Isaac Sim before importing Isaac Lab task configurations."""
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    yield app
    # On Windows, SimulationApp.close() terminates pytest before it can render
    # assertion details and its final summary. The one-shot test process owns
    # Kit, so normal process teardown releases it after pytest reports results.
