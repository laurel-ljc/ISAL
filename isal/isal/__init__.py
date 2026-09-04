"""ISAL-Humanoid package."""

from __future__ import annotations

import os


ISAL_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

__all__ = ["ISAL_ROOT_DIR"]
