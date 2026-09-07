"""Simulator-independent foot interaction sampling (no learning or replay)."""

from .config import SelfSupervisedCfg
from .tracker import FootInteractionTracker, touchdown_query_xy

__all__ = ["SelfSupervisedCfg", "FootInteractionTracker", "touchdown_query_xy"]
