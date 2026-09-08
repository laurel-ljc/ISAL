"""PyTorch models with no environment or simulator imports."""

from .affordance_actor_critic import AffordanceActorCritic
from .affordance_observation_actor_critic import AffordanceObservationActorCritic

__all__ = ["AffordanceActorCritic", "AffordanceObservationActorCritic"]
