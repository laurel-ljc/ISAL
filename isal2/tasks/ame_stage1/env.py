from isal2.tasks.common.ame.ame_env import AMEEnv
from isal2.tasks.common.reference.runtime import ReferenceTaskMixin


def stand_still_training_scale(iteration: int, full_iterations: int, end_iterations: int) -> float:
    if not 0 <= full_iterations < end_iterations:
        raise ValueError("stand_still schedule requires 0 <= full_iterations < end_iterations")
    return max(0.0, min(1.0, (end_iterations - iteration) / (end_iterations - full_iterations)))


class AMEStage1Env(ReferenceTaskMixin, AMEEnv):
    """AME on the pinned reference stage 1 distance curriculum."""

    # Evaluation does not apply the training-only posture penalty.
    stand_still_scale = 0.0

    def set_training_iteration(self, iteration: int) -> None:
        self.stand_still_scale = stand_still_training_scale(
            iteration, self.cfg.stand_still_full_iterations, self.cfg.stand_still_end_iterations
        )
