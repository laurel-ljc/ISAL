"""Load trusted training checkpoints across NumPy versions."""
import pickle
import types

import torch


class _CheckpointUnpickler(pickle.Unpickler):
    """Read NumPy 2 array pickles on older NumPy 1.x installations."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError as exc:
            if module.startswith("numpy._core.") and exc.name in ("numpy._core", module):
                return super().find_class(module.replace("numpy._core.", "numpy.core.", 1), name)
            raise


def load_checkpoint(checkpoint, map_location="cpu"):
    # Keep compatibility local to this trusted checkpoint load, without changing
    # NumPy imports process-wide or modifying the installed Isaac Lab environment.
    compat_pickle = types.ModuleType("checkpoint_pickle")
    compat_pickle.__dict__.update(vars(pickle))
    compat_pickle.Unpickler = _CheckpointUnpickler
    return torch.load(checkpoint, map_location=map_location, weights_only=False, pickle_module=compat_pickle)


