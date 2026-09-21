"""Keep terrain randomness reproducible without consuming training RNG streams."""
from contextlib import contextmanager
import hashlib
import random
import numpy as np
import torch


def tile_seed(cfg, difficulty):
    return (int(getattr(cfg, 'seed', None) or 0) * 1_000_003 +
            int(round(float(difficulty) * 1_000_000))) % 2**32


@contextmanager
def terrain_seed(seed, difficulty, kind):
    key = f'{seed}:{float(difficulty).hex()}:{kind}'.encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:4], 'little')
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        np.random.seed(value)
        random.seed(value)
        # Isaac Lab's random-grid mesh samples torch (CUDA when available).
        # Isolate both CPU and accelerator streams from policy initialization.
        with torch.random.fork_rng():
            torch.manual_seed(value)
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
