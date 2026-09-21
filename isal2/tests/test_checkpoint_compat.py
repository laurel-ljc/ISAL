"""Cross-version NumPy checkpoint loading and runner restore regression."""
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import zipfile

import numpy as np
import torch

from isal2.modified_rsl.runners.on_policy_runner import OnPolicyRunner
from isal2.utils.checkpoint import load_checkpoint


def numpy2_checkpoint(state):
    source, target = io.BytesIO(), io.BytesIO()
    torch.save(state, source, pickle_protocol=2)
    source.seek(0)
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.endswith("data.pkl"):
                data = data.replace(b"cnumpy.core.", b"cnumpy._core.")
            dst.writestr(item, data)
    target.seek(0)
    return target


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_numpy_array_rng_and_tensor_round_trip(self):
        rng = np.random.RandomState(42).get_state()
        loaded = load_checkpoint(numpy2_checkpoint(dict(rng=rng, weight=torch.arange(4))))
        restored = np.random.RandomState()
        restored.set_state(loaded['rng'])
        np.testing.assert_array_equal(restored.rand(8), np.random.RandomState(42).rand(8))
        torch.testing.assert_close(loaded['weight'], torch.arange(4))

    def test_runner_restores_optimizer_rnd_and_iteration(self):
        for load_optimizer in (True, False):
            with self.subTest(load_optimizer=load_optimizer):
                optimizer = Mock(param_groups=[{'lr': .002}])
                policy = Mock()
                policy.load_state_dict.return_value = True
                alg = SimpleNamespace(policy=policy, optimizer=optimizer, rnd=Mock(),
                                      rnd_optimizer=Mock(), learning_rate=.1)
                runner = SimpleNamespace(alg=alg, alg_cfg={'rnd_cfg': True},
                                         env=SimpleNamespace(), device='cpu', current_learning_iteration=0)
                state = dict(model_state_dict={'weight': torch.ones(2)},
                             optimizer_state_dict={'marker': 1}, rnd_state_dict={'marker': 2},
                             rnd_optimizer_state_dict={'marker': 3}, iter=71,
                             infos={'array': np.arange(3)})
                infos = OnPolicyRunner.load(runner, numpy2_checkpoint(state), load_optimizer)
                self.assertEqual(runner.current_learning_iteration, 71)
                np.testing.assert_array_equal(infos['array'], np.arange(3))
                torch.testing.assert_close(policy.load_state_dict.call_args.args[0]['weight'], torch.ones(2))
                alg.rnd.load_state_dict.assert_called_once_with({'marker': 2})
                if load_optimizer:
                    optimizer.load_state_dict.assert_called_once_with({'marker': 1})
                    alg.rnd_optimizer.load_state_dict.assert_called_once_with({'marker': 3})
                    self.assertEqual(alg.learning_rate, .002)
                else:
                    optimizer.load_state_dict.assert_not_called()
                    alg.rnd_optimizer.load_state_dict.assert_not_called()
                    self.assertEqual(alg.learning_rate, .1)


if __name__ == '__main__':
    unittest.main()
