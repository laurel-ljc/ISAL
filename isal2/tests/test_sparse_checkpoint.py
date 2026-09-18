"""Strict real-model initialization and serialized phase admission tests."""
import contextlib
import io
from uuid import uuid4
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from tensordict import TensorDict
from isal2.deprecated_tasks.ame_sparse.agents.ppo_cfg import AMESparseAgentCfg
from isal2.deprecated_tasks.affordance_sparse.agents.ppo_cfg import AffordanceSparseAgentCfg
from isal2.deprecated_tasks.affordance.collection import ContactCollector
from isal2.modified_rsl.runners import OnPolicyRunner
from isal2.modified_rsl.runners.affordance_runner import AffordanceRunner
from isal2.modified_rsl.runners.checkpoint import warm_start, rng_state, restore_rng, model_digest, advance
from isal2.deprecated_tasks.sparse.curriculum import SparseCurriculum
from isal2.deprecated_tasks.sparse.evaluation import check_advance
from isal2.deprecated_tasks.sparse.config import SparseCfg


class FakeEnv:
    num_envs, num_actions, device, cfg = 4, 23, 'cpu', {}
    def __init__(self):
        self.unwrapped = self
        self.collector = ContactCollector(4, 187, (1.6, 1.), .02)
    def get_observations(self):
        return TensorDict({k: torch.zeros(4, n) for k, n in [('policy', 390), ('critic', 1630), ('height_scan', 187)]}, [4])


class SparseCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def runner(self, aff=False):
        cfg = (AffordanceSparseAgentCfg() if aff else AMESparseAgentCfg()).to_dict()
        with contextlib.redirect_stdout(io.StringIO()):
            return (AffordanceRunner if aff else OnPolicyRunner)(FakeEnv(), cfg)

    def test_real_ame_initializations_keep_normalizers_and_override_std(self):
        root = Path(__file__).resolve().parents[1]/'outputs_download/rpo_ame'
        for folder, name in [('ISAL2-RPO-AME-v0-Rough', 'model_9001.pt'), ('ISAL2-RPO-AME-v0-Rough-Hard', 'model_18002.pt')]:
            path = root/folder/name
            if not path.exists():
                self.skipTest('Historical checkpoint not present')
            runner = self.runner()
            saved = torch.load(path, map_location='cpu', weights_only=False)
            warm_start(runner, path)
            self.assertEqual(runner.current_learning_iteration, 0)
            self.assertEqual(len(runner.alg.optimizer.state), 0)
            for key, value in runner.alg.policy.state_dict().items():
                if key != 'std':
                    self.assertTrue(torch.equal(value, saved['model_state_dict'][key]), key)
            self.assertTrue(torch.allclose(runner.alg.policy.std, torch.full((23,), .3)))

    def test_real_affordance_restarts_auxiliary_state(self):
        path = Path(__file__).resolve().parents[1]/'outputs_download/rpo_affordance/ISAL2-RPO-Affordance-v0-Rough/model_9001.pt'
        if not path.exists():
            self.skipTest('Historical checkpoint not present')
        runner = self.runner(True)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        warm_start(runner, path)
        self.assertEqual(float(runner.alg.policy.affordance_alpha), 0.)
        self.assertEqual((len(runner.replay), runner.total_samples, runner.supervised_updates), (0, 0, 0))
        self.assertEqual(len(runner.supervised_optimizer.state), 0)
        for key, value in runner.alg.policy.state_dict().items():
            if key not in ('std', 'affordance_alpha'):
                self.assertTrue(torch.equal(value, saved['model_state_dict'][key]), key)

    def test_cross_family_rejected(self):
        ame, aff = self.runner(), self.runner(True)
        with self.assertRaises(RuntimeError):
            aff.alg.policy.load_state_dict(ame.alg.policy.state_dict())
        with self.assertRaises(RuntimeError):
            ame.alg.policy.load_state_dict(aff.alg.policy.state_dict())

    def test_rng_round_trip(self):
        state = rng_state()
        expected = torch.rand(8)
        restore_rng(state)
        self.assertTrue(torch.equal(expected, torch.rand(8)))

    def test_phase_gate_binds_model_and_requires_two_evaluations(self):
        source = SparseCfg().signature()
        target = SparseCfg(phase='robust').signature()
        state = {'weight': torch.ones(2)}
        groups = [dict(kind=k, level=l, n=32, success_rate=.90, fall_rate=.03, tracking_error=.10)
                  for k, l in [('single_beam', 6), ('radial_beams', 6), ('legacy_star', 0), ('flat', 0)]]
        report = dict(iteration=250, model_digest=model_digest(state), signature=source, condition='stage', groups=groups)
        old_report = dict(report, iteration=0)
        checkpoint = dict(iter=250, model_state_dict=state,
            sparse_state=dict(signature=source, curriculum=dict(validation=[old_report], flat_baseline=.10)))
        check_advance(checkpoint, target, report)
        with self.assertRaises(ValueError):
            check_advance(checkpoint, dict(target, robust_step='height_025'), report)
        with self.assertRaises(ValueError):
            check_advance(checkpoint, target, dict(report, model_digest='wrong'))
        checkpoint['sparse_state']['curriculum']['validation'] = []
        with self.assertRaises(ValueError):
            check_advance(checkpoint, target, report)

    def test_advance_preserves_optimizer_and_clears_auxiliary_replay(self):
        runner = self.runner(True)
        source = SparseCfg().signature()
        target = SparseCfg(phase='robust')
        c = SparseCurriculum(['single_beam']*4)
        c.ability[:] = 6
        c.assigned[:] = 6
        c.unlocked['single_beam'] = 6
        c.flat_baseline = .1
        groups = [dict(kind=k, level=l, n=32, success_rate=.9, fall_rate=.03, tracking_error=.1)
                  for k, l in [('single_beam', 6), ('radial_beams', 6), ('legacy_star', 0), ('flat', 0)]]
        report = dict(iteration=250, model_digest=model_digest(runner.alg.policy.state_dict()),
                      signature=source, condition='stage', groups=groups)
        c.validation = [dict(report, iteration=0)]
        checkpoint = dict(iter=250, model_state_dict=runner.alg.policy.state_dict(),
            optimizer_state_dict=runner.alg.optimizer.state_dict(), rng_state=rng_state(),
            sparse_state=dict(signature=source, terrain='sparse', curriculum=c.state_dict()),
            affordance_state=dict(config=runner.aux_cfg, optimizer=runner.supervised_optimizer.state_dict(),
                                  total_samples=500, supervised_updates=80))
        raw = runner.env
        raw.cfg = SimpleNamespace(sparse=target, terrain_preset='sparse')
        raw.sparse_curriculum = SparseCurriculum(['single_beam']*4)
        raw.scene = SimpleNamespace(terrain=SimpleNamespace(terrain_levels=torch.zeros(4, dtype=torch.long),
            terrain_types=torch.zeros(4, dtype=torch.long), env_origins=torch.zeros(4, 3),
            terrain_origins=torch.zeros(10, 1, 3)))
        raw.reset = lambda: None
        runner.replay.valid[:3] = True
        output = Path(__file__).resolve().parents[1]/'outputs'
        output.mkdir(exist_ok=True)
        folder = output/('sparse_checkpoint_'+uuid4().hex)
        folder.mkdir()
        path = folder/'source.pt'
        try:
            torch.save(checkpoint, path)
            advance(runner, path, report)
        finally:
            path.unlink(missing_ok=True)
            folder.rmdir()
        self.assertEqual(runner.current_learning_iteration, 250)
        self.assertEqual(runner.total_samples, 500)
        self.assertEqual(runner.supervised_updates, 80)
        self.assertEqual(len(runner.replay), 0)
        self.assertEqual(raw.sparse_phase_updates, 0)
        self.assertEqual(raw.sparse_curriculum.ability.tolist(), [6]*4)


if __name__ == '__main__':
    unittest.main()
