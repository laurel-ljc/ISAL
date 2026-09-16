import unittest
import torch
from isal2.tasks.sparse.config import SparseCfg, ROBUST_STEPS
from isal2.tasks.sparse.curriculum import SparseCurriculum
from isal2.tasks.sparse.outcomes import OutcomeTracker
from isal2.tasks.sparse.evaluation import manifest, summarize, wilson


class SparseCurriculumTests(unittest.TestCase):
    def test_success_failure_replay_and_type_gate(self):
        c = SparseCurriculum(['single_beam', 'radial_beams', 'flat'])
        ids = torch.arange(3)
        c.record(ids, torch.ones(3, dtype=torch.bool))
        c.record(ids, torch.ones(3, dtype=torch.bool))
        self.assertEqual(c.ability.tolist(), [0, 0, 0])
        for i in (0, 250):
            c.validate(dict(iteration=i, groups=[dict(kind='single_beam', level=0, n=32, success_rate=.9, fall_rate=.05)]))
        self.assertEqual(c.unlocked['single_beam'], 1)
        self.assertEqual(c.unlocked['radial_beams'], 0)
        c.record(ids, torch.ones(3, dtype=torch.bool))
        c.record(ids, torch.ones(3, dtype=torch.bool))
        self.assertEqual(c.ability.tolist(), [1, 0, 0])
        c.is_replay[0] = True
        for _ in range(5):
            c.record(torch.tensor([0]), torch.tensor([False]))
        self.assertEqual(int(c.ability[0]), 1)
        c.is_replay[0] = False
        c.assigned[0] = 1
        c.record(torch.tensor([0]), torch.tensor([False]))
        c.record(torch.tensor([0]), torch.tensor([False]))
        self.assertEqual(int(c.ability[0]), 0)
        c.record(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.bool))

    def test_validation_samples_and_duplicate_rejected(self):
        c = SparseCurriculum(['single_beam'])
        for i in (0, 250):
            c.validate(dict(iteration=i, groups=[dict(kind='single_beam', level=0, n=4, success_rate=1., fall_rate=0.)]))
        self.assertEqual(c.unlocked['single_beam'], 0)
        with self.assertRaises(ValueError):
            c.validate(dict(iteration=250, groups=[]))

    def test_unverified_narrow_beams_remain_evaluation_only(self):
        c = SparseCurriculum(['single_beam', 'grid_stones'])
        c.unlocked.update(single_beam=6, grid_stones=6)
        for iteration in (0, 250):
            c.validate(dict(iteration=iteration, groups=[dict(kind=k, level=6, n=32, success_rate=.9, fall_rate=0.)
                                                         for k in ('single_beam', 'grid_stones')]))
        self.assertEqual(c.unlocked['single_beam'], 6)
        self.assertEqual(c.unlocked['grid_stones'], 7)

    def test_state_and_sampling(self):
        c = SparseCurriculum(['single_beam']*1000, replay=.2)
        c.ability[:] = 5
        torch.manual_seed(2)
        c.sample(torch.arange(1000))
        self.assertTrue(150 < int(c.is_replay.sum()) < 250)
        self.assertTrue((c.assigned[c.is_replay] < 5).all())
        restored = SparseCurriculum(c.kinds)
        restored.load_state_dict(c.state_dict())
        self.assertTrue(torch.equal(restored.assigned, c.assigned))

    def test_robust_one_factor_at_a_time(self):
        previous = SparseCfg(phase='robust').perturbations()
        for step in ROBUST_STEPS[1:]:
            current = SparseCfg(phase='robust', robust_step=step).perturbations()
            self.assertEqual(sum(previous[k] != current[k] for k in current), 1)
            previous = current

    def test_manifest_core_and_intervals(self):
        scenes = manifest()
        self.assertEqual(sum(s['kind'] in ('single_beam', 'radial_beams', 'legacy_star') for s in scenes), 384)
        self.assertEqual(scenes, manifest())
        low, high = wilson(0, 32)
        self.assertAlmostEqual(low, 0.)
        self.assertGreater(high, 0.)


class SparseOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.tracker = OutcomeTracker(1, .02, SparseCfg())
        self.args = dict(root=torch.tensor([[2., 0., .75]]), feet=torch.tensor([[[2., -.05, 0], [2., .05, 0]]]),
            forces=torch.tensor([[20., 20.]]), supports=torch.tensor([[[2., 0, .5, .5, 0, 0]]]),
            exit_regions=torch.tensor([[2., 0, .5, .5, 0, 0]]), yaw=torch.zeros(1), entry=torch.tensor([-1.5]),
            exit_s=torch.tensor([1.9]), width=torch.tensor([.4]), active=torch.tensor([True]),
            steps=torch.tensor([1]), speed=torch.tensor([.5]), base_failure=torch.tensor([False]))

    def test_success_requires_sustained_support_and_failure_wins(self):
        self.assertFalse(self.tracker.update(**self.args)['success'].item())
        for _ in range(16):
            result = self.tracker.update(**self.args)
        self.assertTrue(result['success'].item())
        self.args['base_failure'][:] = True
        result = self.tracker.update(**self.args)
        self.assertTrue(result['failed'].item())
        self.assertFalse(result['success'].item())

    def test_swing_over_pit_not_failure_but_loaded_pit_is(self):
        self.args['feet'][:] = torch.tensor([[[0, 1., 0], [0, 1., 0]]])
        self.args['forces'].zero_()
        self.assertFalse(self.tracker.update(**self.args)['failed'].item())
        self.args['forces'][:] = 20
        self.tracker.update(**self.args)
        self.assertTrue(self.tracker.update(**self.args)['fall'].item())

    def test_reset_after_inference_step(self):
        with torch.inference_mode():
            self.tracker.update(**self.args)
        self.tracker.reset(torch.tensor([0]), torch.tensor([0.]))

    def test_stuck_is_record_only_and_reset_clears_it(self):
        self.args['root'][:] = torch.tensor([[0., 0., .75]])
        self.args['forces'].zero_()
        for i in range(220):
            self.args['steps'][:] = i+1
            result = self.tracker.update(**self.args)
        self.assertTrue(result['stuck'].item())
        self.assertFalse(result['failed'].item())
        self.tracker.reset(torch.tensor([0]), torch.tensor([0.]))
        self.assertFalse(self.tracker.stuck.any())


if __name__ == '__main__':
    unittest.main()
