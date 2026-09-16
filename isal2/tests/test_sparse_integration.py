"""Cross-component episode semantics and environment adapter cache checks."""
import unittest
from types import SimpleNamespace
import torch
from isal2.tasks.affordance.collection import ContactCollector, world_to_map
from isal2.utils.rsl_env import RslEnvAdapter
from isal2.tasks.sparse.evaluation import summarize


class SparseIntegrationTests(unittest.TestCase):
    def collector(self):
        c = ContactCollector(1, 187, (1.6, 1.), .02)
        f = dict(height=torch.zeros(1, 187), root=torch.zeros(1, 3), yaw=torch.zeros(1),
                 feet=torch.zeros(1, 2, 3), forces=torch.ones(1, 2)*10, speed=torch.zeros(1, 2),
                 support=torch.ones(1, 2), terminated=torch.tensor([False]), truncated=torch.tensor([False]))
        for _ in range(2):
            c.update(**f)
        f['forces'].zero_()
        for _ in range(2):
            c.update(**f)
        f['forces'][:] = 10
        for _ in range(2):
            c.update(**f)
        return c, f

    def test_success_truncation_censors_pending_not_failure_label(self):
        c, f = self.collector()
        f['truncated'][:] = True
        c.update(**f)
        self.assertIsNone(c.pop_samples())
        self.assertFalse(c.active.any())
        self.assertEqual(c.stats['failed'], 0)

    def test_real_failure_emits_failed_pending_contacts(self):
        c, f = self.collector()
        f['terminated'][:] = True
        c.update(**f)
        samples = c.pop_samples()
        self.assertIsNotNone(samples)
        self.assertTrue((samples['label'] == 0).all())

    def test_drift_label_query_uses_sampling_origin(self):
        root = torch.tensor([[.02, -.01, .75]])
        foot = torch.tensor([[.10, .10, 0.]])
        query = world_to_map(foot, root, torch.zeros(1))
        self.assertTrue(torch.allclose(query, torch.tensor([[.08, .11]])))

    def test_metrics_count_all_attempts_and_stuck_can_overlap_timeout(self):
        common = dict(kind='single_beam', level=0, bypass=False, seconds=20., progress=.1, tracking_error=.3)
        data = [dict(common, success=True, fall=False, timeout=False, stuck=False),
                dict(common, success=False, fall=False, timeout=True, stuck=True),
                dict(common, success=False, fall=True, timeout=False, stuck=False)]
        summary = summarize(data)[0]
        self.assertEqual(summary['n'], 3)
        self.assertAlmostEqual(summary['success_rate'], 1/3)
        self.assertAlmostEqual(summary['stuck_rate'], 1/3)


if __name__ == '__main__':
    unittest.main()
