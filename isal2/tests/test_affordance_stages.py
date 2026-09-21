"""Stage gates, successful-contact censoring, checkpoint state and export parity."""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

import torch
from isal2.tests import test_affordance as legacy
from isal2.deprecated_tasks.endpoint_course.common.affordance.collection import ContactCollector
from isal2.deprecated_tasks.endpoint_course.common.course.curriculum import CourseCurriculum
from isal2.deprecated_tasks.endpoint_course.common.course.geometry import STAGES
from isal2.deprecated_tasks.endpoint_course.common.course.runtime import CourseTaskMixin
from isal2.deprecated_tasks.endpoint_course.affordance_stage1.agents.ppo_cfg import AffordanceStage1AgentCfg
from isal2.deprecated_tasks.endpoint_course.affordance_stage2.agents.ppo_cfg import AffordanceStage2AgentCfg
from isal2.modified_rsl.modules.affordance import ActorCriticAffordance
from isal2.modified_rsl.runners.affordance_runner import AffordanceRunner
from isal2.modified_rsl.runners.checkpoint import warm_start, model_digest
from isal2.deployment.export import ActorExport


class StageCollectionTests(legacy.CollectionTests):
    def setUp(self):
        super().setUp()
        self.c = ContactCollector(2, 187, (1.6, 1.), .02)

    def finish_course(self, success=False, failure=False, timeout=False):
        frame = {k: v for k, v in self.frame.items() if k not in ('terminated', 'truncated')}
        self.c.update_course(**frame, result={
            'success': torch.tensor([success, False]), 'failed': torch.tensor([failure, False]),
            'timeout': torch.tensor([timeout, False])})

    def test_success_preserves_mature_label_instead_of_marking_failure(self):
        self.lift(); self.touch(); self.advance(10)
        self.finish_course(success=True)
        samples = self.c.pop_samples()
        self.assertAlmostEqual(samples['label'].item(), 1.)
        self.assertEqual(self.c.stats['failed'], 0)
        self.assertFalse(self.c.active[:2].any())
        self.assertFalse(self.c.swing[:2].any())

    def test_success_censors_short_windows_and_failure_wins(self):
        self.lift(); self.touch()
        self.finish_course(success=True)
        self.assertIsNone(self.c.pop_samples())
        self.assertFalse(self.c.active.any())
        self.setUp()
        self.lift(); self.touch()
        self.finish_course(success=True, failure=True)
        self.assertEqual(self.c.pop_samples()['label'].item(), 0.)

    def test_timeout_preserves_completed_samples_and_local_reset_preserves_other_env(self):
        self.lift(); self.touch(); self.advance(10)
        self.finish_course(timeout=True)
        self.assertEqual(self.c.pop_samples()['label'].item(), 1.)
        self.setUp()
        self.advance(2)
        self.frame['forces'][:, 0] = 0
        self.advance(2)
        self.frame['forces'][:, 0] = 10
        self.advance(2)
        self.c.reset(torch.tensor([0]))
        self.advance(11)
        samples = self.c.pop_samples()
        self.assertEqual(len(samples['label']), 1)
        self.assertEqual(samples['label'].item(), 1.)


class FakeCourseEnv(CourseTaskMixin):
    num_envs, num_actions, device = 4, 23, 'cpu'

    def __init__(self, stage):
        self.unwrapped = self
        self.resets = 0
        self.cfg = SimpleNamespace(seed=42, course=SimpleNamespace(signature=lambda: {'stage': stage}),
            scene_context=SimpleNamespace(terrain_generator=SimpleNamespace(num_cols=len(STAGES[stage])*2)))
        self.course_curriculum = CourseCurriculum(4, STAGES[stage])
        self.collector = ContactCollector(4, 187, (1.6, 1.), .02)

    def get_observations(self):
        return legacy.obs()

    def step(self, actions):
        return legacy.obs(), -actions.square().mean(-1), torch.zeros(4), {}

    def reset(self):
        self.resets += 1
        ids = torch.arange(4)
        self.course_curriculum.sample(ids)
        self.collector.reset(ids)
        return self.get_observations(), {}

    def set_collection_iteration(self, iteration):
        self.collector.iteration = iteration

    def pop_affordance_samples(self):
        return legacy.sample(32, self.collector.iteration)


class StageLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, stage):
        cfg = (AffordanceStage1AgentCfg() if stage == 1 else AffordanceStage2AgentCfg()).to_dict()
        cfg['num_steps_per_env'] = 4
        cfg['algorithm'].update(num_learning_epochs=1, num_mini_batches=1)
        cfg['policy'].update(actor_hidden_dims=[16], critic_hidden_dims=[16], unet_channels=[4, 8, 16])
        cfg['affordance']['capacity'] = 512
        return cfg

    def runner(self, stage):
        with contextlib.redirect_stdout(io.StringIO()):
            return AffordanceRunner(FakeCourseEnv(stage), self.config(stage))

    def checkpoint(self, runner):
        folder = Path(__file__).resolve().parents[1]/'outputs/aff_stage_unit'/uuid4().hex
        folder.mkdir(parents=True)
        path = folder/'model.pt'
        runner.save(path)
        return path

    def test_stage1_gate_boundaries_and_sample_threshold(self):
        runner = self.runner(1)
        self.assertEqual(float(runner.alg.policy.affordance_alpha), 0.)
        runner.total_samples = 256
        for iteration, alpha in ((0, 0.), (499, 0.), (500, 0.), (1000, .5), (1499, .999), (1500, 1.)):
            runner._start_iteration(iteration)
            self.assertAlmostEqual(float(runner.alg.policy.affordance_alpha), alpha, places=6)
        runner.total_samples = 255
        runner._start_iteration(1500)
        self.assertEqual(float(runner.alg.policy.affordance_alpha), 0.)

    def test_stage1_unet_updates_during_warmup_and_optimizers_are_disjoint(self):
        runner = self.runner(1)
        before = model_digest(runner.alg.policy.affordance_net.state_dict())
        with contextlib.redirect_stdout(io.StringIO()):
            runner.learn(1)
        self.assertEqual(runner.supervised_updates, 8)
        self.assertNotEqual(before, model_digest(runner.alg.policy.affordance_net.state_dict()))
        self.assertEqual(float(runner.alg.policy.affordance_alpha), 0.)
        orange = {id(p) for g in runner.alg.optimizer.param_groups for p in g['params']}
        blue = {id(p) for g in runner.supervised_optimizer.param_groups for p in g['params']}
        self.assertFalse(orange & blue)
        frozen = {k: v.clone() for k, v in runner.alg.policy.state_dict().items() if not k.startswith('affordance_net.')}
        runner.supervised_update(legacy.obs())
        for k, v in frozen.items():
            torch.testing.assert_close(v, runner.alg.policy.state_dict()[k], rtol=0, atol=0)

    def test_stage2_first_model_inference_and_empty_replay(self):
        cfg = self.config(2)
        policy = dict(cfg['policy'])
        policy.pop('class_name')
        with contextlib.redirect_stdout(io.StringIO()):
            model = ActorCriticAffordance(legacy.obs(), cfg['obs_groups'], 23, **policy)
        self.assertEqual(float(model.affordance_alpha), 1.)
        runner = self.runner(2)
        before = model_digest(runner.alg.policy.state_dict())
        self.assertEqual(runner.supervised_update(legacy.obs())['affordance_updates'], 0.)
        self.assertEqual(before, model_digest(runner.alg.policy.state_dict()))
        for iteration in (0, 500, 1500):
            runner._start_iteration(iteration)
            self.assertEqual(float(runner.alg.policy.affordance_alpha), 1.)
        with torch.no_grad():
            observation = legacy.obs()
            action = model.act_inference(observation).clone()
            model.affordance_net.output.bias.add_(8.)
            self.assertGreater((model.act_inference(observation)-action).abs().max().item(), 1.e-6)

    def test_stage2_rejects_nonzero_delays(self):
        for key in ('warmup_iterations', 'ramp_iterations', 'gate_min_samples'):
            cfg = self.config(2)
            cfg['affordance'][key] = 1
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Immediate'):
                AffordanceRunner(FakeCourseEnv(2), cfg)

    def test_checkpoint_restores_schedule_replay_both_optimizers_and_course(self):
        for stage in (1, 2):
            source = self.runner(stage)
            with contextlib.redirect_stdout(io.StringIO()):
                source.learn(1)
            source.total_samples = 400
            source._start_iteration(1000)
            source.replay.add(legacy.sample(80, iteration=999))
            source.current_learning_iteration = 1001
            source.env.course_curriculum.levels[0, 1] = 7
            path = self.checkpoint(source)
            restored = self.runner(stage)
            restored.load(path)
            self.assertEqual(restored.current_learning_iteration, 1001)
            self.assertEqual(restored.total_samples, 400)
            self.assertEqual(restored.supervised_updates, source.supervised_updates)
            self.assertEqual(float(restored.alg.policy.affordance_alpha), .5 if stage == 1 else 1.)
            self.assertEqual(restored.env.course_curriculum.levels[0, 1], 7)
            self.assertTrue(restored.alg.optimizer.state)
            self.assertTrue(restored.supervised_optimizer.state)
            self.assertEqual(model_digest(source.alg.policy.state_dict()), model_digest(restored.alg.policy.state_dict()))
            self.assertTrue(torch.equal(source.replay.valid, restored.replay.valid))
            self.assertEqual(len(restored.replay), 80)
            self.assertEqual(source.replay.next, restored.replay.next)
            for key in source.replay.data:
                torch.testing.assert_close(source.replay.data[key][source.replay.valid],
                                           restored.replay.data[key][restored.replay.valid])
            self.assertFalse(restored.env.collector.active.any())
            restored._start_iteration(1001)
            self.assertAlmostEqual(float(restored.alg.policy.affordance_alpha), .501 if stage == 1 else 1., places=6)

    def test_cross_stage_warm_start_resets_target_state_and_gate(self):
        source = self.runner(1)
        with contextlib.redirect_stdout(io.StringIO()):
            source.learn(1)
        path = self.checkpoint(source)
        for stage in (1, 2):
            target = self.runner(stage)
            target.env.course_curriculum.levels[:] = 6
            target.replay.add(legacy.sample(80))
            target.env.collector.active[:] = True
            if stage == 2:
                before = model_digest(target.alg.policy.state_dict())
                with self.assertRaises(ValueError):
                    target.load(path)
                self.assertEqual(before, model_digest(target.alg.policy.state_dict()))
            warm_start(target, path)
            self.assertEqual(float(target.alg.policy.affordance_alpha), 0. if stage == 1 else 1.)
            self.assertEqual(target.current_learning_iteration, 0)
            self.assertEqual(len(target.replay), 0)
            self.assertFalse(target.alg.optimizer.state)
            self.assertFalse(target.supervised_optimizer.state)
            self.assertTrue((target.env.course_curriculum.levels == 0).all())
            self.assertFalse(target.env.collector.active.any())
            for key, value in source.alg.policy.state_dict().items():
                if key not in ('std', 'affordance_alpha'):
                    torch.testing.assert_close(value, target.alg.policy.state_dict()[key], rtol=0, atol=0)

    def test_ame_checkpoint_is_rejected(self):
        from isal2.modified_rsl.modules import ActorCriticAME
        from isal2.deprecated_tasks.endpoint_course.ame_stage1.agents.ppo_cfg import AMEStage1AgentCfg
        cfg = AMEStage1AgentCfg()
        policy = dict(cfg.policy)
        policy.pop('class_name')
        with contextlib.redirect_stdout(io.StringIO()):
            model = ActorCriticAME(legacy.obs(), cfg.obs_groups, 23, **policy)
        with self.assertRaisesRegex(RuntimeError, 'Affordance checkpoint'):
            self.runner(1).alg.policy.load_state_dict(model.state_dict())

    def test_export_wrapper_preserves_stage1_gate_and_stage2_predictions(self):
        for stage in (1, 2):
            model = self.runner(stage).alg.policy.eval()
            for alpha in ((0., .5, 1.) if stage == 1 else (1.,)):
                model.affordance_alpha.fill_(alpha)
                wrapper = ActorExport(model).eval()
                for batch in (1, 4):
                    observation = legacy.obs(batch)
                    with torch.no_grad():
                        torch.testing.assert_close(wrapper(observation['policy'], observation['height_scan']),
                                                   model.act_inference(observation), atol=1e-5, rtol=1e-4)
                self.assertEqual(float(wrapper.alpha), alpha)


if __name__ == '__main__':
    unittest.main()
