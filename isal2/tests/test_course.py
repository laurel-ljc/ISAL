"""CPU acceptance tests for actual meshes, course state and endpoint behavior."""
import copy
import contextlib
import io
from pathlib import Path
from uuid import uuid4
import unittest
from types import SimpleNamespace
import numpy as np
import torch
import trimesh

from isal2.deprecated_tasks.endpoint_course.common.course.geometry import STAGES, build_tile, validate_perception
from isal2.deprecated_tasks.endpoint_course.common.course.curriculum import CourseCurriculum
from isal2.deprecated_tasks.endpoint_course.common.course.commands import endpoint_velocity, push_base_horizontal
from isal2.deprecated_tasks.endpoint_course.common.course.outcomes import OutcomeTracker, polygon_planes, supported, failure_penalty


class CourseGeometryTests(unittest.TestCase):
    def test_all_120_collision_meshes_match_supports_and_platforms(self):
        rng = np.random.default_rng(121)
        for kind in STAGES[1]+STAGES[2]:
            for level in range(10):
                with self.subTest(kind=kind, level=level):
                    tile = build_tile(kind, level)
                    validate_perception(tile)
                    mesh = trimesh.util.concatenate(tile.meshes())
                    self.assertTrue(mesh.is_watertight)
                    xy = np.r_[rng.uniform((-3.9, -1.4), (3.9, 1.4), (150, 2)),
                               [[x, y] for x in (-2.9, -2.7, -2.5, 2.5, 2.7, 2.9) for y in (-.2, 0, .2)]]
                    hits, indices, _ = mesh.ray.intersects_location(
                        np.c_[xy, np.full(len(xy), 4.)], np.tile((0, 0, -1), (len(xy), 1)))
                    heights = np.full(len(xy), -np.inf)
                    np.maximum.at(heights, indices, hits[:, 2])
                    np.testing.assert_allclose(heights, tile.heights(xy), atol=1.e-6)
                    np.testing.assert_allclose(heights[-18:], 0.)
                    self.assertEqual(tile.heights([[0., 2.5]])[0], -1.)
                    planes, tops = polygon_planes(tile.metadata()['supports'])
                    feet = torch.tensor(np.c_[xy, heights], dtype=torch.float32)[None]
                    actual = supported(feet, planes[None], tops[None])[0].numpy()
                    np.testing.assert_array_equal(actual, heights > -.9)

    def test_stepping_stones_visible_across_scan_grid_phases(self):
        for kind in ('grid_stones', 'single_column_stones', 'pentagon_stones'):
            for level in range(10):
                tile = build_tile(kind, level)
                for stone in tile.supports[4:]:
                    center = np.asarray(stone.polygon).mean(axis=0)
                    for dx in (0., .025, .05, .075):
                        for dy in (0., .025, .05, .075):
                            x, y = np.meshgrid(np.arange(-.8, .801, .1)+center[0]+dx,
                                               np.arange(-.5, .501, .1)+center[1]+dy)
                            self.assertGreaterEqual(stone.contains(np.stack((x, y), -1)).sum(), 4)

    def test_actual_route_gaps_and_landings_fit_forward_scan(self):
        for kind in ('pallets', 'narrow_pallets', 'gaps', 'grid_stones', 'pentagon_stones',
                     'single_column_stones', 'consecutive_gaps'):
            for level in range(10):
                tile = build_tile(kind, level)
                x = np.arange(-2., 2.0001, .002)
                walkable = tile.heights(np.c_[x, np.zeros_like(x)]) > -.9
                changes = np.diff(np.r_[True, walkable, True].astype(int))
                starts, ends = np.flatnonzero(changes == -1), np.flatnonzero(changes == 1)
                for start, end in zip(starts, ends):
                    self.assertLessEqual((end-start)*.002 + .2 + .2, .8)
                    # At least .2 m of safe landing lies after each gap.
                    self.assertTrue(walkable[end:min(end+100, len(x))].all())

    def test_pentagon_corners_not_box_support_and_narrow_stair_depth(self):
        tile = build_tile('pentagon_stones', 9)
        stone = tile.supports[4]
        vertices = np.asarray(stone.polygon)
        corner = [vertices[:, 0].max()-.001, vertices[:, 1].max()-.001]
        self.assertFalse(stone.contains([corner])[0])
        a, b = (build_tile('narrow_stairs', k) for k in (0, 9))
        self.assertGreater(a.params['tread'], b.params['tread'])
        self.assertEqual(a.params['width'], b.params['width'])
        for kind in STAGES[1]+STAGES[2]:
            self.assertNotEqual(build_tile(kind, 0).params, build_tile(kind, 9).params)

    def test_actual_scanner_settings_reject_invisible_gaps_or_stones(self):
        with self.assertRaisesRegex(ValueError, 'forward scan'):
            validate_perception(build_tile('gaps', 9), size=(1., 1.))
        with self.assertRaisesRegex(ValueError, 'Stone support'):
            validate_perception(build_tile('pentagon_stones', 9), resolution=.3)
        with self.assertRaisesRegex(ValueError, 'positive'):
            validate_perception(build_tile('stairs', 0), resolution=0.)


class CourseBehaviorTests(unittest.TestCase):
    def test_curriculum_independent_levels_clamps_sampling_and_roundtrip(self):
        c = CourseCurriculum(10000, STAGES[1])
        ids = torch.arange(10000)
        torch.manual_seed(41)
        types, levels = c.sample(ids)
        self.assertTrue((levels == 0).all())
        counts = torch.bincount(types, minlength=7)
        self.assertTrue(((counts > 1250) & (counts < 1600)).all())
        c.record(ids, torch.ones(10000, dtype=torch.bool))
        self.assertTrue((c.levels.sum(-1) == 1).all())
        before = c.levels.clone()
        c.sample(ids)
        self.assertTrue(torch.equal(before, c.levels))
        c.levels[:] = 9
        c.record(ids, torch.ones(10000, dtype=torch.bool))
        self.assertTrue((c.levels == 9).all())
        c.levels[:] = 0
        c.record(ids, torch.zeros(10000, dtype=torch.bool))
        self.assertTrue((c.levels == 0).all())
        c.levels[0, 2] = 4
        state = c.state_dict()
        other = CourseCurriculum(10000, STAGES[1])
        other.load_state_dict(state)
        self.assertTrue(torch.equal(other.levels, c.levels))
        bad = copy.deepcopy(state)
        bad['levels'][0, 0] = 10
        with self.assertRaises(ValueError):
            other.load_state_dict(bad)

    def test_command_rotation_direction_slowdown(self):
        pos = torch.zeros(3, 3)
        goal = torch.tensor([[2., 0., 0.], [2., 0., 0.], [.15, 0., 0.]])
        yaw = torch.tensor([0., torch.pi/2, 0.])
        result = endpoint_velocity(pos, goal, yaw, torch.full((3,), .4))
        torch.testing.assert_close(result[0], torch.tensor([.4, 0., 0.]))
        self.assertAlmostEqual(result[1, 1].item(), -.4, places=6)
        self.assertLess(result[1, 2].item(), 0.)
        self.assertAlmostEqual(result[2, 0].item(), .1, places=6)

    def test_success_failure_timeout_precedence_and_partial_reset(self):
        tile = build_tile('gaps', 0)
        planes, tops = polygon_planes(tile.metadata()['supports'])
        tracker = OutcomeTracker(3, .02)
        root = torch.tensor([[2.7, 0., .7]]).repeat(3, 1)
        feet = torch.tensor([[[2.7, -.1, .0], [2.7, .1, .0]]]).repeat(3, 1, 1)
        args = [root, feet, torch.full((3, 2), 10.), planes[None].repeat(3, 1, 1, 1),
                tops[None].repeat(3, 1), root.clone(), torch.full((3,), .6),
                torch.tensor([False, True, False]), torch.tensor([True, False, False])]
        for i in range(9):
            result = tracker.update(*args)
            self.assertFalse(result['success'].any())
        result = tracker.update(*args)
        self.assertEqual(result['success'].tolist(), [True, False, True])
        self.assertEqual(result['failed'].tolist(), [False, True, False])
        self.assertFalse(result['timeout'].any())
        torch.testing.assert_close(failure_penalty(SimpleNamespace(course_result=result)), torch.tensor([0., 1., 0.]))
        tracker.reset(torch.tensor([0]))
        self.assertEqual(tracker.stable.tolist(), [0, 0, 10])
        root[0, 0] = -2.7
        result = tracker.update(*args)
        self.assertTrue(result['timeout'][0])
        root[2, 0] = 0.
        root[2, 1] = 1.
        self.assertTrue(tracker.update(*args)['bypass'][2])

    def test_push_is_additive_horizontal_bounded_and_partial(self):
        velocity = torch.randn(50, 6)
        saved = {}
        robot = SimpleNamespace(data=SimpleNamespace(root_vel_w=velocity),
            write_root_velocity_to_sim=lambda v, ids: saved.update(v=v, ids=ids))
        env = SimpleNamespace(robot=robot, device='cpu', num_envs=50)
        ids = torch.tensor([0, 3, 7])
        push_base_horizontal(env, ids)
        self.assertTrue(((saved['v'][:, :2]-velocity[ids, :2]).abs() <= .100001).all())
        self.assertTrue(torch.equal(saved['v'][:, 2:], velocity[ids, 2:]))
        self.assertTrue(torch.equal(saved['ids'], ids))

    def test_stage_policy_matches_legacy_architecture(self):
        from isal2.deprecated_tasks.ame.agents.ppo_cfg import AMEAgentCfg
        from isal2.deprecated_tasks.endpoint_course.ame_stage1.agents.ppo_cfg import AMEStage1AgentCfg
        from isal2.deprecated_tasks.endpoint_course.ame_stage2.agents.ppo_cfg import AMEStage2AgentCfg
        old = AMEAgentCfg()
        for cfg in (AMEStage1AgentCfg(), AMEStage2AgentCfg()):
            self.assertEqual(old.policy, cfg.policy)
            self.assertEqual(old.algorithm, cfg.algorithm)
            self.assertEqual(old.obs_groups, cfg.obs_groups)


class CourseCheckpointTests(unittest.TestCase):
    def test_real_runner_resume_and_stage_transfer(self):
        from tensordict import TensorDict
        from isal2.deprecated_tasks.endpoint_course.common.course.runtime import CourseTaskMixin
        from isal2.deprecated_tasks.endpoint_course.ame_stage1.agents.ppo_cfg import AMEStage1AgentCfg
        from isal2.modified_rsl.runners import OnPolicyRunner
        from isal2.modified_rsl.runners.checkpoint import warm_start, model_digest

        class FakeEnv(CourseTaskMixin):
            num_envs, num_actions, device = 4, 23, 'cpu'
            def __init__(self, stage=1):
                self.unwrapped = self
                self.resets = 0
                self.cfg = SimpleNamespace(seed=42, course=SimpleNamespace(signature=lambda: {'stage': stage}),
                    scene_context=SimpleNamespace(terrain_generator=SimpleNamespace(num_cols=len(STAGES[stage])*2)))
                self.course_curriculum = CourseCurriculum(4, STAGES[stage])
            def get_observations(self):
                return TensorDict({k: torch.zeros(4, n) for k, n in
                    [('policy', 390), ('critic', 1630), ('height_scan', 187)]}, [4])
            def reset(self):
                self.resets += 1
                self.course_curriculum.sample(torch.arange(4))
                return self.get_observations(), {}

        torch.set_num_threads(2)
        folder = Path(__file__).resolve().parents[1]/'outputs/course_unit'/uuid4().hex
        folder.mkdir(parents=True)
        path = folder/'checkpoint.pt'
        with contextlib.redirect_stdout(io.StringIO()):
            source = OnPolicyRunner(FakeEnv(), AMEStage1AgentCfg().to_dict())
            source.env.course_curriculum.levels[0, 2] = 7
            source.current_learning_iteration = 19
            source.save(path)
            target = OnPolicyRunner(FakeEnv(), AMEStage1AgentCfg().to_dict())
            target.load(path)
            self.assertEqual(target.current_learning_iteration, 19)
            self.assertEqual(target.env.resets, 1)
            self.assertTrue(torch.equal(source.env.course_curriculum.levels, target.env.course_curriculum.levels))
            self.assertEqual(model_digest(source.alg.policy.state_dict()), model_digest(target.alg.policy.state_dict()))
            stage2 = OnPolicyRunner(FakeEnv(2), AMEStage1AgentCfg().to_dict())
            before = model_digest(stage2.alg.policy.state_dict())
            with self.assertRaisesRegex(ValueError, 'signature'):
                stage2.load(path)
            self.assertEqual(before, model_digest(stage2.alg.policy.state_dict()))
            warm_start(stage2, path)
            self.assertEqual(stage2.current_learning_iteration, 0)
            self.assertTrue((stage2.env.course_curriculum.levels == 0).all())
            self.assertFalse(stage2.alg.optimizer.state)
            for key, value in source.alg.policy.state_dict().items():
                if key != 'std':
                    self.assertTrue(torch.equal(value, stage2.alg.policy.state_dict()[key]), key)


if __name__ == '__main__':
    unittest.main()
