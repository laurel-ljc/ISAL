"""CPU tests for the reference distance curriculum and strict state restore."""
import copy
import contextlib
import io
from pathlib import Path
import random
from uuid import uuid4
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from isal2.tasks.common.reference.curriculum import distance_moves, update_levels
from isal2.tasks.common.reference.runtime import ReferenceTaskMixin
from isal2.tasks.common.reference.seeding import terrain_seed


class FakeTask(ReferenceTaskMixin):
    def __init__(self, n=100, stage=1):
        self.num_envs, self.device = n, 'cpu'
        self.cfg = SimpleNamespace(reference_signature=lambda: dict(stage=stage, version=1))
        terrain = SimpleNamespace(terrain_origins=torch.arange(10*20*3).reshape(10,20,3).float(),
            terrain_levels=torch.zeros(n,dtype=torch.long), terrain_types=torch.zeros(n,dtype=torch.long),
            env_origins=torch.zeros(n,3))
        self.scene = SimpleNamespace(terrain=terrain, env_origins=terrain.env_origins)


class ReferenceTests(unittest.TestCase):
    def test_only_two_active_tasks_and_four_archived_endpoint_ids(self):
        import gymnasium as gym
        import isal2.tasks
        import isal2.deprecated_tasks
        active = {name for name,spec in gym.registry.items() if name.startswith('ISAL2-')
                  and spec.entry_point.startswith('isal2.tasks.')}
        self.assertEqual(active,{'ISAL2-RPO-AME-Stage1-v0','ISAL2-RPO-AME-Stage2-v0'})
        for family in ('AME','Affordance'):
            for stage in (1,2):
                self.assertIn(f'ISAL2-RPO-{family}-Endpoint-Stage{stage}-v0',gym.registry)

    def test_policy_ppo_and_network_shapes_preserved(self):
        from isal2.tasks.ame_stage1.agents.ppo_cfg import AMEStage1AgentCfg
        from isal2.tasks.ame_stage2.agents.ppo_cfg import AMEStage2AgentCfg
        from isal2.deprecated_tasks.endpoint_course.ame_stage1.agents.ppo_cfg import AMEStage1AgentCfg as Old
        for cfg in (AMEStage1AgentCfg(),AMEStage2AgentCfg()):
            for key in ('policy','algorithm','obs_groups','num_steps_per_env'):
                self.assertEqual(getattr(cfg,key),getattr(Old(),key))

    def test_real_runner_checkpoint_restore_warm_start_and_legacy_rejection(self):
        from tensordict import TensorDict
        from isal2.tasks.ame_stage1.agents.ppo_cfg import AMEStage1AgentCfg
        from isal2.modified_rsl.runners import OnPolicyRunner
        from isal2.modified_rsl.runners.checkpoint import warm_start,model_digest
        from isal2.utils.checkpoint import load_checkpoint
        class Env(FakeTask):
            num_actions = 23
            def __init__(self,stage=1):
                super().__init__(4,stage)
                self.unwrapped,self.resets = self,0
            def get_observations(self):
                return TensorDict({k:torch.zeros(4,n) for k,n in
                    [('policy',390),('critic',1630),('height_scan',187)]},[4])
            def reset(self):
                self.resets += 1
                return self.get_observations(),{}
        torch.set_num_threads(2)
        temporary_root = Path(__file__).resolve().parents[1]/'outputs/reference_unit'
        directory = temporary_root/uuid4().hex
        directory.mkdir(parents=True)
        with contextlib.redirect_stdout(io.StringIO()):
            path = directory/'model.pt'
            runner = OnPolicyRunner(Env(),AMEStage1AgentCfg().to_dict())
            # Populate Adam state, including its momentum tensors.
            loss = sum(p.square().mean() for p in runner.alg.policy.parameters())
            loss.backward(); runner.alg.optimizer.step(); runner.alg.optimizer.zero_grad()
            runner.env.scene.terrain.terrain_levels[0] = 8
            runner.current_learning_iteration = 17
            runner.save(path)
            expected_random = (random.random(),np.random.rand(),torch.rand(3))
            target = OnPolicyRunner(Env(),AMEStage1AgentCfg().to_dict())
            target.load(path)
            self.assertEqual(target.current_learning_iteration,17)
            self.assertEqual(target.env.scene.terrain.terrain_levels[0],8)
            self.assertEqual(target.env.resets,1)
            self.assertEqual(model_digest(runner.alg.policy.state_dict()),model_digest(target.alg.policy.state_dict()))
            self.assertEqual(len(runner.alg.optimizer.state),len(target.alg.optimizer.state))
            actual_random = (random.random(),np.random.rand(),torch.rand(3))
            self.assertEqual(expected_random[:2],actual_random[:2])
            self.assertTrue(torch.equal(expected_random[2],actual_random[2]))
            stage2 = OnPolicyRunner(Env(2),AMEStage1AgentCfg().to_dict())
            before = model_digest(stage2.alg.policy.state_dict())
            with self.assertRaises(ValueError):
                stage2.load(path)
            self.assertEqual(before,model_digest(stage2.alg.policy.state_dict()))
            warm_start(stage2,path)
            self.assertEqual(stage2.current_learning_iteration,0)
            self.assertFalse(stage2.alg.optimizer.state)
            self.assertTrue((stage2.env.scene.terrain.terrain_levels<=5).all())
            self.assertTrue(torch.allclose(stage2.alg.policy.std,torch.full((23,),.3)))
            for key,value in runner.alg.policy.state_dict().items():
                if key!='std':
                    self.assertTrue(torch.equal(value,stage2.alg.policy.state_dict()[key]),key)
            old = load_checkpoint(path)
            old['course_state'] = old.pop('reference_state')
            torch.save(old,path)
            with self.assertRaisesRegex(ValueError,'Endpoint/legacy'):
                target.load(path)

    def test_exact_distance_thresholds_and_upgrade_precedence(self):
        d = torch.tensor([4.,4.001,3.99,3.,0.,.01])
        cmd = torch.tensor([[1.5,0],[1.5,0],[.3,.4],[.3,0],[0,0],[0,0]])
        up, down = distance_moves(d, cmd)
        self.assertEqual(up.tolist(), [False,True,False,False,False,False])
        self.assertEqual(down.tolist(), [True,False,True,False,False,False])

    def test_level_limits_and_random_highest_level_reset(self):
        torch.manual_seed(91)
        levels = torch.cat([torch.zeros(100,dtype=torch.long),torch.full((100,),9)])
        result = update_levels(levels, levels==9, levels==0)
        self.assertTrue((result[:100]==0).all())
        self.assertTrue(((result>=0)&(result<=9)).all())
        self.assertEqual(set(result[100:].tolist()), set(range(10)))

    def test_initialization_fixed_columns_and_restore(self):
        task = FakeTask()
        task.reset_reference_curriculum()
        state = task.reference_state_dict()
        self.assertEqual(set(state['levels'].tolist()), set(range(6)))
        self.assertEqual(state['columns'].tolist(), [i//5 for i in range(100)])
        task.reset_reference_curriculum()
        task.load_reference_state_dict(state)
        self.assertTrue(torch.equal(task.scene.terrain.terrain_levels,state['levels']))
        self.assertTrue(torch.equal(task.scene.env_origins,task.scene.terrain.terrain_origins[state['levels'],state['columns']]))
        for target in (FakeTask(99),FakeTask(stage=2)):
            with self.assertRaises(ValueError):
                target.load_reference_state_dict(state)
        for field,value in [('levels',torch.full((100,),10)),('columns',torch.full((100,),20))]:
            invalid = copy.deepcopy(state)
            invalid[field] = value
            with self.assertRaises(ValueError):
                task.load_reference_state_dict(invalid)

    def test_manual_update_does_nothing(self):
        task = FakeTask()
        task._automatic_reset = False
        task._update_curriculum(torch.tensor([0]))
        self.assertFalse(task.scene.terrain.terrain_levels.any())

    def test_seed_repeatability_and_rng_isolation(self):
        np.random.seed(71)
        random.seed(71)
        torch.manual_seed(71)
        expected = (np.random.rand(),random.random(),float(torch.rand(())))
        np.random.seed(71)
        random.seed(71)
        torch.manual_seed(71)
        def draw(seed):
            with terrain_seed(seed,.451,'gap'):
                return np.random.rand(12),[random.random() for _ in range(12)],torch.rand(12)
        a,b,c = draw(42),draw(42),draw(43)
        np.testing.assert_equal(a[0],b[0])
        self.assertEqual(a[1],b[1])
        self.assertTrue(torch.equal(a[2],b[2]))
        self.assertFalse(np.array_equal(a[0],c[0]))
        self.assertEqual((np.random.rand(),random.random(),float(torch.rand(()))),expected)


if __name__ == '__main__':
    unittest.main()
