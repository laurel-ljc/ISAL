"""Deterministic collection, replay and optimizer-boundary regressions."""
from pathlib import Path
import sys
import unittest
from uuid import uuid4
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from tensordict import TensorDict
from isal2.tasks.affordance.collection import ContactCollector, ContactCollectionCfg, world_to_map
from isal2.tasks.affordance.agents.ppo_cfg import AffordanceAgentCfg
from isal2.modified_rsl.modules.affordance import AffordanceUNet
from isal2.modified_rsl.algorithms.affordance_replay import AffordanceReplay
from isal2.modified_rsl.runners.affordance_runner import AffordanceRunner


def obs(n=4):
    return TensorDict({"policy": torch.randn(n, 390), "critic": torch.randn(n, 1630),
                       "height_scan": torch.randn(n, 187)}, [n])


def sample(n=8, iteration=0):
    return dict(height_scan=torch.randn(n, 187), query_xy=torch.zeros(n, 2), label=torch.ones(n)*.8,
                metrics=torch.ones(n, 3), foot_side=torch.arange(n)%2,
                iteration=torch.full((n,), iteration, dtype=torch.long))


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.c = ContactCollector(2, 187, (1.6, 1.), .02)
        self.frame = dict(height=torch.zeros(2,187), root=torch.zeros(2,3), yaw=torch.zeros(2),
            feet=torch.zeros(2,2,3), forces=torch.ones(2,2)*10, speed=torch.zeros(2,2),
            support=torch.ones(2,2), terminated=torch.zeros(2,dtype=torch.bool), truncated=torch.zeros(2,dtype=torch.bool))

    def advance(self, n=1):
        for _ in range(n):
            self.c.update(**self.frame)

    def lift(self):
        self.advance(2)  # initial contact: unpaired, not a training sample
        self.frame["forces"][0,0] = 0
        self.frame["height"][0] = .2
        self.advance()
        self.frame["height"][0] = .7
        self.advance()

    def touch(self):
        self.frame["forces"][0,0] = 10
        self.advance(2)

    def test_first_frame_pairing_and_cross_rollout_label(self):
        self.lift()
        self.c.iteration = 1
        self.touch()
        self.assertIsNone(self.c.pop_samples())
        self.c.iteration = 2
        self.advance(11)
        data = self.c.pop_samples()
        self.assertEqual(len(data["label"]), 1)
        self.assertTrue(torch.allclose(data["height_scan"], torch.full((1,187), .2)))
        self.assertEqual(data["iteration"].item(), 0)
        self.assertEqual(data["foot_side"].item(), 0)
        self.assertAlmostEqual(data["label"].item(), 1.)
        self.assertEqual(self.c.window, 13)
        self.assertIsNone(self.c.pop_samples())

    def test_debounce_and_independent_feet(self):
        self.advance(2)
        self.frame["forces"][0,0] = 0
        self.advance()
        self.frame["forces"][0,0] = 10
        self.advance()
        self.assertEqual(self.c.stats["liftoff"], 0)
        self.frame["forces"][1,1] = 0
        self.advance(2)
        self.assertEqual(self.c.swing.tolist(), [False,False,False,True])
        self.c.reset(torch.tensor([0]))
        self.assertTrue(self.c.swing[3])

    def test_manual_reset_after_inference_collection(self):
        with torch.inference_mode():
            self.lift(); self.touch()
        self.c.reset(torch.tensor([0]))
        self.assertFalse(self.c.active[0].any())
        self.assertFalse(self.c.on_count.is_inference())
        self.assertFalse(self.c.off_count.is_inference())

    def test_transform_boundaries_and_td_first_frame(self):
        root = torch.tensor([[4.,2.,0.]])
        foot = torch.tensor([[4.,2.5,0.]])
        self.assertTrue(torch.allclose(world_to_map(foot,root,torch.tensor([torch.pi/2])), torch.tensor([[.5,0.]]), atol=1e-6))
        self.lift()
        self.frame["forces"][0,0] = 10
        self.frame["feet"][0,0,0] = .8
        self.advance()
        self.frame["feet"][0,0,0] = 2.
        self.advance(12)
        data=self.c.pop_samples()
        self.assertAlmostEqual(data["query_xy"][0,0].item(), .8)

    def test_outside_and_swing_timeout(self):
        self.lift()
        self.frame["feet"][0,0,0] = .81
        self.touch()
        self.assertEqual(self.c.stats["out_of_bounds"],1)
        self.frame["forces"][0,0] = 0
        self.advance(53)
        self.assertEqual(self.c.stats["swing_timeout"],1)
        self.touch()
        self.assertFalse(self.c.active.any())

    def test_failure_precedes_full_window_and_truncation_censors(self):
        self.lift(); self.touch(); self.advance(10)
        self.frame["terminated"][0]=True
        self.advance()
        self.assertEqual(self.c.pop_samples()["label"].item(),0.)
        self.setUp(); self.lift(); self.touch()
        self.frame["truncated"][0]=True
        self.advance()
        self.assertIsNone(self.c.pop_samples())
        self.assertFalse(self.c.active.any())
        self.setUp(); self.lift(); self.touch(); self.advance(10)
        self.frame["truncated"][0]=True
        self.advance()
        self.assertAlmostEqual(self.c.pop_samples()["label"].item(),1.)

    def test_quality_formula_and_manual_reset(self):
        self.lift()
        self.frame["speed"][0,0] = .2
        self.frame["support"][0,0] = .5
        self.touch(); self.advance(11)
        data=self.c.pop_samples()
        self.assertAlmostEqual(data["label"].item(), .4*torch.exp(torch.tensor(-1.)).item()+.3+.15, places=6)
        self.setUp(); self.lift(); self.touch()
        self.c.reset(torch.tensor([0]))
        self.advance(20)
        self.assertIsNone(self.c.pop_samples())

    def test_overlapping_windows_and_overflow(self):
        self.c.cfg.pending_slots = 4
        self.c.window=100
        self.advance(2)
        for _ in range(5):
            self.frame["forces"][0,0]=0; self.advance(2)
            self.frame["forces"][0,0]=10; self.advance(2)
        self.assertEqual(int(self.c.active[0].sum()),4)
        self.assertEqual(self.c.stats["overflow"],1)


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def runner(self):
        class FakeEnv:
            num_envs, num_actions, device, cfg = 4,23,"cpu",{}
            def __init__(self):
                self.unwrapped=self
                self.collector=ContactCollector(4,187,(1.6,1.),.02)
            def get_observations(self): return obs()
            def step(self, actions): return obs(), -actions.square().mean(-1), torch.zeros(4), {}
            def set_collection_iteration(self,i): self.collector.iteration=i
            def pop_affordance_samples(self): return sample(4,self.collector.iteration)
        cfg=AffordanceAgentCfg().to_dict()
        cfg["num_steps_per_env"]=3
        cfg["algorithm"].update(num_learning_epochs=1,num_mini_batches=1)
        cfg["policy"].update(actor_hidden_dims=[16],critic_hidden_dims=[16])
        cfg["affordance"].update(capacity=64,min_samples=4,batch_size=4,gradient_steps=2,
                                  warmup_iterations=0,ramp_iterations=1,gate_min_samples=4)
        return AffordanceRunner(FakeEnv(),cfg)

    def test_unet_odd_size_and_query_sampling(self):
        net=AffordanceUNet()
        self.assertEqual(net(torch.randn(3,1,11,17)).shape,(3,1,11,17))
        x=torch.arange(17).float()/16
        image=x[None,None,None,:].expand(1,1,11,17)
        grid=torch.tensor([[[[-1.,0.],[0.,0.],[1.,0.]]]])
        result=torch.nn.functional.grid_sample(image,grid,align_corners=True)
        self.assertTrue(torch.allclose(result.flatten(),torch.tensor([0.,.5,1.])))

    def test_replay_capacity_expiration_and_restore(self):
        replay=AffordanceReplay(187,5,2)
        with self.assertRaises(ValueError): replay.sample(2)
        replay.add(sample(3,0)); replay.add(sample(3,1))
        self.assertEqual(len(replay),5)
        replay.expire(2)
        self.assertEqual(len(replay),3)
        state=replay.state_dict()
        restored=AffordanceReplay(187,5,2); restored.load_state_dict(state)
        self.assertTrue(torch.equal(replay.valid,restored.valid))
        self.assertEqual(replay.next,restored.next)
        self.assertTrue((restored.sample(20)["iteration"]==1).all())
        restored.expire(3); self.assertEqual(len(restored),0)

    def test_ppo_and_supervision_are_disjoint(self):
        runner=self.runner(); model=runner.alg.policy
        runner._start_iteration(0)
        model.affordance_alpha.fill_(1)
        observation=obs()
        blue={n:p.clone() for n,p in model.affordance_net.named_parameters()}
        with torch.inference_mode():
            for _ in range(3):
                actions=runner.alg.act(observation)
                logp=model.get_actions_log_prob(actions).clone()
                runner.alg.process_env_step(observation,torch.randn(4),torch.zeros(4),{})
                model.act(observation)
                self.assertTrue(torch.allclose(logp,model.get_actions_log_prob(actions),atol=1e-5))
            runner.alg.compute_returns(observation)
        runner.alg.update()
        for n,p in model.affordance_net.named_parameters():
            self.assertTrue(torch.equal(blue[n],p)); self.assertIsNone(p.grad)
        orange={n:p.clone() for n,p in model.state_dict().items() if not n.startswith("affordance_net.")}
        runner.replay.add(sample(8)); model.affordance_alpha.fill_(1)
        orange["affordance_alpha"]=model.affordance_alpha.clone()
        runner.supervised_update(observation)
        self.assertTrue(any(not torch.equal(blue[n],p) for n,p in model.affordance_net.named_parameters()))
        for n,p in model.state_dict().items():
            if n in orange: self.assertTrue(torch.equal(orange[n],p),n)
        self.assertTrue(all(p.grad is None for p in model.affordance_net.parameters()))

    def test_gate_schedule_and_empty_supervision(self):
        runner=self.runner()
        before={k:v.clone() for k,v in runner.alg.policy.state_dict().items()}
        losses=runner.supervised_update(obs())
        self.assertEqual(losses["affordance_updates"],0)
        for k,v in before.items():
            self.assertTrue(torch.equal(v,runner.alg.policy.state_dict()[k]),k)
        runner.aux_cfg.update(warmup_iterations=500,ramp_iterations=1000,gate_min_samples=256)
        runner.total_samples=255
        runner._start_iteration(1500)
        self.assertEqual(float(runner.alg.policy.affordance_alpha),0)
        runner.total_samples=256
        for iteration, expected in ((0,0),(500,0),(1000,.5),(1500,1),(2000,1)):
            runner._start_iteration(iteration)
            self.assertEqual(float(runner.alg.policy.affordance_alpha),expected)

    def test_gate_checkpoint_and_real_two_phase_loop(self):
        runner=self.runner()
        runner.learn(2)
        self.assertEqual(runner.supervised_updates,4)
        self.assertEqual(float(runner.alg.policy.affordance_alpha),1)
        path=Path(__file__).resolve().parents[1]/"outputs"/("aff_unit_"+uuid4().hex)
        path.mkdir(parents=True)
        runner.save(str(path/"model.pt"))
        restored=self.runner(); restored.load(str(path/"model.pt"),map_location="cpu")
        self.assertEqual(restored.current_learning_iteration,2)
        self.assertEqual(restored.total_samples,runner.total_samples)
        self.assertTrue(restored.supervised_optimizer.state)
        self.assertEqual(len(restored.replay),len(runner.replay))
        for n,p in runner.alg.policy.state_dict().items():
            self.assertTrue(torch.equal(p,restored.alg.policy.state_dict()[n]),n)
        restored.learn(1)
        self.assertEqual(restored.supervised_updates,6)


if __name__ == "__main__": unittest.main()
