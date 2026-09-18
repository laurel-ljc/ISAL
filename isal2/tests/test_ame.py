"""AME CPU tests exercise geometry, policy information flow and actual PPO updates."""
from pathlib import Path
import sys
import unittest
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from isal2.modified_rsl.algorithms import PPO
from isal2.modified_rsl.modules import ActorCriticAME
from isal2.modified_rsl.modules.terrain_attention import TerrainAttention
from isal2.modified_rsl.runners import OnPolicyRunner
from isal2.deprecated_tasks.ame.agents.ppo_cfg import AMEAgentCfg


def observations(n=4):
    return TensorDict({"policy": torch.randn(n, 390), "critic": torch.randn(n, 1630),
                       "height_scan": torch.randn(n, 187)}, [n])


class AMETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def model(self, obs):
        cfg = AMEAgentCfg()
        args = dict(cfg.policy)
        args.pop("class_name")
        args.update(actor_hidden_dims=[32], critic_hidden_dims=[32])
        return ActorCriticAME(obs, cfg.obs_groups, 23, **args)

    def test_xy_geometry_and_tokens(self):
        net = TerrainAttention(390)
        coords = net.position_encoding.coordinates[0]
        self.assertEqual(tuple(coords.shape), (2, 11, 17))
        self.assertTrue(torch.allclose(coords[:, 0, 0], torch.tensor([-.8, -.5])))
        self.assertTrue(torch.allclose(coords[:, -1, -1], torch.tensor([.8, .5])))
        # Independent x-fast construction catches a transposed scan reshape.
        scan = torch.tensor([[x + 10 * y for y in range(11) for x in range(17)]], dtype=torch.float32)
        image = net.scan_to_image(scan)
        self.assertEqual(image[0, 0, 3, 8].item(), 38)
        features = net.encode_features(scan)
        self.assertEqual(tuple(features.shape), (1, 32, 11, 17))
        self.assertEqual(tuple(net.position_encoding(features).shape), (1, 187, 32))
        with self.assertRaises(ValueError):
            net.scan_to_image(torch.zeros(2, 186))

    def test_policy_uses_scan_and_ignores_privileged_observations(self):
        obs = observations()
        model = self.model(obs)
        model.update_normalization(obs)
        model.eval()
        with torch.no_grad():
            actions = model.act_inference(obs)
            self.assertEqual(tuple(actions.shape), (4, 23))
            changed = obs.clone()
            changed["critic"] *= 100
            self.assertTrue(torch.equal(actions, model.act_inference(changed)))
            changed["height_scan"] = torch.randn_like(obs["height_scan"]) * 2
            self.assertGreater((actions - model.act_inference(changed)).abs().max().item(), 1e-6)
            self.assertEqual(tuple(model.evaluate(obs).shape), (4, 1))
            self.assertTrue(torch.equal(model.evaluate(obs), model.evaluate(obs.clone().set("height_scan", changed["height_scan"]))))
        model.act(obs)
        self.assertEqual(tuple(model.get_actions_log_prob(actions).shape), (4,))

    def test_critic_gradient_does_not_reach_actor_encoder(self):
        obs = observations()
        model = self.model(obs)
        model.evaluate(obs).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.terrain_attention.parameters()))

    def test_rollout_preserves_height_and_behavior_likelihood(self):
        obs = observations()
        model = self.model(obs)
        model.update_normalization(obs)
        count = model.actor_obs_normalizer.count.clone()
        storage = RolloutStorage("rl", 4, 1, obs, [23])
        alg = PPO(model, storage)
        with torch.inference_mode():
            actions = alg.act(obs)
            likelihood = model.get_actions_log_prob(actions).clone()
            alg.process_env_step(observations(), torch.ones(4), torch.zeros(4), {})
        self.assertTrue(torch.equal(storage.observations["height_scan"][0], obs["height_scan"]))
        model.act(obs)
        self.assertTrue(torch.allclose(likelihood, model.get_actions_log_prob(actions), atol=1e-5))
        self.assertTrue(torch.equal(count, model.actor_obs_normalizer.count))

    def test_ppo_updates_all_actor_modules_and_checkpoint_resumes(self):
        class FakeEnv:
            num_envs, num_actions, device, cfg = 4, 23, "cpu", {}
            def get_observations(self):
                return observations()
            def step(self, actions):
                return observations(), -actions.square().mean(-1), torch.zeros(4), {}
        cfg = AMEAgentCfg().to_dict()
        cfg["num_steps_per_env"] = 4
        cfg["algorithm"].update(num_learning_epochs=2, num_mini_batches=2)
        cfg["policy"].update(actor_hidden_dims=[32], critic_hidden_dims=[32])
        runner = OnPolicyRunner(FakeEnv(), cfg)
        before = {n: p.detach().clone() for n, p in runner.alg.policy.named_parameters()}
        runner.learn(2)
        for prefix in ("terrain_attention.policy_encoder", "terrain_attention.query.",
                       "terrain_attention.position_encoding", "terrain_attention.attention", "actor", "critic"):
            params = [(n, p) for n, p in runner.alg.policy.named_parameters() if n.startswith(prefix)]
            self.assertTrue(any(not torch.equal(before[n], p) for n, p in params), prefix)
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for _, p in params), prefix)
        self.assertEqual(runner.alg.policy.actor_obs_normalizer._mean.shape[-1], 390)
        path = Path(__file__).resolve().parents[1] / "outputs" / ("ame_unit_" + uuid4().hex)
        path.mkdir(parents=True)
        runner.save(str(path / "model.pt"))
        resumed = OnPolicyRunner(FakeEnv(), cfg)
        resumed.load(str(path / "model.pt"), map_location="cpu")
        for name, value in runner.alg.policy.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.alg.policy.state_dict()[name]), name)
        self.assertTrue(resumed.alg.optimizer.state)
        self.assertEqual(resumed.current_learning_iteration, 2)
        resumed.learn(1)
        self.assertEqual(resumed.current_learning_iteration, 3)

    def test_base_and_wrong_geometry_checkpoints_rejected(self):
        obs = observations()
        model = self.model(obs)
        base = ActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 23)
        with self.assertRaisesRegex(RuntimeError, "Base checkpoint"):
            model.load_state_dict(base.state_dict())
        state = model.state_dict()
        state["terrain_attention.position_encoding.coordinates"] = state["terrain_attention.position_encoding.coordinates"] * 2
        with self.assertRaisesRegex(RuntimeError, "geometry differs"):
            model.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
