"""CPU regression tests; run with python -m unittest discover -s isal2/tests."""
import ast
import hashlib
import json
import importlib.util
from pathlib import Path
import sys
from uuid import uuid4
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_isal2_bootstrap", ROOT / "scripts/_bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
bootstrap.bootstrap()

import torch
from tensordict import TensorDict
from isal2.utils.history import HistoryBuffer
from rsl_rl.modules import ActorCritic
from isal2.modified_rsl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from isal2.modified_rsl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO as RslPPO
from rsl_rl.runners import OnPolicyRunner as RslOnPolicyRunner


def load_pure(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


terrain_moves = load_pure("_curriculum", "tasks/base/mdp/curriculum.py").terrain_moves
BaseAgentCfg = load_pure("_agent_cfg", "tasks/base/agents/ppo_cfg.py").BaseAgentCfg


class CoreTests(unittest.TestCase):
    def test_standard_rsl_components_are_external(self):
        self.assertEqual(ActorCritic.__module__, "rsl_rl.modules.actor_critic")
        self.assertEqual(RolloutStorage.__module__, "rsl_rl.storage.rollout_storage")
        self.assertTrue(issubclass(PPO, RslPPO))
        self.assertTrue(issubclass(OnPolicyRunner, RslOnPolicyRunner))
        for name in ("storage", "networks", "utils", "env"):
            self.assertFalse((ROOT / "modified_rsl" / name).exists(), name)

    def test_partial_history_reset(self):
        h = HistoryBuffer(3, 2, 1, "cpu")
        h.append(torch.tensor([[1.], [10.]]))
        h.append(torch.tensor([[2.], [20.]]))
        self.assertEqual(h.buffer[:, :, 0].tolist(), [[1, 1, 2], [10, 10, 20]])
        h.reset(torch.tensor([0]))
        other = h.buffer[1].clone()
        h.append(torch.tensor([[9.], [99.]]), torch.tensor([0]))
        self.assertEqual(h.buffer[0, :, 0].tolist(), [9, 9, 9])
        self.assertTrue(torch.equal(other, h.buffer[1]))

    def test_curriculum_decisions(self):
        distance = torch.tensor([5., 1., 3., 5., 0., 5.])
        commands = torch.tensor([[.4, 0], [.4, 0], [.3, 0], [0., 0], [0., 0], [.4, 0]])
        up, down = terrain_moves(distance, commands, 20., 8., torch.tensor([20, 20, 20, 20, 20, 0]))
        self.assertEqual(up.tolist(), [True, False, False, False, False, False])
        self.assertEqual(down.tolist(), [False, True, False, False, False, False])

    def test_timeout_uses_terminal_state_not_reset_or_previous(self):
        obs = TensorDict({"policy": torch.zeros(3, 2), "critic": torch.ones(3, 1)}, [3])
        policy = ActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 1,
                             actor_hidden_dims=[8], critic_hidden_dims=[8])
        policy.evaluate = lambda data, **kw: data["critic"]
        storage = RolloutStorage("rl", 3, 1, obs, [1])
        alg = PPO(policy, storage, gamma=0.9, normalize_advantage_per_mini_batch=True)
        alg.act(obs)
        terminal = obs.clone()
        terminal["critic"] = torch.tensor([[10.], [20.], [30.]])
        alg.process_env_step(obs, torch.ones(3), torch.ones(3),
            {"time_outs": torch.tensor([True, False, False]), "terminal_observation": terminal})
        self.assertTrue(torch.allclose(storage.rewards[0, :, 0], torch.tensor([10., 1., 1.])))
        alg.compute_returns(obs)
        self.assertTrue(torch.allclose(storage.returns[0, :, 0], torch.tensor([10., 1., 1.])))

    def test_collection_keeps_behavior_normalizer_fixed(self):
        obs = TensorDict({"policy": torch.randn(4, 2), "critic": torch.randn(4, 3)}, [4])
        policy = ActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 1,
            actor_hidden_dims=[8], critic_hidden_dims=[8], actor_obs_normalization=True,
            critic_obs_normalization=True)
        policy.update_normalization(obs)
        count = policy.actor_obs_normalizer.count.clone()
        storage = RolloutStorage("rl", 4, 1, obs, [1])
        alg = PPO(policy, storage)
        actions = alg.act(obs)
        log_prob = policy.get_actions_log_prob(actions).detach().clone()
        new_obs = TensorDict({"policy": torch.randn(4, 2) * 10, "critic": torch.randn(4, 3)}, [4])
        alg.process_env_step(new_obs, torch.ones(4), torch.zeros(4), {})
        self.assertTrue(torch.equal(count, policy.actor_obs_normalizer.count))
        policy.act(obs)
        self.assertTrue(torch.allclose(log_prob, policy.get_actions_log_prob(actions)))

    def test_no_old_imports_and_urdf_assets_exist(self):
        for path in ROOT.rglob("*.py"):
            if any(part in (".cache", "outputs", "build") for part in path.parts):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                self.assertFalse(any(n.split(".")[0] in ("isal", "robolab") for n in names), str(path))
        urdf = ROOT / "assets/data/rpo/urdf/rpo.urdf"
        robot = ET.parse(urdf).getroot()
        joints = [j for j in robot.findall("joint") if j.attrib["type"] != "fixed"]
        self.assertEqual(len(joints), 23)
        for mesh in robot.iter("mesh"):
            self.assertTrue((urdf.parent / mesh.attrib["filename"]).is_file())

    def test_resource_manifest_and_mjcf_references(self):
        data = ROOT / "assets/data/rpo"
        records = json.loads((ROOT / "assets/manifest.json").read_text())
        self.assertEqual(len(records), 31)
        self.assertFalse((data / ".git").exists())
        for record in records:
            self.assertEqual(hashlib.sha256((data / record["file"]).read_bytes()).hexdigest(), record["sha256"])
        for path in (data / "mjcf").glob("*.xml"):
            model = ET.parse(path).getroot()
            meshdir = model.find("compiler").get("meshdir", "")
            for node in model.iter():
                if "file" in node.attrib:
                    directory = path.parent / meshdir if node.tag == "mesh" else path.parent
                    self.assertTrue((directory / node.attrib["file"]).is_file(), str(path))

    def test_runner_resume_iteration_optimizer_and_normalizer(self):
        class FakeEnv:
            num_envs, num_actions, device = 4, 2, "cpu"
            cfg = {}
            def get_observations(self):
                return TensorDict({"policy": torch.randn(4, 6), "critic": torch.randn(4, 8)}, [4])
            def step(self, actions):
                return self.get_observations(), -actions.square().sum(-1), torch.zeros(4), {}
        cfg = BaseAgentCfg().to_dict()
        cfg["num_steps_per_env"] = 4
        cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
        cfg["policy"].update(actor_hidden_dims=[8], critic_hidden_dims=[8])
        runner = OnPolicyRunner(FakeEnv(), cfg)
        self.assertIs(type(runner.alg.policy), ActorCritic)
        self.assertIs(type(runner.alg.storage), RolloutStorage)
        runner.learn(2)
        self.assertEqual(runner.current_learning_iteration, 2)
        (ROOT / "outputs").mkdir(exist_ok=True)
        tmp = ROOT / "outputs" / ("core_" + uuid4().hex)
        tmp.mkdir()
        path = str(Path(tmp) / "model.pt")
        runner.save(path)
        resumed = OnPolicyRunner(FakeEnv(), cfg)
        resumed.load(path, map_location="cpu")
        self.assertEqual(resumed.current_learning_iteration, 2)
        self.assertEqual(runner.alg.learning_rate, resumed.alg.learning_rate)
        self.assertTrue(resumed.alg.optimizer.state)
        for k, value in runner.alg.policy.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.alg.policy.state_dict()[k]), k)
        resumed.learn(1)
        self.assertEqual(resumed.current_learning_iteration, 3)


if __name__ == "__main__":
    unittest.main()
