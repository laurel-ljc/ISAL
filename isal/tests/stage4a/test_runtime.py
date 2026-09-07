"""Configuration parity and bounded scene/runner tests; policy learning is forbidden."""

from collections.abc import Callable
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch


def canonical(value):
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(canonical(item) for item in value)
    if isinstance(value, Callable):
        return value.__name__
    return value


def test_new_task_configuration_parity(isaac_app):
    import gymnasium as gym
    import isal.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isal.learning.config import bind_perceptive_model_config

    base_task, aux_task = "ISAL-Humanoid-Rough-CNN-v0", "ISAL-Humanoid-Rough-CNN-Aux-v0"
    envs = [load_cfg_from_registry(task, "env_cfg_entry_point") for task in (base_task, aux_task)]
    a, b = [cfg.to_dict() for cfg in envs]
    assert not a["self_supervised"]["enabled"] and b["self_supervised"]["enabled"]
    a["self_supervised"]["enabled"] = True
    assert canonical(a) == canonical(b)
    old_env = load_cfg_from_registry("ISAL-Humanoid-Rough-Interaction-v0", "env_cfg_entry_point")
    assert canonical(b) == canonical(old_env.to_dict())
    agents = [load_cfg_from_registry(task, "rsl_rl_cfg_entry_point") for task in (base_task, aux_task)]
    a, b = [cfg.to_dict() for cfg in agents]
    assert not a["policy"]["affordance_head_enabled"] and b["policy"]["affordance_head_enabled"]
    a["policy"]["affordance_head_enabled"] = True
    for cfg in (a, b):
        for key in ("experiment_name", "neptune_project", "wandb_project"):
            cfg.pop(key)
    assert canonical(a) == canonical(b)
    old_agent = load_cfg_from_registry("ISAL-Humanoid-Rough-HeightScan-v0", "rsl_rl_cfg_entry_point")
    assert canonical(agents[0].algorithm.to_dict()) == canonical(old_agent.algorithm.to_dict())
    old_dict = deepcopy(old_agent.to_dict())
    assert bind_perceptive_model_config(None, old_agent) is False
    assert canonical(old_dict) == canonical(old_agent.to_dict())
    # Independent config instances and unchanged older registration defaults.
    agents[0].policy.network["terrain_channels"][0] = 7
    assert agents[1].policy.network["terrain_channels"][0] == 16
    for task in (base_task, aux_task):
        assert gym.spec(task).entry_point.endswith("interaction_env:ISALHumanoidInteractionEnv")


@pytest.mark.parametrize("scenario", ["baseline_default", "aux_small_xy", "aux_large_yx"])
def test_scene_runner_checkpoint_export_without_learning(request, monkeypatch, scenario):
    root = Path(__file__).resolve().parents[3]
    if os.environ.get("ISAL_STAGE4A_SIM_CHILD") != "1":
        node = f"{Path(__file__).as_posix()}::test_scene_runner_checkpoint_export_without_learning[{scenario}]"
        result = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", "env_isaaclab", "python", "-u", "-m", "pytest",
             node, "-q", "--tb=short", "-p", "no:cacheprovider"],
            cwd=root, env={**os.environ, "ISAL_STAGE4A_SIM_CHILD": "1"},
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240,
        )
        output = result.stdout + result.stderr
        directory = root / "outputs/stage4a_acceptance"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{scenario}.log").write_text(output, encoding="utf-8")
        assert result.returncode == 0 and "1 passed" in output and "FAILED" not in output, output[-18000:]
        return

    request.getfixturevalue("isaac_app")
    import gymnasium as gym
    import numpy as np
    import onnxruntime as ort
    import isal.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.algorithms import PPO
    from rsl_rl.runners import OnPolicyRunner
    from isal.learning.config import MODEL_CLASS, bind_perceptive_model_config
    from isal.learning.models import AffordanceActorCritic

    def forbidden(*args, **kwargs):
        raise AssertionError("Stage 4A runtime test must not learn, update PPO or step an optimizer.")

    monkeypatch.setattr(OnPolicyRunner, "learn", forbidden)
    monkeypatch.setattr(PPO, "update", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    aux = scenario.startswith("aux")
    task = "ISAL-Humanoid-Rough-CNN-Aux-v0" if aux else "ISAL-Humanoid-Rough-CNN-v0"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    cfg.seed = 42
    cfg.scene.num_envs = 2 if aux else 1
    if aux:
        cfg.terrain_perception.size = (1.6, .8) if "small" in scenario else (1.8, 1.)
        cfg.robot.actor_obs_history_length = 3
        cfg.robot.critic_obs_history_length = 4
        cfg.scene.height_scanner.pattern_cfg.ordering = "xy" if "small" in scenario else "yx"
    cfg.scene.terrain.terrain_generator.num_rows = 1
    cfg.scene.terrain.terrain_generator.num_cols = 1
    cfg.scene.terrain.max_init_terrain_level = 0
    for name in ("height_scanner", "left_feet_scanner", "right_feet_scanner"):
        getattr(cfg.scene, name).debug_vis = False
    cfg.commands.debug_vis = False
    cfg.log_dir = None
    for key in ("optimizer", "share_cnn_encoders"):
        if hasattr(agent.algorithm, key):
            delattr(agent.algorithm, key)
    directory = root / "outputs/stage4a_acceptance" / scenario
    directory.mkdir(parents=True, exist_ok=True)
    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
    try:
        assert bind_perceptive_model_config(env, agent)
        agent_dict = agent.to_dict()
        assert agent.policy.class_name == MODEL_CLASS
        runner = OnPolicyRunner(env, deepcopy(agent_dict), log_dir=None, device=agent.device)
        model = runner.alg.policy
        assert isinstance(model, AffordanceActorCritic)
        assert type(runner.alg) is PPO and not runner.alg.enable_aux_loss
        assert hasattr(env.unwrapped, "interaction_tracker") == aux
        assert hasattr(model, "affordance_head") == aux
        initial_weights = {name: p.detach().clone() for name, p in model.named_parameters()}
        obs, _ = env.reset()
        obs = obs.to(agent.device)
        model.update_normalization(obs)
        policy = runner.get_inference_policy(device=agent.device)
        with torch.inference_mode():
            expected = policy(obs).clone()
            value = model.evaluate(obs).clone()
            augmented, actions_aug = agent.algorithm.symmetry_cfg.data_augmentation_func(env, obs, expected)
            assert policy(augmented).shape == (2 * env.num_envs, 23)
            assert model.evaluate(augmented).shape == (2 * env.num_envs, 1)
            assert actions_aug.shape == (2 * env.num_envs, 23)
        runner.current_learning_iteration = 7  # Synthetic checkpoint metadata; no iteration was executed.
        checkpoint = directory / "untrained_checkpoint.pt"
        runner.save(str(checkpoint), infos={"stage": "4A", "training_started": False, "synthetic_iteration": 7})
        with torch.no_grad():
            next(model.actor.parameters()).add_(1.)
            model.actor_obs_normalizer._mean.add_(1.)
        original_lr = runner.alg.optimizer.param_groups[0]["lr"]
        runner.alg.optimizer.param_groups[0]["lr"] = .123
        runner.current_learning_iteration = 0
        infos = runner.load(str(checkpoint), map_location=agent.device)
        assert infos["training_started"] is False and runner.current_learning_iteration == 7
        assert runner.alg.optimizer.param_groups[0]["lr"] == original_lr
        with torch.inference_mode():
            assert torch.equal(policy(obs), expected)
            assert torch.equal(model.evaluate(obs), value)
        # Head removal is tested through the independently copied deployment graph.
        model.export_actor(directory / "export")
        exported = torch.jit.load(str(directory / "export/actor.pt")).to(agent.device)
        with torch.inference_mode():
            torch.testing.assert_close(exported(obs["policy"], obs["height_scan"]), expected, rtol=1e-4, atol=1e-5)
        session = ort.InferenceSession(str(directory / "export/actor.onnx"), providers=["CPUExecutionProvider"])
        actual = session.run(None, {key: obs[key].cpu().numpy() for key in ("policy", "height_scan")})[0]
        np.testing.assert_allclose(actual, expected.cpu().numpy(), rtol=1e-4, atol=1e-5)
        sample_count = 0
        with torch.inference_mode():
            for _ in range(40):
                next_obs, rewards, dones, extras = env.step(torch.zeros(env.num_envs, 23, device=env.device))
                assert torch.isfinite(rewards).all()
                assert torch.isfinite(policy(next_obs.to(agent.device))).all()
                if aux:
                    packet = extras["auxiliary"]
                    valid = packet["valid"]
                    assert valid.shape[:2] == (env.num_envs, 2)
                    if valid.any():
                        batch = {key: value[valid] for key, value in packet.items()}
                        pred = model.predict_affordance(batch)
                        assert torch.isfinite(pred).all() and ((0 <= pred) & (pred <= 1)).all()
                        sample_count += int(valid.sum())
                else:
                    assert "auxiliary" not in extras
        for name, p in model.named_parameters():
            assert torch.equal(initial_weights[name], p), name
        counts = {"total": sum(p.numel() for p in model.parameters()),
                  "actor_deployment": sum(p.numel() for p in model.make_actor_export().parameters()),
                  "affordance_head": sum(p.numel() for p in model.affordance_head.parameters()) if aux else 0}
        (directory / "summary.json").write_text(json.dumps({
            "scenario": scenario, "task": task, "num_envs": env.num_envs,
            "control_steps": 40, "environment_steps": env.num_envs * 40,
            "training_started": False, "runner_learn_called": False, "optimizer_steps": 0,
            "sample_count": sample_count, "layout": model.layout, "parameter_counts": counts,
            "checkpoint_roundtrip": True, "export_match": True,
            "ordinary_ppo_aux_loss_enabled": runner.alg.enable_aux_loss,
        }, indent=2), encoding="utf-8")
    finally:
        env.close()
