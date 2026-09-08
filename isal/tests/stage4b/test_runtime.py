"""Real symmetry integration and bounded fixed-action scenes, never learning."""

from collections.abc import Callable
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from .helpers import inputs


def canonical(value):
    if isinstance(value,dict):
        return {k:canonical(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return tuple(canonical(v) for v in value)
    if isinstance(value,Callable):
        return value.__name__
    return value


def test_configuration_parity(isaac_app):
    import gymnasium as gym
    import isal.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    tasks=[f"ISAL-Humanoid-Rough-{suffix}-v0" for suffix in ("AffordanceObs","AffordanceZero","CNN-Aux")]
    envs=[load_cfg_from_registry(t,"env_cfg_entry_point") for t in tasks]
    assert canonical(envs[0].to_dict())==canonical(envs[1].to_dict())==canonical(envs[2].to_dict())
    agents=[load_cfg_from_registry(t,"rsl_rl_cfg_entry_point") for t in tasks]
    a,b=[c.to_dict() for c in agents[:2]]
    assert a["policy"]["affordance_observation"]["input_mode"]=="predicted"
    assert b["policy"]["affordance_observation"]["input_mode"]=="zero"
    b["policy"]["affordance_observation"]["input_mode"]="predicted"
    for c in (a,b):
        for k in ("experiment_name","neptune_project","wandb_project"):
            c.pop(k)
    assert canonical(a)==canonical(b)
    for agent in agents[:2]:
        assert agent.policy.affordance_head_enabled
        assert canonical(agent.algorithm.to_dict())==canonical(agents[2].algorithm.to_dict())
    agents[0].policy.affordance_observation["input_gate"]=1.
    assert agents[1].policy.affordance_observation["input_gate"]==0.
    for task in tasks[:2]:
        assert gym.spec(task).entry_point.endswith("interaction_env:ISALHumanoidInteractionEnv")


class MirrorAnalyticHead(nn.Module):
    def forward(self,v):
        side=v[:,66]-v[:,67]
        score=v[:,64]+side*v[:,65]+v[:,69]*v[:,65]+side*v[:,70]
        return torch.sigmoid(score).unsqueeze(-1)


@pytest.mark.parametrize("ordering",["xy","yx"])
def test_actual_symmetry_with_analytic_head(isaac_app,ordering):
    from isal.learning.models import AffordanceObservationActorCritic
    from isal.learning.models.dense_query import DenseQueryHead
    from isal.tasks.direct.humanoid_rough.height_scan import PerceptiveObservationLayout
    from isal.tasks.direct.humanoid_rough.agents.isal_agent_cfg import data_augmentation_func
    obs,common,extra=inputs(grid=(19,11),ordering=ordering,histories=(3,4),device="cuda")
    model=AffordanceObservationActorCritic(obs,**common,**extra).cuda()
    raw=SimpleNamespace(perceptive_observation_layout=PerceptiveObservationLayout(**common["observation_layout"]))
    augmented,_=data_augmentation_func(raw,obs,None)
    mirrored=augmented[2:]
    signs=torch.tensor([1.,-1.,-1., -1.,1.,-1., 1.,-1.,1.],device="cuda")
    torch.testing.assert_close(model.extract_query_context(mirrored["policy"]),
                               model.extract_query_context(obs["policy"])*signs)
    torch.testing.assert_close(mirrored["height_scan"].reshape(2,19,11),
                               obs["height_scan"].reshape(2,19,11).flip(-1))
    model.affordance_head=DenseQueryHead(MirrorAnalyticHead()).cuda()
    original=model.predict_affordance_grid(obs)
    result=model.predict_affordance_grid(mirrored)
    # This equality is a known property of the analytic head only.
    torch.testing.assert_close(result,original.flip(1).flip(-1),rtol=1e-5,atol=1e-6)
    model.set_affordance_input_gate(1.)
    assert model.act_inference(augmented).shape==(4,23)


@pytest.mark.parametrize("scenario",["predicted_default","zero_default","predicted_dynamic_yx"])
def test_scene_runner_export_and_no_learning(request,monkeypatch,scenario):
    root=Path(__file__).resolve().parents[3]
    if os.environ.get("ISAL_STAGE4B_SIM_CHILD")!="1":
        node=f"{Path(__file__).as_posix()}::test_scene_runner_export_and_no_learning[{scenario}]"
        result=subprocess.run(
            ["conda","run","--no-capture-output","-n","env_isaaclab","python","-u","-m","pytest",
             node,"-q","--tb=short","-p","no:cacheprovider"],
            cwd=root,env={**os.environ,"ISAL_STAGE4B_SIM_CHILD":"1"},capture_output=True,
            text=True,encoding="utf-8",errors="replace",timeout=240)
        output=result.stdout+result.stderr
        directory=root/"outputs/stage4b_acceptance";directory.mkdir(parents=True,exist_ok=True)
        (directory/f"{scenario}.log").write_text(output,encoding="utf-8")
        assert result.returncode==0 and "1 passed" in output and "FAILED" not in output,output[-18000:]
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
    from isal.learning.config import bind_perceptive_model_config
    from isal.learning.models import AffordanceObservationActorCritic

    def forbidden(*args,**kwargs):
        raise AssertionError("Stage 4B prohibits learn, PPO update and optimizer step.")
    monkeypatch.setattr(OnPolicyRunner,"learn",forbidden)
    monkeypatch.setattr(PPO,"update",forbidden)
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD):
        monkeypatch.setattr(cls,"step",forbidden)
    zero=scenario.startswith("zero")
    dynamic="dynamic" in scenario
    task=f"ISAL-Humanoid-Rough-{'AffordanceZero' if zero else 'AffordanceObs'}-v0"
    cfg=load_cfg_from_registry(task,"env_cfg_entry_point")
    agent=load_cfg_from_registry(task,"rsl_rl_cfg_entry_point")
    cfg.seed=42;cfg.scene.num_envs=2 if dynamic else 1
    if dynamic:
        cfg.terrain_perception.size=(1.8,1.)
        cfg.robot.actor_obs_history_length=3;cfg.robot.critic_obs_history_length=4
        cfg.scene.height_scanner.pattern_cfg.ordering="yx"
        cfg.normalization.obs_scales.ang_vel=2.
        cfg.normalization.obs_scales.projected_gravity=3.
        cfg.normalization.obs_scales.commands=4.
    cfg.scene.terrain.terrain_generator.num_rows=1;cfg.scene.terrain.terrain_generator.num_cols=1
    cfg.scene.terrain.max_init_terrain_level=0
    for name in ("height_scanner","left_feet_scanner","right_feet_scanner"):
        getattr(cfg.scene,name).debug_vis=False
    cfg.commands.debug_vis=False;cfg.log_dir=None
    for key in ("optimizer","share_cnn_encoders"):
        if hasattr(agent.algorithm,key):
            delattr(agent.algorithm,key)
    directory=root/"outputs/stage4b_acceptance"/scenario;directory.mkdir(parents=True,exist_ok=True)
    env=RslRlVecEnvWrapper(gym.make(task,cfg=cfg))
    try:
        assert bind_perceptive_model_config(env,agent)
        runner=OnPolicyRunner(env,deepcopy(agent.to_dict()),log_dir=None,device=agent.device)
        model=runner.alg.policy
        assert isinstance(model,AffordanceObservationActorCritic)
        assert type(runner.alg) is PPO and not runner.alg.enable_aux_loss
        assert hasattr(env.unwrapped,"interaction_tracker")
        initial={name:p.detach().clone() for name,p in model.named_parameters()}
        obs,_=env.reset();obs=obs.to(agent.device)
        model.update_normalization(obs);model.set_affordance_input_gate(1.)
        policy=runner.get_inference_policy(device=agent.device)
        with torch.inference_mode():
            expected=policy(obs).clone();value=model.evaluate(obs).clone()
            grid=model.predict_affordance_grid(obs)
            augmented,_=agent.algorithm.symmetry_cfg.data_augmentation_func(env,obs,expected)
            assert policy(augmented).shape==(2*env.num_envs,23)
            learned_mirror_error=float((model.predict_affordance_grid(augmented[env.num_envs:])-
                                        grid.flip(1).flip(-1)).abs().mean())
        runner.current_learning_iteration=7  # synthetic checkpoint field, no training occurred
        path=directory/"untrained_checkpoint.pt"
        runner.save(str(path),infos={"training_started":False,"synthetic_iteration":7})
        with torch.no_grad():
            next(model.actor.parameters()).add_(1.)
            model.actor_obs_normalizer._mean.add_(1.)
        model.set_affordance_input_gate(0.)
        old_lr=runner.alg.optimizer.param_groups[0]["lr"]
        runner.alg.optimizer.param_groups[0]["lr"]=.123;runner.current_learning_iteration=0
        runner.load(str(path),map_location=agent.device)
        assert model.input_gate==1. and runner.current_learning_iteration==7
        assert runner.alg.optimizer.param_groups[0]["lr"]==old_lr
        with torch.inference_mode():
            assert torch.equal(expected,policy(obs)) and torch.equal(value,model.evaluate(obs))
        model.export_actor(directory/"export")
        scripted=torch.jit.load(str(directory/"export/actor.pt")).to(agent.device)
        with torch.inference_mode():
            torch.testing.assert_close(scripted(obs["policy"],obs["height_scan"]),expected,rtol=1e-4,atol=1e-5)
        session=ort.InferenceSession(str(directory/"export/actor.onnx"),providers=["CPUExecutionProvider"])
        result=session.run(None,{key:obs[key].cpu().numpy() for key in ("policy","height_scan")})[0]
        np.testing.assert_allclose(result,expected.cpu().numpy(),rtol=1e-4,atol=1e-5)
        samples=0
        with torch.inference_mode():
            for _ in range(40):
                current,rewards,dones,extras=env.step(torch.zeros(env.num_envs,23,device=env.device))
                assert torch.isfinite(rewards).all() and torch.isfinite(policy(current.to(agent.device))).all()
                assert "auxiliary" in extras and "interaction_stats" in extras
                samples+=int(extras["auxiliary"]["valid"].sum())
        assert all(torch.equal(initial[name],p) for name,p in model.named_parameters())
        counts=dict(total=sum(p.numel() for p in model.parameters()),
                    deployment=sum(p.numel() for p in model.make_actor_export().parameters()),
                    projection=sum(p.numel() for p in model.affordance_projection.parameters()),
                    head=sum(p.numel() for p in model.affordance_head.parameters()))
        (directory/"summary.json").write_text(json.dumps(dict(
            task=task,num_envs=env.num_envs,control_steps=40,training_started=False,
            runner_learn_called=False,optimizer_steps=0,sample_count=samples,
            input_gate=model.input_gate.item(),layout=model.layout,parameter_counts=counts,
            learned_mirror_mae=learned_mirror_error,checkpoint_roundtrip=True,export_match=True,
            context_scales=agent.policy.context_scales,ordinary_ppo_aux_loss_enabled=runner.alg.enable_aux_loss,
        ),indent=2),encoding="utf-8")
    finally:
        env.close()
