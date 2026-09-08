from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch


def test_configuration_fairness(isaac_app):
    import isal.tasks  # noqa
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isal.learning.affordance_runner import canonical_config
    variants=("CNN","CNN-Aux","AffordanceObs","AffordanceZero")
    pairs=[]
    for variant in variants:
        task=f"ISAL-Humanoid-Rough-{variant}-Train-v0"
        env=load_cfg_from_registry(task,"env_cfg_entry_point")
        cfg=load_cfg_from_registry(task,"rsl_rl_cfg_entry_point")
        old=load_cfg_from_registry(f"ISAL-Humanoid-Rough-{variant}-v0","rsl_rl_cfg_entry_point")
        a,b=cfg.algorithm.to_dict(),old.algorithm.to_dict()
        for key in ("auxiliary_learning","affordance_gate_schedule","diagnostics"):
            a.pop(key)
        a.pop("class_name");b.pop("class_name")
        assert canonical_config(a)==canonical_config(b)
        assert canonical_config(cfg.policy)==canonical_config(old.policy)
        assert env.self_supervised.enabled==(variant!="CNN")
        assert env.self_supervised.emit_sample_timing==(variant!="CNN")
        env_dict=env.to_dict();env_dict.pop("self_supervised")
        pairs.append(canonical_config(env_dict))
    assert all(pair==pairs[0] for pair in pairs)


@pytest.mark.parametrize("variant",["CNN","CNN-Aux","AffordanceObs","AffordanceZero"])
def test_real_runner_packet_checkpoint_export(request,variant):
    root=Path(__file__).resolve().parents[3]
    if os.environ.get("ISAL_STAGE5_CHILD")!="1":
        node=f"{Path(__file__).as_posix()}::test_real_runner_packet_checkpoint_export[{variant}]"
        result=subprocess.run(["conda","run","--no-capture-output","-n","env_isaaclab","python","-u","-m","pytest",
                               node,"-q","--tb=short","-p","no:cacheprovider"],cwd=root,
                              env={**os.environ,"ISAL_STAGE5_CHILD":"1"},capture_output=True,
                              text=True,encoding="utf-8",errors="replace",timeout=240)
        out=result.stdout+result.stderr
        directory=root/"outputs/stage5_acceptance";directory.mkdir(parents=True,exist_ok=True)
        (directory/f"{variant}.log").write_text(out,encoding="utf-8")
        assert result.returncode==0 and "1 passed" in out and "FAILED" not in out,out[-14000:]
        return
    request.getfixturevalue("isaac_app")
    import gymnasium as gym
    import numpy as np
    import onnxruntime as ort
    import isal.tasks  # noqa
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isal.learning.config import bind_perceptive_model_config
    from isal.learning.affordance_runner import AffordanceRunner
    from isal.learning.ppo_affordance import PPOWithAffordance

    task=f"ISAL-Humanoid-Rough-{variant}-Train-v0"
    cfg=load_cfg_from_registry(task,"env_cfg_entry_point")
    agent=load_cfg_from_registry(task,"rsl_rl_cfg_entry_point")
    cfg.seed=42;cfg.scene.num_envs=2 if variant=="AffordanceObs" else 1
    if variant=="AffordanceObs":
        cfg.terrain_perception.size=(1.8,1.)
        cfg.scene.height_scanner.pattern_cfg.ordering="yx"
        cfg.robot.actor_obs_history_length=3;cfg.robot.critic_obs_history_length=4
        cfg.normalization.obs_scales.ang_vel=2.
    cfg.scene.terrain.terrain_generator.num_rows=cfg.scene.terrain.terrain_generator.num_cols=1
    cfg.scene.terrain.max_init_terrain_level=0;cfg.log_dir=None;cfg.commands.debug_vis=False
    for name in ("height_scanner","left_feet_scanner","right_feet_scanner"):
        getattr(cfg.scene,name).debug_vis=False
    for key in ("optimizer","share_cnn_encoders"):
        if hasattr(agent.algorithm,key):
            delattr(agent.algorithm,key)
    directory=root/"outputs/stage5_acceptance"/variant;directory.mkdir(parents=True,exist_ok=True)
    env=RslRlVecEnvWrapper(gym.make(task,cfg=cfg))
    try:
        bind_perceptive_model_config(env,agent)
        runner=AffordanceRunner(env,agent.to_dict(),None,agent.device)
        algo=runner.alg;model=algo.policy
        assert type(algo) is PPOWithAffordance
        assert (algo.aux_buffer is None)==(variant=="CNN")
        initial={key:value.detach().clone() for key,value in model.named_parameters()}
        obs,_=env.reset();obs=obs.to(agent.device)
        model.update_normalization(obs)
        if hasattr(model,"set_affordance_input_gate"):
            model.set_affordance_input_gate(.5 if variant=="AffordanceObs" else 0)
        algo.completed_updates=runner.current_learning_iteration=351  # synthetic resume field; no optimization
        algo.environment_steps=123;algo.coefficient=.05 if variant!="CNN" else 0
        path=directory/"untrained_checkpoint.pt"
        runner.save(str(path),infos={"training_started":False,"synthetic_completed_updates":351})
        with torch.inference_mode():
            expected=model.act_inference(obs).clone()
        # Real load clears physical pending state as well as any current buffer.
        if hasattr(env.unwrapped,"interaction_tracker"):
            env.unwrapped.interaction_tracker.active.fill_(True)
        with torch.no_grad():
            next(model.actor.parameters()).add_(1)
        runner.load(path,map_location=agent.device)
        assert all(torch.equal(initial[name],value) for name,value in model.named_parameters())
        if hasattr(env.unwrapped,"interaction_tracker"):
            assert not env.unwrapped.interaction_tracker.active.any()
        with torch.inference_mode():
            assert torch.equal(model.act_inference(obs),expected)
        model.export_actor(directory/"export")
        scripted=torch.jit.load(str(directory/"export/actor.pt"))
        session=ort.InferenceSession(str(directory/"export/actor.onnx"),providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            torch.testing.assert_close(scripted(obs["policy"].cpu(),obs["height_scan"].cpu()),expected.cpu(),rtol=1e-4,atol=1e-5)
        np.testing.assert_allclose(session.run(None,{key:obs[key].cpu().numpy() for key in ("policy","height_scan")})[0],
                                   expected.cpu().numpy(),rtol=1e-4,atol=1e-5)
        algo.begin_rollout()
        # Inference only: keep gate constant for all 40 zero-action checks; no PPO update.
        natural=0
        for _ in range(40):
            with torch.inference_mode():
                current,rewards,dones,extras=env.step(torch.zeros(env.num_envs,23,device=env.device))
                current=current.to(agent.device)
                assert torch.isfinite(model.act_inference(current)).all() and torch.isfinite(rewards).all()
                if variant=="CNN":
                    assert "auxiliary" not in extras
                else:
                    natural+=int(extras["auxiliary"]["valid"].sum())
                algo.collect_auxiliary(extras)
        augmented,_=agent.algorithm.symmetry_cfg.data_augmentation_func(env,obs,None)
        with torch.inference_mode():
            assert model.act_inference(augmented).shape==(2*env.num_envs,23)
        synthetic=0
        if algo.aux_buffer is not None:
            data={key:value.clone() for key,value in extras["auxiliary"].items()}
            data["valid"].zero_();data["valid"][0,0,0]=True
            data["height_scan"][0,0,0]=obs["height_scan"][0].reshape(1,*model.grid_shape)
            data["foot_side"][0,0,0]=torch.tensor([1.,0.],device=env.device)
            data["target"][0,0,0]=.2
            # Synthetic packet supplements the last physical step, so do not count a fake env step.
            algo.aux_buffer.append(data,40);synthetic=1
            batch=algo.aux_buffer.sample(2)
            loss=torch.nn.functional.smooth_l1_loss(model.predict_affordance(batch),batch["target"],beta=.1)
            loss.backward()  # only gradient check; fixture prohibits optimizer.step
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            actions=model.act(obs).clone();values=model.evaluate(obs).clone()
            old_log=model.get_actions_log_prob(actions).unsqueeze(-1).clone()
            old_mu=model.action_mean.clone();old_std=model.action_std.clone()
        losses=algo.compute_minibatch_loss((obs,actions,values,torch.ones_like(values),values+.1,
                                            old_log,old_mu,old_std,(None,None),None),adapt_learning_rate=False)
        losses["ppo"].backward()
        if hasattr(model,"affordance_head"):
            assert all(p.grad is None for p in model.affordance_head.parameters())
        assert all(torch.equal(initial[name],value) for name,value in model.named_parameters())
        (directory/"summary.json").write_text(json.dumps(dict(task=task,num_envs=env.num_envs,zero_action_steps=40,
            training_started=False,optimizer_steps=0,completed_updates_are_synthetic=True,
            environment_steps=algo.environment_steps,natural_samples=natural,synthetic_samples=synthetic,
            coefficient=algo.coefficient,gate=float(getattr(model,"input_gate",0.)),
            buffer_samples=len(algo.aux_buffer) if algo.aux_buffer is not None else 0,
            checkpoint_restored=True,export_aligned=True),indent=2),encoding="utf-8")
    finally:
        env.close()
