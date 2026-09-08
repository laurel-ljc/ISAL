from copy import deepcopy
import json
import random
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from isal.learning.auxiliary_buffer import AuxiliaryRolloutBuffer
from isal.learning.training_config import linear_schedule
from isal.learning.training_diagnostics import auxiliary_metrics,capture_probe,probe_changes
from .helpers import make_algorithm,packet,batch_for,fake_runner


def test_buffer_ownership_overflow_rng_and_inference_tensors(device):
    algo,obs = make_algorithm(device)
    data = packet(obs)
    buffer = AuxiliaryRolloutBuffer(10,device,123)
    rng,crng = torch.get_rng_state(),torch.cuda.get_rng_state()
    with torch.inference_mode():
        buffer.append({k:v.clone() for k,v in data.items()},rollout_step=6)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state())
    assert len(buffer)==10 and buffer.received==16 and buffer.cross_rollout==16
    assert buffer.statistics()["sample_age_mean_steps"]==27
    expected = {key:value.clone() for key,value in buffer.data.items()}
    data["height_scan"].fill_(999)
    assert torch.equal(expected["height_scan"],buffer.data["height_scan"])
    assert not buffer.data["height_scan"].is_inference()
    sample = buffer.sample(8)
    algo.policy.predict_affordance(sample).sum().backward()
    assert next(algo.policy.affordance_head.parameters()).grad is not None
    state = buffer.generator.get_state()
    first = buffer.sample(20)
    buffer.generator.set_state(state)
    second = buffer.sample(20)
    assert all(torch.equal(first[k],second[k]) for k in first)
    buffer.clear()
    assert len(buffer)==0 and buffer.received==0
    with pytest.raises(ValueError,match="empty"):
        buffer.sample(1)


def test_buffer_keeps_both_feet_slots_and_uniform_priority(device):
    algo,obs = make_algorithm(device)
    data = packet(obs)
    data["target"] = torch.arange(16,device=device).reshape(4,2,2,1)/16
    buffer = AuxiliaryRolloutBuffer(32,device,17)
    buffer.append(data,30)
    assert buffer.cross_rollout==0
    assert torch.equal(buffer.data["target"].flatten(),torch.arange(16,device=device)/16)
    assert buffer.data["foot_side"].sum(0).tolist()==[8,8]
    # Chunked arrival retains exactly the global top priorities, not the first contacts.
    buf = AuxiliaryRolloutBuffer(5,device,11)
    ref = torch.Generator(device=device).manual_seed(11)
    scores = torch.cat([torch.rand(16,device=device,generator=ref,dtype=torch.float64) for _ in range(2)])
    buf.append(data,1);buf.append(data,2)
    wanted = scores.topk(5,sorted=False).values.sort().values
    assert torch.equal(buf.priorities.sort().values,wanted)


@pytest.mark.parametrize("mode",["aux","predicted","zero"])
def test_loss_and_gradient_routes(device,mode):
    algo,obs = make_algorithm(device,mode)
    algo.completed_updates=400;algo.begin_rollout()
    algo.collect_auxiliary({"auxiliary":packet(obs)})
    batch = batch_for(algo,obs)
    # The zero-input mode has a projection but permanently supplies zero features.
    losses = algo.compute_minibatch_loss(batch,adapt_learning_rate=False)
    assert losses["auxiliary_reason"]=="active"
    torch.testing.assert_close(losses["total"],losses["ppo"]+.05*losses["auxiliary"])
    losses["auxiliary"].backward(retain_graph=True)
    for name,p in algo.policy.named_parameters():
        if not name.startswith(("actor_terrain_encoder.","affordance_head.")):
            assert p.grad is None,name
    algo.policy.zero_grad(set_to_none=True)
    losses["ppo"].backward()
    assert all(p.grad is None for p in algo.policy.affordance_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in algo.policy.actor.parameters())
    if mode=="predicted":
        assert algo.policy.affordance_projection.weight.grad.abs().sum()>0


@pytest.mark.parametrize("clipped",[False,True])
def test_disabled_and_zero_aux_match_ppo_reference(device,clipped):
    algo,obs=make_algorithm(device,enabled=False,use_clipped_value_loss=clipped)
    algo.begin_rollout()
    assert algo.aux_buffer is None
    batch=list(batch_for(algo,obs))
    batch[2]=batch[2]+1  # old value deliberately differs for the clipped path
    batch[5]=batch[5]+torch.tensor([[.5],[-.6],[.1],[.2]],device=device)
    losses=algo.compute_minibatch_loss(tuple(batch),adapt_learning_rate=False)
    p=algo.policy
    ratio=(p.get_actions_log_prob(batch[1])-batch[5].flatten()).exp()
    expected_surrogate=torch.maximum(-batch[3].flatten()*ratio,-batch[3].flatten()*ratio.clamp(.8,1.2)).mean()
    value=p.evaluate(obs)
    expected_value=(value-batch[4]).square()
    if clipped:
        expected_value=torch.maximum(expected_value,(batch[2]+(value-batch[2]).clamp(-.2,.2)-batch[4]).square())
    torch.testing.assert_close(losses["surrogate"],expected_surrogate)
    torch.testing.assert_close(losses["value"],expected_value.mean())
    torch.testing.assert_close(losses["total"],expected_surrogate+algo.value_loss_coef*expected_value.mean()-algo.entropy_coef*p.entropy.mean())
    assert not losses["auxiliary"].requires_grad
    warm,obs=make_algorithm(device)
    warm.begin_rollout();warm.collect_auxiliary({"auxiliary":packet(obs)})
    assert warm.auxiliary_loss()[1]=="zero_coefficient"
    assert not warm.auxiliary_loss()[0].requires_grad
    warm.coefficient=.05;warm.aux_buffer.clear()
    assert warm.auxiliary_loss()[1]=="insufficient_samples"


def test_schedules_boundary_and_clear(device):
    algo,obs=make_algorithm(device)
    for k,coef,gate in ((0,0,0),(100,0,0),(200,.025,0),(300,.05,0),(350,.05,.5),(400,.05,1)):
        algo.completed_updates=k
        algo.begin_rollout()
        assert algo.coefficient==coef and float(algo.policy.input_gate)==gate
        algo.collect_auxiliary({"auxiliary":packet(obs)})
        with pytest.raises(RuntimeError,match="boundary"):
            algo.begin_rollout()
        # Exercise boundary bookkeeping only; no update or optimizer execution.
        algo.phase="update";algo.finish_update()
        assert algo.completed_updates==k+1 and len(algo.aux_buffer)==0
    assert linear_schedule(10,10,0,.3)==.3 and linear_schedule(9,10,0,.3)==0
    algo.begin_rollout();algo.policy.set_affordance_input_gate(.3)
    with pytest.raises(RuntimeError,match="gate"):
        algo.collect_auxiliary({"auxiliary":packet(obs)})


def test_probe_metrics_no_side_effects_and_constant_targets(device):
    algo,obs=make_algorithm(device)
    algo.completed_updates=400;algo.begin_rollout()
    algo.collect_auxiliary({"auxiliary":packet(obs)})
    sample=algo.aux_buffer.data
    algo.policy.act(obs)
    distribution=algo.policy.distribution
    state=deepcopy(algo.policy.state_dict())
    first=capture_probe(algo.policy,obs,sample)
    second=capture_probe(algo.policy,obs,sample)
    assert all(abs(v)<1e-6 for v in probe_changes(first,second).values())
    quality=auxiliary_metrics(algo.policy,sample,.1)
    assert quality["correlation_valid"] is False and quality["correlation"] is None
    assert quality["groups"]["left"]["count"]==8
    assert auxiliary_metrics(algo.policy,None,.1)["prediction_count"]==0
    assert algo.policy.distribution is distribution
    for key,value in state.items():
        if isinstance(value,torch.Tensor):
            assert torch.equal(value,algo.policy.state_dict()[key])
    with torch.no_grad():
        algo.policy.affordance_head[0][-1].bias.add_(4)
    changes=probe_changes(first,capture_probe(algo.policy,obs,sample))
    assert changes["grid_drift"]>0 and changes["action_mean_drift"]>0 and changes["post_update_kl"]>0


def test_checkpoint_full_training_state_and_rejection(device,tmp_path):
    algo,obs=make_algorithm(device)
    algo.completed_updates=350;algo.begin_rollout()
    algo.collect_auxiliary({"auxiliary":packet(obs)})
    runner=fake_runner(algo)
    with pytest.raises(RuntimeError,match="boundary"):
        runner.save(tmp_path/"bad.pt")
    algo.phase="update";algo.finish_update()  # synthetic boundary; no optimization
    runner.current_learning_iteration=algo.completed_updates
    algo.policy.update_normalization(obs)
    expected=algo.policy.act_inference(obs).detach().clone()
    path=tmp_path/"checkpoint.pt";runner.save(path,infos={"untrained":True})
    rng=(torch.rand(3),torch.rand(3,device="cuda"),random.random(),np.random.rand())
    with torch.no_grad():
        next(algo.policy.actor.parameters()).add_(2)
    algo.begin_rollout();algo.collect_auxiliary({"auxiliary":packet(obs)})
    algo.learning_rate=.9;algo.optimizer.param_groups[0]["lr"]=.9
    reset=[];runner.env.reset=lambda:reset.append(1)
    assert runner.load(path,map_location=device)=={"untrained":True}
    assert reset==[1] and len(algo.aux_buffer)==0 and algo.phase=="boundary"
    assert algo.completed_updates==351 and runner.current_learning_iteration==351
    assert algo.environment_steps==4 and runner.logger.tot_timesteps==4
    assert float(algo.policy.input_gate)==.5 and algo.coefficient==.05
    assert torch.equal(expected,algo.policy.act_inference(obs))
    assert torch.equal(rng[0],torch.rand(3)) and torch.equal(rng[1],torch.rand(3,device="cuda"))
    assert rng[2]==random.random() and rng[3]==np.random.rand()
    algo.begin_rollout()
    assert float(algo.policy.input_gate)==pytest.approx(.51)
    saved=torch.load(path,weights_only=True,map_location="cpu")
    assert "aux_buffer" not in saved["stage5"]
    saved["stage5"]["definition"]["auxiliary_learning"]["loss_coef"] = .1
    torch.save(saved,tmp_path/"mismatch.pt")
    with pytest.raises(ValueError,match="Incompatible"):
        runner.load(tmp_path/"mismatch.pt")
    runner.logger.log_dir=str(tmp_path);algo.last_metrics={"correlation":None,"samples":0}
    runner.write_metrics()
    assert json.loads((tmp_path/"stage5_metrics.jsonl").read_text())["environment_steps"]==4


def test_invalid_configuration_and_multigpu(device):
    with pytest.raises(ValueError,match="single-device"):
        make_algorithm(device,multi_gpu_cfg={"global_rank":0,"world_size":2})
    with pytest.raises(ValueError,match="RND"):
        make_algorithm(device,enable_aux_loss=True)
    for args in ((-1,0,1,1),(0,-1,1,1),(0,0,-1,1),(0,0,1,float("nan"))):
        with pytest.raises(ValueError):
            linear_schedule(*args)


def test_process_env_step_storage_to_loss_without_update(device):
    algo,obs=make_algorithm(device)
    algo.completed_updates=400;algo.begin_rollout()
    original={key:value.detach().clone() for key,value in algo.policy.named_parameters()}
    with torch.inference_mode():
        for _ in range(4):
            algo.act(obs)
            algo.process_env_step(obs,torch.ones(4,device=device),torch.zeros(4,dtype=torch.bool,device=device),
                                  {"auxiliary":packet(obs),"time_outs":torch.zeros(4,device=device)})
        algo.compute_returns(obs)
    assert algo.storage.step==4 and algo.environment_steps==16
    assert len(algo.aux_buffer)==64 and algo.aux_buffer.received==64
    batch=next(algo.storage.mini_batch_generator(2,1))
    losses=algo.compute_minibatch_loss(batch,adapt_learning_rate=False)
    assert losses["auxiliary_reason"]=="active" and torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(p.grad is not None for p in algo.policy.affordance_head.parameters())
    assert all(torch.equal(original[key],value) for key,value in algo.policy.named_parameters())
