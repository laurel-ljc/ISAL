from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from isal.learning.config import OBSERVATION_MODEL_CLASS, bind_perceptive_model_config
from isal.learning.models import AffordanceActorCritic, AffordanceObservationActorCritic
from isal.learning.models.dense_query import DenseQueryHead, canonical_query_coordinates
from .helpers import inputs, aux_sample


class AnalyticHead(nn.Module):
    def forward(self, values):
        # Explicit physical-coordinate/side/context tag, not a learned equivariant head.
        return (values[:, 64] + 10*values[:, 65] + 100*values[:, 66] + 200*values[:, 67]
                + 1000*values[:, 68]).unsqueeze(-1)


@pytest.mark.parametrize("grid,ordering,histories", [
    ((17,11), "xy", (10,10)), ((17,11), "yx", (10,10)),
    ((17,9), "xy", (3,4)), ((17,9), "yx", (3,4)),
    ((19,11), "xy", (3,4)), ((19,11), "yx", (3,4)),
])
def test_grid_binding_context_chunking_and_single_encoder(device, grid, ordering, histories):
    obs, common, extra = inputs(grid, ordering, histories, device=device)
    model = AffordanceObservationActorCritic(obs, **common, **extra).to(device)
    canonical = model.query_xy.clone()
    native = canonical.reshape(*grid,2).transpose(0,1).reshape(-1,2) if ordering == "xy" else canonical
    assert torch.equal(canonical_query_coordinates(native, grid), canonical)
    # Older frames deliberately disagree with the last frame. Keep observation noise.
    obs["policy"].fill_(99.)
    current = obs["policy"].reshape(2, histories[0], 78)[:, -1]
    physical = torch.arange(1., 10., device=device).expand(2, -1)
    noise = torch.linspace(-.05, .05, 9, device=device)
    scales = torch.tensor([2.]*3+[3.]*3+[4.]*3, device=device)
    current[:, :9] = physical*scales + noise
    recovered = current[:, :9]/scales
    expected = torch.cat((recovered[:, 6:9], recovered[:, :3], recovered[:, 3:6]), -1)
    torch.testing.assert_close(model.extract_query_context(obs["policy"]), expected)
    calls = []
    handle = model.actor_terrain_encoder.register_forward_hook(lambda *args: calls.append(1))
    model.set_affordance_input_gate(1)
    assert model.act_inference(obs).shape == (2,23)
    assert len(calls) == 1
    calls.clear()
    predicted = model.predict_affordance_grid(obs)
    assert len(calls) == 1 and predicted.shape == (2,2,*grid)
    assert not predicted.requires_grad and torch.isfinite(predicted).all()
    assert ((0 <= predicted) & (predicted <= 1)).all()
    handle.remove()
    with torch.no_grad():
        z = model.actor_terrain_encoder(model.get_actor_obs(obs)[1])
        for chunk in (1, 2*model.num_rays+10):
            reference = model.affordance_head.grid(z, expected, model.query_xy, model.query_foot_side, chunk)
            torch.testing.assert_close(predicted.flatten(1), reference, rtol=1e-5, atol=1e-6)
    # An analytic head independently proves canonical ordering and side order.
    model.affordance_head = DenseQueryHead(AnalyticHead())
    actual = model.predict_affordance_grid(obs).flatten(2)
    xy_score = canonical[:,0] + 10*canonical[:,1]
    for side in range(2):
        target = xy_score + (side+1)*100 + 1000*expected[:,0,None]
        torch.testing.assert_close(actual[:,side], target)


def test_rng_gate_zero_and_zero_control(device):
    obs, common, extra = inputs(device=device)
    torch.manual_seed(101)
    base = AffordanceActorCritic(obs, **common).to(device)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    torch.manual_seed(101)
    pred = AffordanceObservationActorCritic(obs, **common, **extra).to(device)
    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
    torch.manual_seed(101)
    zero = AffordanceObservationActorCritic(obs, **common, **extra,
               affordance_observation={"input_mode":"zero", "input_gate":1.}).to(device)
    for name, value in pred.named_parameters():
        assert torch.equal(value, dict(zero.named_parameters())[name])
    for name, value in base.named_parameters():
        assert torch.equal(value, dict(pred.named_parameters())[name]), name
    expected = base.act_inference(obs)
    assert torch.equal(expected, pred.act_inference(obs))
    assert torch.equal(expected, zero.act_inference(obs))
    calls = []
    # dense grid calls .forward internally; use a scalar layer hook.
    handle = zero.affordance_head[0].register_forward_hook(lambda *args: calls.append(1))
    zero.act_inference(obs)
    assert not calls
    zero.predict_affordance_grid(obs)
    assert calls  # diagnostic remains available in the zero-input variant
    handle.remove()
    pred.set_affordance_input_gate(1)
    before = pred.act_inference(obs).detach().clone()
    with torch.no_grad():
        pred.affordance_head[0][-1].bias.add_(3)
    assert not torch.allclose(before, pred.act_inference(obs), rtol=1e-6, atol=1e-7)
    for gate in (-.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="input_gate"):
            pred.set_affordance_input_gate(gate)


@pytest.mark.parametrize("loss_source", ["aux", "actor", "symmetry", "value"])
def test_gradient_routes(device, loss_source):
    obs, common, extra = inputs(histories=(3,4), device=device)
    model = AffordanceObservationActorCritic(obs, **common, **extra,
                                           affordance_observation={"input_gate":1.}).to(device)
    if loss_source == "aux":
        sample = aux_sample(obs, (17,11))
        loss = torch.nn.functional.smooth_l1_loss(model.predict_affordance(sample), sample["target"], beta=.1)
        allowed = ("actor_terrain_encoder.", "affordance_head.")
    elif loss_source == "value":
        loss = model.evaluate(obs).square().mean()
        allowed = ("critic.", "critic_state_encoder.", "critic_terrain_encoder.")
    elif loss_source == "actor":
        actions = model.act(obs)
        loss = -model.get_actions_log_prob(actions).mean()-.01*model.entropy.mean()
        allowed = ("actor.", "actor_proprio_encoder.", "actor_terrain_encoder.", "affordance_projection.", "std")
    else:
        mirror = obs.clone()
        mirror["height_scan"] = obs["height_scan"].reshape(2,17,11).flip(-1).flatten(1)
        loss = (model.act_inference(mirror)-model.act_inference(obs).detach()+.1).square().mean()
        allowed = ("actor.", "actor_proprio_encoder.", "actor_terrain_encoder.", "affordance_projection.")
    loss.backward()
    for name, p in model.named_parameters():
        if not name.startswith(allowed):
            assert p.grad is None, (loss_source, name)
    for prefix in allowed:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for name,p in model.named_parameters()
                   if name.startswith(prefix)), (loss_source, prefix)


def test_no_future_data_no_normalizer_mutation_and_policy_recomputation(device):
    obs, common, extra = inputs(device=device)
    model = AffordanceObservationActorCritic(obs, **common, **extra,
                                           affordance_observation={"input_gate":1.}).to(device)
    model.update_normalization(obs)
    model.eval()
    actions = model.act(obs)
    distribution = model.distribution
    old_log = model.get_actions_log_prob(actions).detach()
    state = deepcopy(model.state_dict())
    expected = model.act_inference(obs).detach()
    grid = model.predict_affordance_grid(obs)
    poisoned = obs.clone()
    poisoned["critic"].fill_(float("nan"))
    poisoned["future_target"] = torch.full((2,1), float("nan"), device=device)
    poisoned["tracker_pending"] = torch.randn(2,40,device=device)
    assert torch.equal(grid, model.predict_affordance_grid(poisoned))
    assert torch.equal(expected, model.act_inference(poisoned))
    assert model.distribution is distribution
    for key,value in model.state_dict().items():
        if isinstance(value,torch.Tensor):
            assert torch.equal(state[key],value),key
    model.act(obs)
    torch.testing.assert_close(torch.exp(model.get_actions_log_prob(actions)-old_log),torch.ones(2,device=device))
    with torch.no_grad():
        model.affordance_head[0][-1].bias.add_(5)
    model.act(obs)
    assert not torch.allclose(model.get_actions_log_prob(actions),old_log)


def test_checkpoint_gate_geometry_and_version_rejection(device,tmp_path):
    obs, common, extra = inputs(device=device)
    model = AffordanceObservationActorCritic(obs, **common, **extra).to(device)
    model.update_normalization(obs)
    model.set_affordance_input_gate(.73)
    path=tmp_path/"checkpoint.pt"
    torch.save(model.state_dict(),path)
    other=AffordanceObservationActorCritic(obs, **common, **extra).to(device)
    assert other.load_state_dict(torch.load(path,weights_only=True,map_location=device))
    assert torch.equal(model.input_gate,other.input_gate)
    assert torch.equal(model.act_inference(obs),other.act_inference(obs))
    assert torch.equal(model.predict_affordance_grid(obs),other.predict_affordance_grid(obs))
    before=deepcopy(other.state_dict())
    variants=[]
    for name in ("query_xy","context_scales","query_foot_side"):
        bad=deepcopy(before); bad[name].add_(.1); variants.append(bad)
    bad=deepcopy(before); bad["input_gate"].fill_(float("nan")); variants.append(bad)
    bad=deepcopy(before); bad["_extra_state"]["affordance_observation"]["input_mode"]="zero"; variants.append(bad)
    bad=deepcopy(before); bad["_extra_state"]["layout"]["ordering"]="yx"; variants.append(bad)
    variants.append(AffordanceActorCritic(obs,**common).state_dict())
    for bad in variants:
        with pytest.raises(ValueError):
            other.load_state_dict(bad,strict=False)
    for name,value in other.state_dict().items():
        if isinstance(value,torch.Tensor):
            assert torch.equal(before[name],value)


def test_bind_actual_coordinates_and_input_validation(device):
    from dataclasses import make_dataclass
    obs,common,extra=inputs(grid=(17,9),ordering="xy",histories=(3,4),device=device)
    layout=make_dataclass("Layout",[(key,object) for key in common["observation_layout"]])(**common["observation_layout"])
    canonical=torch.tensor(extra["query_coordinates"],device=device)
    raw=SimpleNamespace(perceptive_observation_layout=layout,
        height_scanner=SimpleNamespace(ray_starts=canonical.reshape(17,9,2).transpose(0,1).reshape(1,-1,2)),
        cfg=SimpleNamespace(terrain_perception=SimpleNamespace(**{k:v for k,v in common["scan_preprocessing"].items()
                                                                if k!="definition"}),
                            normalization=SimpleNamespace(obs_scales=SimpleNamespace(**extra["context_scales"]))))
    agent=SimpleNamespace(policy=SimpleNamespace(class_name=OBSERVATION_MODEL_CLASS))
    assert bind_perceptive_model_config(raw,agent)
    assert agent.policy.query_coordinates==extra["query_coordinates"]
    assert agent.policy.context_scales==extra["context_scales"]
    # A nonzero world yaw only changes the world interpretation, not local queries.
    yaw=torch.tensor(.8,device=device)
    rotation=torch.stack((torch.stack((yaw.cos(),-yaw.sin())),torch.stack((yaw.sin(),yaw.cos()))))
    root=torch.tensor([12.,-7.],device=device)
    world=canonical@rotation.T+root
    recovered=(world-root)@rotation
    torch.testing.assert_close(recovered,canonical,rtol=1e-5,atol=1e-6)
    for scales in (dict(ang_vel=0.,projected_gravity=1.,commands=1.),
                   dict(ang_vel=1.,projected_gravity=float("inf"),commands=1.)):
        with pytest.raises(ValueError,match="scales"):
            AffordanceObservationActorCritic(obs,**common,query_coordinates=extra["query_coordinates"],context_scales=scales)
    with pytest.raises(ValueError,match="chunk"):
        AffordanceObservationActorCritic(obs,**common,**extra,affordance_observation={"query_chunk_size":0})
    with pytest.raises(ValueError,match="canonical"):
        AffordanceObservationActorCritic(obs,**common,context_scales=extra["context_scales"],
                                       query_coordinates=extra["query_coordinates"][::-1])
