from copy import deepcopy

import pytest
import torch

from conftest import auxiliary_input, model_inputs, model_state


@pytest.mark.parametrize("grid,ordering,histories", [
    ((17, 11), "xy", (10, 10)), ((17, 11), "yx", (10, 10)),
    ((17, 9), "xy", (3, 4)), ((17, 9), "yx", (3, 4)),
    ((19, 11), "xy", (3, 4)), ((19, 11), "yx", (3, 4)),
])
def test_shapes_and_native_history_semantics(device, grid, ordering, histories):
    cls, obs, kw = model_inputs(grid, ordering, histories, device=device)
    model = cls(obs, affordance_head_enabled=True, **kw).to(device)
    x, y = torch.meshgrid(torch.arange(grid[0], device=device), torch.arange(grid[1], device=device), indexing="ij")
    canonical = x + 10 * y
    native = canonical.T.flatten() if ordering == "xy" else canonical.flatten()
    history = obs["critic"].reshape(3, histories[1], -1)
    for t in range(histories[1]):
        history[:, t, :139] = t
        history[:, t, 139:] = native + 1000*t
    state, scans = model.get_critic_obs(obs)
    for t in range(histories[1]):
        assert torch.equal(state.reshape(3, histories[1], 139)[:, t], torch.full((3, 139), float(t), device=device))
        assert torch.equal(scans.reshape(3, histories[1], *grid)[:, t], (canonical + 1000*t).expand(3, -1, -1))
    assert model.act(obs).shape == (3, 23)
    assert model.evaluate(obs).shape == (3, 1)
    assert model.entropy.shape == (3,)
    assert model.get_actions_log_prob(model.action_mean).shape == (3,)
    assert torch.isfinite(model.action_mean).all()
    assert torch.equal(model.action_std, torch.ones_like(model.action_std))
    pred = model.predict_affordance(auxiliary_input(obs, grid))
    assert pred.shape == (3, 1) and ((0 <= pred) & (pred <= 1)).all()
    encoder = model.actor_terrain_encoder
    assert encoder.flat_dim == encoder.convolutions(torch.zeros(1, 1, *grid, device=device)).flatten(1).shape[1]
    assert not ({p.data_ptr() for p in model.actor_terrain_encoder.parameters()} &
                {p.data_ptr() for p in model.critic_terrain_encoder.parameters()})


def test_head_isolation_initialization_rng_and_distribution(device):
    cls, obs, kw = model_inputs(device=device)
    torch.manual_seed(731)
    base = cls(obs, **kw).to(device)
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    torch.manual_seed(731)
    aux = cls(obs, affordance_head_enabled=True, **kw).to(device)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    for name, p in base.named_parameters():
        assert torch.equal(p, dict(aux.named_parameters())[name]), name
    assert not hasattr(base, "affordance_head")
    with pytest.raises(RuntimeError, match="no affordance head"):
        base.predict_affordance(auxiliary_input(obs, (17, 11)))
    expected = base.act_inference(obs)
    assert torch.equal(expected, aux.act_inference(obs))
    aux.act(obs)
    distribution = aux.distribution
    before = model_state(aux)
    sample = auxiliary_input(obs, (17, 11))
    prediction = aux.predict_affordance(sample)
    sample["target"] = torch.full_like(sample["target"], float("nan"))
    assert torch.equal(prediction, aux.predict_affordance(sample))
    assert aux.distribution is distribution
    for key, value in aux.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(before[key], value), key
    with torch.no_grad():
        for p in aux.affordance_head.parameters():
            p.add_(100)
    assert torch.equal(expected, aux.act_inference(obs))
    del aux.affordance_head
    assert torch.equal(expected, aux.act_inference(obs))


@pytest.mark.parametrize("source", ["auxiliary", "actor", "value", "symmetry"])
def test_direct_gradient_routes(device, source):
    cls, obs, kw = model_inputs(histories=(3, 4), device=device)
    model = cls(obs, affordance_head_enabled=True, **kw).to(device)
    if source == "auxiliary":
        sample = auxiliary_input(obs, (17, 11))
        loss = torch.nn.functional.smooth_l1_loss(model.predict_affordance(sample), sample["target"], beta=.1)
        allowed = ("actor_terrain_encoder.", "affordance_head.")
    elif source == "value":
        loss = model.evaluate(obs).square().mean()
        allowed = ("critic.", "critic_terrain_encoder.", "critic_state_encoder.")
    elif source == "actor":
        actions = model.act(obs)
        loss = -model.get_actions_log_prob(actions).mean() - .005 * model.entropy.mean()
        allowed = ("actor.", "actor_terrain_encoder.", "actor_proprio_encoder.", "std")
    else:
        # Same differentiable deterministic-actor path as the existing mirror loss.
        target = model.act_inference(obs).detach() + .1
        mirrored = obs.clone()
        mirrored["height_scan"] = obs["height_scan"].reshape(3, 17, 11).flip(-1).flatten(1)
        loss = (model.act_inference(mirrored) - target).square().mean()
        allowed = ("actor.", "actor_terrain_encoder.", "actor_proprio_encoder.")
    loss.backward()
    for name, p in model.named_parameters():
        if not name.startswith(allowed):
            assert p.grad is None, (source, name)
    for prefix in allowed:
        assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0
                   for name, p in model.named_parameters() if name.startswith(prefix)), (source, prefix)


def test_normalization_only_tracks_state_and_not_aux(device):
    cls, obs, kw = model_inputs(histories=(3, 4), device=device)
    model = cls(obs, affordance_head_enabled=True, **kw).to(device)
    peer = cls(obs, affordance_head_enabled=True, **kw).to(device)
    peer.load_state_dict(model.state_dict())
    changed = obs.clone()
    changed["height_scan"].fill_(99)
    changed["critic"].reshape(3, 4, -1)[..., 139:] = -99
    model.update_normalization(obs)
    peer.update_normalization(changed)
    for name in ("actor_obs_normalizer", "critic_obs_normalizer"):
        a, b = getattr(model, name), getattr(peer, name)
        for key, value in a.state_dict().items():
            assert torch.equal(value, b.state_dict()[key])
        assert a.count == 3
    assert model.actor_obs_normalizer.mean.shape == (234,)
    assert model.critic_obs_normalizer.mean.shape == (556,)
    state = model_state(model)
    model.predict_affordance(auxiliary_input(obs, (17, 11)))
    model.act_inference(obs)
    model.evaluate(obs)
    model.eval()
    model.update_normalization(obs)
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, state[key])


def test_checkpoint_and_reject_incompatible_metadata(device, tmp_path):
    cls, obs, kw = model_inputs(device=device)
    model = cls(obs, affordance_head_enabled=True, **kw).to(device)
    model.update_normalization(obs)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = cls(obs, affordance_head_enabled=True, **kw).to(device)
    assert restored.load_state_dict(torch.load(path, weights_only=True, map_location=device)) is True
    assert torch.equal(model.act_inference(obs), restored.act_inference(obs))
    assert torch.equal(model.evaluate(obs), restored.evaluate(obs))
    original = model_state(restored)
    for field, value in (("ordering", "yx"), ("grid_shape", [11, 17])):
        bad = model_state(model)
        bad["_extra_state"]["layout"][field] = value
        with pytest.raises(ValueError, match="metadata"):
            restored.load_state_dict(bad, strict=False)
    bad = model_state(model)
    bad["_extra_state"]["scan_preprocessing"]["min_height"] = -.8
    with pytest.raises(ValueError, match="metadata"):
        restored.load_state_dict(bad)
    with pytest.raises(ValueError, match="old MLP"):
        restored.load_state_dict({"actor.0.weight": torch.zeros(3, 3)})
    for key, value in restored.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, original[key])


def test_layout_and_auxiliary_errors(device):
    cls, obs, kw = model_inputs(device=device)
    missing = deepcopy(kw)
    missing.pop("observation_layout")
    with pytest.raises(ValueError, match="Bind"):
        cls(obs, **missing)
    model = cls(obs, affordance_head_enabled=True, **kw).to(device)
    bad = obs.clone()
    bad["height_scan"] = torch.zeros(3, 153, device=device)
    with pytest.raises(ValueError, match="height_scan"):
        model.act_inference(bad)
    bad["critic"] = torch.zeros(3, 100, device=device)
    with pytest.raises(ValueError, match="critic"):
        model.evaluate(bad)
    sample = auxiliary_input(obs, (17, 11))
    sample["query_xy"] = torch.zeros(3, 3, device=device)
    with pytest.raises(ValueError, match="query_xy"):
        model.predict_affordance(sample)


def test_import_does_not_start_simulator():
    # Run in a clean process because the complete suite also contains real scenes.
    import subprocess
    result = subprocess.run(["conda", "run", "-n", "env_isaaclab", "python", "-c",
                             "from isal.learning.models import AffordanceActorCritic; import sys; "
                             "assert 'isaaclab.app' not in sys.modules; "
                             "assert 'isal.tasks' not in sys.modules; print('PURE_IMPORT_OK')"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and "PURE_IMPORT_OK" in result.stdout, result.stdout + result.stderr


def test_finalized_tracker_packet_flattens_into_head(device):
    """Synthetic simultaneous contacts exercise the actual Stage 3 packet schema."""
    from isal.interaction import FootInteractionTracker, SelfSupervisedCfg

    cls, obs, kw = model_inputs(batch=2, device=device)
    model = cls(obs, affordance_head_enabled=True, **kw).to(device)
    x, y = torch.meshgrid(torch.linspace(-.4, 1.2, 17, device=device),
                          torch.linspace(-.5, .5, 11, device=device), indexing="ij")
    origins = torch.stack((x.flatten(), y.flatten()), -1)
    tracker = FootInteractionTracker(2, (17, 11), origins, .02, SelfSupervisedCfg(), device)
    scan = obs["height_scan"].reshape(2, 1, 17, 11)
    foot = torch.tensor([[[.2, .2, 0.], [.2, -.2, 0.]]], device=device).expand(2, -1, -1).clone()
    force = torch.zeros_like(foot)
    force[..., 2] = 30
    inputs = dict(
        height_scan=scan, root_xy=torch.zeros(2, 2, device=device), root_yaw=torch.zeros(2, device=device),
        command=torch.zeros(2, 3, device=device), base_ang_vel=torch.zeros(2, 3, device=device),
        projected_gravity=torch.tensor([[0., 0., -1.]], device=device).expand(2, -1),
        foot_pos_w=foot, foot_vel_w=torch.zeros_like(foot), foot_force_w=force,
        base_roll_pitch=torch.zeros(2, 2, device=device),
        terminated=torch.zeros(2, dtype=torch.bool, device=device),
        truncated=torch.zeros(2, dtype=torch.bool, device=device),
    )
    tracker.update(**inputs)  # initial stance
    force.zero_()
    tracker.update(**inputs)  # liftoff
    force[..., 2] = 30
    tracker.update(**inputs)  # touchdown; survival age zero
    for _ in range(tracker.survival_steps):
        tracker.update(**inputs)
    packet = tracker.output
    assert packet["valid"].sum() == 4
    batch = {key: value[packet["valid"]] for key, value in packet.items()}
    assert batch["height_scan"].shape == (4, 1, 17, 11)
    pred = model.predict_affordance(batch)
    assert pred.shape == (4, 1) and torch.isfinite(pred).all()
    assert torch.equal(batch["foot_side"].sum(0), torch.tensor([2., 2.], device=device))
    torch.nn.functional.smooth_l1_loss(pred, batch["target"], beta=.1).backward()
    assert any(p.grad is not None for p in model.affordance_head.parameters())
