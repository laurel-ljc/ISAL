"""Synthetic interaction tests. No simulator, optimizer, policy or training."""

from __future__ import annotations

import math

import pytest
import torch

from isal.interaction import FootInteractionTracker, SelfSupervisedCfg, touchdown_query_xy


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; the acceptance record must report this skip.")
    return request.param


def make_tracker(device, **kwargs):
    cfg = SelfSupervisedCfg(outcome_window_s=0.06, survival_window_s=0.10, **kwargs)
    x, y = torch.meshgrid(torch.tensor([-0.4, 0.4, 1.2]), torch.tensor([-0.5, 0.0, 0.5]), indexing="ij")
    origins = torch.stack((x.flatten(), y.flatten()), -1).to(device)
    return FootInteractionTracker(2, (3, 3), origins, 0.02, cfg, device)


def frame(device, contact=True):
    data = {
        "height_scan": torch.full((2, 1, 3, 3), -0.8, device=device),
        "root_xy": torch.zeros((2, 2), device=device), "root_yaw": torch.zeros(2, device=device),
        "command": torch.zeros((2, 3), device=device), "base_ang_vel": torch.zeros((2, 3), device=device),
        "projected_gravity": torch.tensor([[0., 0., -1.], [0., 0., -1.]], device=device),
        "foot_pos_w": torch.zeros((2, 2, 3), device=device),
        "foot_vel_w": torch.zeros((2, 2, 3), device=device),
        "foot_force_w": torch.zeros((2, 2, 3), device=device),
        "base_roll_pitch": torch.zeros((2, 2), device=device),
        "terminated": torch.zeros(2, dtype=torch.bool, device=device),
        "truncated": torch.zeros(2, dtype=torch.bool, device=device),
    }
    data["foot_force_w"][..., 2] = 100.0 if contact else 0.0
    return data


def touchdown(tracker, device):
    tracker.update(**frame(device))
    tracker.update(**frame(device, False))
    return tracker.update(**frame(device))


def test_static_and_initial_air_do_not_create_samples(device):
    tracker = make_tracker(device)
    for _ in range(8):
        assert not tracker.update(**frame(device))["valid"].any()
    assert tracker.counters["touchdown"] == 0
    tracker.reset(torch.arange(2, device=device))
    tracker.update(**frame(device, False))
    tracker.update(**frame(device))
    assert not tracker.active.any()
    assert tracker.counters["touchdown_without_snapshot"] == 4


def test_full_cycle_windows_both_feet_and_packet_ownership(device):
    tracker = make_tracker(device)
    touchdown(tracker, device)
    for _ in range(tracker.survival_steps - 1):
        assert not tracker.update(**frame(device))["valid"].any()
    packet = tracker.update(**frame(device))
    valid = packet["valid"]
    assert valid.shape == (2, 2, tracker.slots)
    assert valid.sum() == 4
    torch.testing.assert_close(packet["target"][valid], torch.ones((4, 1), device=device))
    assert (packet["observed_frames"][valid] == 3).all()
    assert not packet["partial_window"].any()
    assert packet["partial_window"].dtype == torch.bool
    assert packet["valid"].device.type == device
    torch.testing.assert_close(packet["foot_side"][:, 0, 0], torch.tensor([[1., 0.], [1., 0.]], device=device))
    torch.testing.assert_close(packet["foot_side"][:, 1, 0], torch.tensor([[0., 1.], [0., 1.]], device=device))
    for value in packet.values():
        assert torch.isfinite(value).all()
        assert not value[~valid].any()
        assert not value.requires_grad
    tracker.update(**frame(device))
    tracker.reset(torch.arange(2, device=device))
    assert packet["valid"].sum() == 4  # no returned-packet aliasing
    assert not tracker.output["valid"].any()
    assert tracker.counters["samples"] == 4


def test_liftoff_snapshot_is_copied_and_uses_liftoff_pose(device):
    tracker = make_tracker(device)
    tracker.update(**frame(device))
    lift = frame(device, False)
    lift["root_xy"][:] = torch.tensor([10., 20.], device=device)
    lift["root_yaw"][:] = math.pi / 2
    lift["height_scan"].requires_grad_(True)
    lift["command"][:] = 0.3
    tracker.update(**lift)
    with torch.no_grad():
        lift["height_scan"][:] = 7
        lift["root_xy"][:] = -8
    touch = frame(device)
    touch["foot_pos_w"][..., :2] = torch.tensor([10.2, 20.6], device=device)
    touch["terminated"][:] = True
    packet = tracker.update(**touch)
    valid = packet["valid"]
    assert valid.sum() == 4
    torch.testing.assert_close(packet["query_xy"][valid], torch.tensor([[0.6, -0.2]], device=device).expand(4, -1))
    torch.testing.assert_close(packet["height_scan"][valid], torch.full((4, 1, 3, 3), -0.8, device=device))
    assert (packet["command"][valid] == 0.3).all()


def test_query_rotation_and_dynamic_scan_boundaries(device):
    query = touchdown_query_xy(torch.tensor([[4., 7.]], device=device),
                               torch.tensor([[3., 5.]], device=device), torch.tensor([math.pi / 2], device=device))
    torch.testing.assert_close(query, torch.tensor([[2., -1.]], device=device))
    tracker = make_tracker(device)
    tracker.update(**frame(device))
    tracker.update(**frame(device, False))
    touch = frame(device)
    touch["foot_pos_w"][0, :, :2] = torch.tensor([[-0.4, -0.5], [1.2, 0.5]], device=device)
    touch["foot_pos_w"][1, :, :2] = torch.tensor([[-0.41, 0.], [1.21, 0.]], device=device)
    touch["terminated"][:] = True
    packet = tracker.update(**touch)
    assert packet["valid"].sum() == 2
    assert tracker.counters["out_of_bounds"] == 2


def test_overlapping_events_and_simultaneous_fall_flush(device):
    tracker = make_tracker(device)
    touchdown(tracker, device)
    tracker.update(**frame(device, False))
    tracker.update(**frame(device))
    assert tracker.active.sum() == 8
    assert tracker.peak_pending == 2
    fail = frame(device)
    fail["terminated"][:] = True
    packet = tracker.update(**fail)
    assert packet["valid"].sum() == 8
    assert not packet["survival"].any()
    assert packet["partial_window"].sum() == 4
    assert not tracker.active.any()
    tracker.reset(torch.arange(2, device=device), clear_output=False)
    assert tracker.output["valid"].sum() == 8
    assert not tracker.update(**frame(device))["valid"].any()


def test_fastest_contact_alternation_does_not_overflow_default_slots(device):
    tracker = make_tracker(device)
    tracker.update(**frame(device))
    for step in range(40):
        tracker.update(**frame(device, contact=bool(step % 2)))
    assert tracker.counters["samples"] > 0
    assert tracker.counters["overflow"] == 0


def test_explicit_full_slots_do_not_overwrite_prior_sample(device):
    tracker = make_tracker(device, pending_slots=1)
    touchdown(tracker, device)
    lift = frame(device, False)
    lift["height_scan"][:] = 0.5
    tracker.update(**lift)
    tracker.update(**frame(device))
    assert tracker.counters["overflow"] == 4
    assert (tracker.pending["height_scan"] == -0.8).all()


def test_high_slip_is_worse_and_late_motion_does_not_change_outcome(device):
    tracker = make_tracker(device)
    tracker.update(**frame(device))
    tracker.update(**frame(device, False))
    for step in range(tracker.survival_steps + 1):
        data = frame(device)
        data["foot_vel_w"][1, :, 0] = 0.6
        if step >= tracker.outcome_steps:
            data["foot_vel_w"][0, :, 0] = 100
            data["base_roll_pitch"][0] = 2
        packet = tracker.update(**data)
    assert packet["target"][0, 0, 0] == 1
    assert packet["target"][1, 0, 0] < packet["target"][0, 0, 0]
    expected = 0.4 * math.exp(-4) + 0.6
    assert packet["target"][1, 0, 0].item() == pytest.approx(expected)


def test_slip_only_counts_contact_and_persistence_uses_full_window(device):
    tracker = make_tracker(device)
    touchdown(tracker, device)
    no_contact = frame(device, False)
    no_contact["foot_vel_w"][:] = 100
    tracker.update(**no_contact)
    no_contact["terminated"][:] = True
    packet = tracker.update(**no_contact)
    valid = packet["valid"]
    assert (packet["slip_mean"][valid] == 0).all()
    torch.testing.assert_close(packet["persistence"][valid], torch.full((4,), 1 / 3, device=device))
    assert not packet["partial_window"].any()


def test_early_fall_on_touchdown_is_a_partial_negative(device):
    tracker = make_tracker(device)
    tracker.update(**frame(device))
    tracker.update(**frame(device, False))
    data = frame(device)
    data["terminated"][:] = True
    packet = tracker.update(**data)
    valid = packet["valid"]
    assert valid.sum() == 4
    assert (packet["observed_frames"][valid] == 1).all()
    assert packet["partial_window"][valid].all()
    assert not packet["survival"].any()
    torch.testing.assert_close(packet["target"][valid], torch.full((4, 1), 0.65 + 0.2 / 3, device=device))


def test_timeout_incomplete_dropped_mature_survival_retained(device):
    tracker = make_tracker(device)
    touchdown(tracker, device)
    for _ in range(tracker.survival_steps - 1):
        tracker.update(**frame(device))
    data = frame(device)
    data["truncated"][:] = True
    packet = tracker.update(**data)
    assert packet["valid"].sum() == 4
    assert packet["survival"][packet["valid"]].all()
    tracker = make_tracker(device)
    touchdown(tracker, device)
    packet = tracker.update(**data)
    assert not packet["valid"].any()
    assert tracker.counters["incomplete"] == 4


def test_partial_reset_and_left_right_independence(device):
    tracker = make_tracker(device)
    tracker.update(**frame(device))
    data = frame(device)
    data["foot_force_w"][:, 0] = 0
    tracker.update(**data)
    tracker.update(**frame(device))
    assert tracker.active[:, 0].sum() == 2
    assert not tracker.active[:, 1].any()
    tracker.reset(torch.tensor([0], device=device))
    assert not tracker.active[0].any()
    assert tracker.active[1].sum() == 1
    for _ in range(tracker.survival_steps):
        packet = tracker.update(**frame(device))
    assert packet["valid"].sum() == 1
    assert packet["valid"][1, 0, 0]


def test_tilt_angle_wrap_and_force_norm_not_used_for_contact(device):
    tracker = make_tracker(device)
    tracker.update(**frame(device))
    tracker.update(**frame(device, False))
    data = frame(device)
    data["base_roll_pitch"][:, 0] = math.pi - 0.01
    tracker.update(**data)
    data["base_roll_pitch"][:, 0] = -math.pi + 0.01
    data["terminated"][:] = True
    packet = tracker.update(**data)
    torch.testing.assert_close(packet["tilt_change"][packet["valid"]], torch.full((4,), 0.02, device=device))
    tracker = make_tracker(device)
    data = frame(device, False)
    data["foot_force_w"][..., 0] = 1000
    data["foot_force_w"][..., 2] = -100
    tracker.update(**data)
    assert not tracker.previous_contact.any()
    data["foot_force_w"][..., 2] = 20
    tracker.update(**data)
    assert not tracker.previous_contact.any()  # strictly > threshold


@pytest.mark.parametrize("field", ["height_scan", "foot_pos_w", "foot_vel_w", "foot_force_w", "base_roll_pitch"])
def test_nonfinite_records_are_rejected_without_poisoning_packet(device, field):
    tracker = make_tracker(device)
    tracker.update(**frame(device))
    lift = frame(device, False)
    if field == "height_scan":
        lift[field][0] = float("nan")
    tracker.update(**lift)
    touch = frame(device)
    if field != "height_scan":
        touch[field][0] = float("nan")
    touch["terminated"][:] = True
    packet = tracker.update(**touch)
    assert not packet["valid"][0].any()
    assert packet["valid"][1].sum() == 2
    assert tracker.counters["nonfinite"] > 0
    for value in packet.values():
        assert torch.isfinite(value).all()


def test_default_window_rounding_and_configuration_errors():
    assert SelfSupervisedCfg().window_steps(0.02) == (12, 25, 14)
    for cfg in (SelfSupervisedCfg(slip_scale=0), SelfSupervisedCfg(outcome_window_s=1),
                SelfSupervisedCfg(slip_weight=2), SelfSupervisedCfg(pending_slots=0),
                SelfSupervisedCfg(snapshot_mode="pre_touchdown_lag")):
        with pytest.raises(ValueError):
            cfg.window_steps(0.02)


@pytest.mark.parametrize("corner,expected_points", [(False, 9), (True, 4)])
def test_diagnostics_keep_liftoff_masks_through_pending_and_reset(device, corner, expected_points):
    from isal.interaction.scan_diagnostics import raw_height_masks

    tracker = make_tracker(device)
    reference = make_tracker(device)
    tracker.update(**frame(device))
    reference.update(**frame(device))
    lift = frame(device, False)
    lift["root_xy"][:] = torch.tensor([10., 20.], device=device)
    lift["root_yaw"][:] = math.pi / 2
    raw = torch.tensor([[[-2., 0., 0.], [0., float("nan"), 0.], [0., 0., .5]]], device=device).expand(2, -1, -1)
    masks = raw_height_masks(raw, -1.5, .4)
    tracker.update(**lift, scan_diagnostic_masks=masks)
    reference.update(**lift)
    masks[:] = False  # prove ownership; later masks must never replace liftoff data
    for t in range(tracker.survival_steps + 1):
        touch = frame(device)
        local_xy = torch.tensor([-.4, -.5] if corner else [.4, 0.], device=device)
        world_xy = torch.stack((-local_xy[1], local_xy[0])) + torch.tensor([10., 20.], device=device)
        touch["foot_pos_w"][..., :2] = world_xy
        packet = tracker.update(**touch, scan_diagnostic_masks=masks)
        baseline = reference.update(**touch)
    valid = packet["valid"]
    assert valid.sum() == 4
    torch.testing.assert_close(packet["target"], baseline["target"])
    assert packet["scan_diagnostics_available"][valid].all()
    assert (packet["scan_total_count"][valid] == 9).all()
    assert (packet["scan_finite_count"][valid] == 8).all()
    assert (packet["scan_lower_count"][valid] == 1).all()
    assert (packet["scan_upper_count"][valid] == 1).all()
    assert (packet["query_total_count"][valid] == expected_points).all()
    assert (packet["query_invalid_count"][valid] == 1).all()
    assert (packet["query_lower_count"][valid] == 1).all()
    assert (packet["query_upper_count"][valid] == (0 if corner else 1)).all()
    tracker.reset(torch.arange(2, device=device), clear_output=False)
    assert tracker.output["scan_total_count"][valid].sum() == 36
    tracker.reset(torch.arange(2, device=device))
    assert not tracker.output["scan_diagnostics_available"].any()
    assert not tracker.output["scan_total_count"].any()
    assert packet["scan_total_count"][valid].sum() == 36


def test_diagnostics_absent_or_all_invalid_do_not_change_target(device):
    tracker = make_tracker(device)
    touchdown(tracker, device)
    for _ in range(tracker.survival_steps):
        packet = tracker.update(**frame(device))
    assert not packet["scan_diagnostics_available"].any()
    assert not packet["scan_total_count"].any()
    tracker = make_tracker(device)
    tracker.update(**frame(device))
    masks = torch.zeros((2, 3, 3, 3), dtype=torch.bool, device=device)
    tracker.update(**frame(device, False), scan_diagnostic_masks=masks)
    for _ in range(tracker.survival_steps + 1):
        packet = tracker.update(**frame(device))
    valid = packet["valid"]
    assert packet["scan_diagnostics_available"][valid].all()
    assert (packet["target"][valid] == 1).all()
    assert (packet["scan_invalid_count"][valid] == 9).all()
    assert (packet["scan_invalid_fraction"][valid] == 1).all()
    assert not packet["scan_lower_fraction"].any()
    assert not packet["scan_upper_fraction"].any()
