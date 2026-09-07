"""Spatial/temporal correctness, not just shape or double-mirror checks."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict


pytestmark = pytest.mark.usefixtures("isaac_app")


def test_descending_heights_and_raw_diagnostics():
    from isal.tasks.direct.humanoid_rough.height_scan import preprocess_root_relative_height_scan
    from isal.interaction.scan_diagnostics import raw_height_masks, diagnostic_counts, diagnostic_fractions

    result = preprocess_root_relative_height_scan(
        torch.tensor([[0., -.05, -.10, -.20, -.30]]), torch.tensor([.75]),
        min_height=-1.5, max_height=.4, height_scale=.5)
    torch.testing.assert_close(result, torch.tensor([[-1.5, -1.6, -1.7, -1.9, -2.1]]))
    raw = torch.tensor([[[-2., -1.5, 0., .4, .5, float("nan"), float("inf")]]])
    masks = raw_height_masks(raw, -1.5, .4)
    counts = diagnostic_counts(masks)
    assert {key: value.item() for key, value in counts.items()} == {
        "total_count": 7, "finite_count": 5, "lower_count": 2, "upper_count": 2, "invalid_count": 2}
    fractions = diagnostic_fractions(counts)
    assert fractions["lower_fraction"].item() == pytest.approx(.4)
    assert fractions["invalid_fraction"].item() == pytest.approx(2 / 7)
    invalid_counts = diagnostic_counts(raw_height_masks(torch.full((1, 2, 2), float("nan")), -1.5, .4))
    fractions = diagnostic_fractions(invalid_counts)
    assert fractions["lower_fraction"] == fractions["upper_fraction"] == 0
    assert fractions["invalid_fraction"] == 1
    clipped = preprocess_root_relative_height_scan(raw.flatten(1), torch.tensor([0.]),
                                                 min_height=-1.5, max_height=.4, height_scale=.5)
    torch.testing.assert_close(clipped, torch.tensor([[-3., -3., 0., .8, .8, -3., -3.]]))


@pytest.mark.parametrize("shape", [(17, 11), (17, 9), (19, 11)])
@pytest.mark.parametrize("ordering", ["xy", "yx"])
@pytest.mark.parametrize("histories", [(10, 10), (3, 4)])
def test_mirror_spatial_semantics_and_history(shape, ordering, histories):
    from isal.tasks.direct.humanoid_rough.height_scan import (
        PerceptiveObservationLayout, xy_grid_to_flat_ray_order, flat_ray_order_to_xy_grid,
    )
    from isal.tasks.direct.humanoid_rough.agents.isal_agent_cfg import data_augmentation_func

    layout = PerceptiveObservationLayout(shape, ordering, *histories)
    env = SimpleNamespace(perceptive_observation_layout=layout)
    x, y = torch.meshgrid(torch.linspace(-.4, 1.2, shape[0]), torch.linspace(-.5, .5, shape[1]), indexing="ij")
    terrain = x + 10 * y
    actor = torch.zeros(2, histories[0], 78)
    state = torch.zeros(2, histories[1], 139)
    for t in range(histories[0]):
        actor[:, t, 6] = t + 1
        actor[:, t, 7] = t + 2
    for t in range(histories[1]):
        state[:, t, 78] = t + 3
        state[:, t, 79] = t + 4
    grids = terrain[None, None] + 100 * torch.arange(histories[1])[:, None, None, None]
    native = xy_grid_to_flat_ray_order(grids, ordering).expand(2, -1, -1)
    obs = TensorDict({"policy": actor.flatten(1), "height_scan": terrain.flatten().expand(2, -1),
                      "critic": torch.cat((state, native), -1).flatten(1)}, batch_size=[2])
    original = obs.clone()
    actions = torch.arange(46, dtype=torch.float).reshape(2, 23)
    augmented, act_aug = data_augmentation_func(env, obs, actions)
    mirror = augmented[2:]
    for key in obs.keys():
        assert augmented[key].shape == (4, obs[key].shape[1])
        torch.testing.assert_close(augmented[key][:2], original[key])
    torch.testing.assert_close(mirror["height_scan"].reshape(2, *shape), (x - 10 * y).expand(2, -1, -1))
    actor_mirror = mirror["policy"].reshape_as(actor)
    torch.testing.assert_close(actor_mirror[..., 6], actor[..., 6])
    torch.testing.assert_close(actor_mirror[..., 7], -actor[..., 7])
    critic_mirror = mirror["critic"].reshape(2, histories[1], layout.critic_frame_dim)
    torch.testing.assert_close(critic_mirror[..., 78], state[..., 78])
    torch.testing.assert_close(critic_mirror[..., 79], -state[..., 79])
    actual = flat_ray_order_to_xy_grid(critic_mirror[..., 139:].reshape(-1, layout.num_rays), shape, ordering)
    expected = ((x - 10 * y)[None, None] + 100 * torch.arange(histories[1])[:, None, None, None]).repeat(2, 1, 1, 1)
    torch.testing.assert_close(actual, expected)
    restored, _ = data_augmentation_func(env, mirror, None)
    for key in obs.keys():
        torch.testing.assert_close(restored[key][2:], original[key])
    _, action_only = data_augmentation_func(env, None, actions)
    torch.testing.assert_close(action_only, act_aug)
    no_critic, _ = data_augmentation_func(env, obs.select("policy", "height_scan"), None)
    assert no_critic["policy"].shape == augmented["policy"].shape


def test_default_mirror_matches_legacy_and_rejects_bad_layout():
    from isal.tasks.direct.humanoid_rough.height_scan import PerceptiveObservationLayout
    from isal.tasks.direct.humanoid_rough.agents.isal_agent_cfg import (
        data_augmentation_func, mirror_policy_observation, mirror_critic_observation,
    )
    env = SimpleNamespace(perceptive_observation_layout=PerceptiveObservationLayout((17, 11), "xy", 10, 10))
    obs = TensorDict({"policy": torch.randn(2, 780), "critic": torch.randn(2, 3260),
                      "height_scan": torch.randn(2, 187)}, batch_size=[2])
    augmented, _ = data_augmentation_func(env, obs, None)
    torch.testing.assert_close(augmented["policy"][2:], mirror_policy_observation(obs["policy"]))
    torch.testing.assert_close(augmented["critic"][2:], mirror_critic_observation(obs["critic"]))
    obs["critic"] = torch.zeros(2, 3259)
    with pytest.raises(ValueError, match="Critic observation layout mismatch"):
        data_augmentation_func(env, obs, None)
    with pytest.raises(ValueError, match="positive integers"):
        PerceptiveObservationLayout((17, 11), "xy", 0, 10)


def test_final_config_overrides_resolve_dimensions_without_rebuilding_scene():
    from isal.tasks.direct.humanoid_rough.isal_env_cfg import ISALHumanoidRoughHeightScanEnvCfg
    cfg = ISALHumanoidRoughHeightScanEnvCfg()
    scene = cfg.scene
    cfg.scene.num_envs = 2
    cfg.scene.height_scanner.pattern_cfg.ordering = "yx"
    cfg.terrain_perception.size = (1.8, 1.0)
    cfg.terrain_perception.offset_x = .3
    cfg.robot.actor_obs_history_length = 3
    cfg.robot.critic_obs_history_length = 4
    layout = cfg.resolve_perception_layout()
    assert cfg.scene is scene and cfg.scene.num_envs == 2
    assert layout.grid_shape == (19, 11) and layout.ordering == "yx"
    assert cfg.state_space == 139 + 209 and cfg.observation_space == 78
    assert cfg.scene.height_scanner.offset.pos == (.3, 0., 20.)
    cfg.terrain_perception.resolution = .2
    layout = cfg.resolve_perception_layout()
    assert layout.grid_shape == (10, 6) and cfg.state_space == 199
