"""Deployment math, controller state machine, ray geometry and ONNX regression."""
import contextlib
import io
import json
from pathlib import Path
import unittest
import uuid
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
import yaml

from isal2 import PROJECT_ROOT
from isal2.deployment.controller import CommandController
from isal2.deployment.export import ActorExport, TrainingLoader, export_checkpoint, load_actor
from isal2.deployment.runtime import History, Simulator, height_scan, joint_mapping, load_policy, pd_torque, scan_xy
from isal2.deployment.terrain import DEFAULT_TERRAIN, build_scene, tile_heights


RUNS = {"base": "rpo_base/external_rsl_train", "ame": "rpo_ame/ame_train_acceptance",
        "affordance": "rpo_affordance/aff_train_acceptance"}


class DeploymentMathTests(unittest.TestCase):
    def test_history_initialization_order_and_reset(self):
        history = History()
        np.testing.assert_array_equal(history.append(np.arange(78)).reshape(5, 78), np.tile(np.arange(78), (5, 1)))
        frame = np.ones(78) * 4
        stacked = history.append(frame).reshape(5, 78)
        np.testing.assert_array_equal(stacked[0], np.arange(78))
        np.testing.assert_array_equal(stacked[-1], frame)
        history.reset()
        np.testing.assert_array_equal(history.append(frame).reshape(5, 78), np.tile(frame, (5, 1)))

    def test_pd_saturation_per_joint(self):
        cfg = {"kp": [100, 40], "kd": [3.3, 2], "effort_limit": [120, 27]}
        torque, saturated = pd_torque(np.zeros(2), np.array([0, 1]), np.array([2, -2]), cfg)
        np.testing.assert_array_equal(torque, [120, -27])
        self.assertTrue(saturated.all())

    def test_safe_yaml_does_not_construct_objects(self):
        self.assertEqual(yaml.load("a: !!python/tuple [1, 2]", Loader=TrainingLoader), {"a": [1, 2]})
        with self.assertRaises(yaml.constructor.ConstructorError):
            yaml.load("!!python/object/apply:os.system ['echo unexpected']", Loader=TrainingLoader)

    def test_controller_edges_bounds_disconnect_and_reconnect(self):
        controller = CommandController({"lin_vel_x": [-.6, 1], "lin_vel_y": [-.5, .5], "ang_vel_z": [-1.57, 1.57]})
        state = {"buttons": 0, "lx": 0, "ly": 0, "rx": 0, "ry": 0}
        self.assertTrue(controller.paused)
        controller.update(state)
        command, events = controller.update({**state, "buttons": 16, "ly": 32767, "lx": 32767, "lt": 255})
        np.testing.assert_allclose(command, [1, -.5, 1.57])
        self.assertTrue(events["pause"])
        controller.update({**state, "buttons": 16})
        self.assertFalse(controller.paused)
        command, _ = controller.update({**state, "ly": 100})
        self.assertFalse(command.any())
        controller.update(None)
        self.assertTrue(controller.paused)
        controller.update({**state, "buttons": 16})
        self.assertTrue(controller.paused)
        controller.update(state)
        controller.update({**state, "buttons": 16})
        self.assertFalse(controller.paused)
        controller.update(state)
        command, events = controller.update({**state, "buttons": 0x1000, "ly": 32767})
        self.assertFalse(command.any())
        self.assertTrue(events["zero"])

    def test_xy_ray_yaw_exclusion_and_miss(self):
        # A tilted plane z=.1*x+.2*y gives an independently known asymmetric map.
        normal = np.array([-.1, -.2, 1.])
        normal /= np.linalg.norm(normal)
        zaxis = " ".join(map(str, normal))
        xml = f'<mujoco><worldbody><geom type="plane" size="10 10 .1" zaxis="{zaxis}" group="0"/>' \
              '<body pos="2 3 2"><geom type="box" size="2 2 .1" group="1"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        config = {"shape": [11, 17], "resolution": .1, "ray_offset": [0, 0, 20],
                  "height_offset": .75, "clip": [-1, 1], "miss_value": 1}
        xy = scan_xy(config)
        np.testing.assert_allclose(xy[:2], [[-.8, -.5], [-.7, -.5]])
        for yaw in (0, np.pi / 2):
            c, s = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            world = xy @ rotation[:2, :2].T + [2, 3]
            expected = -.1 * world[:, 0] - .2 * world[:, 1]
            actual = height_scan(model, data, [2, 3, .75], rotation, config)[0]
            np.testing.assert_allclose(actual, np.clip(expected, -1, 1), atol=1e-6)
        model.geom_group[:] = 1
        self.assertTrue((height_scan(model, data, [0, 0, .75], np.eye(3), config) == 1).all())

    def test_terrain_seed_boundaries_and_real_pits(self):
        for kind in ("flat", "rough", "slope", "inv_slope", "stairs", "inv_stairs", "boxes", "stones_gaps"):
            a = tile_heights(kind, .5, DEFAULT_TERRAIN, np.random.default_rng(9))
            b = tile_heights(kind, .5, DEFAULT_TERRAIN, np.random.default_rng(9))
            np.testing.assert_array_equal(a, b)
            self.assertTrue(np.isfinite(a).all())
            self.assertTrue((a[0] == 0).all() and (a[-1] == 0).all())
            if kind == "stones_gaps":
                self.assertEqual(float(a.min()), -1.)


class DeploymentSceneRegressionTests(unittest.TestCase):
    """Physical frame and compiled terrain checks that require no checkpoint."""

    @classmethod
    def setUpClass(cls):
        source = ET.parse(PROJECT_ROOT / "assets/data/rpo/mjcf/rpo.xml").getroot()
        names = [motor.get("joint") for motor in source.find("actuator")]
        cls.metadata = {
            "joint_names": names, "physics_dt": .001, "control_dt": .02,
            "joints": {"default_pos": [0.] * 23, "default_vel": [0.] * 23,
                       "effort_limit": [120.] * 23, "armature": [.01] * 23},
            "root_pos": [0, 0, .75], "root_quat_wxyz": [1, 0, 0, 0],
            "history_length": 5, "clip_observations": 100,
            "obs_scales": dict.fromkeys(("ang_vel", "projected_gravity", "commands",
                                         "joint_pos", "joint_vel", "actions"), 1.),
            "inputs": {"policy": [None, 390]},
        }
        cls.output = PROJECT_ROOT / "outputs/deployment_validation" / ("scene_" + uuid.uuid4().hex)
        cls.model, cls.details = build_scene(cls.metadata, output=cls.output)

    def test_angular_velocity_uses_link_frame_at_arbitrary_orientation(self):
        sim = Simulator(self.model, self.metadata, None)
        # The actual RPO base has non-diagonal inertia: BODY's local frame is
        # different from base_link even with an identity floating-base quaternion.
        self.assertFalse(np.allclose(sim.data.xmat[sim.base], sim.data.ximat[sim.base]))
        for quat in ([1., 0, 0, 0], np.array([1., 2., -3., 4.]) / np.sqrt(30)):
            sim.reset()
            sim.data.qpos[sim.free_qpos + 3:sim.free_qpos + 7] = quat
            for axis in range(3):
                omega = np.eye(3)[axis] * (axis + 1)
                sim.data.qvel[sim.free_dof + 3:sim.free_dof + 6] = omega
                mujoco.mj_forward(sim.model, sim.data)
                frame = sim.observation([0, 0, 0])["policy"].reshape(5, 78)[-1]
                np.testing.assert_allclose(frame[:3], omega, atol=1e-6)
                # Independent cross-check against the link-attached IMU gyro.
                np.testing.assert_allclose(frame[:3], sim.data.sensor("angular-velocity").data, atol=1e-6)

    def test_generated_heightfields_match_named_tiles_and_saved_model(self):
        saved = mujoco.MjModel.from_binary_path(str(self.output / "scene.mjb"))
        rng = np.random.default_rng(42)
        for tile in self.details["tiles"]:
            expected = tile_heights(tile["kind"], .5, DEFAULT_TERRAIN, rng)
            for model in (self.model, saved):
                hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, tile["name"])
                start = model.hfield_adr[hid]
                actual = model.hfield_data[start:start + expected.size].reshape(expected.shape)
                actual = actual * model.hfield_size[hid, 2] + tile["center"][2]
                np.testing.assert_allclose(actual, expected, atol=1e-7, err_msg=tile["name"])


class DeploymentCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        # Ordinary output directories inherit the workspace ACL on Windows; mkdtemp's
        # private ACL can exclude the restricted test process on this host.
        cls.temp = PROJECT_ROOT / "outputs" / "deployment_validation" / ("unit_" + uuid.uuid4().hex)
        cls.temp.mkdir(parents=True)
        cls.export_paths = {}
        for kind, run in RUNS.items():
            checkpoint = PROJECT_ROOT / "outputs" / run / "model_5.pt"
            if not checkpoint.exists():
                raise unittest.SkipTest("Local acceptance checkpoints are required for integration tests")
            output = cls.temp / kind
            with contextlib.redirect_stdout(io.StringIO()):
                export_checkpoint(checkpoint, output)
            cls.export_paths[kind] = output

    def test_dynamic_batch_and_no_critic_in_graph(self):
        import onnx
        for path in self.export_paths.values():
            metadata = json.loads((path / "deployment.json").read_text())
            self.assertEqual(set(metadata["validation"]["max_absolute_error_by_batch"]), {"1", "4", "32"})
            graph = onnx.load(path / "model.onnx").graph
            self.assertFalse(any("critic" in init.name for init in graph.initializer))

    def test_affordance_alpha_and_normalization(self):
        checkpoint = PROJECT_ROOT / "outputs" / RUNS["affordance"] / "model_5.pt"
        with contextlib.redirect_stdout(io.StringIO()):
            model, _ = load_actor(checkpoint)
        obs = {"policy": torch.randn(4, 390), "height_scan": torch.randn(4, 187) * .1}
        with torch.inference_mode():
            for alpha in (0., 1.):
                model.affordance_alpha.fill_(alpha)
                wrapper = ActorExport(model).eval()
                torch.testing.assert_close(wrapper(**obs), model.act_inference(obs), atol=1e-5, rtol=1e-4)
                for key, value in model.actor_obs_normalizer.state_dict().items():
                    torch.testing.assert_close(wrapper.normalizer.state_dict()[key], value)
                actual = wrapper(**obs)
                changed = wrapper(obs["policy"], obs["height_scan"] + .5)
                self.assertGreater(float((actual - changed).abs().max()), 1e-6)
                if alpha == 0:
                    for p in wrapper.unet.parameters():
                        p.add_(.1)
                    torch.testing.assert_close(actual, wrapper(**obs), atol=0, rtol=0)
            original = wrapper(**obs).clone()
            for p in wrapper.unet.parameters():
                p.add_(.1)
            self.assertGreater(float((original - wrapper(**obs)).abs().max()), 1e-6)

    def test_named_joints_reset_gravity_and_mixed_rays(self):
        session, metadata = load_policy(self.export_paths["ame"] / "model.onnx")
        model, details = build_scene(metadata, output=self.temp / "mixed")
        sim = Simulator(model, metadata, session)
        ids, dofs, motors = joint_mapping(model, metadata["joint_names"][::-1])
        np.testing.assert_array_equal(ids, sim.qpos_ids[::-1])
        np.testing.assert_array_equal(dofs, sim.qvel_ids[::-1])
        np.testing.assert_array_equal(motors, sim.motor_ids[::-1])
        first = sim.observation([0, 0, 0])["policy"].reshape(5, 78)
        np.testing.assert_array_equal(first[0], first[-1])
        np.testing.assert_allclose(first[-1, 3:6], [0, 0, -1], atol=1e-6)
        # A 90 degree roll rotates gravity onto negative local y.
        sim.data.qpos[sim.free_qpos+3:sim.free_qpos+7] = [np.sqrt(.5), np.sqrt(.5), 0, 0]
        mujoco.mj_forward(model, sim.data)
        np.testing.assert_allclose(sim.observation([0, 0, 0])["policy"][0, -75:-72], [0, -1, 0], atol=1e-6)
        sim.previous_action[:] = 1
        sim.reset()
        self.assertFalse(sim.previous_action.any())
        for tile in details["tiles"]:
            pos = [*tile["center"][:2], tile["center_height"] + .75]
            scan = height_scan(model, sim.data, pos, np.eye(3), metadata["height_scan"])
            self.assertTrue(np.isfinite(scan).all())
        # Query the entire stone tile from above: both the pit floor and stone tops must be hit.
        tile = details["tiles"][-1]
        config = {**metadata["height_scan"], "shape": [81, 81], "resolution": .1, "height_offset": 0, "clip": [-10, 10]}
        scan = height_scan(model, sim.data, [*tile["center"][:2], 2], np.eye(3), config)
        self.assertGreater(float(scan.max() - scan.min()), .9)


if __name__ == "__main__":
    unittest.main()
