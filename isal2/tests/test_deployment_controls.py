"""Gamepad transport, independent movement/camera input, and rendered camera geometry."""
import ctypes
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from isal2.deployment.camera import FollowCamera
from isal2.deployment.controller import CommandController, State, XInput


class GameControlsTests(unittest.TestCase):
    def test_xinput_preserves_unsigned_trigger_bytes(self):
        def read_state(index, pointer):
            gamepad = ctypes.cast(pointer, ctypes.POINTER(State)).contents.gamepad
            gamepad.lt, gamepad.rt, gamepad.rx = 127, 255, -32768
            return 0
        pad = XInput.__new__(XInput)
        pad.index, pad.dll = 0, SimpleNamespace(XInputGetState=read_state)
        state = pad.poll()
        self.assertEqual((state["lt"], state["rt"], state["rx"]), (127, 255, -32768))

    def test_trigger_direction_proportion_cancellation_and_camera_independence(self):
        ctrl = CommandController({"lin_vel_x": [-.6, 1], "lin_vel_y": [-.5, .5], "ang_vel_z": [-1., 1.5]})
        neutral = {"buttons": 0}
        ctrl.update(neutral)
        ctrl.update({"buttons": 16})
        for state, expected in [({"lt": 255}, 1.5), ({"rt": 255}, -1.),
                                ({"lt": 255, "rt": 255}, 0.), ({"lt": 10, "rt": 5}, 0.),
                                ({"rx": 32767, "ry": -32768}, 0.)]:
            command, _ = ctrl.update({**neutral, **state})
            np.testing.assert_allclose(command, [0, 0, expected])
        command, _ = ctrl.update({**neutral, "lt": 128})
        self.assertGreater(command[2], 0)
        self.assertLess(command[2], 1.5)
        command, _ = ctrl.update({**neutral, "lt": 255, "rx": 32767, "ly": 32767})
        np.testing.assert_allclose(command, [1., 0., 1.5])
        command, _ = ctrl.update(None)
        self.assertTrue(ctrl.paused)
        self.assertFalse(command.any())


class FollowCameraTests(unittest.TestCase):
    @staticmethod
    def rotation(degrees):
        angle = np.radians(degrees)
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])

    def test_default_view_really_is_behind_and_faces_robot_heading(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="10 10 .1"/></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        scene = mujoco.MjvScene(model, maxgeom=10)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        follow = FollowCamera()
        for heading in (0, 90, 179, -179, -90):
            rotation = self.rotation(heading)
            target = np.array([1., 2., .75])
            follow.update(camera, target, rotation, None, .02)
            mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            forward = scene.camera[0].forward[:2]
            np.testing.assert_allclose(forward / np.linalg.norm(forward), rotation[:2, 0], atol=1e-6)
            position = np.mean([eye.pos for eye in scene.camera], axis=0)
            self.assertLess(np.dot(position - target, rotation[:, 0]), -3.)
        follow.reset()
        follow.update(camera, target, self.rotation(179), None, 0.)
        azimuth = camera.azimuth
        follow.update(camera, target, self.rotation(-179), None, .02)
        self.assertAlmostEqual(camera.azimuth - azimuth, 2.)

    def test_orbit_limits_hold_rb_return_and_reset(self):
        follow, camera = FollowCamera(), mujoco.MjvCamera()
        update = lambda state, dt: follow.update(camera, [0, 0, .75], np.eye(3), state, dt)
        update({"rx": 32767, "ry": 32767}, .1)
        self.assertLess(camera.azimuth, 0)
        self.assertGreater(camera.elevation, -20)
        old = np.array([follow.yaw_offset, follow.pitch_offset])
        update({"rx": 100, "ry": -100}, .1)
        np.testing.assert_array_equal([follow.yaw_offset, follow.pitch_offset], old)
        # Release, stick noise, and disconnect must all preserve the chosen view.
        update(None, 8.)
        np.testing.assert_array_equal([follow.yaw_offset, follow.pitch_offset], old)
        update({"buttons": 0x0200}, .1)
        self.assertTrue((np.abs([follow.yaw_offset, follow.pitch_offset]) < abs(old)).all())
        self.assertTrue(follow.recentering)
        update({"buttons": 0}, 8.)
        np.testing.assert_allclose([camera.azimuth, camera.elevation], [0, -20], atol=.001)
        self.assertFalse(follow.recentering)
        update({"rx": -32768, "ry": -32768}, 10.)
        self.assertEqual(camera.azimuth, 160)
        self.assertEqual(camera.elevation, -75)
        follow.reset()
        update(None, 0.)
        np.testing.assert_allclose([camera.azimuth, camera.elevation], [0, -20])

    def test_stick_interrupts_return_and_held_rb_does_not_restart_it(self):
        follow, camera = FollowCamera(), mujoco.MjvCamera()
        update = lambda state: follow.update(camera, [0, 0, .75], np.eye(3), state, .1)
        update({"rx": 32767})
        update({"buttons": 0x0200})
        self.assertTrue(follow.recentering)
        update({"buttons": 0x0200, "rx": -32768})
        self.assertFalse(follow.recentering)
        angle = camera.azimuth
        update({"buttons": 0x0200})
        self.assertEqual(camera.azimuth, angle)
        update({"buttons": 0})
        self.assertEqual(camera.azimuth, angle)
        # Holding an offset still follows the robot heading.
        follow.update(camera, [0, 0, .75], self.rotation(90), {}, .1)
        self.assertAlmostEqual(camera.azimuth, angle + 90)
        update({"buttons": 0x0200})
        self.assertTrue(follow.recentering)
        follow.reset()
        self.assertFalse(follow.recentering)

    def test_return_is_independent_of_frame_rate(self):
        results = []
        for fps in (25, 50, 100):
            follow, camera = FollowCamera(), mujoco.MjvCamera()
            follow.update(camera, [0, 0, 0], np.eye(3), {"rx": 32767, "ry": -32768}, .5)
            follow.update(camera, [0, 0, 0], np.eye(3), {"buttons": 0x0200}, 0.)
            for _ in range(fps * 2):
                follow.update(camera, [0, 0, 0], np.eye(3), None, 1 / fps)
            results.append([camera.azimuth, camera.elevation])
        np.testing.assert_allclose(results, np.tile(results[0], (3, 1)), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
