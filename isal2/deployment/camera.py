"""Heading-relative third-person orbit camera, independent of simulation time."""
import numpy as np

from .controller import stick_value


DEFAULT_CAMERA = {"distance": 4., "elevation": -20., "yaw_speed": 120., "pitch_speed": 60.,
                  "yaw_limit": 160., "pitch_limits": [-75., -5.], "return_time": .8, "deadzone": .15}


class FollowCamera:
    def __init__(self, config=None):
        if config and set(config) - set(DEFAULT_CAMERA):
            raise ValueError("Unknown camera configuration keys")
        self.config = {**DEFAULT_CAMERA, **(config or {})}
        cfg = self.config
        for key in ("distance", "yaw_speed", "pitch_speed", "yaw_limit", "return_time"):
            if not np.isfinite(cfg[key]) or cfg[key] <= 0:
                raise ValueError(f"Camera {key} must be finite and positive")
        if not 0 <= cfg["deadzone"] < 1:
            raise ValueError("Camera deadzone must be in [0,1)")
        limits = cfg["pitch_limits"]
        if (len(limits) != 2 or not np.isfinite([*limits, cfg["elevation"]]).all()
                or not -89 <= limits[0] <= cfg["elevation"] <= limits[1] < 0):
            raise ValueError("Camera pitch limits must contain elevation and stay in [-89,0) degrees")
        self.reset()

    def reset(self):
        self.yaw_offset = self.pitch_offset = 0.
        self.heading = None
        self.recentering = False
        self.previous_buttons = 0

    def update(self, camera, position, rotation, state, dt):
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("Camera dt must be finite and nonnegative")
        cfg = self.config
        state = state or {}
        buttons = state.get("buttons", 0)
        # XInput RIGHT_SHOULDER (RB): one press starts a complete smooth return.
        if buttons & ~self.previous_buttons & 0x0200:
            self.recentering = True
        self.previous_buttons = buttons
        horizontal = stick_value(state, "rx", cfg["deadzone"])
        vertical = stick_value(state, "ry", cfg["deadzone"])
        if horizontal or vertical:
            self.recentering = False
            self.yaw_offset = float(np.clip(self.yaw_offset - horizontal * cfg["yaw_speed"] * dt,
                                            -cfg["yaw_limit"], cfg["yaw_limit"]))
            self.pitch_offset = float(np.clip(self.pitch_offset + vertical * cfg["pitch_speed"] * dt,
                cfg["pitch_limits"][0] - cfg["elevation"], cfg["pitch_limits"][1] - cfg["elevation"]))
        elif self.recentering:
            # Exponential relaxation gives the same return speed at any frame rate,
            # and continues after RB is released, also while physics is paused.
            decay = np.exp(-dt / cfg["return_time"])
            self.yaw_offset *= decay
            self.pitch_offset *= decay
            if max(abs(self.yaw_offset), abs(self.pitch_offset)) < .01:
                self.yaw_offset = self.pitch_offset = 0.
                self.recentering = False
        rotation = np.asarray(rotation).reshape(3, 3)
        heading = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
        self.heading = (heading if self.heading is None else
                        self.heading + (heading - self.heading + 180.) % 360. - 180.)
        # MuJoCo azimuth is the viewing direction: azimuth=body yaw places the
        # camera behind the robot, looking along its forward (+X) axis.
        camera.lookat[:] = position
        camera.distance = cfg["distance"]
        camera.azimuth = self.heading + self.yaw_offset
        camera.elevation = cfg["elevation"] + self.pitch_offset
