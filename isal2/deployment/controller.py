"""Windows XInput transport separated from deterministic command/button mapping."""
import ctypes
import sys
import numpy as np


DEFAULT_CONTROLLER = {"deadzone": .15, "axes": ["ly", "lx", "rx"], "signs": [1, -1, -1],
                      "sensitivity": [1., 1., 1.], "buttons": {"zero": 0x1000, "reset": 0x8000,
                      "pause": 0x0010, "exit": 0x0020}}


class Gamepad(ctypes.Structure):
    _fields_ = [("buttons", ctypes.c_ushort), ("lt", ctypes.c_ubyte), ("rt", ctypes.c_ubyte),
                ("lx", ctypes.c_short), ("ly", ctypes.c_short), ("rx", ctypes.c_short), ("ry", ctypes.c_short)]


class State(ctypes.Structure):
    _fields_ = [("packet", ctypes.c_uint32), ("gamepad", Gamepad)]


class XInput:
    def __init__(self, index=0):
        if sys.platform != "win32":
            raise RuntimeError("Interactive gamepad control currently requires Windows XInput")
        if index not in range(4):
            raise ValueError("XInput index must be 0..3")
        self.index = index
        self.dll = ctypes.WinDLL("xinput1_4.dll")
        self.dll.XInputGetState.argtypes = [ctypes.c_uint32, ctypes.POINTER(State)]
        self.dll.XInputGetState.restype = ctypes.c_uint32

    def poll(self):
        state = State()
        if self.dll.XInputGetState(self.index, ctypes.byref(state)) != 0:
            return None
        g = state.gamepad
        return {"buttons": g.buttons, **{key: getattr(g, key) for key in ("lx", "ly", "rx", "ry")}}


class CommandController:
    def __init__(self, ranges, config=None):
        self.config = {**DEFAULT_CONTROLLER, **(config or {})}
        if config and set(config) - set(DEFAULT_CONTROLLER):
            raise ValueError("Unknown controller configuration keys")
        self.config["buttons"] = {**DEFAULT_CONTROLLER["buttons"], **(config or {}).get("buttons", {})}
        self.bounds = np.array([ranges[key] for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")])
        if not 0 <= self.config["deadzone"] < 1:
            raise ValueError("Controller deadzone must be in [0,1)")
        if len(self.config["axes"]) != 3 or any(a not in ("lx", "ly", "rx", "ry") for a in self.config["axes"]):
            raise ValueError("Expected three XInput stick axes")
        if len(self.config["signs"]) != 3 or len(self.config["sensitivity"]) != 3:
            raise ValueError("Expected three signs and sensitivity values")
        if not all(s in (-1, 1) for s in self.config["signs"]) or not all(
                np.isfinite(s) and s > 0 for s in self.config["sensitivity"]):
            raise ValueError("Signs must be +/-1 and sensitivities finite and positive")
        self.paused, self.connected, self.previous_buttons = True, False, 0

    def update(self, state):
        events = {key: False for key in ("zero", "reset", "pause", "exit")}
        if state is None:
            self.paused, self.connected, self.previous_buttons = True, False, 0
            return np.zeros(3, dtype=np.float32), events
        buttons = state["buttons"]
        # Holding Start through reconnection cannot resume the robot.
        if not self.connected:
            self.previous_buttons = buttons
        pressed = buttons & ~self.previous_buttons
        self.previous_buttons, self.connected = buttons, True
        events = {key: bool(pressed & self.config["buttons"][key]) for key in events}
        if events["pause"]:
            self.paused = not self.paused
        values = np.array([state[a] / (32767 if state[a] >= 0 else 32768) for a in self.config["axes"]])
        dead = self.config["deadzone"]
        values = np.sign(values) * np.maximum(np.abs(values) - dead, 0) / (1 - dead)
        values = np.clip(values * self.config["signs"] * self.config["sensitivity"], -1, 1)
        commands = np.where(values >= 0, values * self.bounds[:, 1], -values * self.bounds[:, 0])
        if self.paused or events["zero"] or events["reset"]:
            commands[:] = 0
        return commands.astype(np.float32), events
