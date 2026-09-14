"""NumPy/MuJoCo/ONNX Runtime inference. No training packages are imported here."""
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import onnxruntime as ort


def load_policy(path, metadata_path=None):
    path = Path(path)
    metadata = json.loads(Path(metadata_path or path.parent / "deployment.json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != 1 or metadata["onnx_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Deployment metadata version or ONNX checksum mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if {i.name: i.shape[1:] for i in session.get_inputs()} != {k: v[1:] for k, v in metadata["inputs"].items()}:
        raise ValueError("ONNX inputs do not match deployment metadata")
    return session, metadata


def joint_mapping(model, names):
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    if len(names) != 23 or len(set(names)) != 23 or min(joints) < 0:
        raise ValueError("Expected all 23 uniquely named training joints in the MuJoCo model")
    motors = []
    for joint in joints:
        found = np.flatnonzero((model.actuator_trnid[:, 0] == joint) &
                              (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT))
        if len(found) != 1 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError("Each training joint must be a hinge with exactly one joint actuator")
        actuator = found[0]
        if not np.array_equal(model.actuator_gear[actuator], [1, 0, 0, 0, 0, 0]):
            raise ValueError("Expected unit-gear torque actuators")
        motors.append(actuator)
    return model.jnt_qposadr[joints], model.jnt_dofadr[joints], np.array(motors)


class History:
    def __init__(self, length=5):
        self.length, self.frames = length, None

    def reset(self):
        self.frames = None

    def append(self, frame):
        if self.frames is None:
            self.frames = np.tile(frame, (self.length, 1))
        else:
            self.frames[:-1] = self.frames[1:]
            self.frames[-1] = frame
        return self.frames.reshape(1, -1).astype(np.float32)


def scan_xy(config):
    rows, cols = config["shape"]
    x = (np.arange(cols) - (cols - 1) / 2) * config["resolution"]
    y = (np.arange(rows) - (rows - 1) / 2) * config["resolution"]
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def height_scan(model, data, root_position, rotation, config, scale=1):
    xy = scan_xy(config)
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    xy = xy @ np.array([[c, s], [-s, c]])
    offset = np.array(config["ray_offset"], dtype=float)
    # Offset is expressed in the yaw-aligned sensor frame too.
    offset[:2] = offset[:2] @ np.array([[c, s], [-s, c]])
    start = np.array(root_position, dtype=float) + offset
    group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    direction = np.array([0., 0., -1.])
    result = np.empty(len(xy), dtype=np.float32)
    geom_id = np.empty(1, dtype=np.int32)
    for i, point in enumerate(xy):
        origin = start + [point[0], point[1], 0]
        distance = mujoco.mj_ray(model, data, origin, direction, group, True, -1, geom_id)
        if distance < 0 or not np.isfinite(distance):
            result[i] = config["miss_value"]
        else:
            ground_z = origin[2] - distance
            result[i] = np.clip(root_position[2] - ground_z - config["height_offset"], *config["clip"])
    return result[None] * scale


def pd_torque(position, velocity, target, cfg):
    raw = np.array(cfg["kp"]) * (target - position) - np.array(cfg["kd"]) * velocity
    limits = np.array(cfg["effort_limit"])
    return np.clip(raw, -limits, limits), np.abs(raw) > limits


class Simulator:
    def __init__(self, model, metadata, session, spawn_offset=(0, 0, 0)):
        self.model, self.meta, self.session = model, metadata, session
        self.spawn_offset = np.array(spawn_offset, dtype=float)
        self.data = mujoco.MjData(model)
        self.qpos_ids, self.qvel_ids, self.motor_ids = joint_mapping(model, metadata["joint_names"])
        self.base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base < 0:
            raise ValueError("Missing base_link body")
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("Expected exactly one floating base")
        self.free_qpos, self.free_dof = model.jnt_qposadr[free[0]], model.jnt_dofadr[free[0]]
        self.default = np.array(metadata["joints"]["default_pos"])
        self.default_velocity = np.array(metadata["joints"]["default_vel"])
        self.history = History(metadata["history_length"])
        ratio = metadata["control_dt"] / model.opt.timestep
        self.decimation = round(ratio)
        if self.decimation < 1 or not np.isclose(self.decimation, ratio):
            raise ValueError("Control period must be an integer multiple of physics timestep")
        self.steps, self.resets, self.saturations, self.falls = 0, 0, 0, 0
        self.records = []
        self.reset(count=False)

    def reset(self, count=True):
        mujoco.mj_resetData(self.model, self.data)
        address = self.free_qpos
        self.data.qpos[address:address + 3] = np.array(self.meta["root_pos"]) + self.spawn_offset
        self.data.qpos[address + 3:address + 7] = self.meta["root_quat_wxyz"]
        self.data.qpos[self.qpos_ids] = self.default
        self.data.qvel[self.qvel_ids] = self.default_velocity
        self.previous_action = np.zeros(23, dtype=np.float32)
        self.target = self.default.copy()
        self.history.reset()
        mujoco.mj_forward(self.model, self.data)
        self.resets += int(count)

    def observation(self, command):
        data = self.data
        rotation = data.xmat[self.base].reshape(3, 3)
        velocity = np.zeros(6)
        # BODY uses the principal-inertia frame (ximat), which is rotated relative
        # to base_link. Training root_ang_vel_b and gravity both use the link frame.
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_XBODY, self.base, velocity, 1)
        scale = self.meta["obs_scales"]
        frame = np.concatenate([velocity[:3] * scale["ang_vel"],
                                rotation.T @ np.array([0., 0., -1.]) * scale["projected_gravity"],
                                np.array(command) * scale["commands"],
                                (data.qpos[self.qpos_ids] - self.default) * scale["joint_pos"],
                                (data.qvel[self.qvel_ids] - self.default_velocity) * scale["joint_vel"],
                                self.previous_action * scale["actions"]])
        limit = self.meta["clip_observations"]
        obs = {"policy": np.clip(self.history.append(frame), -limit, limit)}
        if "height_scan" in self.meta["inputs"]:
            obs["height_scan"] = height_scan(self.model, data, data.xpos[self.base], rotation,
                                              self.meta["height_scan"], scale["height_scan"])
        return obs

    def step(self, command):
        obs = self.observation(command)
        if not all(np.isfinite(value).all() for value in obs.values()):
            raise FloatingPointError("Non-finite policy observation")
        start = time.perf_counter()
        action = self.session.run(["actions"], obs)[0][0]
        latency = time.perf_counter() - start
        if action.shape != (23,) or not np.isfinite(action).all():
            raise FloatingPointError("Invalid ONNX action")
        self.previous_action = np.clip(action, -self.meta["clip_actions"], self.meta["clip_actions"])
        self.target = self.default + self.meta["action_scale"] * self.previous_action
        saturated = 0
        for _ in range(self.decimation):
            torque, saturation = pd_torque(self.data.qpos[self.qpos_ids], self.data.qvel[self.qvel_ids],
                                           self.target, self.meta["joints"])
            self.data.ctrl[self.motor_ids] = torque
            saturated += int(saturation.sum())
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise FloatingPointError("Non-finite MuJoCo state")
        mujoco.mj_forward(self.model, self.data)
        if any(w.number for w in self.data.warning):
            raise RuntimeError("MuJoCo issued a simulation warning; inspect saved run context")
        self.steps += 1
        self.saturations += saturated
        self.records.append({"step": self.steps, "command": list(map(float, command)),
                             "action": self.previous_action.tolist(), "torque": torque.tolist(),
                             "saturation_count": saturated, "inference_ms": latency * 1000,
                             "root_pos": self.data.xpos[self.base].tolist()})
        # This deployment recovery heuristic is separate from the training termination rule.
        return bool(self.data.xmat[self.base].reshape(3, 3)[2, 2] < .35)

    def result(self):
        latencies = [r["inference_ms"] for r in self.records]
        return {"control_steps": self.steps, "reset_count": self.resets, "fall_count": self.falls,
                "saturation_count": self.saturations, "finite": True,
                "inference_ms_mean": float(np.mean(latencies)) if latencies else None,
                "inference_ms_p95": float(np.percentile(latencies, 95)) if latencies else None}
