"""Trusted local training checkpoint -> deterministic, actor-only ONNX artifact."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from isal2.utils.checkpoint import load_checkpoint


class TrainingLoader(yaml.SafeLoader):
    """Decode only the two inert Python tags emitted by our training config writer."""


TrainingLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: loader.construct_sequence(node))
TrainingLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice",
                               lambda loader, node: loader.construct_sequence(node))


def resolve_value(value, name, default=None):
    if isinstance(value, dict):
        matches = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous configuration for {name}: {value}")
        value = matches[0] if matches else default
    if value is None:
        raise ValueError(f"Missing configuration for {name}")
    return float(value)


def deployment_metadata(env, names, kind, checkpoint, policy_cfg, alpha):
    if len(names) != 23 or len(set(names)) != 23:
        raise ValueError("Expected 23 unique joint names")
    robot = env["scene"]["robot"]
    initial = robot["init_state"]
    joint = {key: [] for key in ("default_pos", "default_vel", "kp", "kd", "effort_limit", "velocity_limit", "armature")}
    for name in names:
        actuators = [a for a in robot["actuators"].values()
                     if any(re.fullmatch(p, name) for p in a["joint_names_expr"])]
        if len(actuators) != 1:
            raise ValueError(f"Expected one actuator configuration for {name}")
        a = actuators[0]
        joint["default_pos"].append(resolve_value(initial["joint_pos"], name, 0))
        joint["default_vel"].append(resolve_value(initial["joint_vel"], name, 0))
        for key, source in (("kp", "stiffness"), ("kd", "damping"), ("effort_limit", "effort_limit_sim"),
                            ("velocity_limit", "velocity_limit_sim"), ("armature", "armature")):
            joint[key].append(resolve_value(a[source], name))
    norm, rcfg = env["normalization"], env["robot"]
    if env["actor_frame_dim"] != 78 or rcfg["actor_obs_history_length"] != 5:
        raise ValueError("Deployment currently requires 78-dimensional frames and 5 history frames")
    scan = env["scene_context"]["height_scanner"]
    shape = list(policy_cfg.get("map_shape", (11, 17)))
    if kind != "base" and (shape != [11, 17] or scan["prim_body_name"] != "base_link"):
        raise ValueError("Deployment requires the centered base_link 11x17 scan")
    if kind != "base" and (not np.isclose(pcfg_resolution := policy_cfg.get("map_resolution", .1), scan["resolution"])
                           or not np.allclose(scan["size"], [(shape[1] - 1) * pcfg_resolution,
                                                             (shape[0] - 1) * pcfg_resolution])):
        raise ValueError("Training scanner geometry disagrees with the policy's metric position encoding")
    family = kind.capitalize() if kind != 'ame' else 'AME'
    stage = env.get('reference', env.get('course', {})).get('stage')
    suffix = 'Endpoint-' if 'course' in env else ''
    task = f'ISAL2-RPO-{family}-{suffix}Stage{stage}-v0' if stage in (1, 2) else f'ISAL2-RPO-{family}-v0'
    return {"schema_version": 1, "model_type": kind, "task": task,
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "inputs": {"policy": [None, 390], **({"height_scan": [None, 187]} if kind != "base" else {})},
            "output": {"actions": [None, 23]}, "joint_names": names, "joints": joint,
            "root_pos": initial["pos"], "root_quat_wxyz": initial["rot"],
            "frame_order": ["ang_vel_b", "projected_gravity_b", "commands", "joint_pos_delta", "joint_vel_delta", "previous_action"],
            "history_length": 5, "frame_dim": 78, "obs_scales": norm["obs_scales"],
            "clip_observations": norm["clip_observations"], "clip_actions": norm["clip_actions"],
            "action_scale": rcfg["action_scale"], "control_dt": env["sim"]["dt"] * env["decimation"],
            "command_ranges": {key: env["commands"]["ranges"][key] for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")},
            "command_generator": {key: env['commands'][key] for key in ('resampling_time_range', 'heading_command', 'heading_control_stiffness', 'rel_standing_envs')},
            "heading_range": env['commands']['ranges']['heading'],
            "curriculum": env.get('reference', env.get('course')),
            "height_scan": {"shape": shape, "ordering": "xy", "resolution": scan["resolution"],
                            "size": scan["size"], "ray_offset": scan["offset"],
                            "height_offset": norm["height_scan_offset"], "clip": [-1, 1], "miss_value": 1},
            "affordance_alpha": alpha, "physics_dt": .001, "actuator_delay_steps": 0}


class ActorExport(nn.Module):
    """Only inference modules are retained; attention is expressed with ONNX primitives."""
    def __init__(self, model):
        super().__init__()
        self.normalizer = deepcopy(model.actor_obs_normalizer)
        self.actor = deepcopy(model.actor)
        self.terrain = deepcopy(getattr(model, "terrain_attention", None))
        self.unet = deepcopy(getattr(model, "affordance_net", None))
        self.register_buffer("alpha", getattr(model, "affordance_alpha", torch.tensor(0.)).detach().clone())

    def forward(self, policy, height_scan=None):
        proprio = self.normalizer(policy)
        if self.terrain is None:
            return self.actor(proprio)
        t = self.terrain
        image = height_scan.reshape(height_scan.shape[0], 1, *t.map_shape)
        features = t.policy_encoder(image)
        if self.unet is not None:
            quality = .5 + self.alpha * (self.unet(image).sigmoid() - .5)
            quality = F.interpolate(quality, size=features.shape[-2:], mode="bilinear", align_corners=True)
            features = torch.cat([features, quality], 1)
        tokens = t.token_norm(t.position_encoding(features))
        query = t.query_norm(t.query(proprio)).unsqueeze(1)
        a = t.attention
        weights = a.in_proj_weight.chunk(3)
        biases = a.in_proj_bias.chunk(3)
        dim, heads = a.embed_dim, a.num_heads
        head_dim = dim // heads
        q = F.linear(query, weights[0], biases[0]).reshape(policy.shape[0], 1, heads, head_dim).transpose(1, 2)
        k = F.linear(tokens, weights[1], biases[1]).reshape(policy.shape[0], -1, heads, head_dim).transpose(1, 2)
        v = F.linear(tokens, weights[2], biases[2]).reshape(policy.shape[0], -1, heads, head_dim).transpose(1, 2)
        probabilities = torch.softmax((q @ k.transpose(-2, -1)) * head_dim ** -.5, dim=-1)
        attended = (probabilities @ v).transpose(1, 2).reshape(policy.shape[0], 1, dim)
        terrain = t.output_norm(a.out_proj(attended).squeeze(1))
        return self.actor(torch.cat([proprio, terrain], -1))


def load_actor(checkpoint, env_path=None, joints_path=None, agent_path=None):
    from rsl_rl.modules import ActorCritic
    from isal2.modified_rsl.modules import ActorCriticAME, ActorCriticAffordance
    checkpoint = Path(checkpoint)
    state = load_checkpoint(checkpoint)
    env = yaml.load(Path(env_path or checkpoint.parent / "env.yaml").read_text(encoding="utf-8"), Loader=TrainingLoader)
    names = json.loads(Path(joints_path or checkpoint.parent / "joint_names.json").read_text(encoding="utf-8"))
    cfg = (yaml.load(Path(agent_path).read_text(encoding="utf-8"), Loader=TrainingLoader)
           if agent_path else state["train_cfg"])
    pcfg = deepcopy(cfg["policy"])
    class_name = pcfg.pop("class_name")
    if pcfg.get("state_dependent_std", False):
        raise ValueError("Deployment currently supports the project's state-independent action distribution only")
    classes = {"ActorCritic": (ActorCritic, "base"), "rsl_rl.modules:ActorCritic": (ActorCritic, "base"),
               "isal2.modified_rsl.modules:ActorCriticAME": (ActorCriticAME, "ame"),
               "isal2.modified_rsl.modules:ActorCriticAffordance": (ActorCriticAffordance, "affordance")}
    if class_name not in classes:
        raise ValueError(f"Unsupported checkpoint policy class: {class_name}")
    cls, kind = classes[class_name]
    obs = {"policy": torch.zeros(1, 390),
           "critic": torch.zeros(1, env["critic_frame_dim"] * env["robot"]["critic_obs_history_length"])}
    if kind != "base":
        obs["height_scan"] = torch.zeros(1, 187)
    model = cls(obs, cfg["obs_groups"], len(names), **pcfg).cpu().eval()
    model.load_state_dict(state["model_state_dict"], strict=True)
    metadata = deployment_metadata(env, names, kind, checkpoint, pcfg,
                                   float(model.affordance_alpha) if kind == "affordance" else None)
    if 'reference_state' in state:
        metadata['curriculum'] = state['reference_state']['signature']
    return model, metadata


def export_checkpoint(checkpoint, output=None, env_path=None, joints_path=None, agent_path=None):
    import onnx
    import onnxruntime as ort
    torch.set_num_threads(1)
    model, metadata = load_actor(checkpoint, env_path, joints_path, agent_path)
    wrapper = ActorExport(model).eval()
    output = Path(output or Path(checkpoint).parent / "export")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "model.onnx"
    inputs = list(metadata["inputs"])
    example = tuple(torch.zeros(1, metadata["inputs"][key][1]) for key in inputs)
    with torch.inference_mode():
        torch.onnx.export(wrapper, example, str(path), input_names=inputs, output_names=["actions"],
                          dynamic_axes={key: {0: "batch"} for key in inputs + ["actions"]},
                          opset_version=17, dynamo=False)
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    errors = {}
    generator = torch.Generator().manual_seed(71)
    with torch.inference_mode():
        for batch in (1, 4, 32):
            obs = {key: torch.randn(batch, size[1], generator=generator) * .3 for key, size in metadata["inputs"].items()}
            native = model.act_inference(obs).numpy()
            wrapped = wrapper(*(obs[key] for key in inputs)).numpy()
            actual = session.run(None, {key: value.numpy() for key, value in obs.items()})[0]
            np.testing.assert_allclose(wrapped, native, atol=1e-5, rtol=1e-4)
            np.testing.assert_allclose(actual, native, atol=1e-5, rtol=1e-4)
            errors[str(batch)] = float(np.max(np.abs(actual - native)))
    metadata["onnx_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["validation"] = {"max_absolute_error_by_batch": errors, "atol": 1e-5, "rtol": 1e-4}
    (output / "deployment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
