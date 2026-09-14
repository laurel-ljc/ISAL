"""Deterministic MuJoCo heightfield tiles; original MJCF and meshes stay untouched."""
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from isal2 import PROJECT_ROOT


DEFAULT_TERRAIN = {
    "tile_size": 8., "resolution": .05, "border_width": 1., "platform_width": 2.,
    "step_height_range": [.05, .15], "hard_step_height_range": [.05, .2], "step_width": .25,
    "slope_range": [.1, .3], "grid_width": .45, "noise_range": [-.02, .04], "noise_step": .02,
    "stone_width_range": [.3, .4], "stone_distance_range": [.1, .2], "gap_width_range": [.1, .3],
    "pit_depth": 1.,
}


def tile_heights(kind, difficulty, cfg, rng, hard=False):
    size, spacing = cfg["tile_size"], cfg["resolution"]
    n = int(round(size / spacing)) + 1
    x, y = np.meshgrid(np.linspace(-size / 2, size / 2, n), np.linspace(-size / 2, size / 2, n), indexing="xy")
    radial = np.maximum(abs(x), abs(y))
    inner = size / 2 - cfg["border_width"]
    active = radial < inner
    distance = np.maximum(inner - radial, 0)
    height = np.zeros_like(x)
    lerp = lambda pair: pair[0] + difficulty * (pair[1] - pair[0])
    step = lerp(cfg["hard_step_height_range"] if hard else cfg["step_height_range"])
    if kind == "rough":
        low, high = cfg["noise_range"]
        height = np.round(rng.uniform(low, high, x.shape) / cfg["noise_step"]) * cfg["noise_step"]
    elif kind in ("slope", "inv_slope", "stairs", "inv_stairs"):
        rise = np.minimum(distance, inner - cfg["platform_width"] / 2)
        height = rise * lerp(cfg["slope_range"]) if "slope" in kind else np.floor(rise / cfg["step_width"]) * step
        if kind.startswith("inv_"):
            height = -height
    elif kind == "boxes":
        indices_x = np.floor((x + size / 2) / cfg["grid_width"]).astype(int)
        indices_y = np.floor((y + size / 2) / cfg["grid_width"]).astype(int)
        random_grid = rng.uniform(0, step, (indices_y.max() + 1, indices_x.max() + 1))
        height = random_grid[indices_y, indices_x]
        height[radial < cfg["platform_width"] / 2] = 0
    elif kind == "stones_gaps":
        width = lerp(cfg["stone_width_range"])
        gap = lerp(cfg["stone_distance_range"])
        period = width + gap
        on_stone = ((x + size / 2) % period < width) & ((y + size / 2) % period < width)
        height[~on_stone] = -cfg["pit_depth"]
        # An uninterrupted trench supplements the stepping-stone gaps.
        height[abs(x - 1.5) < lerp(cfg["gap_width_range"]) / 2] = -cfg["pit_depth"]
        height[radial < cfg["platform_width"] / 2] = 0
    elif kind != "flat":
        raise ValueError(f"Unknown tile type: {kind}")
    height[~active] = 0
    return height.astype(np.float32)


def build_scene(metadata, preset="mixed", seed=42, difficulty=.5, config=None, output=None):
    import mujoco
    if preset not in ("flat", "rough", "rough_hard", "mixed") or not 0 <= difficulty <= 1:
        raise ValueError("Expected a supported terrain and difficulty in [0,1]")
    cfg = deepcopy(DEFAULT_TERRAIN)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown terrain settings: {sorted(unknown)}")
        cfg.update(config)
    if not (0 < cfg["resolution"] <= cfg["tile_size"] / 4 and
            0 < cfg["platform_width"] < cfg["tile_size"] - 2 * cfg["border_width"]):
        raise ValueError("Invalid terrain resolution, platform or border")
    output = Path(output or PROJECT_ROOT / "outputs" / "sim2sim" / "scene")
    output.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / "assets" / "data" / "rpo"
    root = ET.parse(source / "mjcf" / "rpo.xml").getroot()
    root.find("compiler").set("meshdir", str(source / "meshes"))
    root.find("option").set("timestep", str(metadata["physics_dt"]))
    world, assets = root.find("worldbody"), root.find("asset")
    for geom in list(world.findall("geom")):
        world.remove(geom)
    for body in world.findall("body"):
        for geom in body.iter("geom"):
            geom.set("group", "1")
    kinds = ["flat"] if preset == "flat" else ["flat", "rough", "slope", "inv_slope", "stairs", "inv_stairs", "boxes",
                                               "stones_gaps" if preset in ("rough_hard", "mixed") else "rough"]
    rng = np.random.default_rng(seed)
    fields, tile_info = [], []
    for i, kind in enumerate(kinds):
        heights = tile_heights(kind, difficulty, cfg, rng, preset == "rough_hard")
        minimum = float(heights.min())
        span = max(float(heights.max()) - minimum, .01)
        center = [(i % 4) * cfg["tile_size"], (i // 4) * cfg["tile_size"], minimum]
        name = f"tile_{i}_{kind}"
        ET.SubElement(assets, "hfield", name=name, nrow=str(heights.shape[0]), ncol=str(heights.shape[1]),
                      size=f"{cfg['tile_size']/2} {cfg['tile_size']/2} {span} .1")
        ET.SubElement(world, "geom", name=name, type="hfield", hfield=name, pos=" ".join(map(str, center)),
                      group="0", contype="1", conaffinity="15", condim="3", friction=".9 .2 .2",
                      rgba=".32 .40 .30 1")
        fields.append((name, ((heights - minimum) / span).ravel()))
        tile_info.append({"name": name, "kind": kind, "center": center, "min_height": minimum,
                          "max_height": float(heights.max()), "shape": list(heights.shape),
                          "center_height": float(heights[heights.shape[0] // 2, heights.shape[1] // 2])})
    limits = dict(zip(metadata["joint_names"], metadata["joints"]["effort_limit"]))
    armatures = dict(zip(metadata["joint_names"], metadata["joints"]["armature"]))
    for motor in root.find("actuator"):
        limit = limits[motor.attrib["joint"]]
        motor.set("ctrlrange", f"{-limit} {limit}")
    for body in world.findall("body"):
        for joint in body.iter("joint"):
            if joint.get("name") in armatures:
                joint.set("armature", str(armatures[joint.get("name")]))
    path = output / "scene.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    for name, field in fields:
        # The source MJCF already has an hf0 asset. Compiled IDs therefore do not
        # start at zero for the generated tiles; resolve each heightfield by name.
        field_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, name)
        if field_id < 0 or model.hfield_nrow[field_id] * model.hfield_ncol[field_id] != len(field):
            raise ValueError(f"Heightfield geometry mismatch for {name}")
        start = model.hfield_adr[field_id]
        model.hfield_data[start:start + len(field)] = field
    # MJB contains the populated heightfield, unlike the XML template alone.
    mujoco.mj_saveModel(model, str(output / "scene.mjb"), None)
    details = {"preset": preset, "seed": seed, "difficulty": difficulty, "config": cfg, "tiles": tile_info}
    (output / "terrain.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return model, details
