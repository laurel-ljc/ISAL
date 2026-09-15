"""Render the current training terrain functions in Isaac Sim, without running training."""
import argparse
import importlib.util
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
args.kit_args += f" --portable-root={args.output.as_posix()}/kit --/log/file={args.output.as_posix()}/kit.log"
app = AppLauncher(args).app

import numpy as np
import trimesh
from PIL import Image
import omni.usd
import omni.replicator.core as rep
from pxr import UsdGeom, UsdLux, Gf
from isaacsim.core.utils.viewports import set_camera_view
from isaaclab.terrains.height_field import HfTerrainBaseCfg

source = Path(__file__).resolve().parents[1] / "tasks/base/terrain_generator_cfg.py"
spec = importlib.util.spec_from_file_location("catalog_terrain_cfg", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
light = UsdLux.DomeLight.Define(stage, "/World/Sky")
light.CreateIntensityAttr(650)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(2500)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(25, -35, -30))
camera = rep.create.camera(position=(10, -10, 15), look_at=(0, 0, 0), focal_length=30,
                           clipping_range=(0.1, 10000))
camera_path = camera.get_output_prims()["prims"][0].GetPath().pathString
product = rep.create.render_product(camera, (800, 700))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([product])
manifest = {"source": str(source), "renderer": "Isaac Sim RTX", "difficulty": 0.5,
            "seed": 42, "note": "Individual 8x8 m tiles; original geometry, illustrative colors; no training changes.", "terrains": []}

def put_mesh(path, mesh, color):
    prim = UsdGeom.Mesh.Define(stage, path)
    prim.CreatePointsAttr(mesh.vertices.astype(np.float32))
    prim.CreateFaceVertexCountsAttr(np.full(len(mesh.faces), 3))
    prim.CreateFaceVertexIndicesAttr(mesh.faces.flatten())
    prim.CreateSubdivisionSchemeAttr("none")
    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    prim.CreateDoubleSidedAttr(True)
    return prim

def capture(path, eye, target):
    set_camera_view(eye=np.array(eye), target=np.array(target), camera_prim_path=camera_path)
    for _ in range(5):
        rep.orchestrator.step(rt_subframes=4, pause_timeline=True)
    frame = np.asarray(rgb.get_data())
    if frame.size == 0:
        raise RuntimeError("Empty rendered image")
    Image.fromarray(frame[:, :, :3]).save(path)
    print(f"CAPTURE {path.name}", flush=True)

try:
    for preset, cfg in [("rough", module.ROUGH_TERRAINS_CFG), ("rough_hard", module.ROUGH_HARD_TERRAINS_CFG)]:
        for index, (name, original) in enumerate(cfg.sub_terrains.items()):
            sub = original.copy()
            sub.size = cfg.size
            if isinstance(sub, HfTerrainBaseCfg):
                sub.horizontal_scale = cfg.horizontal_scale
                sub.vertical_scale = cfg.vertical_scale
                sub.slope_threshold = cfg.slope_threshold
            np.random.seed(42)
            meshes, origin = sub.function(0.5, sub)
            mesh = trimesh.util.concatenate(meshes)
            mesh.apply_translation([-4, -4, 0])
            prim = put_mesh("/World/Tile", mesh, (0.38, 0.58, 0.66) if preset == "rough" else (0.67, 0.49, 0.28))
            top = float(mesh.bounds[1, 2])
            filename = f"{preset}__{name}.png"
            capture(args.output / filename, (8, -10, top + 13), (0, 0, top - 0.2))
            if name in ("star", "gap", "stepping_stones"):
                capture(args.output / f"{preset}__{name}__top.png", (0, -0.01, top + 17), (0, 0, top))
            manifest["terrains"].append({"preset": preset, "name": name, "proportion": sub.proportion,
                "bounds": mesh.bounds.tolist(), "origin": np.asarray(origin).tolist(), "image": filename})
            stage.RemovePrim("/World/Tile")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("CATALOG_COMPLETE", flush=True)
finally:
    app.close()
