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
args.kit_args += (f" --portable-root={args.output.as_posix()}/kit --/log/file={args.output.as_posix()}/kit.log"
                 f" --/app/userConfigPath={args.output.as_posix()}/user.config.json"
                 f" --/app/userConfigPathOverride={args.output.as_posix()}/user.config.json"
                 f" --/app/tokens/cache={args.output.as_posix()}/cache")
launcher = AppLauncher(args, fast_shutdown=False)
app = launcher.app
print("APP_READY", flush=True)
import sys
import traceback

def report_error(kind, value, tb):
    message = "".join(traceback.format_exception(kind, value, tb))
    (args.output / "error.txt").write_text(message, encoding="utf-8")
    print(message, flush=True)

sys.excepthook = report_error

import numpy as np
import trimesh
from PIL import Image
import omni.usd
import omni.replicator.core as rep
from pxr import UsdGeom, UsdLux, Gf
from isaaclab.terrains.height_field import HfTerrainBaseCfg

source = Path(__file__).resolve().parents[1] / "tasks/base/terrain_generator_cfg.py"
spec = importlib.util.spec_from_file_location("catalog_terrain_cfg", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
light = UsdLux.DomeLight.Define(stage, "/World/Sky")
light.CreateIntensityAttr(650)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(2500)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(25, -35, -30))
camera_path = "/World/CatalogCamera"
camera = UsdGeom.Camera.Define(stage, camera_path)
camera.CreateFocalLengthAttr(30)
camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 10000))
camera_transform = UsdGeom.Xformable(camera).AddTransformOp()
product = rep.create.render_product(camera_path, (800, 700))
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
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    camera_transform.Set(view.GetInverse())
    print(f"RENDER_START {path.name}", flush=True)
    for _ in range(40):
        app.update()
    frame = np.asarray(rgb.get_data())
    if frame.size == 0:
        raise RuntimeError("Empty rendered image")
    if np.std(frame[:, :, :3].astype(float), axis=(0, 1)).max() < 1:
        raise RuntimeError("Rendered frame has no visible geometry")
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
            np.savez_compressed(args.output / f"{preset}__{name}.npz", vertices=mesh.vertices, faces=mesh.faces)
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
except Exception:
    report_error(*sys.exc_info())
finally:
    app.close(skip_cleanup=True)
