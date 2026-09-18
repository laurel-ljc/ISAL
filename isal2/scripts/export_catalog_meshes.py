"""Export exact terrain meshes without starting a rendering viewport."""
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
args.kit_args += f" --/app/userConfigPath={args.output.as_posix()}/export_user.json --/log/file={args.output.as_posix()}/export_kit.log"
launcher = AppLauncher(args)
app = launcher.app
import numpy as np
import trimesh
from isaaclab.terrains.height_field import HfTerrainBaseCfg

try:
    source = Path(__file__).resolve().parents[1] / "deprecated_tasks/base/terrain_generator_cfg.py"
    spec = importlib.util.spec_from_file_location("catalog_cfg", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = {"source": str(source), "difficulty": 0.5, "seed": 42, "terrains": []}
    for preset, cfg in [("rough", module.ROUGH_TERRAINS_CFG), ("rough_hard", module.ROUGH_HARD_TERRAINS_CFG)]:
        for name, original in cfg.sub_terrains.items():
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
            stem = f"{preset}__{name}"
            np.savez_compressed(args.output / f"{stem}.npz", vertices=mesh.vertices, faces=mesh.faces)
            mesh.export(args.output / f"{stem}.obj")
            manifest["terrains"].append({"preset": preset, "name": name, "proportion": sub.proportion,
                "bounds": mesh.bounds.tolist(), "origin": np.asarray(origin).tolist(), "image": f"{stem}.png"})
            print(f"EXPORTED {stem}", flush=True)
    (args.output / "mesh_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("EXPORT_COMPLETE", flush=True)
except Exception:
    import traceback
    traceback.print_exc()
finally:
    app.close()
