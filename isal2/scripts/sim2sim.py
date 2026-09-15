"""Run an ISAL2 ONNX actor in MuJoCo, with Windows XInput or a headless command."""
import argparse
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import sys
import time
try:
    from ._bootstrap import bootstrap
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("_isal2_bootstrap", Path(__file__).with_name("_bootstrap.py"))
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    bootstrap = _module.bootstrap

ROOT = bootstrap()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--terrain", choices=["flat", "rough", "rough_hard", "mixed"], default="mixed")
    parser.add_argument("--terrain-config", help="JSON overrides for terrain parameters")
    parser.add_argument("--controller-config", help="JSON overrides for XInput mapping")
    parser.add_argument("--camera-config", help="JSON overrides for the following/orbit camera")
    parser.add_argument("--controller-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=float, default=.5)
    parser.add_argument("--spawn-tile", type=int, default=0, help="Spawn tile index; default 0 is flat")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, help="Simulated seconds; default 10 headless, unlimited interactive")
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.headless and args.command is None:
        parser.error("--headless requires an explicit --command VX VY WZ")
    if not args.headless and args.command is not None:
        parser.error("--command is reserved for --headless; interactive commands come from XInput")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    from isal2.deployment.controller import CommandController, XInput
    from isal2.deployment.camera import FollowCamera
    from isal2.deployment.runtime import Simulator, load_policy
    from isal2.deployment.terrain import build_scene
    import numpy as np
    session, metadata = load_policy(args.model, args.metadata)
    output = Path(args.output or ROOT / "outputs" / "sim2sim" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=True)
    read_config = lambda path: json.loads(Path(path).read_text(encoding="utf-8")) if path else None
    model, terrain = build_scene(metadata, args.terrain, args.seed, args.difficulty,
                                 read_config(args.terrain_config), output / "scene")
    if args.spawn_tile not in range(len(terrain["tiles"])):
        parser.error("--spawn-tile is outside the generated tile range")
    tile = terrain["tiles"][args.spawn_tile]
    simulator = Simulator(model, metadata, session, tile["spawn_position"])
    active_tile = args.spawn_tile
    controller = CommandController(metadata["command_ranges"], read_config(args.controller_config))
    camera = FollowCamera(read_config(args.camera_config))
    duration = args.duration if args.duration is not None else (10 if args.headless else float("inf"))
    max_steps = int(np.ceil(duration / metadata["control_dt"])) if np.isfinite(duration) else float("inf")
    fixed = np.array(args.command or [0, 0, 0], dtype=float)
    if np.any(fixed < controller.bounds[:, 0]) or np.any(fixed > controller.bounds[:, 1]):
        parser.error("--command is outside the trained command ranges")
    keyboard_events = set()
    def keyboard(key):
        keyboard_events.add(key)
    if args.headless:
        context, pad = nullcontext(None), None
    else:
        import mujoco.viewer
        context = mujoco.viewer.launch_passive(model, simulator.data, key_callback=keyboard)
        pad = XInput(args.controller_index)
        print("Paused. XInput: Start resume/pause, A zero, Y reset, Back exit. Keyboard: Space pause, R reset, Esc exit.")
        print("Left stick: move. LT/RT: turn left/right. Right stick: orbit/look and hold angle. RB: smoothly return behind the robot.")
        print("Terrain bays (N/P: next/previous bay, teleport to its approach and pause):")
        print("\n".join(f"  {t['index']:2d}: {t['label']}" for t in terrain["tiles"]))
        print(f"Current bay: {active_tile} - {tile['label']}")
    error = None
    try:
        with context as viewer:
            if viewer:
                with viewer.lock():
                    camera.update(viewer.cam, simulator.data.xpos[simulator.base],
                                  simulator.data.xmat[simulator.base], None, 0.)
            camera_time = time.perf_counter()
            while simulator.steps < max_steps and (viewer is None or viewer.is_running()):
                start = time.perf_counter()
                pad_state = None if args.headless else pad.poll()
                command, events = (fixed, {}) if args.headless else controller.update(pad_state)
                keys = keyboard_events.copy()
                keyboard_events.difference_update(keys)
                if events.get("exit") or 256 in keys:
                    break
                if 32 in keys and controller.connected:
                    controller.paused = not controller.paused
                switch_tile = viewer is not None and (78 in keys or 80 in keys)
                if switch_tile:
                    active_tile = (active_tile + (1 if 78 in keys else -1)) % len(terrain["tiles"])
                    tile = terrain["tiles"][active_tile]
                    simulator.spawn_offset[:] = tile["spawn_position"]
                    print(f"Current bay: {active_tile} - {tile['label']}; paused. Press Start to resume.")
                if events.get("reset") or 82 in keys or switch_tile:
                    simulator.reset()
                    camera.reset()
                    controller.paused = True
                if args.headless or not controller.paused:
                    fallen = simulator.step(command)
                    if fallen:
                        simulator.falls += 1
                        if args.headless:
                            simulator.reset()
                        else:
                            controller.paused = True
                            print("Fall detected; paused. Press Y/R to reset.")
                if viewer:
                    now = time.perf_counter()
                    with viewer.lock():
                        camera.update(viewer.cam, simulator.data.xpos[simulator.base],
                                      simulator.data.xmat[simulator.base], pad_state, min(now - camera_time, .1))
                    camera_time = now
                    viewer.sync()
                    time.sleep(max(0, metadata["control_dt"] - (time.perf_counter() - start)))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result = {**simulator.result(), "error": error, "finite": error is None,
                  "active_tile": active_tile,
                  "model_type": metadata["model_type"], "terrain": args.terrain,
                  "arguments": vars(args), "providers": session.get_providers(),
                  "forbidden_imports": [name for name in ("torch", "rsl_rl", "isaaclab", "isal", "robolab") if name in sys.modules]}
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output / "telemetry.json").write_text(json.dumps(simulator.records), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
