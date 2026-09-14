"""Real viewer smoke test with synthetic XInput transport, not a hardware gamepad test."""
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _bootstrap import bootstrap
bootstrap()
from isal2.deployment.controller import XInput


class SyntheticPad:
    def __init__(self, index):
        self.polls = 0

    def poll(self):
        self.polls += 1
        # Resume, pause, reset, resume, then exit through the real application loop.
        buttons = {2: 16, 8: 16, 10: 0x8000, 12: 16, 28: 32}.get(self.polls, 0)
        return {"buttons": buttons, "lx": 0, "ly": 10000, "rx": 0, "ry": 0}


if __name__ == "__main__":
    connected = [i for i in range(4) if XInput(i).poll() is not None]
    output = ROOT / "outputs" / "deployment_validation" / "viewer"
    sys.argv = ["sim2sim.py", "--model", str(ROOT / "outputs/rpo_affordance/aff_gate_acceptance/export/model.onnx"),
                "--output", str(output)]
    with patch("isal2.deployment.controller.XInput", SyntheticPad):
        runpy.run_path(str(ROOT / "scripts/sim2sim.py"), run_name="__main__")
    result = json.loads((output / "result.json").read_text())
    assert result["reset_count"] == 1 and result["control_steps"] == 22, result
    result["synthetic_input"] = True
    result["physical_controller_indices"] = connected
    (output / "viewer_check.json").write_text(json.dumps(result, indent=2))
    print("Viewer opened, paused, reset, resumed and closed; input was SYNTHETIC.")
