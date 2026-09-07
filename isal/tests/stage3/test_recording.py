import csv
import json

import pytest
import torch

from isal.interaction.recording import InteractionRecorder


def test_recording_counts_all_samples_but_bounds_saved_examples(tmp_path):
    prefix = (1, 2, 2)
    packet = {"valid": torch.ones(prefix, dtype=torch.bool), "target": torch.tensor([0., 0.5, 0.8, 1.]).reshape(*prefix, 1),
              "query_xy": torch.zeros((*prefix, 2)), "height_scan": torch.zeros((*prefix, 1, 3, 3)),
              "partial_window": torch.zeros(prefix, dtype=torch.bool)}
    for key in InteractionRecorder.METRICS:
        if key != "target":
            packet[key] = torch.ones(prefix)
    recorder = InteractionRecorder(max_samples=3)
    for _ in range(2):
        recorder.append(packet, {"samples": torch.tensor(8)})
    result = recorder.write(tmp_path, metadata={"synthetic": True})
    assert result["total_samples"] == 8
    assert result["saved_samples"] == 3
    assert result["target_categories"] == {"bad": 2, "middle": 2, "good": 4}
    assert sum(result["target_histogram"]["counts"]) == 8
    assert result["metrics"]["target"]["mean"] == pytest.approx(0.575)
    assert len(torch.load(tmp_path / "samples.pt", weights_only=True)["target"]) == 3
    with (tmp_path / "samples.csv").open(encoding="utf-8", newline="") as file:
        assert len(list(csv.DictReader(file))) == 3
    assert json.loads((tmp_path / "summary.json").read_text())["metadata"]["synthetic"]


def test_empty_debug_export_is_valid_json_and_csv(tmp_path):
    result = InteractionRecorder().write(tmp_path, metadata={"synthetic": True})
    assert result["metrics"]["target"]["mean"] is None
    assert result["saved_samples"] == result["total_samples"] == 0
    assert torch.load(tmp_path / "samples.pt", weights_only=True) == {}
    with pytest.raises(ValueError):
        InteractionRecorder(1001)


def test_diagnostic_point_weighting_and_mixed_legacy_packets(tmp_path):
    packet = {"valid": torch.ones((1, 2, 1), dtype=torch.bool),
              "target": torch.ones(1, 2, 1, 1), "query_xy": torch.zeros(1, 2, 1, 2),
              "partial_window": torch.zeros(1, 2, 1, dtype=torch.bool)}
    for name in InteractionRecorder.METRICS:
        if name != "target":
            packet[name] = torch.ones(1, 2, 1)
    recorder = InteractionRecorder(6)
    recorder.append(packet, {})  # retained old examples must align with new fields
    packet["scan_diagnostics_available"] = torch.ones(1, 2, 1, dtype=torch.bool)
    for scope in ("scan", "query"):
        for name, values in {"total_count": [9, 4], "finite_count": [8, 4], "lower_count": [4, 0],
                             "upper_count": [0, 1], "invalid_count": [1, 0]}.items():
            packet[f"{scope}_{name}"] = torch.tensor(values).reshape(1, 2, 1)
    recorder.append(packet, {})
    summary = recorder.write(tmp_path, metadata={"synthetic": True})
    diagnostics = summary["scan_diagnostics"]
    assert diagnostics["samples"] == diagnostics["missing_samples"] == 2
    for scope in ("scan", "query"):
        assert diagnostics[scope]["lower_fraction"] == pytest.approx(4 / 12)
        assert diagnostics[scope]["invalid_fraction"] == pytest.approx(1 / 13)
    saved = torch.load(tmp_path / "samples.pt", weights_only=True)
    assert all(len(value) == 4 for value in saved.values())
    assert saved["scan_diagnostics_available"].tolist() == [False, False, True, True]
    with (tmp_path / "samples.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["scan_lower_fraction"] == ""
    assert float(rows[2]["scan_lower_fraction"]) == .5


def test_empty_diagnostics_are_unavailable(tmp_path):
    summary = InteractionRecorder().write(tmp_path, metadata={})
    assert summary["scan_diagnostics"]["available"] is False
    assert summary["scan_diagnostics"]["scan"]["total_count"] == 0
    assert summary["scan_diagnostics"]["query"]["lower_fraction"] is None
