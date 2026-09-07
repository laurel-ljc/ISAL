"""Bounded, explicitly CPU-side debug export; never used by the learning path."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from .scan_diagnostics import COUNT_NAMES, FRACTION_NAMES, SCOPES, DIAGNOSTIC_FIELDS


class InteractionRecorder:
    """Stream summary statistics for all samples; retain at most 1000 examples."""

    METRICS = ("target", "slip_mean", "tilt_change", "persistence", "survival", "peak_force", "observed_frames")

    def __init__(self, max_samples: int = 1000, *, bad_threshold: float = 0.35, good_threshold: float = 0.70):
        if not 0 <= max_samples <= 1000:
            raise ValueError("Debug retention must be between 0 and 1000 samples.")
        self.max_samples, self.bad_threshold, self.good_threshold = max_samples, bad_threshold, good_threshold
        self.count, self.saved = 0, 0
        self.parts: dict[str, list[torch.Tensor]] = {}
        self.moments = {key: [0.0, 0.0, float("inf"), -float("inf")] for key in (*self.METRICS, "query_x", "query_y")}
        self.histogram = [0] * 10
        self.categories = {"bad": 0, "middle": 0, "good": 0}
        self.partial_samples = 0
        self.latest_stats = {}
        self.diagnostic_samples = 0
        self.diagnostic_totals = {scope: dict.fromkeys(COUNT_NAMES, 0) for scope in SCOPES}

    def append(self, packet: dict[str, torch.Tensor], statistics: dict[str, torch.Tensor]) -> None:
        self.latest_stats = {key: int(value.detach().cpu()) for key, value in statistics.items()}
        valid = packet["valid"]
        # CPU conversion belongs only to this offline debugging boundary.
        batch = {key: value[valid].detach().cpu() for key, value in packet.items() if key != "valid"}
        count = batch["target"].shape[0]
        if count == 0:
            return
        # Normalize optional diagnostics so old and new packets can be mixed
        # without producing different column lengths in retained examples.
        available = batch.get("scan_diagnostics_available", torch.zeros(count, dtype=torch.bool))
        if not all(f"{scope}_{name}" in batch for scope in SCOPES for name in COUNT_NAMES):
            available = torch.zeros(count, dtype=torch.bool)
        batch["scan_diagnostics_available"] = available
        self.diagnostic_samples += int(available.sum())
        for scope in SCOPES:
            for name in COUNT_NAMES:
                key = f"{scope}_{name}"
                batch[key] = torch.where(available, batch.get(key, torch.zeros(count, dtype=torch.long)), 0)
                self.diagnostic_totals[scope][name] += int(batch[key].sum())
            for name in FRACTION_NAMES:
                numerator = "invalid_count" if name == "invalid_fraction" else name.replace("fraction", "count")
                denominator = "total_count" if name == "invalid_fraction" else "finite_count"
                batch[f"{scope}_{name}"] = batch[f"{scope}_{numerator}"] / batch[f"{scope}_{denominator}"].clamp_min(1)
        self.count += count
        self.partial_samples += int(batch["partial_window"].sum())
        values = {key: batch[key].flatten().double() for key in self.METRICS}
        values.update(query_x=batch["query_xy"][:, 0].double(), query_y=batch["query_xy"][:, 1].double())
        for key, value in values.items():
            m = self.moments[key]
            m[0] += float(value.sum())
            m[1] += float(value.square().sum())
            m[2], m[3] = min(m[2], float(value.min())), max(m[3], float(value.max()))
        target = values["target"]
        bins = torch.histc(target, bins=10, min=0, max=1).long().tolist()
        self.histogram = [a + b for a, b in zip(self.histogram, bins)]
        self.categories["bad"] += int((target < self.bad_threshold).sum())
        self.categories["middle"] += int(((target >= self.bad_threshold) & (target < self.good_threshold)).sum())
        self.categories["good"] += int((target >= self.good_threshold).sum())
        keep = min(count, self.max_samples - self.saved)
        if keep:
            for key, value in batch.items():
                self.parts.setdefault(key, []).append(value[:keep].clone())
            # Index columns identify the original fixed output slot.
            self.parts.setdefault("source_index", []).append(valid.nonzero().detach().cpu()[:keep].clone())
            self.saved += keep

    def write(self, output_dir: str | Path, *, metadata: dict) -> dict:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        metrics = {}
        for key, (total, squared, low, high) in self.moments.items():
            mean = total / self.count if self.count else None
            metrics[key] = {"mean": mean, "std": max(0, squared / self.count - mean**2)**0.5 if self.count else None,
                            "min": low if self.count else None, "max": high if self.count else None}
        summary = {"metadata": metadata, "total_samples": self.count, "saved_samples": self.saved,
                   "partial_samples": self.partial_samples, "counters": self.latest_stats,
                   "metrics": metrics, "target_histogram": {"edges": [i / 10 for i in range(11)], "counts": self.histogram},
                   "target_categories": self.categories}
        summary["scan_diagnostics"] = {"available": self.diagnostic_samples > 0,
                                       "samples": self.diagnostic_samples, "missing_samples": self.count - self.diagnostic_samples}
        for scope, counts in self.diagnostic_totals.items():
            # Ratios of accumulated point counts, not a mean of per-sample ratios.
            fractions = {"lower_fraction": counts["lower_count"] / max(counts["finite_count"], 1),
                         "upper_fraction": counts["upper_count"] / max(counts["finite_count"], 1),
                         "invalid_fraction": counts["invalid_count"] / max(counts["total_count"], 1)}
            summary["scan_diagnostics"][scope] = {
                **counts, **{key: value if self.diagnostic_samples else None for key, value in fractions.items()}}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        samples = {key: torch.cat(parts) for key, parts in self.parts.items()}
        torch.save(samples, output / "samples.pt")
        columns = ["env", "foot", "slot", "query_x", "query_y", "partial_window", *self.METRICS,
                   "scan_diagnostics_available", *DIAGNOSTIC_FIELDS]
        with (output / "samples.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for i in range(self.saved):
                env, foot, slot = samples["source_index"][i].tolist()
                row = {"env": env, "foot": foot, "slot": slot,
                       "query_x": float(samples["query_xy"][i, 0]), "query_y": float(samples["query_xy"][i, 1]),
                       "partial_window": bool(samples["partial_window"][i])}
                row.update({key: float(samples[key][i].item()) for key in self.METRICS})
                available = bool(samples["scan_diagnostics_available"][i])
                row["scan_diagnostics_available"] = available
                row.update({key: samples[key][i].item() if available else "" for key in DIAGNOSTIC_FIELDS})
                writer.writerow(row)
        return summary
