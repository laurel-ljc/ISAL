"""Raw-height diagnostics, independent of the simulator and label scoring."""

import torch


COUNT_NAMES = ("total_count", "finite_count", "lower_count", "upper_count", "invalid_count")
FRACTION_NAMES = ("lower_fraction", "upper_fraction", "invalid_fraction")
SCOPES = ("scan", "query")
DIAGNOSTIC_FIELDS = tuple(f"{scope}_{name}" for scope in SCOPES for name in (*COUNT_NAMES, *FRACTION_NAMES))


def raw_height_masks(relative_height: torch.Tensor, min_height: float, max_height: float) -> torch.Tensor:
    """Return (...,3,H,W) finite/lower/upper masks from (...,H,W) raw heights."""
    finite = torch.isfinite(relative_height)
    return torch.stack((finite, finite & (relative_height <= min_height),
                        finite & (relative_height >= max_height)), dim=-3)


def diagnostic_counts(masks: torch.Tensor, selection: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Count points, using an optional spatial subset (never duplicate border points)."""
    if masks.dtype != torch.bool or masks.shape[-3] != 3:
        raise ValueError("Diagnostic masks must be boolean (...,3,H,W).")
    finite, lower, upper = masks.unbind(-3)
    if selection is None:
        selection = torch.ones_like(finite)
    total = selection.sum(dim=(-2, -1))
    finite_count = (finite & selection).sum(dim=(-2, -1))
    return {"total_count": total, "finite_count": finite_count,
            "lower_count": (lower & selection).sum(dim=(-2, -1)),
            "upper_count": (upper & selection).sum(dim=(-2, -1)),
            "invalid_count": total - finite_count}


def diagnostic_fractions(counts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"lower_fraction": counts["lower_count"] / counts["finite_count"].clamp_min(1),
            "upper_fraction": counts["upper_count"] / counts["finite_count"].clamp_min(1),
            "invalid_fraction": counts["invalid_count"] / counts["total_count"].clamp_min(1)}
