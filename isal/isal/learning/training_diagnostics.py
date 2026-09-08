"""Read-only update probes and contact-quality metrics."""

import torch
import torch.nn.functional as F


def gaussian_kl(old_mean, old_std, mean, std):
    return (torch.log(std/old_std) + (old_std.square()+(old_mean-mean).square())/(2*std.square())-.5).sum(-1)


def tensor_summary(value):
    flat = value.detach().float().flatten()
    return {"mean": float(flat.mean()), "min": float(flat.min()), "max": float(flat.max()),
            "histogram_10_bins": torch.histc(flat, bins=10, min=0, max=1).long().cpu().tolist()}


@torch.no_grad()
def auxiliary_metrics(model, batch, beta):
    if not batch or batch["target"].numel() == 0:
        return {"prediction_count": 0, "correlation_valid": False, "correlation": None}
    pred, target = model.predict_affordance(batch), batch["target"]
    p, t = pred.flatten(), target.flatten()
    pd, td = p.double(), t.double()
    pc, tc = pd-pd.mean(), td-td.mean()
    denom = pc.square().sum().sqrt()*tc.square().sum().sqrt()
    valid = bool(p.numel() >= 2 and p.max()-p.min() > 1e-8 and t.max()-t.min() > 1e-8 and denom > 1e-12)
    result = {"prediction_count": p.numel(), "prediction": tensor_summary(pred), "target": tensor_summary(target),
              "mae": float((pred-target).abs().mean()), "smooth_l1": float(F.smooth_l1_loss(pred,target,beta=beta)),
              "correlation_valid": valid, "correlation": float((pc*tc).sum()/denom) if valid else None}
    groups = {"left": batch["foot_side"][:, 0] > .5, "right": batch["foot_side"][:, 1] > .5}
    if "partial_window" in batch:
        groups.update(partial=batch["partial_window"].bool(), full=~batch["partial_window"].bool())
    if "survival" in batch:
        groups.update(survived=batch["survival"] > .5, failed=batch["survival"] <= .5)
    result["groups"] = {key: {"count": int(mask.sum()),
                                "target_mean": float(t[mask].mean()) if mask.any() else None,
                                "mae": float((p[mask]-t[mask]).abs().mean()) if mask.any() else None}
                        for key, mask in groups.items()}
    return result


@torch.no_grad()
def capture_probe(model, obs, auxiliary=None):
    result = {"mean": model.act_inference(obs).detach().clone(), "std": model.std.detach().clone()}
    if auxiliary:
        result["aux_prediction"] = model.predict_affordance(auxiliary).detach().clone()
    if getattr(model, "input_mode", None) == "predicted":
        result["grid"] = model.predict_affordance_grid(obs).detach().clone()
    return result


def contact_label_metrics(batch):
    """All retained labels, separate from the bounded prediction diagnostic probe."""
    if not batch:
        return {"count":0}
    result = {"count":batch["target"].shape[0], "target":tensor_summary(batch["target"])}
    for key in ("slip_score","tilt_score","persistence","survival"):
        if key in batch:
            result[key] = tensor_summary(batch[key])
    result["left_count"] = int((batch["foot_side"][:,0]>.5).sum())
    result["right_count"] = int((batch["foot_side"][:,1]>.5).sum())
    if "partial_window" in batch:
        result["partial_count"] = int(batch["partial_window"].sum())
        result["full_count"] = result["count"]-result["partial_count"]
    return result


def probe_changes(before, after):
    result = {"action_mean_drift": float((before["mean"]-after["mean"]).abs().mean()),
              "post_update_kl": float(gaussian_kl(before["mean"], before["std"], after["mean"], after["std"]).mean())}
    for key in ("aux_prediction", "grid"):
        if key in before:
            result[f"{key}_drift"] = float((before[key]-after[key]).abs().mean())
    return result
