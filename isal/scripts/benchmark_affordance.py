"""Measure the complete Stage 4B gate=1 inference path; never updates weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from tensordict import TensorDict

from export_actor import load_checkpoint_model


def benchmark(checkpoint: str, device: str, warmup: int, repeats: int) -> dict:
    model = load_checkpoint_model(checkpoint).to(device)
    if model.get_extra_state().get("stage") != "4B" or model.input_mode != "predicted":
        raise ValueError("Benchmark requires a Stage 4B predicted-input checkpoint.")
    model.set_affordance_input_gate(1.0)
    layout = model.get_extra_state()["layout"]
    rays = layout["grid_shape"][0] * layout["grid_shape"][1]
    policy = torch.zeros(1, layout["actor_history_length"] * layout["actor_frame_dim"], device=device)
    policy[:, -layout["actor_frame_dim"] + 5] = -model.context_scales[1]
    obs = TensorDict({"policy": policy, "height_scan": torch.full((1, rays), -1.5, device=device)},
                     batch_size=[1], device=device)
    is_cuda = torch.device(device).type == "cuda"
    with torch.inference_mode():
        for _ in range(warmup):
            model.act_inference(obs)
        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device) if is_cuda else None
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            action = model.act_inference(obs)
            if is_cuda:
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - started) * 1000)
        peak = torch.cuda.max_memory_allocated(device) if is_cuda else None
        if not torch.isfinite(action).all():
            raise RuntimeError("Non-finite benchmark output.")
    return {
        "device": device,
        "device_name": torch.cuda.get_device_name(device) if is_cuda else "CPU",
        "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
        "batch_size": 1, "input_gate": float(model.input_gate),
        "query_chunk_size": model.query_chunk_size, "grid_shape": layout["grid_shape"],
        "warmup": warmup, "repeats": repeats, "training_started": False,
        "latency_ms": {"median": statistics.median(timings), "mean": statistics.mean(timings),
                       "p95": sorted(timings)[min(repeats - 1, int(0.95 * repeats))]},
        "cuda_baseline_allocated_bytes": baseline, "cuda_peak_allocated_bytes": peak,
        "cuda_incremental_peak_bytes": peak - baseline if is_cuda else None,
        "parameters": {"model": sum(p.numel() for p in model.parameters()),
                       "actor_export": sum(p.numel() for p in model.make_actor_export().parameters()),
                       "projection": sum(p.numel() for p in model.affordance_projection.parameters())},
        "measurement": "Synthetic fixed observation; complete act_inference; CUDA synchronized host latency. "
                       "Peak allocation includes the resident full model; not process/driver memory or learning evidence.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1 or args.threads < 1:
        parser.error("warmup, repeats and threads must be positive")
    torch.set_num_threads(args.threads)
    results = [benchmark(args.checkpoint, device, args.warmup, args.repeats) for device in args.devices]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
