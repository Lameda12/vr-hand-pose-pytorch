"""
Torch profiler wrapper for inference loop analysis.

Identifies the top time-consuming ops in the full forward pass.
Use this to find bottlenecks before optimizing.

Usage:
    python benchmarks/profile_inference.py --device cuda --resolution 720p
    # Outputs: profiler trace + top-20 ops table + chrome://tracing JSON

Output files:
    benchmarks/profile_trace.json   — open in chrome://tracing
    benchmarks/profile_summary.txt  — top ops by self_cuda_time_total
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import HandPoseNet

RESOLUTIONS = {
    "480p":  (640, 480),
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile HandPoseNet inference")
    p.add_argument("--device",     default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--resolution", default="720p", choices=list(RESOLUTIONS.keys()))
    p.add_argument("--backbone",   default="mobilenetv3")
    p.add_argument("--fp16",       action="store_true")
    p.add_argument("--warmup",     type=int, default=5)
    p.add_argument("--active",     type=int, default=10,
                   help="Number of profiled iterations")
    p.add_argument("--output-dir", default="benchmarks")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )
    print(f"Profiling on: {device}")

    model = HandPoseNet(backbone_name=args.backbone, pretrained_backbone=False)
    model.to(device).eval()
    if args.fp16 and device.type == "cuda":
        model = model.half()

    W, H = RESOLUTIONS[args.resolution]
    frame = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    if args.fp16 and device.type == "cuda":
        tensor = tensor.half()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    trace_path = str(out_dir / "profile_trace.json")
    summary_path = out_dir / "profile_summary.txt"

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    schedule = torch.profiler.schedule(
        wait=0,
        warmup=args.warmup,
        active=args.active,
        repeat=1,
    )

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out_dir)),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for _ in range(args.warmup + args.active):
            with torch.no_grad():
                model.infer(tensor)
            prof.step()

    # Sort by CUDA time (or CPU time if no CUDA)
    sort_key = (
        "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    )
    table = prof.key_averages().table(sort_by=sort_key, row_limit=20)
    print("\n" + table)

    with open(summary_path, "w") as f:
        f.write(f"Device: {device}  Resolution: {W}×{H}  FP16: {args.fp16}\n\n")
        f.write(table)

    # Also export chrome trace
    prof.export_chrome_trace(trace_path)

    print(f"\nTrace:   {trace_path}  (open in chrome://tracing)")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
