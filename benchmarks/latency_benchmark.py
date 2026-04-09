"""
Latency benchmark for HandPoseNet inference pipeline.

Measures:
  - Preprocessing time (BGR→tensor)
  - Model forward pass time
  - Full pipeline end-to-end
  - Memory usage

Target: <50ms total @720p/30fps on mid-range CPU.

Usage:
    python benchmarks/latency_benchmark.py --device cpu --resolution 720p --runs 100
    python benchmarks/latency_benchmark.py --device cuda --fp16
"""

import argparse
import time
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import HandPoseNet


RESOLUTIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HandPoseNet latency benchmark")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--resolution", default="720p", choices=list(RESOLUTIONS.keys()))
    p.add_argument("--backbone", default="mobilenetv3", choices=["mobilenetv3", "lightweight"])
    p.add_argument("--fp16", action="store_true", help="Use FP16 (CUDA only)")
    p.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations (excluded)")
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


def benchmark_preprocessing(frame_bgr: np.ndarray, n: int) -> list[float]:
    """Measure BGR→tensor preprocessing time."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        times.append(time.perf_counter() - t0)
    return times


def benchmark_inference(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    n: int,
    warmup: int,
    device: torch.device,
) -> list[float]:
    """Measure model.infer() time (GPU-synced if CUDA)."""
    times = []
    model.eval()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.infer(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()

    for _ in range(n):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.infer(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return times


def report(label: str, times_s: list[float]) -> None:
    ms = [t * 1000 for t in times_s]
    print(f"\n{label}")
    print(f"  Mean:   {statistics.mean(ms):7.2f} ms")
    print(f"  Median: {statistics.median(ms):7.2f} ms")
    print(f"  P95:    {sorted(ms)[int(0.95*len(ms))]:7.2f} ms")
    print(f"  P99:    {sorted(ms)[int(0.99*len(ms))]:7.2f} ms")
    print(f"  Min:    {min(ms):7.2f} ms")
    print(f"  Max:    {max(ms):7.2f} ms")
    fps = 1000 / statistics.mean(ms)
    print(f"  FPS:    {fps:7.1f}")
    target_met = statistics.mean(ms) < 50.0
    print(f"  Target <50ms: {'PASS' if target_met else 'FAIL'}")


def main() -> None:
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Model
    model = HandPoseNet(
        backbone_name=args.backbone,
        pretrained_backbone=False,
        use_depth_head=True,
    ).to(device)
    model.eval()

    use_fp16 = args.fp16 and device.type == "cuda"
    if use_fp16:
        model = model.half()
        print("FP16 enabled")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Backbone: {args.backbone} | Parameters: {n_params:,}")

    # Synthetic input
    W, H = RESOLUTIONS[args.resolution]
    print(f"Resolution: {W}×{H}")
    frame_bgr = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    # Preprocessing benchmark
    pre_times = benchmark_preprocessing(frame_bgr, args.runs)
    report(f"Preprocessing (CPU) — {args.resolution}", pre_times)

    # Build inference tensor
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    if use_fp16:
        tensor = tensor.half()

    # Inference benchmark
    infer_times = benchmark_inference(model, tensor, args.runs, args.warmup, device)
    report(f"Model Inference ({device}) — {args.resolution}", infer_times)

    # End-to-end (preprocess + infer)
    e2e_ms = [p + i for p, i in zip(pre_times[:args.runs], infer_times)]
    report(f"End-to-End (preprocess + inference)", e2e_ms)

    # Memory
    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"\nGPU Peak Memory: {mem_mb:.1f} MB")
    else:
        import psutil, os
        proc = psutil.Process(os.getpid())
        mem_mb = proc.memory_info().rss / 1e6
        print(f"\nCPU RSS Memory: {mem_mb:.1f} MB")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
