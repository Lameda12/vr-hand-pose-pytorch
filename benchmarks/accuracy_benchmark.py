"""
Accuracy benchmark: PCK@0.2 + AUC on a validation set.

Requires:
  - A trained checkpoint (best.pt from export_checkpoint.py)
  - FreiHAND dataset at a known path

Usage:
    python benchmarks/accuracy_benchmark.py \
        --checkpoint checkpoints/best.pt \
        --data /data/freihand \
        --split val \
        --max-samples 100

Outputs:
  - Metrics printed to stdout
  - Appended to benchmarks/accuracy_results.csv
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import HandPoseNet
from src.evaluate import PoseEvaluator, save_metrics_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Accuracy benchmark on FreiHAND val set")
    p.add_argument("--checkpoint", required=True, help="Path to best.pt (exported)")
    p.add_argument("--data",       required=True, help="FreiHAND dataset root")
    p.add_argument("--config",     default="configs/model_config.yaml")
    p.add_argument("--split",      default="val", choices=["train", "val", "eval"])
    p.add_argument("--max-samples",type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device",     default="auto")
    p.add_argument("--output",     default="benchmarks/accuracy_results.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )
    print(f"Device: {device}")

    # Load model
    model = HandPoseNet(
        backbone_name=cfg["model"]["backbone"],
        pretrained_backbone=False,
        use_depth_head=cfg["model"]["use_depth_head"],
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).eval()

    if cfg["inference"]["fp16"] and device.type == "cuda":
        model = model.half()

    print(f"Loaded checkpoint: {args.checkpoint}")
    if "epoch" in ckpt:
        print(f"  Epoch: {ckpt['epoch']}")
    if "val_pck_02" in ckpt and ckpt["val_pck_02"] is not None:
        print(f"  Saved val PCK@0.2: {ckpt['val_pck_02']:.4f}")

    # Dataloader
    from src.data.freihand import build_dataloader
    loader = build_dataloader(
        args.data,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=tuple(cfg["model"]["input_size"]),
        max_samples=args.max_samples,
    )
    print(f"Evaluating on {len(loader.dataset)} samples ({args.split})...")

    # Evaluate
    evaluator = PoseEvaluator()
    t0 = time.perf_counter()

    for batch in loader:
        images = batch["image"].to(device)
        if cfg["inference"]["fp16"] and device.type == "cuda":
            images = images.half()

        with torch.no_grad():
            preds = model.infer(images)

        # Move preds to CPU for evaluator
        preds_cpu = {k: v.float().cpu() for k, v in preds.items()}
        batch_cpu = {k: v.cpu() for k, v in batch.items()}
        evaluator.update(preds_cpu, batch_cpu)

    elapsed = time.perf_counter() - t0
    metrics = evaluator.compute()

    print(f"\n{'='*40}")
    print(f"Accuracy Results ({args.split} split)")
    print(f"{'='*40}")
    print(metrics)
    print(f"Wall time: {elapsed:.1f}s  ({elapsed/metrics.n_samples*1000:.1f}ms/sample)")

    # PCK target check
    target_met = metrics.pck_at_02 >= 0.6
    print(f"\nTarget PCK@0.2 >0.60: {'PASS' if target_met else 'FAIL'}")

    # Save to CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    write_header = not out_path.exists()

    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "checkpoint", "split", "n_samples",
            "pck_at_02", "auc", "elapsed_s", "target_met",
        ])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "checkpoint": args.checkpoint,
            "split":      args.split,
            "n_samples":  metrics.n_samples,
            "pck_at_02":  f"{metrics.pck_at_02:.6f}",
            "auc":        f"{metrics.auc:.6f}",
            "elapsed_s":  f"{elapsed:.2f}",
            "target_met": target_met,
        })

    print(f"Results appended to {out_path}")


if __name__ == "__main__":
    main()
