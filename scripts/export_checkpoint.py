"""
Export best PyTorch Lightning checkpoint to a plain .pt state dict.

Lightning saves .ckpt files (includes optimizer state, epoch, etc.).
This script extracts just the model weights for deployment.

Usage:
    python scripts/export_checkpoint.py \
        --ckpt checkpoints/best.ckpt \
        --output checkpoints/best.pt \
        --config configs/model_config.yaml

Output best.pt can be loaded directly:
    model = HandPoseNet(...)
    model.load_state_dict(torch.load("checkpoints/best.pt"))
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Lightning checkpoint to plain .pt")
    p.add_argument("--ckpt",   required=True, help="Path to .ckpt file")
    p.add_argument("--output", required=True, help="Output .pt path")
    p.add_argument("--config", default="configs/model_config.yaml")
    p.add_argument("--verify", action="store_true",
                   help="Run a forward pass to verify the exported weights")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Lightning stores model weights under "state_dict" with "model." prefix
    if "state_dict" not in ckpt:
        print("ERROR: 'state_dict' key not found — is this a Lightning checkpoint?")
        sys.exit(1)

    raw_sd = ckpt["state_dict"]
    # Strip "model." prefix added by LightningModule
    model_sd = {
        k[len("model."):]: v
        for k, v in raw_sd.items()
        if k.startswith("model.")
    }
    if not model_sd:
        print("ERROR: No keys with 'model.' prefix found.")
        print(f"Keys found: {list(raw_sd.keys())[:10]}")
        sys.exit(1)

    # Metadata to embed in the export
    epoch = ckpt.get("epoch", -1)
    val_pck = None
    if "callbacks" in ckpt:
        for v in ckpt["callbacks"].values():
            if isinstance(v, dict) and "best_model_score" in v:
                val_pck = float(v["best_model_score"])
                break

    output = {
        "model_state_dict": model_sd,
        "config": cfg["model"],
        "epoch": epoch,
        "val_pck_02": val_pck,
    }

    # Verify by loading into HandPoseNet
    if args.verify:
        from src.models import HandPoseNet
        model = HandPoseNet(
            backbone_name=cfg["model"]["backbone"],
            pretrained_backbone=False,
            use_depth_head=cfg["model"]["use_depth_head"],
        )
        missing, unexpected = model.load_state_dict(model_sd, strict=True)
        if missing:
            print(f"WARNING: missing keys: {missing}")
        if unexpected:
            print(f"WARNING: unexpected keys: {unexpected}")

        model.eval()
        W, H = cfg["model"]["input_size"]
        x = torch.randn(1, 3, H, W)
        with torch.no_grad():
            out = model.infer(x)
        print(f"Forward pass OK — keypoints shape: {out['keypoints'].shape}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Exported: {out_path}  ({size_mb:.1f} MB)")
    print(f"  Epoch: {epoch}")
    if val_pck is not None:
        print(f"  Val PCK@0.2: {val_pck:.4f}")


if __name__ == "__main__":
    main()
