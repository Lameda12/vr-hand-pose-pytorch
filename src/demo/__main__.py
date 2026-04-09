"""
python -m src.demo [--camera N] [--checkpoint PATH] [--backbone NAME] [--no-fp16]
"""

import argparse
from .webcam_demo import WebcamDemo


def main() -> None:
    p = argparse.ArgumentParser(description="Turin hand pose webcam demo")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--backbone", default="mobilenetv3", choices=["mobilenetv3", "lightweight"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    args = p.parse_args()

    demo = WebcamDemo(
        model_path=args.checkpoint,
        backbone=args.backbone,
        device=args.device,
    )
    demo.run(camera_idx=args.camera)


if __name__ == "__main__":
    main()
