"""
Training script for HandPoseNet on FreiHAND.

Uses PyTorch Lightning for training loop management.
Config-driven: all hyperparams from configs/model_config.yaml.

Usage:
    python -m src.train --config configs/model_config.yaml --data /data/freihand
    python -m src.train --config configs/model_config.yaml --data /data/freihand --fast-dev-run

Outputs:
    checkpoints/best.pt   — best val PCK@0.2
    checkpoints/last.pt   — most recent epoch
    logs/metrics.csv      — epoch-level metrics log
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

# Allow running as a module from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HandPoseNet on FreiHAND")
    p.add_argument("--config", default="configs/model_config.yaml")
    p.add_argument("--data", required=True, help="Path to freihand/ dataset root")
    p.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap dataset size for quick iteration")
    p.add_argument("--fast-dev-run", action="store_true",
                   help="Run 1 train + 1 val batch only (smoke test)")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_trainer(cfg: Dict, args: argparse.Namespace):
    """
    Build PyTorch Lightning Trainer from config.

    Deferred import: pytorch_lightning optional dep (extras[train]).
    """
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
        from pytorch_lightning.loggers import WandbLogger, CSVLogger
    except ImportError:
        raise ImportError(
            "pytorch-lightning required for training. "
            "Install with: pip install 'turin-hand-pose[train]'"
        )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="best",
            monitor="val/pck_02",
            mode="max",
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/pck_02",
            patience=15,
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    loggers = [CSVLogger(args.log_dir, name="turin")]
    if not args.no_wandb:
        try:
            loggers.append(WandbLogger(project="turin-hand-pose", save_dir=args.log_dir))
        except Exception:
            logger.warning("W&B not available — falling back to CSV only")

    train_cfg = cfg["training"]
    return pl.Trainer(
        max_epochs=1 if args.fast_dev_run else train_cfg["epochs"],
        accelerator="auto",
        precision="16-mixed" if cfg["inference"]["fp16"] else "32",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
        gradient_clip_val=1.0,
    )


class HandPoseLightningModule:
    """
    PyTorch Lightning module for HandPoseNet training.

    Wraps model, loss, optimizer, and scheduler.
    Logs train/val loss and PCK@0.2 to W&B/CSV.

    Constructor-level deferred import of pl.LightningModule
    to keep train.py importable without pytorch_lightning installed.
    """

    @staticmethod
    def build(cfg: Dict, args: argparse.Namespace):
        """
        Factory: returns a pl.LightningModule instance.

        Deferred import keeps the module importable without pl installed.
        """
        import pytorch_lightning as pl
        from src.models import HandPoseNet
        from src.models.losses import HandPoseLoss
        from src.data.freihand import build_dataloader
        from src.evaluate import PoseEvaluator, save_metrics_csv

        class _Module(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.save_hyperparameters(cfg)
                self.model = HandPoseNet(
                    backbone_name=cfg["model"]["backbone"],
                    pretrained_backbone=cfg["model"]["pretrained_backbone"],
                    use_depth_head=cfg["model"]["use_depth_head"],
                )
                loss_cfg = cfg["training"]["loss"]
                self.criterion = HandPoseLoss(
                    lambda_heatmap=loss_cfg["lambda_heatmap"],
                    lambda_depth=loss_cfg["lambda_depth"],
                )
                self._evaluator = PoseEvaluator()
                self._log_dir = args.log_dir
                self._data_root = args.data
                self._max_samples = args.max_samples

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                images = batch["image"]
                preds = self.model(images)
                losses = self.criterion(preds, batch)
                self.log("train/loss_total", losses["total"], prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/loss_det",   losses["det"],   on_step=False, on_epoch=True)
                self.log("train/loss_hm",    losses["heatmap"], on_step=False, on_epoch=True)
                if "depth" in losses:
                    self.log("train/loss_depth", losses["depth"], on_step=False, on_epoch=True)
                return losses["total"]

            def validation_step(self, batch, batch_idx):
                images = batch["image"]
                with torch.no_grad():
                    preds = self.model.infer(images)
                    losses_raw = self.model(images)
                    losses = self.criterion(losses_raw, batch)

                self.log("val/loss_total", losses["total"], prog_bar=True, on_epoch=True)
                self._evaluator.update(preds, batch)
                return losses["total"]

            def on_validation_epoch_end(self):
                metrics = self._evaluator.compute()
                self.log("val/pck_02", metrics.pck_at_02, prog_bar=True)
                self.log("val/auc",    metrics.auc)
                self._evaluator.reset()

                save_metrics_csv(
                    metrics,
                    Path(self._log_dir) / "metrics.csv",
                    self.current_epoch,
                )
                logger.info(f"\nEpoch {self.current_epoch} validation:\n{metrics}")

            def configure_optimizers(self):
                t_cfg = cfg["training"]
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=t_cfg["learning_rate"],
                    weight_decay=t_cfg["weight_decay"],
                )
                sched_name = t_cfg.get("lr_scheduler", "cosine")
                if sched_name == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=t_cfg["epochs"],
                        eta_min=1e-6,
                    )
                elif sched_name == "step":
                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, step_size=30, gamma=0.1
                    )
                else:
                    return optimizer
                return {"optimizer": optimizer, "lr_scheduler": scheduler}

            def train_dataloader(self):
                return build_dataloader(
                    self._data_root, "train",
                    batch_size=cfg["training"]["batch_size"],
                    num_workers=min(4, os.cpu_count() or 1),
                    image_size=tuple(cfg["model"]["input_size"]),
                    max_samples=self._max_samples,
                )

            def val_dataloader(self):
                return build_dataloader(
                    self._data_root, "val",
                    batch_size=cfg["training"]["batch_size"],
                    num_workers=min(4, os.cpu_count() or 1),
                    image_size=tuple(cfg["model"]["input_size"]),
                    max_samples=self._max_samples // 10 if self._max_samples else None,
                )

        return _Module()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_config(args.config)

    module = HandPoseLightningModule.build(cfg, args)
    trainer = build_trainer(cfg, args)

    ckpt = args.checkpoint
    trainer.fit(module, ckpt_path=ckpt)

    logger.info("Training complete.")
    logger.info(f"Best checkpoint: {args.output_dir}/best.ckpt")


if __name__ == "__main__":
    main()
