"""
Evaluation metrics for hand pose estimation.

Metrics:
  - PCK  (Percentage of Correct Keypoints) — primary 2D accuracy metric
  - MPJPE (Mean Per-Joint Position Error) — standard 3D metric (mm)
  - AUC  (Area Under PCK curve) — threshold-independent summary
  - Tracking ID switch rate — temporal consistency

All metrics logged to CSV and (optionally) Weights & Biases.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# PCK thresholds as fraction of hand bounding box diagonal
PCK_THRESHOLDS = np.linspace(0.05, 0.5, 10)   # 10 points for AUC


@dataclass
class EvalMetrics:
    """
    Aggregated evaluation results for one epoch.

    Attributes:
        pck_at_02:    PCK@0.2 — primary metric (target >0.6)
        pck_curve:    PCK values at each threshold in PCK_THRESHOLDS
        auc:          area under PCK curve [0, 1]
        mpjpe_mm:     mean per-joint position error in mm (if depth available)
        n_samples:    number of evaluated samples
    """
    pck_at_02: float = 0.0
    pck_curve: np.ndarray = field(default_factory=lambda: np.zeros(len(PCK_THRESHOLDS)))
    auc: float = 0.0
    mpjpe_mm: Optional[float] = None
    n_samples: int = 0

    def __str__(self) -> str:
        lines = [
            f"Samples:   {self.n_samples}",
            f"PCK@0.2:   {self.pck_at_02:.4f}  (target >0.60)",
            f"AUC:       {self.auc:.4f}",
        ]
        if self.mpjpe_mm is not None:
            lines.append(f"MPJPE:     {self.mpjpe_mm:.2f} mm")
        return "\n".join(lines)


def compute_pck(
    pred_kp: np.ndarray,
    gt_kp: np.ndarray,
    visibility: np.ndarray,
    threshold_frac: float = 0.2,
) -> float:
    """
    Percentage of Correct Keypoints @ threshold.

    Threshold is threshold_frac × hand bounding box diagonal.

    Args:
        pred_kp:        [21, 2] predicted (u, v) in image pixels
        gt_kp:          [21, 2] ground truth (u, v) in image pixels
        visibility:     [21]    float {0, 1}
        threshold_frac: fraction of bbox diagonal

    Returns:
        PCK in [0, 1]
    """
    vis = visibility > 0.5
    if vis.sum() == 0:
        return 0.0

    # Bounding box diagonal of GT visible keypoints
    gt_vis = gt_kp[vis]
    bbox_diag = np.linalg.norm(gt_vis.max(0) - gt_vis.min(0))
    if bbox_diag < 1.0:
        return 0.0

    threshold_px = threshold_frac * bbox_diag
    dist = np.linalg.norm(pred_kp[vis] - gt_kp[vis], axis=-1)   # [n_vis]
    correct = (dist <= threshold_px).sum()
    return float(correct) / float(vis.sum())


def compute_pck_curve(
    pred_kp: np.ndarray,
    gt_kp: np.ndarray,
    visibility: np.ndarray,
    thresholds: np.ndarray = PCK_THRESHOLDS,
) -> np.ndarray:
    """
    PCK at multiple thresholds for AUC computation.

    Args:
        pred_kp:    [21, 2]
        gt_kp:      [21, 2]
        visibility: [21]
        thresholds: [T] threshold fractions

    Returns:
        pck_curve: [T] float
    """
    return np.array([compute_pck(pred_kp, gt_kp, visibility, t) for t in thresholds])


def compute_mpjpe(
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    visibility: np.ndarray,
) -> float:
    """
    Mean Per-Joint Position Error in mm (3D).

    Args:
        pred_xyz:   [21, 3] predicted 3D positions in mm
        gt_xyz:     [21, 3] ground truth 3D positions in mm
        visibility: [21]    float {0, 1}

    Returns:
        MPJPE in mm
    """
    vis = visibility > 0.5
    if vis.sum() == 0:
        return float("nan")
    dist = np.linalg.norm(pred_xyz[vis] - gt_xyz[vis], axis=-1)
    return float(dist.mean())


class PoseEvaluator:
    """
    Stateful evaluator: accumulates per-frame metrics over a DataLoader.

    Usage:
        evaluator = PoseEvaluator()
        for batch in val_loader:
            pred = model.infer(batch["image"].to(device))
            evaluator.update(pred, batch)
        metrics = evaluator.compute()
        print(metrics)
        evaluator.reset()
    """

    def __init__(self) -> None:
        self._pck_curves: List[np.ndarray] = []
        self._n_samples = 0

    def update(
        self,
        pred: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> None:
        """
        Accumulate metrics for one batch.

        Args:
            pred:  dict from HandPoseNet.infer() — keys: "keypoints" [B, 21, 2]
            batch: dict from FreiHANDDataset — keys: "keypoints" [B, 21, 2], "kp_mask" [B, 21]
        """
        pred_kp = pred["keypoints"].cpu().numpy()    # [B, 21, 2] — heatmap space
        gt_kp   = batch["keypoints"].cpu().numpy()   # [B, 21, 2] — image space (need rescaling)
        vis     = batch["kp_mask"].cpu().numpy()     # [B, 21]

        B = pred_kp.shape[0]
        for i in range(B):
            curve = compute_pck_curve(pred_kp[i], gt_kp[i], vis[i])
            self._pck_curves.append(curve)
        self._n_samples += B

    def compute(self) -> EvalMetrics:
        """Aggregate accumulated metrics into EvalMetrics."""
        if self._n_samples == 0:
            return EvalMetrics()

        curves = np.stack(self._pck_curves)   # [N, T]
        mean_curve = curves.mean(0)           # [T]
        auc = float(np.trapz(mean_curve, PCK_THRESHOLDS) / (PCK_THRESHOLDS[-1] - PCK_THRESHOLDS[0]))

        # PCK@0.2 = index closest to 0.2 in thresholds
        idx_02 = int(np.argmin(np.abs(PCK_THRESHOLDS - 0.2)))
        pck_02 = float(mean_curve[idx_02])

        return EvalMetrics(
            pck_at_02=pck_02,
            pck_curve=mean_curve,
            auc=auc,
            n_samples=self._n_samples,
        )

    def reset(self) -> None:
        self._pck_curves = []
        self._n_samples = 0


class TrackingEvaluator:
    """
    Tracks ID switch rate across a sequence of frames.

    ID switch: when a track that was assigned to a hand changes its track_id
    between frames, even though the hand was continuously visible.

    Args:
        n_hands: maximum number of simultaneously tracked hands

    Usage:
        te = TrackingEvaluator()
        for tracks in frame_tracks:          # tracks: List[HandTrack]
            te.update(tracks)
        print(te.id_switch_rate())           # target <5%
    """

    def __init__(self) -> None:
        self._prev_assignment: Dict[int, int] = {}   # hand_slot → track_id
        self._switches = 0
        self._total = 0

    def update(self, tracks: list) -> None:
        """
        Update with tracks for one frame.

        Args:
            tracks: list of HandTrack objects (sorted by position for slot assignment)
        """
        # Simple slot assignment: sort by u-coordinate of wrist
        sorted_tracks = sorted(tracks, key=lambda t: t.state[0])

        curr = {}
        for slot, t in enumerate(sorted_tracks):
            curr[slot] = t.track_id
            if slot in self._prev_assignment:
                self._total += 1
                if self._prev_assignment[slot] != t.track_id:
                    self._switches += 1

        self._prev_assignment = curr

    def id_switch_rate(self) -> float:
        """ID switch rate in [0, 1]. Target: <0.05."""
        if self._total == 0:
            return 0.0
        return self._switches / self._total

    def reset(self) -> None:
        self._prev_assignment = {}
        self._switches = 0
        self._total = 0


def save_metrics_csv(metrics: EvalMetrics, path: str | Path, epoch: int) -> None:
    """
    Append epoch metrics to a CSV log file.

    Args:
        metrics: EvalMetrics from PoseEvaluator.compute()
        path:    output CSV path
        epoch:   current epoch number
    """
    path = Path(path)
    write_header = not path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "pck_at_02", "auc", "mpjpe_mm", "n_samples"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "epoch":     epoch,
            "pck_at_02": f"{metrics.pck_at_02:.6f}",
            "auc":       f"{metrics.auc:.6f}",
            "mpjpe_mm":  f"{metrics.mpjpe_mm:.4f}" if metrics.mpjpe_mm else "",
            "n_samples": metrics.n_samples,
        })
    logger.info(f"Metrics saved to {path}")
