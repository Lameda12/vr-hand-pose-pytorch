"""
FreiHAND dataset loader for egocentric hand pose training.

Dataset structure (after download + extraction):
  freihand/
    training/
      rgb/          00000000.jpg ... 32559.jpg   (32,560 images × 4 augmented = 130,240 total)
      mask/         00000000.jpg ...
    evaluation/
      rgb/          00000000.jpg ... 3959.jpg
    training_K.json       # [N, 3, 3] camera intrinsics
    training_mano.json    # [N, 51] MANO pose params (not used directly)
    training_xyz.json     # [N, 21, 3] world-space 3D keypoints
    evaluation_K.json
    evaluation_xyz.json

Reference: Zimmermann et al., ICCV 2019
Download:  https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html

This loader:
  - Loads RGB + 21 keypoints (2D projected + normalized depth)
  - Applies spatial augmentation via HandAugmentation
  - Returns tensors ready for HandPoseNet + HandPoseLoss
  - Generates Gaussian heatmap GT at heatmap_size = (H/4, W/4)
  - Builds detection mask GT at feature_size = (H/8, W/8)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .augmentation import HandAugmentation
from .heatmap_utils import build_heatmaps, keypoints_to_heatmap_coords

logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 21
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FreiHANDDataset(Dataset):
    """
    FreiHAND egocentric hand pose dataset.

    Returns per-sample dicts with:
        "image":       [3, H, W] float32 RGB normalized [0,1]
        "det_mask":    [Hf, Wf] float32 — 1 at hand center grid cell
        "heatmaps_gt": [21, Hh, Wh] float32 Gaussian heatmaps
        "kp_mask":     [21] float32 visibility
        "depth_gt":    [21] float32 normalized depth in [-1, 1]
        "keypoints":   [21, 2] float32 (u, v) in image pixel space (for eval)
        "K":           [3, 3] float32 camera intrinsics

    Args:
        root:           path to freihand/ directory
        split:          "train" | "val" | "eval"
                        train/val split from training set (90/10)
        image_size:     (W, H) to resize images
        heatmap_sigma:  Gaussian blob sigma in heatmap pixels
        augment:        apply data augmentation (train only)
        max_samples:    cap dataset size (None = full)
        val_fraction:   fraction of training set used for validation

    Usage:
        ds = FreiHANDDataset("/data/freihand", split="train", image_size=(640, 480))
        loader = DataLoader(ds, batch_size=16, num_workers=4, shuffle=True)
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: Tuple[int, int] = (640, 480),
        heatmap_sigma: float = 2.0,
        augment: bool = True,
        max_samples: Optional[int] = None,
        val_fraction: float = 0.1,
    ) -> None:
        assert split in ("train", "val", "eval"), f"Invalid split: {split!r}"
        self.root = Path(root)
        self.split = split
        self.image_size = image_size          # (W, H)
        self.heatmap_sigma = heatmap_sigma
        self.max_samples = max_samples

        # Derived sizes
        W, H = image_size
        self.heatmap_size = (H // 4, W // 4)     # stride-4 heatmaps
        self.feature_size = (H // 8, W // 8)     # stride-8 detection feature map

        # Load annotation JSONs
        if split in ("train", "val"):
            self._img_dir = self.root / "training" / "rgb"
            K_path   = self.root / "training_K.json"
            xyz_path = self.root / "training_xyz.json"
        else:
            self._img_dir = self.root / "evaluation" / "rgb"
            K_path   = self.root / "evaluation_K.json"
            xyz_path = self.root / "evaluation_xyz.json"

        logger.info(f"Loading FreiHAND {split} from {self.root}")
        self._K_list   = self._load_json(K_path)    # List[[3,3]]
        self._xyz_list = self._load_json(xyz_path)  # List[[21,3]]

        # FreiHAND: training set is 4× augmented (indices 0..N-1, N..2N-1, ...)
        # We use all 4 augmentation folds for training
        n_total = len(self._K_list)

        if split in ("train", "val"):
            n_val = max(1, int(n_total * val_fraction))
            if split == "val":
                self._indices = list(range(n_total - n_val, n_total))
            else:
                self._indices = list(range(0, n_total - n_val))
        else:
            self._indices = list(range(n_total))

        if max_samples is not None:
            self._indices = self._indices[:max_samples]

        # Augmentation
        if augment and split == "train":
            self.aug = HandAugmentation()
        else:
            self.aug = HandAugmentation.identity()

        logger.info(f"FreiHAND {split}: {len(self._indices)} samples")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = self._indices[idx]
        img_path = self._img_dir / f"{sample_idx:08d}.jpg"

        # Load image (BGR)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load annotations
        K   = np.array(self._K_list[sample_idx], dtype=np.float32)   # [3, 3]
        xyz = np.array(self._xyz_list[sample_idx], dtype=np.float32)  # [21, 3]

        # Project 3D world → 2D image using camera intrinsics
        # K @ xyz^T → [3, 21]; divide by z
        xyz_cam = xyz   # FreiHAND xyz is already in camera space
        uv1 = (K @ xyz_cam.T)                # [3, 21]
        z_cam = uv1[2, :]                    # [21]
        uv = (uv1[:2, :] / (z_cam + 1e-6)).T  # [21, 2] in original image coords

        # All keypoints visible (FreiHAND has full annotations)
        visibility = np.ones(NUM_KEYPOINTS, dtype=np.float32)

        # Resize image to target input size
        orig_H, orig_W = img_bgr.shape[:2]
        img_bgr = cv2.resize(img_bgr, self.image_size)

        # Scale keypoints to resized image coords
        uv[:, 0] *= self.image_size[0] / orig_W
        uv[:, 1] *= self.image_size[1] / orig_H
        uv = uv.astype(np.float32)

        # Apply augmentation
        img_bgr, uv, visibility = self.aug(img_bgr, uv, visibility)

        # BGR → RGB (always explicit)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Normalize to float [0,1] then ImageNet normalize
        img_f = img_rgb.astype(np.float32) / 255.0
        img_f = (img_f - _IMAGENET_MEAN) / _IMAGENET_STD

        # CHW tensor
        image_tensor = torch.from_numpy(img_f.transpose(2, 0, 1))  # [3, H, W]

        # Build heatmap GT
        uv_hm = keypoints_to_heatmap_coords(
            uv, self.image_size[::-1], self.heatmap_size  # image_size=(W,H), pass (H,W)
        )
        heatmaps_gt = build_heatmaps(uv_hm, self.heatmap_size, self.heatmap_sigma, visibility)

        # Build detection mask GT (1 at grid cell containing hand center)
        det_mask = self._build_det_mask(uv, visibility)

        # Relative depth: normalize z_cam to [-1, 1] within the sample
        z_rel = self._normalize_depth(z_cam, visibility)

        return {
            "image":       image_tensor,                         # [3, H, W]
            "det_mask":    torch.from_numpy(det_mask),           # [Hf, Wf]
            "heatmaps_gt": torch.from_numpy(heatmaps_gt),        # [21, Hh, Ww]
            "kp_mask":     torch.from_numpy(visibility),         # [21]
            "depth_gt":    torch.from_numpy(z_rel),              # [21]
            "keypoints":   torch.from_numpy(uv),                 # [21, 2]
            "K":           torch.from_numpy(K),                  # [3, 3]
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_det_mask(self, uv: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        """
        Build detection ground truth mask at feature_size resolution.

        Places a 1 at the grid cell containing the wrist keypoint (index 0),
        which serves as the hand center anchor.

        Returns: [Hf, Wf] float32
        """
        Hf, Wf = self.feature_size
        W_img, H_img = self.image_size
        mask = np.zeros((Hf, Wf), dtype=np.float32)

        wrist = uv[0]   # wrist = keypoint 0
        if visibility[0] > 0.5:
            u_f = int(wrist[0] * Wf / W_img)
            v_f = int(wrist[1] * Hf / H_img)
            u_f = np.clip(u_f, 0, Wf - 1)
            v_f = np.clip(v_f, 0, Hf - 1)
            mask[v_f, u_f] = 1.0

        return mask

    @staticmethod
    def _normalize_depth(z_cam: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        """
        Normalize per-keypoint camera-space depth to [-1, 1].

        Subtracts wrist depth (anchor), divides by visible range.
        Invisible keypoints get depth = 0.

        Args:
            z_cam:      [21] float32 camera-space z (meters)
            visibility: [21] float32 {0, 1}

        Returns:
            z_rel: [21] float32 in [-1, 1]
        """
        z_rel = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        vis_mask = visibility > 0.5
        if vis_mask.sum() < 2:
            return z_rel

        z_anchor = z_cam[0]   # wrist depth as reference
        z_centered = z_cam - z_anchor

        z_vis = z_centered[vis_mask]
        z_range = max(z_vis.max() - z_vis.min(), 1e-3)
        z_rel[vis_mask] = np.clip(z_centered[vis_mask] / z_range, -1.0, 1.0)
        return z_rel

    @staticmethod
    def _load_json(path: Path) -> list:
        if not path.exists():
            raise FileNotFoundError(
                f"FreiHAND annotation file not found: {path}\n"
                "Download the dataset from: "
                "https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html"
            )
        with open(path) as f:
            return json.load(f)


def build_dataloader(
    root: str | Path,
    split: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (640, 480),
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Convenience factory for train/val/eval DataLoaders.

    Args:
        root:        freihand/ dataset root
        split:       "train" | "val" | "eval"
        batch_size:  samples per batch
        num_workers: DataLoader worker processes
        image_size:  (W, H) model input size
        max_samples: cap dataset (useful for quick iteration)

    Returns:
        DataLoader with correct shuffle/drop_last settings per split
    """
    ds = FreiHANDDataset(
        root=root,
        split=split,
        image_size=image_size,
        augment=(split == "train"),
        max_samples=max_samples,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )
