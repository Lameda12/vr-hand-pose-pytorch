"""
HO3D (Hand-Object 3D) dataset loader.

Dataset for hand-object interaction — supplements FreiHAND with challenging
occlusion scenarios where the hand grasps or manipulates objects.

Dataset structure (after download + extraction):
  ho3d/
    train/
      ABF10/
        rgb/       00000.jpg ...
        meta/      00000.pkl  (camera intrinsics K, 3D joints xyz)
      ...
    evaluation/
      ...
    train.txt   # list of sequence/frame IDs
    evaluation.txt

Reference: Hampali et al., CVPR 2020
Download:  https://www.tugraz.at/index.php?id=40231

Differences from FreiHAND:
  - RGB-D (depth channel available but not required)
  - Hand annotated alongside object pose (we use hand-only subset)
  - Stronger occlusion → important for robustness evaluation
  - Per-frame pickle metadata (K, joints, mano_params)
"""

from __future__ import annotations

import logging
import pickle
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


class HO3DDataset(Dataset):
    """
    HO3D hand-object interaction dataset — hand pose subset.

    Same output format as FreiHANDDataset for drop-in compatibility.

    Args:
        root:         path to ho3d/ directory
        split:        "train" | "val" | "eval"
        image_size:   (W, H) model input size
        heatmap_sigma:Gaussian blob sigma
        augment:      apply augmentation (train only)
        max_samples:  cap dataset size
        val_fraction: fraction of training sequences used for validation

    Returns same keys as FreiHANDDataset:
        image, det_mask, heatmaps_gt, kp_mask, depth_gt, keypoints, K
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
        self.image_size = image_size
        self.heatmap_sigma = heatmap_sigma

        W, H = image_size
        self.heatmap_size = (H // 4, W // 4)
        self.feature_size = (H // 8, W // 8)

        # Load split file
        split_file = self.root / ("train.txt" if split != "eval" else "evaluation.txt")
        self._samples = self._load_split(split_file)

        # Train/val partition
        n = len(self._samples)
        n_val = max(1, int(n * val_fraction))
        if split == "val":
            self._samples = self._samples[n - n_val:]
        elif split == "train":
            self._samples = self._samples[:n - n_val]

        if max_samples is not None:
            self._samples = self._samples[:max_samples]

        self.aug = HandAugmentation() if (augment and split == "train") else HandAugmentation.identity()
        logger.info(f"HO3D {split}: {len(self._samples)} samples")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq, frame_id = self._samples[idx]

        # Paths
        if self.split == "eval":
            img_path  = self.root / "evaluation" / seq / "rgb"   / f"{frame_id:05d}.jpg"
            meta_path = self.root / "evaluation" / seq / "meta"  / f"{frame_id:05d}.pkl"
        else:
            img_path  = self.root / "train" / seq / "rgb"  / f"{frame_id:05d}.jpg"
            meta_path = self.root / "train" / seq / "meta" / f"{frame_id:05d}.pkl"

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load metadata
        with open(meta_path, "rb") as f:
            meta = pickle.load(f, encoding="latin1")

        K   = np.array(meta["camMat"], dtype=np.float32)         # [3, 3]
        xyz = np.array(meta["handJoints3D"], dtype=np.float32)   # [21, 3] world-space

        # Project 3D → 2D
        uv1 = (K @ xyz.T)
        z_cam = uv1[2, :]
        uv = (uv1[:2] / (z_cam + 1e-6)).T.astype(np.float32)   # [21, 2]

        # All joints annotated
        visibility = np.ones(NUM_KEYPOINTS, dtype=np.float32)

        # Resize + scale keypoints
        orig_H, orig_W = img_bgr.shape[:2]
        img_bgr = cv2.resize(img_bgr, self.image_size)
        uv[:, 0] *= self.image_size[0] / orig_W
        uv[:, 1] *= self.image_size[1] / orig_H

        # Augment
        img_bgr, uv, visibility = self.aug(img_bgr, uv, visibility)

        # BGR → RGB → normalize
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_f   = (img_rgb.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        image_tensor = torch.from_numpy(img_f.transpose(2, 0, 1))

        # Heatmap GT
        uv_hm = keypoints_to_heatmap_coords(uv, self.image_size[::-1], self.heatmap_size)
        heatmaps_gt = build_heatmaps(uv_hm, self.heatmap_size, self.heatmap_sigma, visibility)

        # Detection mask + depth
        det_mask = self._build_det_mask(uv, visibility)
        z_rel    = self._normalize_depth(z_cam, visibility)

        return {
            "image":       image_tensor,
            "det_mask":    torch.from_numpy(det_mask),
            "heatmaps_gt": torch.from_numpy(heatmaps_gt),
            "kp_mask":     torch.from_numpy(visibility),
            "depth_gt":    torch.from_numpy(z_rel),
            "keypoints":   torch.from_numpy(uv),
            "K":           torch.from_numpy(K),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_det_mask(self, uv: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        Hf, Wf = self.feature_size
        W_img, H_img = self.image_size
        mask = np.zeros((Hf, Wf), dtype=np.float32)
        wrist = uv[0]
        if visibility[0] > 0.5:
            u_f = np.clip(int(wrist[0] * Wf / W_img), 0, Wf - 1)
            v_f = np.clip(int(wrist[1] * Hf / H_img), 0, Hf - 1)
            mask[v_f, u_f] = 1.0
        return mask

    @staticmethod
    def _normalize_depth(z_cam: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        z_rel = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        vis   = visibility > 0.5
        if vis.sum() < 2:
            return z_rel
        z_centered = z_cam - z_cam[0]
        z_vis  = z_centered[vis]
        z_range = max(z_vis.max() - z_vis.min(), 1e-3)
        z_rel[vis] = np.clip(z_centered[vis] / z_range, -1.0, 1.0)
        return z_rel

    def _load_split(self, split_file: Path) -> List[Tuple[str, int]]:
        """Parse train.txt / evaluation.txt → list of (sequence_name, frame_id)."""
        if not split_file.exists():
            raise FileNotFoundError(
                f"HO3D split file not found: {split_file}\n"
                "Download from: https://www.tugraz.at/index.php?id=40231"
            )
        samples = []
        for line in split_file.read_text().strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("/")
            if len(parts) == 2:
                seq, frame_str = parts
                samples.append((seq, int(frame_str)))
        return samples


def build_combined_dataloader(
    freihand_root: Optional[str | Path],
    ho3d_root: Optional[str | Path],
    split: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (640, 480),
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Combine FreiHAND + HO3D into a single DataLoader.

    Useful for robustness fine-tuning: FreiHAND gives clean egocentric
    samples; HO3D adds hard occlusion cases from hand-object interaction.

    At least one root must be provided.
    """
    from torch.utils.data import ConcatDataset
    from .freihand import FreiHANDDataset

    datasets = []
    if freihand_root is not None:
        datasets.append(FreiHANDDataset(
            freihand_root, split=split, image_size=image_size,
            augment=(split == "train"), max_samples=max_samples,
        ))
    if ho3d_root is not None:
        datasets.append(HO3DDataset(
            ho3d_root, split=split, image_size=image_size,
            augment=(split == "train"), max_samples=max_samples,
        ))

    if not datasets:
        raise ValueError("At least one dataset root (freihand or ho3d) must be provided")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )
