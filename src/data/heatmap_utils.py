"""
Gaussian heatmap generation and soft-argmax utilities.

FreiHAND ground-truth pipeline:
  (u, v) keypoint coords → Gaussian blobs → [21, Hh, Wh] float32 heatmaps

Used at training time only; inference uses the raw heatmap logits.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


def build_heatmaps(
    keypoints: np.ndarray,
    heatmap_size: Tuple[int, int],
    sigma: float = 2.0,
    visibility: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build 21-channel Gaussian heatmaps from 2D keypoint coordinates.

    Each visible keypoint generates a 2D Gaussian blob of std=sigma.
    Occluded keypoints (visibility=0) produce all-zero channels.

    Args:
        keypoints:    [21, 2] float32 (u, v) in heatmap pixel coords
        heatmap_size: (H, W) of output heatmap
        sigma:        Gaussian standard deviation in pixels (default=2)
        visibility:   [21] float32 in {0, 1}; None = all visible

    Returns:
        heatmaps: [21, H, W] float32, values in [0, 1]

    Performance: ~0.2ms for 21 keypoints @(60,80) heatmap on CPU
    """
    H, W = heatmap_size
    K = keypoints.shape[0]
    heatmaps = np.zeros((K, H, W), dtype=np.float32)

    if visibility is None:
        visibility = np.ones(K, dtype=np.float32)

    # Build coordinate grids (reusable across keypoints)
    gy = np.arange(H, dtype=np.float32)   # [H]
    gx = np.arange(W, dtype=np.float32)   # [W]
    grid_y, grid_x = np.meshgrid(gy, gx, indexing="ij")  # both [H, W]

    for k in range(K):
        if visibility[k] < 0.5:
            continue   # occluded: leave channel as zeros

        u, v = keypoints[k, 0], keypoints[k, 1]

        # Skip if keypoint outside heatmap bounds
        if u < 0 or u >= W or v < 0 or v >= H:
            continue

        gauss = np.exp(-((grid_x - u) ** 2 + (grid_y - v) ** 2) / (2 * sigma ** 2))
        heatmaps[k] = gauss   # already in [0, 1] (max=1 at center)

    return heatmaps


def softargmax2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Differentiable soft-argmax: heatmaps → (u, v) coordinates.

    Converts a batch of heatmap logits to expected 2D keypoint positions
    using the softmax-weighted spatial expectation.

    Args:
        heatmaps: [B, K, H, W] float32 — raw logits or probabilities

    Returns:
        keypoints: [B, K, 2] float32 — (u, v) in pixel coords [0, W-1] × [0, H-1]

    Notes:
        Gradient flows through softmax — suitable for end-to-end training.
        At inference, prefer argmax for speed (HandPoseNet.infer does this).
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device

    # Normalize across spatial dims
    hm_flat = heatmaps.flatten(2)                  # [B, K, H*W]
    hm_prob = torch.softmax(hm_flat, dim=-1)       # [B, K, H*W]

    gy = torch.linspace(0, H - 1, H, device=device)
    gx = torch.linspace(0, W - 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")  # [H, W]

    kp_v = (hm_prob * grid_y.flatten()).sum(-1)    # [B, K]
    kp_u = (hm_prob * grid_x.flatten()).sum(-1)    # [B, K]

    return torch.stack([kp_u, kp_v], dim=-1)       # [B, K, 2]


def keypoints_to_heatmap_coords(
    keypoints_px: np.ndarray,
    image_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
) -> np.ndarray:
    """
    Scale keypoint pixel coordinates from image space to heatmap space.

    Args:
        keypoints_px:  [21, 2] (u, v) in image pixel coords
        image_size:    (H_img, W_img)
        heatmap_size:  (H_hm, W_hm)

    Returns:
        keypoints_hm: [21, 2] (u, v) in heatmap pixel coords
    """
    H_img, W_img = image_size
    H_hm, W_hm = heatmap_size
    scale = np.array([W_hm / W_img, H_hm / H_img], dtype=np.float32)
    return keypoints_px * scale


def heatmap_coords_to_image(
    keypoints_hm: np.ndarray,
    image_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
) -> np.ndarray:
    """
    Scale keypoint coordinates from heatmap space back to image space.

    Inverse of keypoints_to_heatmap_coords.

    Args:
        keypoints_hm:  [21, 2] (u, v) in heatmap pixel coords
        image_size:    (H_img, W_img)
        heatmap_size:  (H_hm, W_hm)

    Returns:
        keypoints_px: [21, 2] (u, v) in image pixel coords
    """
    H_img, W_img = image_size
    H_hm, W_hm = heatmap_size
    scale = np.array([W_img / W_hm, H_img / H_hm], dtype=np.float32)
    return keypoints_hm * scale
