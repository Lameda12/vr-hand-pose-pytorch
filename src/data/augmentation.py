"""
Data augmentation pipeline for hand pose training.

Applies spatially-consistent transforms to both the image and keypoints.
All transforms preserve the [21, 2] keypoint array in pixel coords.

Design:
  - Compose-style API (list of transforms applied in sequence)
  - Each transform returns (image, keypoints, visibility) unchanged if skipped
  - NumPy-based for direct use with OpenCV frames before tensor conversion
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

import cv2
import numpy as np


ImageKpVis = Tuple[np.ndarray, np.ndarray, np.ndarray]
# image: [H, W, 3] uint8 BGR
# keypoints: [21, 2] float32 (u, v) pixel coords
# visibility: [21] float32 {0, 1}


class HorizontalFlip:
    """
    Random horizontal flip with keypoint mirroring.

    Swaps left/right finger indices to preserve semantic labeling.
    FreiHAND/MediaPipe mapping: right-hand = standard, left-hand = mirror.

    Args:
        p: probability of applying flip
    """

    # MediaPipe 21-kp: these indices are symmetric — swap thumb side with pinky side
    # For egocentric capture both hands appear; flip swaps the wrist/palm handedness
    # Simple approach: mirror u-coordinate, keep indices (dataset is egocentric single-hand)
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: np.ndarray, keypoints: np.ndarray, visibility: np.ndarray) -> ImageKpVis:
        if random.random() >= self.p:
            return image, keypoints, visibility

        H, W = image.shape[:2]
        image = cv2.flip(image, 1)                # flip horizontal (axis=1)
        kp = keypoints.copy()
        kp[:, 0] = W - 1 - kp[:, 0]              # mirror u: u' = W - 1 - u
        return image, kp, visibility.copy()


class RandomRotation:
    """
    Random in-plane rotation around image center.

    Rotates image and keypoints by the same angle.
    Keypoints outside the frame after rotation are marked as occluded.

    Args:
        max_degrees: maximum rotation angle in either direction
        p:           probability of applying rotation
    """

    def __init__(self, max_degrees: float = 30.0, p: float = 0.8) -> None:
        self.max_degrees = max_degrees
        self.p = p

    def __call__(self, image: np.ndarray, keypoints: np.ndarray, visibility: np.ndarray) -> ImageKpVis:
        if random.random() >= self.p:
            return image, keypoints, visibility

        H, W = image.shape[:2]
        angle = random.uniform(-self.max_degrees, self.max_degrees)
        cx, cy = W / 2.0, H / 2.0

        # Rotation matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale=1.0)  # [2, 3]
        image = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

        # Transform keypoints: [u, v, 1] @ M^T
        kp = keypoints.copy()
        ones = np.ones((kp.shape[0], 1), dtype=np.float32)
        kp_h = np.hstack([kp, ones])              # [21, 3]
        kp_rot = (M @ kp_h.T).T                   # [21, 2]

        # Mask out-of-bounds keypoints
        vis = visibility.copy()
        out_u = (kp_rot[:, 0] < 0) | (kp_rot[:, 0] >= W)
        out_v = (kp_rot[:, 1] < 0) | (kp_rot[:, 1] >= H)
        vis[out_u | out_v] = 0.0

        return image, kp_rot.astype(np.float32), vis


class RandomScale:
    """
    Random scale (zoom in/out) around hand center.

    Scales image + keypoints so hand appears larger or smaller.
    Preserves aspect ratio.

    Args:
        scale_range: (min_scale, max_scale) multiplicative factors
        p:           probability of applying scale
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.8) -> None:
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image: np.ndarray, keypoints: np.ndarray, visibility: np.ndarray) -> ImageKpVis:
        if random.random() >= self.p:
            return image, keypoints, visibility

        H, W = image.shape[:2]
        scale = random.uniform(*self.scale_range)

        # Scale around center of visible keypoints (hand centroid)
        vis_kp = keypoints[visibility > 0.5]
        if len(vis_kp) == 0:
            return image, keypoints, visibility
        cx, cy = vis_kp.mean(0)

        M = cv2.getRotationMatrix2D((float(cx), float(cy)), 0.0, scale)
        image = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

        kp = keypoints.copy()
        ones = np.ones((kp.shape[0], 1), dtype=np.float32)
        kp_h = np.hstack([kp, ones])
        kp_scaled = (M @ kp_h.T).T

        vis = visibility.copy()
        out_u = (kp_scaled[:, 0] < 0) | (kp_scaled[:, 0] >= W)
        out_v = (kp_scaled[:, 1] < 0) | (kp_scaled[:, 1] >= H)
        vis[out_u | out_v] = 0.0

        return image, kp_scaled.astype(np.float32), vis


class ColorJitter:
    """
    Random brightness, contrast, saturation, and hue jitter.

    Operates on BGR uint8 images. Does NOT touch keypoints.

    Args:
        brightness: max absolute brightness delta [0, 255]
        contrast:   multiplicative contrast range (1±contrast)
        saturation: multiplicative saturation range (1±saturation)
        hue:        max hue shift in degrees
        p:          probability of applying jitter
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 10.0,
        p: float = 0.8,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, image: np.ndarray, keypoints: np.ndarray, visibility: np.ndarray) -> ImageKpVis:
        if random.random() >= self.p:
            return image, keypoints, visibility

        img = image.astype(np.float32)

        # Brightness
        if self.brightness > 0:
            delta = random.uniform(-self.brightness * 255, self.brightness * 255)
            img += delta

        # Contrast
        if self.contrast > 0:
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            img = img * factor

        img = np.clip(img, 0, 255).astype(np.uint8)

        # Saturation + Hue: operate in HSV
        if self.saturation > 0 or self.hue > 0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            if self.saturation > 0:
                s_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)
            if self.hue > 0:
                h_delta = random.uniform(-self.hue, self.hue)
                hsv[:, :, 0] = (hsv[:, :, 0] + h_delta) % 180
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return img, keypoints, visibility


class GaussianNoise:
    """
    Additive Gaussian noise to image pixels.

    Args:
        std:  noise standard deviation (pixel units, float)
        p:    probability of applying noise
    """

    def __init__(self, std: float = 5.0, p: float = 0.3) -> None:
        self.std = std
        self.p = p

    def __call__(self, image: np.ndarray, keypoints: np.ndarray, visibility: np.ndarray) -> ImageKpVis:
        if random.random() >= self.p:
            return image, keypoints, visibility
        noise = np.random.randn(*image.shape).astype(np.float32) * self.std
        img = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img, keypoints, visibility


class RandomOcclusion:
    """
    Simulates occlusion by painting a random rectangle over the hand.

    Forces the model to handle partial visibility — critical for VR use.

    Args:
        max_cover_frac: maximum fraction of hand bbox to occlude
        p:              probability of applying occlusion
    """

    def __init__(self, max_cover_frac: float = 0.4, p: float = 0.2) -> None:
        self.max_cover_frac = max_cover_frac
        self.p = p

    def __call__(self, image: np.ndarray, keypoints: np.ndarray, visibility: np.ndarray) -> ImageKpVis:
        if random.random() >= self.p:
            return image, keypoints, visibility

        vis_kp = keypoints[visibility > 0.5]
        if len(vis_kp) < 2:
            return image, keypoints, visibility

        x1, y1 = vis_kp.min(0).astype(int)
        x2, y2 = vis_kp.max(0).astype(int)
        hand_w, hand_h = max(x2 - x1, 1), max(y2 - y1, 1)

        # Random occluder within hand bbox
        occ_w = int(hand_w * random.uniform(0.1, self.max_cover_frac))
        occ_h = int(hand_h * random.uniform(0.1, self.max_cover_frac))
        occ_x = random.randint(x1, max(x1, x2 - occ_w))
        occ_y = random.randint(y1, max(y1, y2 - occ_h))

        img = image.copy()
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))
        cv2.rectangle(img, (occ_x, occ_y), (occ_x + occ_w, occ_y + occ_h), color, -1)

        # Mark keypoints inside occluder as invisible
        vis = visibility.copy()
        in_occ_u = (keypoints[:, 0] >= occ_x) & (keypoints[:, 0] <= occ_x + occ_w)
        in_occ_v = (keypoints[:, 1] >= occ_y) & (keypoints[:, 1] <= occ_y + occ_h)
        vis[in_occ_u & in_occ_v] = 0.0

        return img, keypoints, vis


class HandAugmentation:
    """
    Compose augmentation pipeline from CLAUDE.md spec.

    Default config matches configs/model_config.yaml augmentation section.

    Args:
        horizontal_flip:   bool
        rotation_max_deg:  float
        scale_range:       (min, max)
        brightness_jitter: float  (maps to ColorJitter brightness)
        color_jitter:      float  (maps to ColorJitter contrast/saturation)
        add_occlusion:     bool   (random rect occlusion)

    Usage:
        aug = HandAugmentation(horizontal_flip=True, rotation_max_deg=30)
        img_aug, kp_aug, vis_aug = aug(image, keypoints, visibility)
    """

    def __init__(
        self,
        horizontal_flip: bool = True,
        rotation_max_deg: float = 30.0,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        brightness_jitter: float = 0.3,
        color_jitter: float = 0.2,
        add_occlusion: bool = True,
    ) -> None:
        self.transforms: List = []
        if horizontal_flip:
            self.transforms.append(HorizontalFlip(p=0.5))
        if rotation_max_deg > 0:
            self.transforms.append(RandomRotation(max_degrees=rotation_max_deg, p=0.8))
        if scale_range != (1.0, 1.0):
            self.transforms.append(RandomScale(scale_range=scale_range, p=0.8))
        if brightness_jitter > 0 or color_jitter > 0:
            self.transforms.append(ColorJitter(
                brightness=brightness_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                p=0.8,
            ))
        self.transforms.append(GaussianNoise(std=5.0, p=0.3))
        if add_occlusion:
            self.transforms.append(RandomOcclusion(p=0.2))

    def __call__(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        visibility: np.ndarray,
    ) -> ImageKpVis:
        """
        Apply full augmentation pipeline.

        Args:
            image:      [H, W, 3] uint8 BGR
            keypoints:  [21, 2] float32 (u, v) pixel coords (image space)
            visibility: [21] float32 {0, 1}

        Returns:
            Augmented (image, keypoints, visibility) triple
        """
        for t in self.transforms:
            image, keypoints, visibility = t(image, keypoints, visibility)
        return image, keypoints, visibility

    @classmethod
    def from_config(cls, cfg: dict) -> "HandAugmentation":
        """Build from configs/model_config.yaml augmentation dict."""
        aug = cfg.get("augmentation", {})
        return cls(
            horizontal_flip=aug.get("horizontal_flip", True),
            rotation_max_deg=aug.get("rotation_max_deg", 30.0),
            scale_range=tuple(aug.get("scale_range", [0.8, 1.2])),
            brightness_jitter=aug.get("brightness_jitter", 0.3),
            color_jitter=aug.get("color_jitter", 0.2),
        )

    @classmethod
    def identity(cls) -> "HandAugmentation":
        """No-op augmentation for validation."""
        return cls(
            horizontal_flip=False,
            rotation_max_deg=0.0,
            scale_range=(1.0, 1.0),
            brightness_jitter=0.0,
            color_jitter=0.0,
            add_occlusion=False,
        )
