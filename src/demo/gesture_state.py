"""
Gesture state machine for VR interaction.

Detects:
  - pinch:      thumb tip ↔ index tip distance < threshold
  - point:      index finger extended, others curled
  - open_palm:  all fingers extended
  - fist:       all fingers curled

Outputs per-frame gesture labels + VR-relevant quantities:
  - pinch_distance: Euclidean thumb-index tip distance (pixels or mm if depth known)
  - point_ray:      (dx, dy) direction from wrist to index tip (normalized)
  - drag_delta:     (Δx, Δy) pinch displacement since pinch start
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple


class Gesture(Enum):
    NONE = auto()
    PINCH = auto()
    POINT = auto()
    OPEN_PALM = auto()
    FIST = auto()


# MediaPipe/FreiHAND keypoint indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# Finger tip + PIP pairs for extension check
_FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
_FINGER_PIPS = [2, 6, 10, 14, 18]   # PIP joints


@dataclass
class GestureResult:
    """
    Per-frame gesture output.

    Attributes:
        gesture:        detected Gesture enum
        pinch_distance: thumb-index tip distance in keypoint pixel space
        point_ray:      unit vector (dx, dy) from wrist to index tip (None if not pointing)
        drag_delta:     cumulative (Δu, Δv) while pinch is active (None if not pinching)
        confidence:     heuristic confidence in [0, 1]
    """
    gesture: Gesture = Gesture.NONE
    pinch_distance: float = float("inf")
    point_ray: Optional[np.ndarray] = None    # [2]
    drag_delta: Optional[np.ndarray] = None   # [2]
    confidence: float = 0.0


class GestureState:
    """
    Stateful gesture classifier operating on 21-keypoint hand poses.

    Maintains pinch origin for drag tracking across frames.

    Args:
        pinch_px_threshold:   pixel distance below which pinch is active
        extend_ratio:         tip-to-PIP ratio above which finger counts as extended

    Usage:
        gs = GestureState()
        for kp in keypoints_per_frame:   # kp: [21, 2] or [21, 3]
            result = gs.update(kp)
    """

    def __init__(
        self,
        pinch_px_threshold: float = 30.0,
        extend_ratio: float = 1.2,
    ) -> None:
        self.pinch_px_threshold = pinch_px_threshold
        self.extend_ratio = extend_ratio
        self._pinch_origin: Optional[np.ndarray] = None   # [2] midpoint at pinch start

    def update(self, keypoints: np.ndarray) -> GestureResult:
        """
        Classify gesture from keypoints.

        Args:
            keypoints: [21, 2] or [21, 3] float32 — (u, v) or (u, v, z_rel)

        Returns:
            GestureResult with all VR-relevant fields populated
        """
        if keypoints.shape[0] != 21:
            raise ValueError(f"Expected 21 keypoints, got {keypoints.shape[0]}")

        kp2d = keypoints[:, :2]   # use only (u, v) for 2D gesture

        pinch_dist = float(np.linalg.norm(kp2d[THUMB_TIP] - kp2d[INDEX_TIP]))
        is_pinching = pinch_dist < self.pinch_px_threshold

        extended = self._finger_extensions(kp2d)

        # Point ray: wrist → index tip
        wrist = kp2d[WRIST]
        index_tip = kp2d[INDEX_TIP]
        ray_vec = index_tip - wrist
        ray_norm = np.linalg.norm(ray_vec)
        point_ray = (ray_vec / (ray_norm + 1e-6)).astype(np.float32) if ray_norm > 1 else None

        # Classify
        n_extended = sum(extended)
        if is_pinching:
            gesture = Gesture.PINCH
            confidence = 1.0 - pinch_dist / self.pinch_px_threshold
        elif extended[1] and not extended[2] and not extended[3] and not extended[4]:
            gesture = Gesture.POINT
            confidence = 0.8
        elif n_extended >= 4:
            gesture = Gesture.OPEN_PALM
            confidence = n_extended / 5.0
        elif n_extended == 0:
            gesture = Gesture.FIST
            confidence = 0.9
        else:
            gesture = Gesture.NONE
            confidence = 0.0

        # Drag delta tracking
        drag_delta: Optional[np.ndarray] = None
        if is_pinching:
            pinch_midpoint = (kp2d[THUMB_TIP] + kp2d[INDEX_TIP]) / 2.0
            if self._pinch_origin is None:
                self._pinch_origin = pinch_midpoint.copy()
            drag_delta = (pinch_midpoint - self._pinch_origin).astype(np.float32)
        else:
            self._pinch_origin = None

        return GestureResult(
            gesture=gesture,
            pinch_distance=pinch_dist,
            point_ray=point_ray,
            drag_delta=drag_delta,
            confidence=confidence,
        )

    def _finger_extensions(self, kp2d: np.ndarray) -> list[bool]:
        """
        Heuristic: finger is extended if tip is farther from wrist than PIP.

        Returns: list[bool] of length 5 (thumb, index, middle, ring, pinky)
        """
        wrist = kp2d[WRIST]
        extended = []
        for tip_idx, pip_idx in zip(_FINGER_TIPS, _FINGER_PIPS):
            d_tip = float(np.linalg.norm(kp2d[tip_idx] - wrist))
            d_pip = float(np.linalg.norm(kp2d[pip_idx] - wrist))
            extended.append(d_tip > d_pip * self.extend_ratio)
        return extended
