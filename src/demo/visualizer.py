"""
Visualization utilities for hand skeleton and gesture overlays.

All functions mutate the provided frame in-place (OpenCV convention).
Coordinate system: (u, v) = (col, row) pixel indices.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Optional

from .gesture_state import GestureResult, Gesture

# Per-finger colors (BGR) for skeleton rendering
_FINGER_COLORS = {
    "thumb":  (147, 20, 255),   # purple
    "index":  (255, 0, 0),      # blue
    "middle": (0, 255, 0),      # green
    "ring":   (0, 165, 255),    # orange
    "pinky":  (0, 0, 255),      # red
    "palm":   (200, 200, 200),  # gray
}

# Edge index → finger name mapping
_EDGE_TO_FINGER = {
    (0, 1): "thumb",  (1, 2): "thumb",  (2, 3): "thumb",  (3, 4): "thumb",
    (0, 5): "index",  (5, 6): "index",  (6, 7): "index",  (7, 8): "index",
    (0, 9): "middle", (9, 10): "middle",(10,11): "middle",(11,12): "middle",
    (0,13): "ring",  (13,14): "ring",  (14,15): "ring",  (15,16): "ring",
    (0,17): "pinky", (17,18): "pinky", (18,19): "pinky", (19,20): "pinky",
    (5, 9): "palm",  (9,13): "palm",   (13,17): "palm",
}

# Gesture → display color
_GESTURE_COLORS = {
    Gesture.PINCH:     (0, 200, 255),
    Gesture.POINT:     (0, 255, 0),
    Gesture.OPEN_PALM: (255, 200, 0),
    Gesture.FIST:      (0, 0, 255),
    Gesture.NONE:      (180, 180, 180),
}

# Track ID → hue (for multi-hand color coding)
_ID_HUES = [60, 120, 180, 240, 300]   # degrees in HSV


def viz_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    edges: List[Tuple[int, int]],
    track_id: int = 0,
    dot_radius: int = 4,
    line_thickness: int = 2,
) -> None:
    """
    Draw hand skeleton on frame in-place.

    Args:
        frame:      [H, W, 3] uint8 BGR — mutated
        keypoints:  [21, 2] float32 (u, v) pixel coords
        edges:      list of (i, j) keypoint connectivity
        track_id:   used to modulate colors for multi-hand
        dot_radius: keypoint dot radius in pixels
        line_thickness: skeleton line width
    """
    kp = keypoints.astype(int)

    # Draw edges
    for i, j in edges:
        pt1 = tuple(kp[i])
        pt2 = tuple(kp[j])
        color = _EDGE_TO_FINGER.get((min(i,j), max(i,j)), "palm")
        bgr = _FINGER_COLORS[color]
        # Dim color for secondary hand (odd track_id)
        if track_id % 2 == 1:
            bgr = tuple(c // 2 for c in bgr)
        cv2.line(frame, pt1, pt2, bgr, line_thickness, cv2.LINE_AA)

    # Draw joints
    for u, v in kp:
        cv2.circle(frame, (u, v), dot_radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (u, v), dot_radius, (0, 0, 0), 1, cv2.LINE_AA)

    # Track ID label at wrist
    wrist_u, wrist_v = kp[0]
    cv2.putText(
        frame,
        f"#{track_id}",
        (wrist_u + 5, wrist_v + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def viz_gesture_overlay(
    frame: np.ndarray,
    result: GestureResult,
    keypoints: np.ndarray,
) -> None:
    """
    Draw gesture label and VR-relevant overlays on frame in-place.

    Overlays:
      - Gesture label + confidence at top-left of hand bbox
      - Pinch line between thumb and index tips
      - Point ray arrow from wrist

    Args:
        frame:     [H, W, 3] uint8 BGR — mutated
        result:    GestureResult from GestureState.update()
        keypoints: [21, 2] float32 (u, v) pixel coords
    """
    kp = keypoints.astype(int)
    color = _GESTURE_COLORS.get(result.gesture, (180, 180, 180))

    # Gesture label
    x_min = int(kp[:, 0].min()) - 5
    y_min = int(kp[:, 1].min()) - 20
    label = f"{result.gesture.name}  {result.confidence:.2f}"
    cv2.putText(frame, label, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)

    # Pinch: draw line between thumb tip and index tip
    if result.gesture == Gesture.PINCH:
        thumb = tuple(kp[4])
        index = tuple(kp[8])
        cv2.line(frame, thumb, index, color, 2, cv2.LINE_AA)
        dist_text = f"{result.pinch_distance:.0f}px"
        mid = ((kp[4][0] + kp[8][0]) // 2, (kp[4][1] + kp[8][1]) // 2)
        cv2.putText(frame, dist_text, mid, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)

    # Point: draw ray arrow
    if result.gesture == Gesture.POINT and result.point_ray is not None:
        wrist = kp[0]
        ray_len = 80
        tip = (int(wrist[0] + result.point_ray[0] * ray_len),
               int(wrist[1] + result.point_ray[1] * ray_len))
        cv2.arrowedLine(frame, tuple(wrist), tip, color, 2, cv2.LINE_AA, tipLength=0.3)
