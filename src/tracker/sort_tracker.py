"""
SORT (Simple Online and Realtime Tracking) adapter for hand poses.

Wraps the Kalman tracker with:
  - Confidence score filtering before association
  - Hungarian algorithm assignment (scipy.linear_sum_assignment) for
    globally optimal matching vs. greedy IoU

Reference: Bewley et al. 2016 (arXiv:1602.00763)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

from .kalman_tracker import KalmanHandTracker, HandTrack


def _hungarian_assign(
    cost_matrix: np.ndarray,
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Optimal assignment via Hungarian algorithm (scipy).

    Falls back to greedy if scipy unavailable (shouldn't happen in practice).

    Args:
        cost_matrix:   [D, T] IoU scores (higher = better match)
        iou_threshold: minimum IoU to accept a match

    Returns:
        matched:        [(det_idx, trk_idx), ...]
        unmatched_dets: [det_idx, ...]
        unmatched_trks: [trk_idx, ...]
    """
    D, T = cost_matrix.shape
    if D == 0 or T == 0:
        return [], list(range(D)), list(range(T))

    try:
        from scipy.optimize import linear_sum_assignment
        # linear_sum_assignment minimizes cost; negate IoU to maximize
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    except ImportError:
        # Greedy fallback
        row_ind, col_ind = [], []
        used_rows, used_cols = set(), set()
        flat_order = np.argsort(-cost_matrix.flatten())
        for idx in flat_order:
            r = idx // T
            c = idx % T
            if r in used_rows or c in used_cols:
                continue
            row_ind.append(r)
            col_ind.append(c)
            used_rows.add(r)
            used_cols.add(c)

    matched, unmatched_dets, unmatched_trks = [], [], []
    assigned_d: set = set()
    assigned_t: set = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] >= iou_threshold:
            matched.append((int(r), int(c)))
            assigned_d.add(int(r))
            assigned_t.add(int(c))

    unmatched_dets = [i for i in range(D) if i not in assigned_d]
    unmatched_trks = [i for i in range(T) if i not in assigned_t]
    return matched, unmatched_dets, unmatched_trks


class SORTTracker:
    """
    SORT-style tracker: Kalman predict + IoU associate + Hungarian assign.

    Improvements over KalmanHandTracker:
      1. Globally-optimal assignment via Hungarian algorithm
      2. Confidence score filtering before association
      3. Deterministic behavior regardless of insertion order

    Args:
        min_confidence:  detection score threshold (default 0.4)
        max_misses:      frames before track deletion (default 5)
        min_hits:        detections before track confirmed (default 2)
        iou_threshold:   minimum IoU to accept assignment (default 0.3)

    Usage:
        tracker = SORTTracker()
        tracks = tracker.update(detections, scores)
    """

    def __init__(
        self,
        min_confidence: float = 0.4,
        max_misses: int = 5,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
    ) -> None:
        self.min_confidence = min_confidence
        self.iou_threshold = iou_threshold
        self._kalman = KalmanHandTracker(
            max_misses=max_misses,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
        )
        # Monkey-patch the association method to use Hungarian
        self._kalman._associate = self._associate_hungarian  # type: ignore[method-assign]

    def update(
        self,
        detections: List[Tuple[np.ndarray, np.ndarray]],
        scores: Optional[List[float]] = None,
    ) -> List[HandTrack]:
        """
        Update tracks with new detections, filtered by confidence.

        Args:
            detections: list of (keypoints[21,2], bbox[4])
            scores:     optional per-detection confidence scores

        Returns:
            confirmed active tracks
        """
        if scores is not None:
            detections = [
                d for d, s in zip(detections, scores) if s >= self.min_confidence
            ]
        return self._kalman.update(detections)

    def reset(self) -> None:
        """Clear all tracks."""
        self._kalman.reset()

    @property
    def active_tracks(self) -> List[HandTrack]:
        """Currently confirmed tracks."""
        return [t for t in self._kalman._tracks if t.hits >= self._kalman.min_hits]

    # ------------------------------------------------------------------
    # Hungarian association (replaces greedy in KalmanHandTracker)
    # ------------------------------------------------------------------

    def _associate_hungarian(
        self,
        detections: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Globally-optimal IoU matching using scipy.linear_sum_assignment.

        Replaces KalmanHandTracker._associate (greedy) via monkey-patch.
        Same signature/return contract as the parent method.
        """
        tracks = self._kalman._tracks
        if not tracks or not detections:
            return [], list(range(len(detections))), list(range(len(tracks)))

        # Collect predicted track boxes
        trk_boxes = []
        for t in tracks:
            if t.last_bbox is not None:
                trk_boxes.append(t.last_bbox)
            else:
                kp = t.state[:42].reshape(21, 2)
                x1, y1 = kp.min(0)
                x2, y2 = kp.max(0)
                trk_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))

        det_boxes = [d[1] for d in detections]
        iou_matrix = KalmanHandTracker._iou_matrix(det_boxes, trk_boxes)  # [D, T]

        return _hungarian_assign(iou_matrix, self.iou_threshold)
