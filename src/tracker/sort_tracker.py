"""
SORT (Simple Online and Realtime Tracking) adapter for hand poses.

Wraps the Kalman tracker with per-frame confidence filtering and
provides a clean interface matching KalmanHandTracker.update().

Reference: Bewley et al. 2016 (arXiv:1602.00763)
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .kalman_tracker import KalmanHandTracker, HandTrack


class SORTTracker:
    """
    SORT-style tracker: Kalman predict + IoU associate + Hungarian assign.

    Difference from KalmanHandTracker: uses confidence scores to filter
    detections before association; implements minimum-confidence gating.

    Args:
        min_confidence:  detection score threshold
        max_misses:      frames before track deletion
        min_hits:        detections before track confirmed
        iou_threshold:   association IoU gate

    Usage is identical to KalmanHandTracker:
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
        self._kalman = KalmanHandTracker(
            max_misses=max_misses,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
        )

    def update(
        self,
        detections: List[Tuple[np.ndarray, np.ndarray]],
        scores: Optional[List[float]] = None,
    ) -> List[HandTrack]:
        """
        Update tracks with new detections, optionally filtered by confidence.

        Args:
            detections: list of (keypoints[21,2], bbox[4])
            scores:     optional detection confidence scores, same length

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
        """Currently active (confirmed) tracks."""
        return [t for t in self._kalman._tracks if t.hits >= self._kalman.min_hits]
