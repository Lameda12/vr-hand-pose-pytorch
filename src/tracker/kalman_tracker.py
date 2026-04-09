"""
Kalman filter tracker for 21-keypoint hand pose.

State vector: [u1, v1, u2, v2, ..., u21, v21,   # 42 position dims
               du1, dv1, ..., du21, dv21]         # 42 velocity dims
Total state dim: 84

Handles:
- Smooth tracking during rapid hand motion
- Predict-only during occlusion (no observation update)
- Re-identification via IoU matching on bounding boxes
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class HandTrack:
    """
    Single tracked hand instance.

    Attributes:
        track_id:      unique integer ID persisting across frames
        state:         Kalman state [84] = [pos×42, vel×42]
        cov:           Kalman covariance [84, 84]
        age:           frames since track was created
        hits:          consecutive detection frames
        misses:        consecutive missed detection frames
        last_bbox:     (x1, y1, x2, y2) in pixel coords, last observed
    """
    track_id: int
    state: np.ndarray       # [84]
    cov: np.ndarray         # [84, 84]
    age: int = 0
    hits: int = 0
    misses: int = 0
    last_bbox: Optional[np.ndarray] = None   # [4]


# Constant-velocity motion model
_DIM_X = 84   # state dim
_DIM_Z = 42   # observation dim (keypoint positions only)

# F: state transition [84, 84]
_F = np.eye(_DIM_X, dtype=np.float32)
_F[:_DIM_Z, _DIM_Z:] = np.eye(_DIM_Z, dtype=np.float32)   # pos += vel * dt

# H: observation matrix [42, 84] — observe positions only
_H = np.zeros((_DIM_Z, _DIM_X), dtype=np.float32)
_H[:, :_DIM_Z] = np.eye(_DIM_Z, dtype=np.float32)

# Process noise Q [84, 84]
_Q = np.eye(_DIM_X, dtype=np.float32)
_Q[:_DIM_Z, :_DIM_Z] *= 0.01    # position noise
_Q[_DIM_Z:, _DIM_Z:] *= 0.1     # velocity noise

# Measurement noise R [42, 42]
_R = np.eye(_DIM_Z, dtype=np.float32) * 2.0   # ~2px measurement noise


class KalmanHandTracker:
    """
    Multi-hand Kalman filter tracker with IoU-based data association.

    Usage:
        tracker = KalmanHandTracker(max_misses=5)
        for frame in video:
            detections = model.infer(frame)   # list of (keypoints[21,2], bbox[4])
            tracks = tracker.update(detections)
            for t in tracks:
                draw(t.state[:42].reshape(21, 2), t.track_id)

    Args:
        max_misses:      frames without detection before track is deleted
        min_hits:        detections before track is confirmed
        iou_threshold:   minimum IoU for bbox association
    """

    def __init__(
        self,
        max_misses: int = 5,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: List[HandTrack] = []
        self._next_id = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Tuple[np.ndarray, np.ndarray]],
    ) -> List[HandTrack]:
        """
        Update tracker with new detections.

        Args:
            detections: list of (keypoints, bbox) where
                keypoints: [21, 2] float32, pixel coords
                bbox:      [4] float32, (x1, y1, x2, y2) pixel coords

        Returns:
            confirmed active tracks, each with updated state
        """
        # Predict step for all existing tracks
        for t in self._tracks:
            self._predict(t)

        # Associate detections to tracks via bbox IoU
        matched, unmatched_dets, unmatched_trks = self._associate(detections)

        # Update matched tracks
        for det_idx, trk_idx in matched:
            kp, bbox = detections[det_idx]
            obs = kp.flatten()   # [42]
            self._update(self._tracks[trk_idx], obs, bbox)

        # Spawn new tracks for unmatched detections
        for det_idx in unmatched_dets:
            kp, bbox = detections[det_idx]
            self._spawn(kp.flatten(), bbox)

        # Increment misses for unmatched tracks
        for trk_idx in unmatched_trks:
            self._tracks[trk_idx].misses += 1

        # Prune dead tracks
        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]

        # Return confirmed tracks only
        return [t for t in self._tracks if t.hits >= self.min_hits]

    def reset(self) -> None:
        """Clear all tracks. Call on scene change."""
        self._tracks = []

    # ------------------------------------------------------------------
    # Kalman internals
    # ------------------------------------------------------------------

    def _predict(self, t: HandTrack) -> None:
        """Advance state estimate one step (constant-velocity model)."""
        t.state = _F @ t.state
        t.cov = _F @ t.cov @ _F.T + _Q
        t.age += 1

    def _update(self, t: HandTrack, obs: np.ndarray, bbox: np.ndarray) -> None:
        """
        Kalman update step.

        Args:
            t:   track to update
            obs: [42] observed keypoint positions
            bbox:[4]  observed bounding box
        """
        S = _H @ t.cov @ _H.T + _R           # innovation cov [42, 42]
        K = t.cov @ _H.T @ np.linalg.inv(S)  # Kalman gain [84, 42]
        innovation = obs - _H @ t.state
        t.state = t.state + K @ innovation
        t.cov = (np.eye(_DIM_X) - K @ _H) @ t.cov
        t.last_bbox = bbox
        t.hits += 1
        t.misses = 0

    def _spawn(self, obs: np.ndarray, bbox: np.ndarray) -> None:
        """
        Create a new track from a detection.

        Args:
            obs: [42] initial keypoint positions
            bbox:[4]  bounding box
        """
        state = np.zeros(_DIM_X, dtype=np.float32)
        state[:_DIM_Z] = obs
        cov = np.eye(_DIM_X, dtype=np.float32) * 100.0   # high initial uncertainty
        track = HandTrack(
            track_id=self._next_id,
            state=state,
            cov=cov,
            hits=1,
            last_bbox=bbox,
        )
        self._tracks.append(track)
        self._next_id += 1

    # ------------------------------------------------------------------
    # Data association
    # ------------------------------------------------------------------

    def _associate(
        self,
        detections: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Greedy IoU matching between detections and tracks.

        Returns:
            matched:        [(det_idx, trk_idx), ...]
            unmatched_dets: [det_idx, ...]
            unmatched_trks: [trk_idx, ...]
        """
        if not self._tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self._tracks)))

        # Predicted bbox from track state (derive from keypoint bounding box)
        trk_boxes = []
        for t in self._tracks:
            if t.last_bbox is not None:
                # Use last observed bbox shifted by velocity estimate
                vel = t.state[_DIM_Z:][:4]   # approx velocity of first 2 kps
                trk_boxes.append(t.last_bbox)
            else:
                kp = t.state[:_DIM_Z].reshape(21, 2)
                x1, y1 = kp.min(0)
                x2, y2 = kp.max(0)
                trk_boxes.append(np.array([x1, y1, x2, y2]))

        det_boxes = [d[1] for d in detections]
        iou_mat = self._iou_matrix(det_boxes, trk_boxes)   # [D, T]

        matched, unmatched_dets, unmatched_trks = [], [], []
        assigned_trks = set()
        assigned_dets = set()

        # Greedy: match highest IoU pairs
        flat_order = np.argsort(-iou_mat.flatten())
        for idx in flat_order:
            d = idx // len(trk_boxes)
            t = idx % len(trk_boxes)
            if iou_mat[d, t] < self.iou_threshold:
                break
            if d in assigned_dets or t in assigned_trks:
                continue
            matched.append((d, t))
            assigned_dets.add(d)
            assigned_trks.add(t)

        unmatched_dets = [i for i in range(len(detections)) if i not in assigned_dets]
        unmatched_trks = [i for i in range(len(self._tracks)) if i not in assigned_trks]
        return matched, unmatched_dets, unmatched_trks

    @staticmethod
    def _iou_matrix(
        boxes_a: List[np.ndarray],
        boxes_b: List[np.ndarray],
    ) -> np.ndarray:
        """
        Compute pairwise IoU between two sets of (x1,y1,x2,y2) boxes.

        Returns: [len(a), len(b)] float32
        """
        iou = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
        for i, a in enumerate(boxes_a):
            for j, b in enumerate(boxes_b):
                ix1 = max(a[0], b[0])
                iy1 = max(a[1], b[1])
                ix2 = min(a[2], b[2])
                iy2 = min(a[3], b[3])
                inter_w = max(0.0, ix2 - ix1)
                inter_h = max(0.0, iy2 - iy1)
                inter = inter_w * inter_h
                area_a = (a[2] - a[0]) * (a[3] - a[1])
                area_b = (b[2] - b[0]) * (b[3] - b[1])
                union = area_a + area_b - inter
                iou[i, j] = inter / (union + 1e-6)
        return iou
