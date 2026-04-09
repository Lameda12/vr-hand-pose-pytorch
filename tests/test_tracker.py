"""
Unit tests for src/tracker/.

Run: pytest tests/test_tracker.py -v
"""

import pytest
import numpy as np
from src.tracker.kalman_tracker import KalmanHandTracker, HandTrack
from src.tracker.sort_tracker import SORTTracker


def _make_detection(
    center: tuple[float, float] = (100.0, 100.0),
    size: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a fake (keypoints[21,2], bbox[4]) detection."""
    cx, cy = center
    kp = np.tile(np.array([[cx, cy]], dtype=np.float32), (21, 1))
    kp += np.random.randn(21, 2).astype(np.float32) * 5.0   # small jitter
    bbox = np.array([cx - size/2, cy - size/2, cx + size/2, cy + size/2], dtype=np.float32)
    return kp, bbox


class TestKalmanHandTracker:
    def test_spawn_on_first_detection(self):
        tracker = KalmanHandTracker(min_hits=1)
        dets = [_make_detection()]
        tracks = tracker.update(dets)
        assert len(tracks) == 1
        assert tracks[0].track_id == 0

    def test_track_id_persistence(self):
        tracker = KalmanHandTracker(min_hits=1)
        for _ in range(5):
            dets = [_make_detection(center=(100, 100))]
            tracks = tracker.update(dets)
        assert len(tracks) == 1
        assert tracks[0].track_id == 0
        assert tracks[0].hits >= 5

    def test_min_hits_gating(self):
        tracker = KalmanHandTracker(min_hits=3)
        dets = [_make_detection()]
        # Hit 1: not confirmed
        tracks = tracker.update(dets)
        assert len(tracks) == 0
        # Hit 2: not confirmed
        tracks = tracker.update(dets)
        assert len(tracks) == 0
        # Hit 3: confirmed
        tracks = tracker.update(dets)
        assert len(tracks) == 1

    def test_occlusion_predict_only(self):
        tracker = KalmanHandTracker(min_hits=1, max_misses=3)
        dets = [_make_detection(center=(100, 100))]
        tracker.update(dets)
        tracker.update(dets)

        # No detections for 2 frames — predict-only
        tracks_1 = tracker.update([])
        tracks_2 = tracker.update([])
        # Track still alive (misses < max_misses)
        # (May not be in confirmed list if hits requirement resets — just check alive)
        assert len(tracker._tracks) == 1

    def test_track_deletion_on_max_misses(self):
        tracker = KalmanHandTracker(min_hits=1, max_misses=2)
        dets = [_make_detection()]
        tracker.update(dets)
        tracker.update(dets)
        tracker.update([])
        tracker.update([])
        tracker.update([])   # 3rd miss → deleted
        assert len(tracker._tracks) == 0

    def test_two_hands_separate_ids(self):
        tracker = KalmanHandTracker(min_hits=1, iou_threshold=0.1)
        dets = [
            _make_detection(center=(100, 100)),
            _make_detection(center=(400, 300)),
        ]
        tracks = tracker.update(dets)
        assert len(tracks) == 2
        ids = {t.track_id for t in tracks}
        assert len(ids) == 2   # distinct IDs

    def test_state_shape(self):
        tracker = KalmanHandTracker(min_hits=1)
        tracks = tracker.update([_make_detection()])
        assert tracks[0].state.shape == (84,)
        assert tracks[0].cov.shape == (84, 84)

    def test_reset(self):
        tracker = KalmanHandTracker(min_hits=1)
        tracker.update([_make_detection()])
        tracker.reset()
        assert len(tracker._tracks) == 0

    def test_iou_matrix_overlap(self):
        a = [np.array([0, 0, 10, 10], dtype=np.float32)]
        b = [np.array([5, 5, 15, 15], dtype=np.float32)]
        iou = KalmanHandTracker._iou_matrix(a, b)
        # Intersection: 5×5=25, Union: 100+100-25=175
        expected = 25.0 / 175.0
        assert abs(iou[0, 0] - expected) < 1e-4

    def test_iou_matrix_no_overlap(self):
        a = [np.array([0, 0, 10, 10], dtype=np.float32)]
        b = [np.array([20, 20, 30, 30], dtype=np.float32)]
        iou = KalmanHandTracker._iou_matrix(a, b)
        assert iou[0, 0] == pytest.approx(0.0)


class TestSORTTracker:
    def test_confidence_filtering(self):
        tracker = SORTTracker(min_confidence=0.8, min_hits=1)
        dets = [_make_detection()]
        scores = [0.5]   # below threshold
        tracks = tracker.update(dets, scores)
        assert len(tracks) == 0

    def test_confidence_passes(self):
        tracker = SORTTracker(min_confidence=0.4, min_hits=1)
        dets = [_make_detection()]
        scores = [0.9]
        tracks = tracker.update(dets, scores)
        assert len(tracks) == 1

    def test_no_scores_passthrough(self):
        tracker = SORTTracker(min_hits=1)
        dets = [_make_detection()]
        tracks = tracker.update(dets, scores=None)
        assert len(tracks) == 1

    def test_reset(self):
        tracker = SORTTracker(min_hits=1)
        tracker.update([_make_detection()])
        tracker.reset()
        assert len(tracker.active_tracks) == 0
