"""
Unit tests for gesture state machine.

Run: pytest tests/test_gesture.py -v
"""

import pytest
import numpy as np
from src.demo.gesture_state import GestureState, GestureResult, Gesture, WRIST, THUMB_TIP, INDEX_TIP


def _make_keypoints(wrist=(200, 300), scale=100.0) -> np.ndarray:
    """
    Build a neutral open-palm keypoint array.
    Fingers are straight up from the wrist (decreasing v = up).
    """
    kp = np.zeros((21, 2), dtype=np.float32)
    kp[WRIST] = wrist

    # Thumb: spread to the left
    for i, idx in enumerate([1, 2, 3, 4]):
        kp[idx] = (wrist[0] - (i+1)*15, wrist[1] - (i+1)*20)

    # Index–Pinky: straight up, evenly spaced
    for finger_i, base_idx in enumerate([5, 9, 13, 17]):
        for j in range(4):
            kp[base_idx + j] = (wrist[0] + (finger_i - 2) * 20, wrist[1] - (j+1) * scale/4)

    return kp


class TestGestureState:
    def test_pinch_detected(self):
        gs = GestureState(pinch_px_threshold=30.0)
        kp = _make_keypoints()
        midpoint = (kp[WRIST][0] + 50, kp[WRIST][1] - 50)
        # Place thumb and index tips very close together
        kp[THUMB_TIP] = np.array([midpoint[0], midpoint[1]], dtype=np.float32)
        kp[INDEX_TIP] = np.array([midpoint[0] + 5, midpoint[1]], dtype=np.float32)
        result = gs.update(kp)
        assert result.gesture == Gesture.PINCH
        assert result.pinch_distance < 30.0
        assert result.confidence > 0.0

    def test_no_pinch_when_far(self):
        gs = GestureState(pinch_px_threshold=30.0)
        kp = _make_keypoints()
        kp[THUMB_TIP] = np.array([100.0, 100.0], dtype=np.float32)
        kp[INDEX_TIP] = np.array([300.0, 300.0], dtype=np.float32)
        result = gs.update(kp)
        assert result.gesture != Gesture.PINCH

    def test_pinch_drag_delta_origin(self):
        gs = GestureState(pinch_px_threshold=30.0)
        kp = _make_keypoints()
        kp[THUMB_TIP] = np.array([200.0, 200.0], dtype=np.float32)
        kp[INDEX_TIP] = np.array([210.0, 200.0], dtype=np.float32)

        result1 = gs.update(kp)
        assert result1.gesture == Gesture.PINCH
        assert result1.drag_delta is not None
        # First frame: delta should be near zero (origin set this frame)
        assert np.linalg.norm(result1.drag_delta) < 1.0

    def test_pinch_drag_delta_moves(self):
        gs = GestureState(pinch_px_threshold=30.0)
        kp = _make_keypoints()

        # Frame 1: pinch at (200, 200)
        kp[THUMB_TIP] = np.array([200.0, 200.0], dtype=np.float32)
        kp[INDEX_TIP] = np.array([205.0, 200.0], dtype=np.float32)
        gs.update(kp)

        # Frame 2: pinch moved to (250, 200)
        kp2 = kp.copy()
        kp2[THUMB_TIP] = np.array([250.0, 200.0], dtype=np.float32)
        kp2[INDEX_TIP] = np.array([255.0, 200.0], dtype=np.float32)
        result2 = gs.update(kp2)

        assert result2.drag_delta is not None
        assert result2.drag_delta[0] > 40.0   # moved ~50px in u

    def test_pinch_origin_resets_after_release(self):
        gs = GestureState(pinch_px_threshold=30.0)
        kp = _make_keypoints()

        # Pinch
        kp[THUMB_TIP] = np.array([200.0, 200.0], dtype=np.float32)
        kp[INDEX_TIP] = np.array([205.0, 200.0], dtype=np.float32)
        gs.update(kp)

        # Release
        kp2 = kp.copy()
        kp2[THUMB_TIP] = np.array([100.0, 100.0], dtype=np.float32)
        kp2[INDEX_TIP] = np.array([300.0, 300.0], dtype=np.float32)
        gs.update(kp2)
        assert gs._pinch_origin is None

    def test_point_ray_direction(self):
        gs = GestureState()
        kp = _make_keypoints(wrist=(200, 300))
        # Set index tip straight up
        kp[INDEX_TIP] = np.array([200.0, 100.0], dtype=np.float32)
        result = gs.update(kp)

        if result.gesture == Gesture.POINT and result.point_ray is not None:
            # Ray should point upward (negative v direction)
            assert result.point_ray[1] < 0

    def test_invalid_keypoints_raises(self):
        gs = GestureState()
        bad_kp = np.zeros((15, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="21 keypoints"):
            gs.update(bad_kp)

    def test_accepts_3d_keypoints(self):
        gs = GestureState(pinch_px_threshold=30.0)
        kp3d = np.zeros((21, 3), dtype=np.float32)
        kp3d[:, :2] = _make_keypoints()
        # Pinch position
        kp3d[THUMB_TIP] = [200, 200, 0.1]
        kp3d[INDEX_TIP] = [205, 200, 0.1]
        result = gs.update(kp3d)
        assert result.gesture == Gesture.PINCH

    def test_confidence_in_range(self):
        gs = GestureState()
        for _ in range(10):
            kp = _make_keypoints()
            kp += np.random.randn(21, 2).astype(np.float32) * 20
            result = gs.update(kp)
            assert 0.0 <= result.confidence <= 1.0
