"""
Shared pytest fixtures for Turin test suite.

Provides reusable model instances, synthetic frames, and keypoint arrays
without requiring dataset downloads or GPU.
"""

import pytest
import numpy as np
import torch

from src.models.hand_pose_net import HandPoseNet, NUM_KEYPOINTS
from src.tracker.kalman_tracker import KalmanHandTracker
from src.tracker.sort_tracker import SORTTracker
from src.demo.gesture_state import GestureState


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def lightweight_model() -> HandPoseNet:
    """HandPoseNet with lightweight backbone — shared across all tests."""
    m = HandPoseNet(backbone_name="lightweight", pretrained_backbone=False, use_depth_head=True)
    m.eval()
    return m


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Consistent device selection for all tests (CPU in CI, CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def frame_480p() -> np.ndarray:
    """Synthetic 480p BGR uint8 frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def frame_720p() -> np.ndarray:
    """Synthetic 720p BGR uint8 frame."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def batch_tensor_480p() -> torch.Tensor:
    """Random [2, 3, 480, 640] float32 tensor (2-sample batch)."""
    return torch.rand(2, 3, 480, 640)


@pytest.fixture
def hand_keypoints_2d() -> np.ndarray:
    """
    21 synthetic hand keypoints [21, 2] in a plausible layout.
    Wrist at (320, 400), fingers spread upward.
    """
    kp = np.zeros((21, 2), dtype=np.float32)
    kp[0] = [320, 400]   # wrist

    finger_bases = [(280, 350), (300, 340), (320, 335), (340, 340), (360, 350)]
    for finger_i, (bx, by) in enumerate(finger_bases):
        base_idx = 1 + finger_i * 4
        for joint_j in range(4):
            kp[base_idx + joint_j] = (bx, by - joint_j * 25)

    return kp


@pytest.fixture
def visibility_full() -> np.ndarray:
    """All 21 keypoints visible."""
    return np.ones(NUM_KEYPOINTS, dtype=np.float32)


@pytest.fixture
def visibility_partial() -> np.ndarray:
    """Only wrist + index finger visible (rest occluded)."""
    vis = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
    vis[0] = 1.0    # wrist
    vis[5:9] = 1.0  # index finger
    return vis


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def make_detection(
    center: tuple[float, float] = (320.0, 240.0),
    size: float = 120.0,
    jitter: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic (keypoints[21,2], bbox[4]) detection tuple.

    Args:
        center:  (u, v) centroid
        size:    approximate hand size in pixels
        jitter:  random noise on keypoints

    Returns:
        (keypoints[21,2], bbox[4]) suitable for tracker.update()
    """
    cx, cy = center
    kp = np.tile(np.array([[cx, cy]], dtype=np.float32), (21, 1))
    kp += np.random.randn(21, 2).astype(np.float32) * jitter
    bbox = np.array([cx - size/2, cy - size/2, cx + size/2, cy + size/2], dtype=np.float32)
    return kp, bbox


@pytest.fixture
def kalman_tracker() -> KalmanHandTracker:
    """Fresh KalmanHandTracker for each test."""
    return KalmanHandTracker(max_misses=5, min_hits=1)


@pytest.fixture
def sort_tracker() -> SORTTracker:
    """Fresh SORTTracker for each test."""
    return SORTTracker(min_confidence=0.0, min_hits=1)


@pytest.fixture
def gesture_state() -> GestureState:
    """Fresh GestureState for each test."""
    return GestureState(pinch_px_threshold=30.0)
