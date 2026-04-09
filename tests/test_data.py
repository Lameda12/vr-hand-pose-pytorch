"""
Unit tests for src/data/ (heatmap utils + augmentation).

FreiHANDDataset tests skipped if dataset not present (CI-safe).

Run: pytest tests/test_data.py -v
"""

import pytest
import numpy as np
import cv2
import torch

from src.data.heatmap_utils import (
    build_heatmaps,
    softargmax2d,
    keypoints_to_heatmap_coords,
    heatmap_coords_to_image,
)
from src.data.augmentation import (
    HandAugmentation,
    HorizontalFlip,
    RandomRotation,
    RandomScale,
    ColorJitter,
    GaussianNoise,
    RandomOcclusion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_image():
    """200×200 gray BGR image."""
    return np.full((200, 200, 3), 128, dtype=np.uint8)


@pytest.fixture
def hand_keypoints():
    """21 keypoints arranged vertically up from wrist."""
    kp = np.zeros((21, 2), dtype=np.float32)
    kp[0] = [100, 160]   # wrist
    for i in range(1, 21):
        kp[i] = [100 + (i % 5) * 5 - 10, 160 - i * 5]
    return kp


@pytest.fixture
def visibility_all():
    return np.ones(21, dtype=np.float32)


# ---------------------------------------------------------------------------
# Heatmap utils
# ---------------------------------------------------------------------------

class TestBuildHeatmaps:
    def test_output_shape(self, hand_keypoints, visibility_all):
        hm = build_heatmaps(hand_keypoints, heatmap_size=(60, 80), sigma=2.0, visibility=visibility_all)
        assert hm.shape == (21, 60, 80)

    def test_value_range(self, hand_keypoints, visibility_all):
        hm = build_heatmaps(hand_keypoints, heatmap_size=(60, 80), visibility=visibility_all)
        assert hm.min() >= 0.0
        assert hm.max() <= 1.0 + 1e-6

    def test_peak_at_keypoint(self):
        kp = np.array([[40.0, 30.0]], dtype=np.float32)   # u=40, v=30
        kp = np.tile(kp, (21, 1))
        kp[0] = [40.0, 30.0]
        vis = np.ones(21, dtype=np.float32)
        hm = build_heatmaps(kp, heatmap_size=(60, 80), sigma=2.0, visibility=vis)
        # Peak of keypoint 0 should be at (v=30, u=40)
        peak_v, peak_u = np.unravel_index(hm[0].argmax(), hm[0].shape)
        assert abs(peak_u - 40) <= 1
        assert abs(peak_v - 30) <= 1

    def test_occluded_is_zero(self):
        kp = np.tile(np.array([[40.0, 30.0]], dtype=np.float32), (21, 1))
        vis = np.zeros(21, dtype=np.float32)
        hm = build_heatmaps(kp, heatmap_size=(60, 80), visibility=vis)
        assert hm.sum() == 0.0

    def test_none_visibility_treats_all_visible(self):
        kp = np.tile(np.array([[40.0, 30.0]], dtype=np.float32), (21, 1))
        hm = build_heatmaps(kp, heatmap_size=(60, 80), visibility=None)
        assert hm.sum() > 0.0

    def test_out_of_bounds_skipped(self):
        kp = np.tile(np.array([[-5.0, -5.0]], dtype=np.float32), (21, 1))
        vis = np.ones(21, dtype=np.float32)
        hm = build_heatmaps(kp, heatmap_size=(60, 80), visibility=vis)
        assert hm.sum() == 0.0


class TestSoftArgmax2D:
    def test_output_shape(self):
        hm = torch.randn(2, 21, 60, 80)
        kp = softargmax2d(hm)
        assert kp.shape == (2, 21, 2)

    def test_peak_recovery(self):
        """Soft-argmax should recover peak location of a sharp heatmap."""
        hm = torch.zeros(1, 1, 30, 40)
        hm[0, 0, 15, 20] = 10.0   # sharp peak at (v=15, u=20)
        kp = softargmax2d(hm)
        assert abs(kp[0, 0, 0].item() - 20.0) < 1.0   # u
        assert abs(kp[0, 0, 1].item() - 15.0) < 1.0   # v

    def test_differentiable(self):
        hm = torch.randn(1, 21, 30, 40, requires_grad=True)
        kp = softargmax2d(hm)
        kp.sum().backward()
        assert hm.grad is not None


class TestKeypointScaling:
    def test_roundtrip(self):
        kp = np.random.rand(21, 2).astype(np.float32) * np.array([640, 480])
        kp_hm = keypoints_to_heatmap_coords(kp, image_size=(480, 640), heatmap_size=(120, 160))
        kp_back = heatmap_coords_to_image(kp_hm, image_size=(480, 640), heatmap_size=(120, 160))
        np.testing.assert_allclose(kp, kp_back, atol=1e-4)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class TestHorizontalFlip:
    def test_flip_applied(self, blank_image, hand_keypoints, visibility_all):
        import random
        random.seed(0)   # seed 0 may or may not flip — run 20 times
        flipped_any = False
        flip = HorizontalFlip(p=1.0)   # always flip
        img2, kp2, vis2 = flip(blank_image.copy(), hand_keypoints.copy(), visibility_all.copy())
        W = blank_image.shape[1]
        # u' = W - 1 - u
        np.testing.assert_allclose(kp2[:, 0], W - 1 - hand_keypoints[:, 0], atol=1)

    def test_flip_preserves_shape(self, blank_image, hand_keypoints, visibility_all):
        flip = HorizontalFlip(p=1.0)
        img2, kp2, vis2 = flip(blank_image, hand_keypoints, visibility_all)
        assert img2.shape == blank_image.shape
        assert kp2.shape == hand_keypoints.shape

    def test_no_flip_at_p0(self, blank_image, hand_keypoints, visibility_all):
        flip = HorizontalFlip(p=0.0)
        img2, kp2, vis2 = flip(blank_image.copy(), hand_keypoints.copy(), visibility_all.copy())
        np.testing.assert_array_equal(kp2, hand_keypoints)


class TestRandomRotation:
    def test_output_shapes(self, blank_image, hand_keypoints, visibility_all):
        rot = RandomRotation(max_degrees=30, p=1.0)
        img2, kp2, vis2 = rot(blank_image.copy(), hand_keypoints.copy(), visibility_all.copy())
        assert img2.shape == blank_image.shape
        assert kp2.shape == (21, 2)
        assert vis2.shape == (21,)

    def test_no_rotate_at_p0(self, blank_image, hand_keypoints, visibility_all):
        rot = RandomRotation(p=0.0)
        img2, kp2, _ = rot(blank_image.copy(), hand_keypoints.copy(), visibility_all.copy())
        np.testing.assert_array_equal(kp2, hand_keypoints)


class TestColorJitter:
    def test_image_unchanged_at_p0(self, blank_image, hand_keypoints, visibility_all):
        jitter = ColorJitter(p=0.0)
        img2, _, _ = jitter(blank_image.copy(), hand_keypoints, visibility_all)
        np.testing.assert_array_equal(img2, blank_image)

    def test_keypoints_unchanged(self, blank_image, hand_keypoints, visibility_all):
        jitter = ColorJitter(p=1.0)
        _, kp2, _ = jitter(blank_image.copy(), hand_keypoints.copy(), visibility_all.copy())
        np.testing.assert_array_equal(kp2, hand_keypoints)

    def test_output_uint8(self, blank_image, hand_keypoints, visibility_all):
        jitter = ColorJitter(p=1.0)
        img2, _, _ = jitter(blank_image.copy(), hand_keypoints, visibility_all)
        assert img2.dtype == np.uint8


class TestHandAugmentation:
    def test_identity_no_change(self, blank_image, hand_keypoints, visibility_all):
        aug = HandAugmentation.identity()
        img2, kp2, vis2 = aug(blank_image.copy(), hand_keypoints.copy(), visibility_all.copy())
        np.testing.assert_array_equal(img2, blank_image)
        np.testing.assert_array_equal(kp2, hand_keypoints)

    def test_full_aug_shape_preserved(self, blank_image, hand_keypoints, visibility_all):
        aug = HandAugmentation()
        img2, kp2, vis2 = aug(blank_image.copy(), hand_keypoints.copy(), visibility_all.copy())
        assert img2.shape == blank_image.shape
        assert kp2.shape == (21, 2)
        assert vis2.shape == (21,)

    def test_from_config(self):
        cfg = {
            "augmentation": {
                "horizontal_flip": True,
                "rotation_max_deg": 20,
                "scale_range": [0.9, 1.1],
                "brightness_jitter": 0.2,
                "color_jitter": 0.2,
            }
        }
        aug = HandAugmentation.from_config(cfg)
        assert aug is not None

    def test_visibility_stays_binary(self, blank_image, hand_keypoints, visibility_all):
        aug = HandAugmentation()
        for _ in range(5):
            _, _, vis2 = aug(blank_image.copy(), hand_keypoints.copy(), visibility_all.copy())
            assert ((vis2 == 0.0) | (vis2 == 1.0)).all(), "Visibility must be binary {0, 1}"


# ---------------------------------------------------------------------------
# FreiHAND (skipped if dataset absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("pathlib").Path("/data/freihand/training_K.json").exists(),
    reason="FreiHAND dataset not found at /data/freihand"
)
class TestFreiHANDDataset:
    def test_sample_shapes(self):
        from src.data.freihand import FreiHANDDataset
        ds = FreiHANDDataset("/data/freihand", split="train", max_samples=4)
        sample = ds[0]
        assert sample["image"].shape == (3, 480, 640)
        assert sample["heatmaps_gt"].shape == (21, 120, 160)
        assert sample["det_mask"].shape == (60, 80)
        assert sample["kp_mask"].shape == (21,)
        assert sample["depth_gt"].shape == (21,)

    def test_image_normalized(self):
        from src.data.freihand import FreiHANDDataset
        ds = FreiHANDDataset("/data/freihand", split="train", max_samples=2)
        img = ds[0]["image"]
        # After ImageNet normalization values should span negative range
        assert img.min() < -1.0

    def test_heatmap_range(self):
        from src.data.freihand import FreiHANDDataset
        ds = FreiHANDDataset("/data/freihand", split="train", max_samples=2)
        hm = ds[0]["heatmaps_gt"]
        assert hm.min() >= 0.0
        assert hm.max() <= 1.0 + 1e-5

    def test_depth_range(self):
        from src.data.freihand import FreiHANDDataset
        ds = FreiHANDDataset("/data/freihand", split="train", max_samples=2)
        depth = ds[0]["depth_gt"]
        assert depth.abs().max() <= 1.0 + 1e-5
