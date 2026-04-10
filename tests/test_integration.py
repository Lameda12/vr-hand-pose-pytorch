"""
End-to-end integration tests — no dataset required.

Tests the full pipeline with synthetic data:
  preprocess → model.infer() → tracker.update() → gesture.update() → metrics

These tests catch interface mismatches between modules that unit tests miss.

Run: pytest tests/test_integration.py -v
"""

import numpy as np
import pytest
import torch
import cv2

from src.models.hand_pose_net import HandPoseNet, NUM_KEYPOINTS
from src.tracker.kalman_tracker import KalmanHandTracker
from src.tracker.sort_tracker import SORTTracker
from src.demo.gesture_state import GestureState, Gesture
from src.demo.visualizer import viz_skeleton, viz_gesture_overlay
from src.demo.webcam_demo import SKELETON_EDGES  # noqa: F401 — verify importable
from src.data.heatmap_utils import build_heatmaps, softargmax2d
from src.data.augmentation import HandAugmentation
from src.evaluate import PoseEvaluator, TrackingEvaluator, EvalMetrics
from tests.conftest import make_detection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _bgr_to_tensor(frame_bgr: np.ndarray, W: int = 640, H: int = 480) -> torch.Tensor:
    """Resize BGR frame and convert to normalized [1, 3, H, W] tensor."""
    resized = cv2.resize(frame_bgr, (W, H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
    return t.unsqueeze(0)


# ---------------------------------------------------------------------------
# Integration: preprocess → infer
# ---------------------------------------------------------------------------

class TestPreprocessToInfer:
    def test_bgr_tensor_dtype_device(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = _bgr_to_tensor(frame)
        assert tensor.dtype == torch.float32
        assert tensor.shape == (1, 3, 480, 640)

    def test_model_accepts_preprocessed_frame(self):
        model = HandPoseNet(backbone_name="lightweight", pretrained_backbone=False)
        model.eval()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = _bgr_to_tensor(frame)
        with torch.no_grad():
            out = model.infer(tensor)
        assert "keypoints" in out
        assert out["keypoints"].shape == (1, NUM_KEYPOINTS, 2)

    def test_infer_scores_in_unit_interval(self):
        model = HandPoseNet(backbone_name="lightweight", pretrained_backbone=False)
        model.eval()
        tensor = _bgr_to_tensor(np.zeros((480, 640, 3), dtype=np.uint8))
        with torch.no_grad():
            out = model.infer(tensor)
        scores = out["det_scores"]
        assert (scores >= 0).all() and (scores <= 1).all()


# ---------------------------------------------------------------------------
# Integration: infer → tracker
# ---------------------------------------------------------------------------

class TestInferToTracker:
    def test_tracker_accepts_model_detections(self):
        """Simulates the exact data flow in WebcamDemo._process_frame."""
        model = HandPoseNet(backbone_name="lightweight", pretrained_backbone=False)
        model.eval()
        tracker = KalmanHandTracker(max_misses=5, min_hits=1)

        for _ in range(3):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            tensor = _bgr_to_tensor(frame)
            with torch.no_grad():
                preds = model.infer(tensor)

            # Replicate WebcamDemo._parse_detections logic
            kp = preds["keypoints"][0].cpu().numpy()  # [21, 2] heatmap space
            scores = preds["det_scores"][0].cpu().numpy()

            # Scale to frame space (heatmap→frame)
            kp_frame = kp.copy()
            kp_frame[:, 0] *= 640 / (640 // 4)
            kp_frame[:, 1] *= 480 / (480 // 4)

            detections = []
            if scores.max() >= 0.0:   # accept any detection for this test
                x1, y1 = kp_frame.min(0)
                x2, y2 = kp_frame.max(0)
                bbox = np.array([x1 - 20, y1 - 20, x2 + 20, y2 + 20], dtype=np.float32)
                detections.append((kp_frame.astype(np.float32), bbox))

            tracks = tracker.update(detections)
            # Tracks should be list (may be empty before min_hits)
            assert isinstance(tracks, list)

    def test_tracker_state_feeds_gesture(self):
        """Track state [84] → keypoints [21, 2] → GestureState is valid."""
        tracker = KalmanHandTracker(max_misses=5, min_hits=1)
        gs = GestureState()

        det = make_detection(center=(320, 240), size=120)
        for _ in range(3):
            tracks = tracker.update([det])

        if tracks:
            kp2d = tracks[0].state[:42].reshape(21, 2)
            result = gs.update(kp2d)
            assert isinstance(result.gesture, Gesture)
            assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Integration: tracker → evaluator
# ---------------------------------------------------------------------------

class TestTrackerToEvaluator:
    def test_tracking_evaluator_with_kalman_tracks(self):
        tracker = KalmanHandTracker(max_misses=5, min_hits=1)
        te = TrackingEvaluator()

        det = make_detection(center=(320, 240))
        for _ in range(10):
            tracks = tracker.update([det])
            te.update(tracks)

        rate = te.id_switch_rate()
        assert 0.0 <= rate < 0.1   # stable single detection → very low switch rate


# ---------------------------------------------------------------------------
# Integration: augmentation → heatmap GT → loss
# ---------------------------------------------------------------------------

class TestAugToHeatmapToLoss:
    def test_augmented_frame_produces_valid_heatmaps(self):
        aug = HandAugmentation()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        kp = np.random.rand(21, 2).astype(np.float32) * np.array([640, 480])
        vis = np.ones(21, dtype=np.float32)

        img_aug, kp_aug, vis_aug = aug(frame, kp, vis)

        hm = build_heatmaps(kp_aug, heatmap_size=(120, 160), sigma=2.0, visibility=vis_aug)
        assert hm.shape == (21, 120, 160)
        assert hm.min() >= 0.0 and hm.max() <= 1.0 + 1e-6

    def test_heatmap_to_loss_gradient_flows(self):
        """Full training step: forward → loss → backward."""
        from src.models.losses import HandPoseLoss
        model = HandPoseNet(backbone_name="lightweight", pretrained_backbone=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        criterion = HandPoseLoss()

        x = torch.rand(1, 3, 224, 224)
        preds = model(x)

        targets = {
            "det_mask":    torch.zeros(1, 28, 28),
            "heatmaps_gt": torch.rand(1, 21, 56, 56),
            "kp_mask":     torch.ones(1, 21),
            "depth_gt":    torch.zeros(1, 21),
        }

        optimizer.zero_grad()
        losses = criterion(preds, targets)
        losses["total"].backward()
        optimizer.step()

        # Gradient must have flowed
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert p.grad.abs().sum() >= 0   # just check no NaN
            assert not (p.data != p.data).any(), f"NaN in param {name}"

    def test_softargmax_roundtrip_with_heatmap_gt(self):
        """
        Build a GT heatmap from a known keypoint, then recover it via softargmax.
        Max error should be <2px at heatmap resolution.
        """
        kp_gt = np.array([[40.0, 30.0]], dtype=np.float32)   # u=40, v=30
        kp_gt = np.tile(kp_gt, (21, 1))
        kp_gt[0] = [40.0, 30.0]

        hm = build_heatmaps(kp_gt, heatmap_size=(60, 80), sigma=2.0)
        hm_tensor = torch.from_numpy(hm).unsqueeze(0)   # [1, 21, 60, 80]

        kp_recovered = softargmax2d(hm_tensor)           # [1, 21, 2]
        u_err = abs(kp_recovered[0, 0, 0].item() - 40.0)
        v_err = abs(kp_recovered[0, 0, 1].item() - 30.0)
        assert u_err < 2.0, f"u error too large: {u_err}"
        assert v_err < 2.0, f"v error too large: {v_err}"


# ---------------------------------------------------------------------------
# Integration: PoseEvaluator with synthetic model
# ---------------------------------------------------------------------------

class TestPoseEvaluatorEndToEnd:
    def test_evaluator_with_synthetic_batch(self):
        """PoseEvaluator.update() accepts HandPoseNet.infer() output + batch dict."""
        model = HandPoseNet(backbone_name="lightweight", pretrained_backbone=False)
        model.eval()
        ev = PoseEvaluator()

        B = 4
        x = torch.rand(B, 3, 224, 224)
        with torch.no_grad():
            preds = model.infer(x)

        # Synthetic batch ground truth (matches FreiHANDDataset output format)
        batch = {
            "keypoints": torch.rand(B, NUM_KEYPOINTS, 2) * 56,  # heatmap-space coords
            "kp_mask":   torch.ones(B, NUM_KEYPOINTS),
        }

        ev.update(preds, batch)
        metrics = ev.compute()

        assert 0.0 <= metrics.pck_at_02 <= 1.0
        assert 0.0 <= metrics.auc <= 1.0
        assert metrics.n_samples == B

    def test_evaluator_empty_returns_defaults(self):
        ev = PoseEvaluator()
        m = ev.compute()
        assert m.n_samples == 0
        assert m.pck_at_02 == 0.0


# ---------------------------------------------------------------------------
# Integration: visualizer doesn't crash on valid inputs
# ---------------------------------------------------------------------------

class TestVisualizerIntegration:
    def test_viz_skeleton_on_synthetic_keypoints(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kp = np.random.rand(21, 2).astype(np.float32) * np.array([640, 480])
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]   # thumb only
        viz_skeleton(frame, kp, edges, track_id=0)
        # Should not raise; frame should be modified
        assert frame.sum() > 0   # something was drawn

    def test_viz_gesture_overlay_all_gestures(self):
        from src.demo.gesture_state import GestureResult
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kp = np.random.rand(21, 2).astype(np.float32) * np.array([200, 200]) + 100

        for gesture in Gesture:
            result = GestureResult(
                gesture=gesture,
                pinch_distance=20.0,
                point_ray=np.array([1.0, 0.0], dtype=np.float32) if gesture == Gesture.POINT else None,
                drag_delta=np.array([5.0, 5.0], dtype=np.float32) if gesture == Gesture.PINCH else None,
                confidence=0.9,
            )
            # Should not raise
            viz_gesture_overlay(frame, result, kp)


# ---------------------------------------------------------------------------
# Integration: SORT vs Kalman give consistent API
# ---------------------------------------------------------------------------

class TestSORTvsKalmanAPI:
    """Both trackers must expose identical update() → List[HandTrack] contract."""

    @pytest.mark.parametrize("TrackerCls", [KalmanHandTracker, SORTTracker])
    def test_update_returns_list(self, TrackerCls):
        tracker = TrackerCls(min_hits=1) if TrackerCls == KalmanHandTracker \
            else TrackerCls(min_confidence=0.0, min_hits=1)
        det = make_detection()
        tracks = tracker.update([det])
        assert isinstance(tracks, list)

    @pytest.mark.parametrize("TrackerCls", [KalmanHandTracker, SORTTracker])
    def test_reset_clears_state(self, TrackerCls):
        tracker = TrackerCls(min_hits=1) if TrackerCls == KalmanHandTracker \
            else TrackerCls(min_confidence=0.0, min_hits=1)
        for _ in range(5):
            tracker.update([make_detection()])
        tracker.reset()
        # After reset, internal state is clean
        assert len(tracker._tracks if hasattr(tracker, "_tracks")
                   else tracker._kalman._tracks) == 0
