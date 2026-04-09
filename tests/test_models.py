"""
Unit tests for src/models/.

Run: pytest tests/test_models.py -v
"""

import pytest
import torch
import torch.nn as nn
from src.models.backbone import MobileNetV3Backbone, LightweightBackbone, build_backbone
from src.models.hand_pose_net import HandPoseNet, NUM_KEYPOINTS
from src.models.losses import FocalLoss, HeatmapLoss, DepthLoss, HandPoseLoss


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class TestMobileNetV3Backbone:
    def test_output_shape_224(self):
        model = MobileNetV3Backbone(pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 96, 28, 28), f"Unexpected shape: {out.shape}"

    def test_output_shape_480(self):
        model = MobileNetV3Backbone(pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 480, 640)
        with torch.no_grad():
            out = model(x)
        # stride=8: 480/8=60, 640/8=80
        assert out.shape == (1, 96, 60, 80), f"Unexpected shape: {out.shape}"

    def test_out_channels_attr(self):
        model = MobileNetV3Backbone(pretrained=False)
        assert model.out_channels == 96


class TestLightweightBackbone:
    def test_output_shape(self):
        model = LightweightBackbone()
        model.eval()
        x = torch.randn(2, 3, 480, 640)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 64, 60, 80), f"Unexpected shape: {out.shape}"

    def test_out_channels(self):
        assert LightweightBackbone().out_channels == 64


class TestBuildBackbone:
    def test_mobilenetv3(self):
        m = build_backbone("mobilenetv3", pretrained=False)
        assert isinstance(m, MobileNetV3Backbone)

    def test_lightweight(self):
        m = build_backbone("lightweight")
        assert isinstance(m, LightweightBackbone)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            build_backbone("resnet50")


# ---------------------------------------------------------------------------
# HandPoseNet
# ---------------------------------------------------------------------------

class TestHandPoseNet:
    @pytest.fixture
    def model(self):
        m = HandPoseNet(backbone_name="lightweight", pretrained_backbone=False, use_depth_head=True)
        m.eval()
        return m

    def test_forward_keys(self, model):
        x = torch.randn(1, 3, 480, 640)
        with torch.no_grad():
            out = model(x)
        assert "det_logits" in out
        assert "heatmaps" in out
        assert "depth" in out

    def test_det_logits_shape(self, model):
        x = torch.randn(2, 3, 480, 640)
        with torch.no_grad():
            out = model(x)
        # stride=8: 480/8=60, 640/8=80
        assert out["det_logits"].shape == (2, 5, 60, 80)

    def test_heatmap_shape(self, model):
        x = torch.randn(2, 3, 480, 640)
        with torch.no_grad():
            out = model(x)
        # stride=4: 480/4=120, 640/4=160
        assert out["heatmaps"].shape == (2, NUM_KEYPOINTS, 120, 160)

    def test_depth_shape(self, model):
        x = torch.randn(2, 3, 480, 640)
        with torch.no_grad():
            out = model(x)
        assert out["depth"].shape == (2, NUM_KEYPOINTS)
        # Tanh output in [-1, 1]
        assert out["depth"].abs().max() <= 1.0

    def test_no_depth_head(self):
        model = HandPoseNet(backbone_name="lightweight", pretrained_backbone=False, use_depth_head=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert "depth" not in out

    def test_infer_output_keys(self, model):
        x = torch.randn(1, 3, 480, 640)
        out = model.infer(x)
        assert "det_scores" in out
        assert "det_boxes" in out
        assert "keypoints" in out

    def test_infer_keypoints_shape(self, model):
        x = torch.randn(1, 3, 480, 640)
        out = model.infer(x)
        assert out["keypoints"].shape == (1, NUM_KEYPOINTS, 2)

    def test_infer_scores_range(self, model):
        x = torch.randn(1, 3, 480, 640)
        out = model.infer(x)
        scores = out["det_scores"]
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_batch_consistency(self, model):
        """Batched and single-sample inference should give same results."""
        torch.manual_seed(0)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out_batch = model(x)
            out_single_0 = model(x[:1])
        # Det logits should be close between batch and individual
        torch.testing.assert_close(
            out_batch["det_logits"][:1],
            out_single_0["det_logits"],
            atol=1e-4, rtol=1e-3,
        )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class TestFocalLoss:
    def test_positive_loss(self):
        loss_fn = FocalLoss()
        logits = torch.zeros(4, 10, 10)
        targets = torch.ones(4, 10, 10)
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_zero_loss_perfect(self):
        loss_fn = FocalLoss()
        # Large positive logit → high confidence positive → near-zero focal weight
        logits = torch.full((2, 5, 5), 10.0)
        targets = torch.ones(2, 5, 5)
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01


class TestHeatmapLoss:
    def test_shape_agnostic(self):
        loss_fn = HeatmapLoss()
        pred = torch.randn(2, 21, 60, 80)
        target = torch.rand(2, 21, 60, 80)
        mask = torch.ones(2, 21)
        loss = loss_fn(pred, target, mask)
        assert loss.shape == ()

    def test_masked_keypoints_ignored(self):
        loss_fn = HeatmapLoss()
        pred = torch.randn(2, 21, 60, 80)
        target = torch.zeros(2, 21, 60, 80)
        mask_all = torch.ones(2, 21)
        mask_none = torch.zeros(2, 21)
        loss_all = loss_fn(pred, target, mask_all)
        loss_none = loss_fn(pred, target, mask_none)
        # All occluded → loss should be 0
        assert loss_none.item() == pytest.approx(0.0)
        assert loss_all.item() > loss_none.item()


class TestHandPoseLoss:
    def test_combined_loss_keys(self):
        model = HandPoseNet(backbone_name="lightweight", pretrained_backbone=False)
        model.eval()
        loss_fn = HandPoseLoss()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            preds = model(x)

        targets = {
            "det_mask": torch.zeros(2, 28, 28),
            "heatmaps_gt": torch.rand(2, 21, 56, 56),
            "kp_mask": torch.ones(2, 21),
            "depth_gt": torch.zeros(2, 21),
        }
        losses = loss_fn(preds, targets)
        assert "total" in losses
        assert "det" in losses
        assert "heatmap" in losses

    def test_total_positive(self):
        model = HandPoseNet(backbone_name="lightweight", pretrained_backbone=False)
        loss_fn = HandPoseLoss()
        x = torch.randn(1, 3, 224, 224)
        preds = model(x)
        targets = {
            "det_mask": torch.zeros(1, 28, 28),
            "heatmaps_gt": torch.rand(1, 21, 56, 56),
            "kp_mask": torch.ones(1, 21),
        }
        losses = loss_fn(preds, targets)
        assert losses["total"].item() > 0
