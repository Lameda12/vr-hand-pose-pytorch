"""
HandPoseNet: Single-stage hand detection + 21-keypoint 2D pose estimation.

Architecture:
  Backbone (MobileNetV3) → Detection Head → Heatmap Head → Depth Head
  Output: bounding boxes + 21 keypoints (u, v, z_rel)

Keypoint layout (MediaPipe/FreiHAND convention):
  0: Wrist
  1-4:  Thumb  (MCP, PIP, DIP, Tip)
  5-8:  Index  (MCP, PIP, DIP, Tip)
  9-12: Middle (MCP, PIP, DIP, Tip)
  13-16:Ring   (MCP, PIP, DIP, Tip)
  17-20:Pinky  (MCP, PIP, DIP, Tip)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .backbone import build_backbone

NUM_KEYPOINTS = 21


class DepthwiseSepConv(nn.Module):
    """Depthwise-separable conv + BN + ReLU6."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, padding=pad, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(self.bn(self.pw(self.dw(x))), inplace=True)


class DetectionHead(nn.Module):
    """
    Anchor-free hand detection head.

    Predicts per-location: objectness score + bbox (cx, cy, w, h) in
    normalized [0,1] coordinates relative to the feature cell.

    Input:  [B, C, Hf, Wf]
    Output: [B, 5, Hf, Wf]  (obj, cx, cy, w, h)
    """

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.conv1 = DepthwiseSepConv(in_ch, 128)
        self.conv2 = DepthwiseSepConv(128, 128)
        self.out = nn.Conv2d(128, 5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.out(x)


class HeatmapHead(nn.Module):
    """
    21-keypoint heatmap regression head.

    Produces Gaussian heatmaps per keypoint; argmax gives (u,v) coords.
    Heatmap resolution = input_resolution / 4 for accuracy vs speed balance.

    Input:  [B, C, Hf, Wf]
    Output: [B, 21, Hf*2, Wf*2]  (bilinear upsample ×2 from stride-8 feat)
    """

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.conv1 = DepthwiseSepConv(in_ch, 128)
        self.conv2 = DepthwiseSepConv(128, 128)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
        )
        self.out = nn.Conv2d(64, NUM_KEYPOINTS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return self.out(x)   # raw logits; sigmoid in loss


class DepthHead(nn.Module):
    """
    Per-keypoint relative depth regression.

    Regresses z_rel in [-1, 1] (normalized to hand bounding box depth).
    Applied to ROI-pooled features from the detected hand bbox.

    Input:  [B, C, Hf, Wf]
    Output: [B, 21]  (relative depth per keypoint)
    """

    def __init__(self, in_ch: int, pool_size: int = 7) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        flat_dim = in_ch * pool_size * pool_size
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_KEYPOINTS),
            nn.Tanh(),   # output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, Hf, Wf]

        Returns:
            depth: [B, 21]
        """
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


class HandPoseNet(nn.Module):
    """
    Full hand pose network: detect + 2D keypoints + relative depth.

    Args:
        backbone_name: "mobilenetv3" | "lightweight"
        pretrained_backbone: load ImageNet weights for backbone
        use_depth_head: enable relative-depth regression (adds ~2ms)

    Input:  [B, 3, H, W]  float32, RGB, normalized [0,1]
    Outputs:
        "det_logits":  [B, 5, H/8, W/8]   detection map (obj + bbox deltas)
        "heatmaps":    [B, 21, H/4, W/4]  keypoint heatmaps (raw logits)
        "depth":       [B, 21]            relative depth (optional, Tanh)

    End-to-end latency target: <25ms @720p on RTX 3070 (FP16)
    """

    def __init__(
        self,
        backbone_name: str = "mobilenetv3",
        pretrained_backbone: bool = True,
        use_depth_head: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_name, pretrained=pretrained_backbone)
        in_ch = self.backbone.out_channels

        self.det_head = DetectionHead(in_ch)
        self.heatmap_head = HeatmapHead(in_ch)
        self.depth_head = DepthHead(in_ch) if use_depth_head else None
        self.use_depth_head = use_depth_head

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]

        Returns dict with:
            "det_logits": [B, 5, H/8, W/8]
            "heatmaps":   [B, 21, H/4, W/4]
            "depth":      [B, 21]  (only if use_depth_head=True)
        """
        feat = self.backbone(x)  # [B, C, H/8, W/8]

        out: Dict[str, torch.Tensor] = {
            "det_logits": self.det_head(feat),
            "heatmaps": self.heatmap_head(feat),
        }

        if self.use_depth_head:
            out["depth"] = self.depth_head(feat)

        return out

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        det_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-only forward. Applies sigmoid to detection objectness,
        converts heatmaps to (u, v) pixel coords.

        Args:
            x: [B, 3, H, W]
            det_threshold: objectness score threshold

        Returns dict with:
            "det_scores":  [B, Hf*Wf]     objectness in [0,1]
            "det_boxes":   [B, Hf*Wf, 4]  (cx, cy, w, h) normalized
            "keypoints":   [B, 21, 2]     (u, v) in pixel coords (H/4 space)
            "depth":       [B, 21]        (optional)
        """
        self.eval()
        raw = self.forward(x)

        B, _, Hf, Wf = raw["det_logits"].shape
        det = raw["det_logits"]

        scores = torch.sigmoid(det[:, 0])           # [B, Hf, Wf]
        boxes = torch.sigmoid(det[:, 1:])            # [B, 4, Hf, Wf]

        # Heatmap → (u, v): softargmax for differentiability
        hm = raw["heatmaps"]                         # [B, 21, Hh, Hw]
        _, _, Hh, Hw = hm.shape
        hm_flat = hm.flatten(2)                      # [B, 21, Hh*Hw]
        hm_prob = torch.softmax(hm_flat, dim=-1)

        gy = torch.linspace(0, Hh - 1, Hh, device=x.device)
        gx = torch.linspace(0, Hw - 1, Hw, device=x.device)
        gy, gx = torch.meshgrid(gy, gx, indexing="ij")
        gy = gy.flatten()
        gx = gx.flatten()

        kp_v = (hm_prob * gy).sum(-1)               # [B, 21]
        kp_u = (hm_prob * gx).sum(-1)               # [B, 21]
        keypoints = torch.stack([kp_u, kp_v], dim=-1)  # [B, 21, 2]

        result: Dict[str, torch.Tensor] = {
            "det_scores": scores.flatten(1),
            "det_boxes": boxes.permute(0, 2, 3, 1).flatten(1, 2),
            "keypoints": keypoints,
        }

        if self.use_depth_head:
            result["depth"] = raw["depth"]

        return result
