"""
Loss functions for hand pose training.

Combined loss: L_det + λ_hm * L_heatmap + λ_depth * L_depth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class FocalLoss(nn.Module):
    """
    Focal loss for detection objectness (class imbalance: sparse hands).

    Args:
        alpha: weighting factor for positive class
        gamma: focusing exponent (2.0 standard)

    Input:  logits [B, ...], targets [B, ...] in {0, 1}
    Output: scalar loss
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * ce).mean()


class HeatmapLoss(nn.Module):
    """
    MSE loss on Gaussian heatmaps.

    Targets: Gaussian blobs of sigma=2 centered on keypoint locations.
    Applied only where hand is visible (mask).

    Input:
        pred:   [B, 21, Hh, Wh] raw logits
        target: [B, 21, Hh, Wh] Gaussian heatmaps in [0,1]
        mask:   [B, 21]         visibility mask (1=visible, 0=occluded)

    Output: scalar loss
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        sq_err = (pred_sigmoid - target) ** 2         # [B, 21, Hh, Wh]
        # Weight by mask: occluded keypoints contribute less
        mask_spatial = mask.unsqueeze(-1).unsqueeze(-1)  # [B, 21, 1, 1]
        weighted = sq_err * mask_spatial
        return weighted.mean()


class DepthLoss(nn.Module):
    """
    Smooth-L1 loss for relative depth regression.

    Input:
        pred:   [B, 21] Tanh output in [-1, 1]
        target: [B, 21] normalized depth ground truth in [-1, 1]
        mask:   [B, 21] visibility mask

    Output: scalar loss
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.smooth_l1_loss(pred, target, reduction="none")  # [B, 21]
        return (loss * mask).sum() / (mask.sum() + 1e-6)


class HandPoseLoss(nn.Module):
    """
    Combined loss for HandPoseNet training.

    L_total = L_det + λ_hm * L_heatmap + λ_depth * L_depth

    Args:
        lambda_heatmap: weight for heatmap loss (default 10.0)
        lambda_depth:   weight for depth loss (default 1.0)

    Inputs (all from model forward + ground truth):
        preds:  dict from HandPoseNet.forward()
        targets: dict with keys:
            "det_mask":    [B, Hf, Wf]   1 where hand center lies
            "det_boxes_gt":[B, 4]        (cx, cy, w, h) normalized
            "heatmaps_gt": [B, 21, Hh, Wh]
            "kp_mask":     [B, 21]       visibility
            "depth_gt":    [B, 21]       (optional)

    Output: dict with "total", "det", "heatmap", "depth" losses
    """

    def __init__(
        self,
        lambda_heatmap: float = 10.0,
        lambda_depth: float = 1.0,
    ) -> None:
        super().__init__()
        self.focal = FocalLoss()
        self.heatmap = HeatmapLoss()
        self.depth = DepthLoss()
        self.lambda_hm = lambda_heatmap
        self.lambda_depth = lambda_depth

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # Detection loss: objectness channel only
        det_obj_logits = preds["det_logits"][:, 0]   # [B, Hf, Wf]
        l_det = self.focal(det_obj_logits, targets["det_mask"].float())

        # Heatmap loss
        l_hm = self.heatmap(
            preds["heatmaps"],
            targets["heatmaps_gt"],
            targets["kp_mask"],
        )

        losses: Dict[str, torch.Tensor] = {
            "det": l_det,
            "heatmap": l_hm,
        }

        # Depth loss (optional)
        l_depth = torch.tensor(0.0, device=l_det.device)
        if "depth" in preds and "depth_gt" in targets:
            l_depth = self.depth(preds["depth"], targets["depth_gt"], targets["kp_mask"])
            losses["depth"] = l_depth

        losses["total"] = l_det + self.lambda_hm * l_hm + self.lambda_depth * l_depth
        return losses
