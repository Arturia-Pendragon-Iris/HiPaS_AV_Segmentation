"""
Loss functions for STS artery-vein segmentation training.

WeightedDiceLoss: Dice on the cumulative mask + volume-weighted dice on
    each incremental vessel level, following Eq. 13-15 of the paper.

OverlapLoss: penalises predictions where artery overlaps vein GT and vice
    versa, following Eq. 16 of the paper.

STSLoss: WeightedDiceLoss + OverlapLoss (Eq. 17).
"""

import torch
import torch.nn as nn


def _dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft dice over all spatial dimensions. pred and target: (B, ...) floats in [0,1]."""
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    return 1.0 - (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)


class WeightedDiceLoss(nn.Module):
    """
    Computes the weighted dice loss for STS stage i (Eq. 13-15).

    For stage i, we have:
        P_i : predicted cumulative mask (B, 2, D, H, W)
        P_prev : predicted cumulative mask from stage i-1 (or zeros if i==0)
        T_0 .. T_i : ground-truth cumulative masks

    Loss:
        - Dice on (P_i, T_i)  [level-0 term acts as the base Dice]
        - w_i * Dice(ΔP_i, ΔT_i) where ΔP_i = P_i - P_prev, ΔT_i = T_i - T_{i-1}
        - w_i = V(T_0) / V(ΔT_i)   (Eq. 15)

    Both artery (ch 0) and vein (ch 1) losses are averaged.

    max_weight: upper bound on w_i. Fine vessels at level 3 can have very small
        ΔT volume, making w very large and destabilising training.
    """

    def __init__(self, smooth: float = 1.0, max_weight: float = 20.0):
        super().__init__()
        self.smooth = smooth
        self.max_weight = max_weight

    def forward(
        self,
        pred: torch.Tensor,          # (B, 2, D, H, W) — current stage output
        masks: list[torch.Tensor],   # 4 × (B, 2, D, H, W) — cumulative GT masks
        stage: int,
        prior: torch.Tensor | None = None,  # (B, 2, D, H, W) — prev stage output / zeros
    ) -> torch.Tensor:

        target = masks[stage]           # cumulative GT for this stage
        target_prev = masks[stage - 1] if stage > 0 else torch.zeros_like(target)
        prior = prior if prior is not None else torch.zeros_like(pred)

        device = pred.device
        target = target.to(device)
        target_prev = target_prev.to(device)
        prior = prior.to(device)

        total = torch.tensor(0.0, device=device)

        for ch in range(2):  # artery, vein
            p = pred[:, ch]
            t = target[:, ch]
            t_prev = target_prev[:, ch]
            p_prev = prior[:, ch]

            # Base dice: cumulative prediction vs cumulative GT
            total = total + _dice(p, t, self.smooth).mean()

            if stage > 0:
                # Incremental terms
                delta_p = torch.clamp(p - p_prev, 0.0, 1.0)
                delta_t = torch.clamp(t - t_prev, 0.0, 1.0)

                # Weight: volume of level-0 GT / volume of current delta GT
                vol_t0 = masks[0][:, ch].to(device).sum(dim=(-3, -2, -1)) + 1.0
                vol_delta = delta_t.sum(dim=(-3, -2, -1)) + 1.0
                w = torch.clamp((vol_t0 / vol_delta).mean(), max=self.max_weight)

                total = total + w * _dice(delta_p, delta_t, self.smooth).mean()

        return total / 2.0


class OverlapLoss(nn.Module):
    """
    Penalises artery predictions overlapping the vein GT and vice versa (Eq. 16).

        L_overlap = (P_A * T_V + P_V * T_A) / (P + T + eps)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,        # (B, 2, D, H, W)
        target: torch.Tensor,      # (B, 2, D, H, W)
    ) -> torch.Tensor:
        pred = pred.to(target.device)
        pa = pred[:, 0]   # artery prediction
        pv = pred[:, 1]   # vein prediction
        ta = target[:, 0]
        tv = target[:, 1]

        numerator = (pa * tv + pv * ta).sum()
        denominator = (pred + target).sum() + 1e-6
        return numerator / denominator


class STSLoss(nn.Module):
    """Combined loss: WeightedDice + OverlapLoss (Eq. 17)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.dice_loss = WeightedDiceLoss(smooth)
        self.overlap_loss = OverlapLoss()

    def forward(
        self,
        pred: torch.Tensor,
        masks: list[torch.Tensor],
        stage: int,
        prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        target = masks[stage].to(pred.device)
        dice = self.dice_loss(pred, masks, stage, prior)
        overlap = self.overlap_loss(pred, target)
        total = dice + overlap
        return total, {"dice": dice.item(), "overlap": overlap.item()}
