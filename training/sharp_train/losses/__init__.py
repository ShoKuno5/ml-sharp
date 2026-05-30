"""Symmetric render-supervision loss for dual-branch SHARP.

The image distance ``D = alpha*L1 + beta*LPIPS + gamma*DISTS`` is applied with a valid mask
(warp holes excluded). The symmetric image loss matches each branch to its own camera:

    L_img = w_PP * D(render(G_P, cam_pinhole),   GT_pinhole)
          + w_DD * D(render(G_D, cam_distorted), GT_distorted)
          [+ w_PD * D(render(G_P, cam_distorted), GT_distorted)]   # cross (diagnostic)
          [+ w_DP * D(render(G_D, cam_pinhole),   GT_pinhole)]     # cross (diagnostic)

plus a single shared depth term ``lambda * L_depth(shared_depth, GT_depth)``.

The renders are produced by the trainer; this module only combines pre-rendered tensors so it
is free of any CUDA / gsplat dependency and is unit-testable on CPU.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from .perceptual import DISTSLoss, LPIPSLoss, _apply_mask


def masked_l1(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean L1 over valid pixels (``mask`` broadcasts over channels)."""
    diff = (pred - target).abs()
    if mask is None:
        return diff.mean()
    mask = mask.expand_as(diff)
    denom = mask.sum().clamp(min=1.0)
    return (diff * mask).sum() / denom


def depth_l1(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor | None = None,
    layer: int = 0,
) -> torch.Tensor:
    """Masked L1 on a single depth layer (front surface by default).

    Args:
        pred_depth: Predicted metric depth, shape [B, L, H, W] or [B, 1, H, W].
        gt_depth: Ground-truth metric depth, shape [B, 1, H, W] or [B, H, W].
        mask: Optional valid mask broadcastable to [B, 1, H, W].
        layer: Which predicted depth layer to supervise (default front surface, 0).
    """
    if pred_depth.dim() == 4 and pred_depth.shape[1] > 1:
        pred_depth = pred_depth[:, layer : layer + 1]
    elif pred_depth.dim() == 4:
        pred_depth = pred_depth[:, :1]
    if gt_depth.dim() == 3:
        gt_depth = gt_depth[:, None]
    return masked_l1(pred_depth, gt_depth, mask)


@dataclasses.dataclass
class LossWeights:
    """Weights for the symmetric render-supervision loss."""

    # Image-distance term weights.
    alpha_l1: float = 1.0
    beta_lpips: float = 1.0
    gamma_dists: float = 1.0
    # Symmetric / cross branch-camera weights.
    w_pp: float = 1.0
    w_dd: float = 1.0
    w_pd: float = 0.0  # cross: render(G_P, distorted) vs GT_distorted (ablation)
    w_dp: float = 0.0  # cross: render(G_D, pinhole)   vs GT_pinhole   (ablation)
    # Shared depth term.
    lambda_depth: float = 0.05


class ImageDistance(nn.Module):
    """D(pred, target) = alpha*L1 + beta*LPIPS + gamma*DISTS, mask-aware.

    The LPIPS/DISTS sub-modules are only constructed when their weight is non-zero, so a pure
    L1 configuration needs no pretrained backbones.
    """

    def __init__(self, weights: LossWeights, lpips_net: str = "alex") -> None:
        """Initialize ImageDistance from :class:`LossWeights`."""
        super().__init__()
        self.weights = weights
        self.lpips = LPIPSLoss(net=lpips_net) if weights.beta_lpips > 0 else None
        self.dists = DISTSLoss() if weights.gamma_dists > 0 else None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(distance, term_dict)`` for masked ``pred`` vs ``target`` in [0, 1]."""
        weights = self.weights
        terms: dict[str, torch.Tensor] = {}
        total = pred.new_zeros(())

        if weights.alpha_l1 > 0:
            l1 = masked_l1(pred, target, mask)
            terms["l1"] = l1
            total = total + weights.alpha_l1 * l1
        if self.lpips is not None:
            lp = self.lpips(pred, target, mask)
            terms["lpips"] = lp
            total = total + weights.beta_lpips * lp
        if self.dists is not None:
            ds = self.dists(pred, target, mask)
            terms["dists"] = ds
            total = total + weights.gamma_dists * ds
        return total, terms


class SymmetricLoss(nn.Module):
    """Combine pre-rendered branch images + shared depth into the total training loss."""

    def __init__(self, weights: LossWeights, lpips_net: str = "alex") -> None:
        """Initialize SymmetricLoss from :class:`LossWeights`."""
        super().__init__()
        self.weights = weights
        self.image_distance = ImageDistance(weights, lpips_net=lpips_net)

    def forward(
        self,
        *,
        render_p_pin: torch.Tensor,
        render_d_dist: torch.Tensor,
        gt_pin: torch.Tensor,
        gt_dist: torch.Tensor,
        mask_pin: torch.Tensor | None = None,
        mask_dist: torch.Tensor | None = None,
        render_p_dist: torch.Tensor | None = None,
        render_d_pin: torch.Tensor | None = None,
        pred_depth: torch.Tensor | None = None,
        gt_depth: torch.Tensor | None = None,
        depth_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(total_loss, term_dict)``.

        Matched cells (P x pinhole, D x distorted) are always included; the cross cells are
        included only when their weight is non-zero and the corresponding render is supplied.
        """
        weights = self.weights
        terms: dict[str, torch.Tensor] = {}
        total = render_p_pin.new_zeros(())

        d_pp, t_pp = self.image_distance(render_p_pin, gt_pin, mask_pin)
        total = total + weights.w_pp * d_pp
        terms["D_pp"] = d_pp
        terms.update({f"pp_{k}": v for k, v in t_pp.items()})

        d_dd, t_dd = self.image_distance(render_d_dist, gt_dist, mask_dist)
        total = total + weights.w_dd * d_dd
        terms["D_dd"] = d_dd
        terms.update({f"dd_{k}": v for k, v in t_dd.items()})

        if weights.w_pd > 0 and render_p_dist is not None:
            d_pd, _ = self.image_distance(render_p_dist, gt_dist, mask_dist)
            total = total + weights.w_pd * d_pd
            terms["D_pd"] = d_pd
        if weights.w_dp > 0 and render_d_pin is not None:
            d_dp, _ = self.image_distance(render_d_pin, gt_pin, mask_pin)
            total = total + weights.w_dp * d_dp
            terms["D_dp"] = d_dp

        if weights.lambda_depth > 0 and pred_depth is not None and gt_depth is not None:
            l_depth = depth_l1(pred_depth, gt_depth, depth_mask)
            total = total + weights.lambda_depth * l_depth
            terms["depth"] = l_depth

        terms["total"] = total
        return total, terms


__all__ = [
    "LossWeights",
    "ImageDistance",
    "SymmetricLoss",
    "LPIPSLoss",
    "DISTSLoss",
    "masked_l1",
    "depth_l1",
    "_apply_mask",
]
