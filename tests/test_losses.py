"""CPU tests for the symmetric render-supervision loss (no perceptual backbones needed).

Perceptual weights are set to zero so no pretrained networks are downloaded; the L1 / depth /
masking / combination logic and gradient flow are validated directly.
"""

from __future__ import annotations

import torch
from sharp_train.losses import LossWeights, SymmetricLoss, depth_l1, masked_l1


def test_masked_l1_excludes_invalid() -> None:
    """masked_l1 ignores masked-out pixels."""
    pred = torch.zeros(1, 3, 4, 4)
    target = torch.ones(1, 3, 4, 4)
    mask = torch.ones(1, 1, 4, 4)
    mask[..., 2:, :] = 0.0  # invalidate the bottom half (where pred==target would differ)
    # Valid region: |0 - 1| = 1 everywhere -> mean over valid = 1.0
    assert abs(masked_l1(pred, target, mask).item() - 1.0) < 1e-6
    # If we instead make the valid region match, loss is 0 regardless of invalid region.
    pred2 = target.clone()
    pred2[..., 2:, :] = 5.0  # large error, but masked out
    assert masked_l1(pred2, target, mask).item() < 1e-6


def test_depth_l1_layer_selection() -> None:
    """depth_l1 supervises the requested layer of a 2-layer depth map."""
    pred = torch.stack([torch.full((1, 4, 4), 2.0), torch.full((1, 4, 4), 9.0)], dim=1).squeeze(2)
    # pred shape [1, 2, 4, 4]; layer 0 == 2.0, layer 1 == 9.0
    gt = torch.full((1, 1, 4, 4), 2.0)
    assert depth_l1(pred, gt, layer=0).item() < 1e-6
    assert abs(depth_l1(pred, gt, layer=1).item() - 7.0) < 1e-6


def test_symmetric_loss_combination_and_grad() -> None:
    """SymmetricLoss combines matched terms with weights and is differentiable."""
    weights = LossWeights(alpha_l1=1.0, beta_lpips=0.0, gamma_dists=0.0, w_pp=1.0, w_dd=2.0,
                          lambda_depth=0.5)
    loss_fn = SymmetricLoss(weights)

    render_p = torch.zeros(1, 3, 8, 8, requires_grad=True)
    render_d = torch.zeros(1, 3, 8, 8, requires_grad=True)
    gt_pin = torch.full((1, 3, 8, 8), 0.5)
    gt_dist = torch.full((1, 3, 8, 8), 0.25)
    pred_depth = torch.zeros(1, 2, 8, 8, requires_grad=True)
    gt_depth = torch.ones(1, 1, 8, 8)

    total, terms = loss_fn(
        render_p_pin=render_p,
        render_d_dist=render_d,
        gt_pin=gt_pin,
        gt_dist=gt_dist,
        pred_depth=pred_depth,
        gt_depth=gt_depth,
    )
    # Expected: w_pp*|0-0.5| + w_dd*|0-0.25| + lambda*|0-1| = 0.5 + 2*0.25 + 0.5*1 = 1.5
    assert abs(total.item() - 1.5) < 1e-6
    assert "D_pp" in terms and "D_dd" in terms and "depth" in terms
    assert "D_pd" not in terms and "D_dp" not in terms  # cross terms off by default

    total.backward()
    assert render_p.grad.abs().sum() > 0
    assert render_d.grad.abs().sum() > 0
    assert pred_depth.grad.abs().sum() > 0


def test_cross_terms_enabled() -> None:
    """Cross terms appear only when their weight is non-zero and renders are supplied."""
    weights = LossWeights(beta_lpips=0.0, gamma_dists=0.0, w_pd=1.0, w_dp=1.0, lambda_depth=0.0)
    loss_fn = SymmetricLoss(weights)
    z = torch.zeros(1, 3, 4, 4)
    total, terms = loss_fn(
        render_p_pin=z,
        render_d_dist=z,
        gt_pin=z + 0.1,
        gt_dist=z + 0.2,
        render_p_dist=z,
        render_d_pin=z,
    )
    assert "D_pd" in terms and "D_dp" in terms
