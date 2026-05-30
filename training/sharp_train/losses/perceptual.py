"""Perceptual losses (LPIPS + DISTS) with valid-mask support.

The perceptual networks are lazily constructed on first use so the rest of the harness (and
the CPU test suite) can run with the perceptual terms disabled and without downloading any
pretrained backbones. Masking is applied by zeroing the invalid regions of *both* the
prediction and the target before the metric, which keeps warp holes from polluting the score.
"""

from __future__ import annotations

import torch
from torch import nn


def _apply_mask(image: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Zero the invalid (mask == 0) regions; ``mask`` broadcasts over channels."""
    if mask is None:
        return image
    return image * mask


class LPIPSLoss(nn.Module):
    """LPIPS perceptual distance (expects images in [0, 1], internally rescaled to [-1, 1])."""

    def __init__(self, net: str = "alex") -> None:
        """Initialize LPIPSLoss.

        Args:
            net: Backbone for LPIPS, one of {"alex", "vgg", "squeeze"}.
        """
        super().__init__()
        self.net = net
        self._model: nn.Module | None = None

    def _lazy(self, device: torch.device) -> nn.Module:
        if self._model is None:
            import lpips

            model = lpips.LPIPS(net=self.net, verbose=False)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._model = model
        return self._model.to(device)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the mean LPIPS distance between masked ``pred`` and ``target`` in [0, 1]."""
        model = self._lazy(pred.device)
        pred = _apply_mask(pred, mask) * 2.0 - 1.0
        target = _apply_mask(target, mask) * 2.0 - 1.0
        return model(pred, target).mean()


class DISTSLoss(nn.Module):
    """DISTS perceptual distance (expects images in [0, 1])."""

    def __init__(self) -> None:
        """Initialize DISTSLoss."""
        super().__init__()
        self._model: nn.Module | None = None

    def _lazy(self, device: torch.device) -> nn.Module:
        if self._model is None:
            from DISTS_pytorch import DISTS

            model = DISTS()
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._model = model
        return self._model.to(device)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the mean DISTS distance between masked ``pred`` and ``target`` in [0, 1]."""
        model = self._lazy(pred.device)
        pred = _apply_mask(pred, mask)
        target = _apply_mask(target, mask)
        return model(pred, target).mean()
