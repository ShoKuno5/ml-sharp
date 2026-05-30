"""Shape-correct synthetic data so the harness runs end-to-end with no real dataset.

The default sample is an identity-pose self-reconstruction at xi=0: the GT target for both
branches is the (avg-pooled) input image and the target pose is identity, so a healthy pipeline
should drive the loss towards zero (the overfit sanity check). The internal resolution is fixed
at 1536 because SHARP's sliding-pyramid encoder only accepts that size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .batch import BatchSample

INTERNAL_RESOLUTION = 1536


def _intrinsics(focal: float, width: int, height: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[focal, 0, width / 2, 0], [0, focal, height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        device=device,
    )


def make_dummy_sample(
    render_size: int = 512,
    internal_resolution: int = INTERNAL_RESOLUTION,
    focal_ratio: float = 1.0,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> BatchSample:
    """Build one identity-pose self-reconstruction sample at xi=0.

    Args:
        render_size: Output (target) render resolution (square).
        internal_resolution: Encoder internal resolution (must be 1536 for the real model).
        focal_ratio: Focal length as a multiple of the internal width.
        seed: RNG seed for the random input image.
        device: Device for the tensors.

    Returns:
        A single-example :class:`BatchSample`.
    """
    device = torch.device(device)
    generator = torch.Generator().manual_seed(seed)
    image = torch.rand(1, 3, internal_resolution, internal_resolution, generator=generator).to(
        device
    )

    focal = focal_ratio * internal_resolution
    input_intrinsics = _intrinsics(focal, internal_resolution, internal_resolution, device)[None]
    target_viewmat = torch.eye(4, device=device)[None]
    target_intrinsics = _intrinsics(
        focal_ratio * render_size, render_size, render_size, device
    )[None]

    target = F.interpolate(image, size=(render_size, render_size), mode="bilinear",
                           align_corners=False)
    mask = torch.ones(1, 1, render_size, render_size, device=device)

    return BatchSample(
        image=image,
        disparity_factor=torch.tensor([focal_ratio], device=device),
        input_intrinsics=input_intrinsics,
        target_viewmat=target_viewmat,
        target_intrinsics=target_intrinsics,
        render_width=render_size,
        render_height=render_size,
        gt_pinhole=target,
        gt_distorted=target,
        mask_pinhole=mask,
        mask_distorted=mask,
        camera_model="pinhole",
        radial_coeffs=None,
        gt_depth=None,
    )


class DummyDataset(Dataset):
    """A dataset of identical dummy samples (length controls the number of steps per epoch)."""

    def __init__(self, length: int = 1, render_size: int = 512, seed: int = 0) -> None:
        """Initialize DummyDataset with a fixed sample repeated ``length`` times."""
        self.length = length
        self.render_size = render_size
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> BatchSample:
        # Return an unbatched-by-convention single example (B=1 already baked in).
        return make_dummy_sample(render_size=self.render_size, seed=self.seed)
