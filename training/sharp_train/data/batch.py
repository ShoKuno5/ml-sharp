"""The batch contract shared by every dataset and the trainer.

A :class:`BatchSample` carries one (input, target) view pair plus the camera/distortion
parameters needed to render and supervise both branches. Tensors are batched (leading B dim).
The real ScanNet++ / ScanNet-raycast datasets (Step 3) emit the same structure.
"""

from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class BatchSample:
    """One render-supervision training example (batched).

    Attributes:
        image: Input (distorted) image at internal resolution, [B, 3, Hin, Win], in [0, 1].
        disparity_factor: Depth->disparity factor (``f_px / W``), [B].
        input_intrinsics: Input-camera intrinsics at internal resolution, [B, 4, 4].
        target_viewmat: Input->target relative extrinsics (world == input cam), [B, 4, 4].
        target_intrinsics: Target-camera intrinsics at render resolution, [B, 4, 4].
        render_width: Target render width in pixels.
        render_height: Target render height in pixels.
        gt_pinhole: GT target rendered with the pinhole camera, [B, 3, Hr, Wr], in [0, 1].
        gt_distorted: GT target rendered with the distorted camera, [B, 3, Hr, Wr], in [0, 1].
        mask_pinhole: Valid mask for ``gt_pinhole`` (1 = valid), [B, 1, Hr, Wr].
        mask_distorted: Valid mask for ``gt_distorted``, [B, 1, Hr, Wr].
        camera_model: gsplat camera model for the distorted branch.
        radial_coeffs: Distorted-branch radial coefficients, [B, K] or None (xi=0).
        tangential_coeffs: Distorted-branch tangential coefficients, [B, 2] or None.
        gt_depth: GT metric depth at internal resolution for the shared depth loss + alignment,
            [B, 1, Hin, Win] or None.
    """

    image: torch.Tensor
    disparity_factor: torch.Tensor
    input_intrinsics: torch.Tensor
    target_viewmat: torch.Tensor
    # Distorted-branch (D) target intrinsics. The pinhole branch (P) uses
    # ``target_intrinsics_pinhole`` when set, else falls back to this (they coincide at xi=0).
    target_intrinsics: torch.Tensor
    render_width: int
    render_height: int
    gt_pinhole: torch.Tensor
    gt_distorted: torch.Tensor
    mask_pinhole: torch.Tensor
    mask_distorted: torch.Tensor
    target_intrinsics_pinhole: torch.Tensor | None = None
    camera_model: str = "pinhole"
    radial_coeffs: torch.Tensor | None = None
    tangential_coeffs: torch.Tensor | None = None
    gt_depth: torch.Tensor | None = None

    def to(self, device: torch.device) -> BatchSample:
        """Move all tensor fields to ``device`` (non-tensor fields pass through)."""

        def _move(value):
            return value.to(device) if isinstance(value, torch.Tensor) else value

        return BatchSample(
            **{f.name: _move(getattr(self, f.name)) for f in dataclasses.fields(self)}
        )
