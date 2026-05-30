"""Differentiable lift of SHARP's NDC-perspective Gaussians to a metric world frame.

The SHARP predictor emits Gaussians in a canonical NDC-perspective frame
(``mean = [z·x_ndc, z·y_ndc, z]``). For render-supervised training we must place them in a
metric world frame (input camera at identity) so a target view at a metric relative pose can
be rendered through gsplat. The inference-time unprojection
(:func:`sharp.utils.gaussians.unproject_gaussians`) is **non-differentiable** (it does a
detached CPU SVD in ``decompose_covariance_matrices``), so it cannot sit in the training graph.

This module provides a fully differentiable equivalent:

- means and covariances are transformed by the constant linear part ``M`` of the unprojection
  matrix ``inv(ndc @ K @ E)`` (reusing :func:`get_unprojection_matrix` and
  :func:`compose_covariance_matrices`);
- branch **P** (classic pinhole) renders by passing ``covars=`` straight to gsplat;
- branch **D** (Unscented-Transform / distorted) requires ``quats`` + ``scales`` (gsplat rejects
  ``covars`` on the UT path), which we recover from the metric covariance via the differentiable
  symmetric eigendecomposition :func:`torch.linalg.eigh` (with a small jitter and a
  determinant-sign flip to guarantee a proper rotation).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from sharp.utils import linalg
from sharp.utils.gaussians import (
    Gaussians3D,
    compose_covariance_matrices,
    get_unprojection_matrix,
)


class WorldGaussians(NamedTuple):
    """Metric world-frame Gaussians for rendering.

    ``covars`` is always populated (3x3 metric covariance). ``quats``/``scales`` are populated
    lazily by :func:`covars_to_quats_scales` for the UT/distorted render path.
    """

    means: torch.Tensor  # [B, N, 3]
    covars: torch.Tensor  # [B, N, 3, 3]
    colors: torch.Tensor  # [B, N, 3]
    opacities: torch.Tensor  # [B, N]
    quats: torch.Tensor | None = None  # [B, N, 4]
    scales: torch.Tensor | None = None  # [B, N, 3]


def lift_to_world(
    gaussians: Gaussians3D,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
) -> WorldGaussians:
    """Differentiably lift NDC-perspective Gaussians to the metric world (input-cam) frame.

    Args:
        gaussians: Predicted Gaussians in the canonical NDC-perspective frame, with
            ``mean_vectors`` [B, N, 3], ``quaternions`` [B, N, 4], ``singular_values`` [B, N, 3],
            ``colors`` [B, N, 3] and ``opacities`` [B, N].
        intrinsics: The input-camera 4x4 (or 3x3) intrinsics at the internal resolution, in the
            same convention as :func:`sharp.cli.predict.predict_image` (centered principal point).
        image_shape: ``(width, height)`` of the internal frame the Gaussians were predicted at.

    Returns:
        World-frame means + covariance (and colors/opacities passed through).
    """
    device = gaussians.mean_vectors.device
    dtype = gaussians.mean_vectors.dtype

    if intrinsics.shape[-1] == 3:
        intrinsics_4x4 = torch.eye(4, device=device, dtype=dtype)
        intrinsics_4x4[:3, :3] = intrinsics
        intrinsics = intrinsics_4x4

    extrinsics = torch.eye(4, device=device, dtype=dtype)
    unprojection = get_unprojection_matrix(extrinsics, intrinsics.to(dtype), image_shape)
    linear = unprojection[:3, :3].to(dtype)  # [3, 3]
    offset = unprojection[:3, 3].to(dtype)  # [3]

    # means_world = means @ linear^T + offset   (differentiable in means).
    means_world = gaussians.mean_vectors @ linear.transpose(-1, -2) + offset

    # Sigma_world = linear @ Sigma_ndc @ linear^T   (differentiable).
    covars_ndc = compose_covariance_matrices(gaussians.quaternions, gaussians.singular_values)
    covars_world = linear @ covars_ndc @ linear.transpose(-1, -2)

    return WorldGaussians(
        means=means_world,
        covars=covars_world,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def covars_to_quats_scales(
    covars: torch.Tensor, jitter: float = 1e-9
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiably decompose symmetric 3x3 covariances into quaternions and scales.

    Uses :func:`torch.linalg.eigh` (differentiable, runs on GPU) instead of the inference
    code's detached CPU SVD. A small ``jitter`` is added to the eigenvalues for gradient
    stability under near-degeneracy, and the eigenvector basis is flipped to a proper rotation
    (det = +1) when ``eigh`` returns a reflection.

    Args:
        covars: Symmetric covariance matrices, shape [..., 3, 3].
        jitter: Added to eigenvalues before the square root.

    Returns:
        ``(quaternions [..., 4], scales [..., 3])``.
    """
    # Symmetrize defensively (numerical asymmetry would make eigh complain).
    covars = 0.5 * (covars + covars.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(covars)  # eigvals ascending; eigvecs columns

    # Flip to a proper rotation (det = +1) without an in-place op on the autograd graph.
    det = torch.linalg.det(eigvecs)  # [...]
    flip = torch.ones_like(eigvecs)
    flip[..., :, 2] = torch.sign(det).unsqueeze(-1)
    rotations = eigvecs * flip

    scales = torch.sqrt(eigvals.clamp(min=0.0) + jitter)
    quaternions = linalg.quaternions_from_rotation_matrices(rotations)
    return quaternions, scales


def lift_for_render(
    gaussians: Gaussians3D,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
    need_quats_scales: bool,
    jitter: float = 1e-9,
) -> WorldGaussians:
    """Lift to world and, for the UT/distorted path, also produce quats + scales.

    Args:
        gaussians: NDC-perspective Gaussians.
        intrinsics: Input-camera intrinsics (internal resolution).
        image_shape: ``(width, height)`` of the internal frame.
        need_quats_scales: If True (UT/distorted render), decompose the covariance into
            ``quats``/``scales`` (gsplat's UT path rejects ``covars``). If False (classic
            pinhole render), only ``covars`` is needed.
        jitter: Eigenvalue jitter for the decomposition.

    Returns:
        World-frame Gaussians, with ``quats``/``scales`` populated iff ``need_quats_scales``.
    """
    world = lift_to_world(gaussians, intrinsics, image_shape)
    if not need_quats_scales:
        return world
    quats, scales = covars_to_quats_scales(world.covars, jitter=jitter)
    return world._replace(quats=quats, scales=scales)
