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

    # The lift's linear algebra (``torch.linalg.inv`` here, ``eigh`` in the decomposition) does
    # not support low-precision dtypes, so under a bf16 autocast training step we must run it in
    # fp32. The lift is cheap relative to the encoder/decoder, so we keep the whole thing fp32
    # (autocast disabled); the bf16 memory savings live upstream in the heavy network.
    with torch.autocast(device_type=device.type, enabled=False):
        means = gaussians.mean_vectors.float()
        quaternions = gaussians.quaternions.float()
        scales = gaussians.singular_values.float()

        intrinsics = intrinsics.float()
        if intrinsics.shape[-1] == 3:
            intrinsics_4x4 = torch.eye(4, device=device, dtype=torch.float32)
            intrinsics_4x4[:3, :3] = intrinsics
            intrinsics = intrinsics_4x4

        extrinsics = torch.eye(4, device=device, dtype=torch.float32)
        unprojection = get_unprojection_matrix(extrinsics, intrinsics, image_shape)
        linear = unprojection[..., :3, :3]  # [3, 3]
        offset = unprojection[..., :3, 3]  # [3]

        # means_world = means @ linear^T + offset   (differentiable in means).
        means_world = means @ linear.transpose(-1, -2) + offset

        # Sigma_world = linear @ Sigma_ndc @ linear^T   (differentiable).
        covars_ndc = compose_covariance_matrices(quaternions, scales)
        covars_world = linear @ covars_ndc @ linear.transpose(-1, -2)

        return WorldGaussians(
            means=means_world,
            covars=covars_world,
            colors=gaussians.colors.float(),
            opacities=gaussians.opacities.float(),
        )


_MAGMA_SELECTED = False


def _ensure_magma_eigh_backend(device: torch.device) -> None:
    """Prefer MAGMA for CUDA linalg (set once, idempotent).

    A backend sweep on this torch 2.8+cu128 build showed cuSOLVER's batched symmetric solver
    (``Xsyevbatched``) fails at the bufferSize query for batch >= ~65536 — independent of the
    data, conditioning, or scale — while MAGMA (bundled in this build) succeeds for every batch,
    including the full ~1.18M decomposition in a single call. eigh emits ~1.18M 3x3 matrices per
    branch, so we route CUDA eigendecompositions through MAGMA.
    """
    global _MAGMA_SELECTED
    if device.type == "cuda" and not _MAGMA_SELECTED:
        try:
            torch.backends.cuda.preferred_linalg_library("magma")
        except Exception:  # noqa: BLE001 - backend selection is best-effort
            pass
        _MAGMA_SELECTED = True


def covars_to_quats_scales(
    covars: torch.Tensor, jitter: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiably decompose symmetric 3x3 covariances into quaternions and scales.

    Uses :func:`torch.linalg.eigh` (differentiable, runs on GPU) instead of the inference code's
    detached CPU SVD. CUDA eigendecompositions are routed through MAGMA
    (:func:`_ensure_magma_eigh_backend`) because cuSOLVER's batched solver rejects the large
    (~1.18M) batch. A small diagonal ``jitter`` is added before the decomposition for strict
    positive-definiteness / gradient stability, inputs are sanitized, and the eigenvector basis
    is flipped to a proper rotation (det = +1) when ``eigh`` returns a reflection.

    Args:
        covars: Symmetric covariance matrices, shape [..., 3, 3].
        jitter: Diagonal floor added to the covariance before the eigendecomposition.

    Returns:
        ``(quaternions [..., 4], scales [..., 3])``.
    """
    # eigh / det reject low-precision dtypes; force fp32 (autocast disabled) so this is safe
    # inside a bf16 training step.
    with torch.autocast(device_type=covars.device.type, enabled=False):
        covars = covars.float()
        _ensure_magma_eigh_backend(covars.device)
        covars = torch.nan_to_num(covars, nan=0.0, posinf=0.0, neginf=0.0)
        # Symmetrize defensively (numerical asymmetry would make eigh complain).
        covars = 0.5 * (covars + covars.transpose(-1, -2))
        # Diagonal floor BEFORE eigh (positive-definiteness + sqrt-grad stability).
        if jitter > 0:
            eye = torch.eye(3, device=covars.device, dtype=covars.dtype)
            covars = covars + jitter * eye

        eigvals, eigvecs = torch.linalg.eigh(covars)  # eigvals ascending; eigvecs columns

        # Flip to a proper rotation (det = +1) without an in-place op on the autograd graph.
        det = torch.linalg.det(eigvecs)  # [...]
        flip = torch.ones_like(eigvecs)
        flip[..., :, 2] = torch.sign(det).unsqueeze(-1)
        rotations = eigvecs * flip

        scales = torch.sqrt(eigvals.clamp(min=0.0))
        quaternions = linalg.quaternions_from_rotation_matrices(rotations)
        return quaternions, scales


def lift_for_render(
    gaussians: Gaussians3D,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
    need_quats_scales: bool,
    jitter: float = 1e-8,
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
