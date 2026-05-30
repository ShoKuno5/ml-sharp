"""GPU-only: the Unscented-Transform render path reproduces classic pinhole at xi=0.

Requires a CUDA device with a gsplat-compatible architecture (e.g. the A100 on iruka2);
skipped on CPU / incompatible GPUs. Run with:

    PYTHONPATH=src:training python -m pytest tests/test_renderer_ut_xi0.py -v

Validates the B3 renderer extension: with no distortion (``radial_coeffs=None``,
``camera_model="pinhole"``), enabling the 3DGUT path (``with_ut``/``with_eval3d``) must match
the classic EWA rasterization to high PSNR.
"""

from __future__ import annotations

import math

import pytest
import torch
from sharp.utils.gaussians import Gaussians3D
from sharp.utils.gsplat import GSplatRenderer


def _cuda_usable() -> bool:
    """True only for a CUDA device gsplat can actually launch kernels on (>= sm_70)."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 7


CUDA_OK = _cuda_usable()
PSNR_THRESHOLD_DB = 50.0


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).square().mean().item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _toy_scene(device: torch.device, n: int = 2000) -> Gaussians3D:
    gen = torch.Generator(device="cpu").manual_seed(0)
    means = torch.randn(n, 3, generator=gen) * 0.3
    means[:, 2] = means[:, 2].abs() + 2.0  # in front of the camera
    quats = torch.randn(n, 4, generator=gen)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = (torch.rand(n, 3, generator=gen) * 0.02 + 0.01)
    colors = torch.rand(n, 3, generator=gen)
    opacities = torch.rand(n, generator=gen) * 0.5 + 0.5
    return Gaussians3D(
        mean_vectors=means[None].to(device),
        quaternions=quats[None].to(device),
        singular_values=scales[None].to(device),
        colors=colors[None].to(device),
        opacities=opacities[None].to(device),
    )


@pytest.mark.skipif(not CUDA_OK, reason="gsplat rasterization requires a CUDA device")
def test_ut_pinhole_matches_classic() -> None:
    """The UT (3DGUT) pinhole render matches classic EWA rasterization at xi=0."""
    device = torch.device("cuda")
    width = height = 256
    f = 200.0
    intrinsics = torch.tensor(
        [[f, 0, width / 2, 0], [0, f, height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        device=device,
    )[None]
    extrinsics = torch.eye(4, device=device)[None]
    gaussians = _toy_scene(device)
    renderer = GSplatRenderer(color_space="linearRGB", background_color="black")

    classic = renderer(gaussians, extrinsics, intrinsics, width, height)
    ut = renderer(
        gaussians,
        extrinsics,
        intrinsics,
        width,
        height,
        camera_model="pinhole",
        with_ut=True,
        with_eval3d=True,
    )

    psnr = _psnr(classic.color, ut.color)
    assert psnr > PSNR_THRESHOLD_DB, f"UT vs classic PSNR={psnr:.1f}dB <= {PSNR_THRESHOLD_DB}dB"
