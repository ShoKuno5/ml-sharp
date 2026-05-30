"""Correctness + differentiability of the training-time world lift (CPU-runnable, float32).

The reused ``sharp.utils`` math (linalg, covariance composition, unprojection) is float32-only,
which matches the training regime. The gsplat UT-vs-classic equality at xi=0 needs a CUDA device
and lives in a GPU-only test (run on iruka2). Here we check the parts that do not need gsplat:

- the covariance round-trip ``compose(decompose(Sigma)) == Sigma`` (the eigh decomposition is a
  correct inverse of the quaternion/scale composition);
- the metric covariance equals ``M Sigma_ndc M^T``;
- gradients flow through the full lift (means, quats, scales -> world means/covars/quats/scales).
"""

from __future__ import annotations

import torch
from sharp.utils.gaussians import (
    Gaussians3D,
    compose_covariance_matrices,
    get_unprojection_matrix,
)
from sharp_train.model.world_lift import covars_to_quats_scales, lift_for_render, lift_to_world

_INTRINSICS = torch.tensor(
    [[700.0, 0, 512, 0], [0, 700.0, 512, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32
)
_IMAGE_SHAPE = (1024, 1024)


def _random_spd(n: int, generator: torch.Generator) -> torch.Tensor:
    """Random symmetric positive-definite 3x3 matrices with distinct eigenvalues (float32)."""
    q, _ = torch.linalg.qr(torch.randn(n, 3, 3, generator=generator))
    eigvals = torch.tensor([0.5, 1.0, 2.0]).expand(n, 3)
    eigvals = eigvals + 0.1 * torch.rand(n, 3, generator=generator)
    return q @ torch.diag_embed(eigvals) @ q.transpose(-1, -2)


def _gaussians(n: int, generator: torch.Generator, requires_grad: bool = False) -> Gaussians3D:
    means = torch.randn(1, n, 3, generator=generator)
    quats = torch.randn(1, n, 4, generator=generator)
    scales = torch.rand(1, n, 3, generator=generator) + 0.3
    if requires_grad:
        means.requires_grad_(True)
        quats.requires_grad_(True)
        scales.requires_grad_(True)
    return Gaussians3D(
        mean_vectors=means,
        quaternions=quats,
        singular_values=scales,
        colors=torch.rand(1, n, 3, generator=generator),
        opacities=torch.rand(1, n, generator=generator),
    )


def test_covariance_decompose_roundtrip() -> None:
    """compose(decompose(Sigma)) reproduces Sigma for well-conditioned SPD covariances."""
    gen = torch.Generator().manual_seed(0)
    covars = _random_spd(64, gen)
    quats, scales = covars_to_quats_scales(covars, jitter=0.0)
    recomposed = compose_covariance_matrices(quats, scales)
    assert torch.allclose(recomposed, covars, atol=1e-4, rtol=1e-3)


def test_metric_covariance_formula() -> None:
    """lift_to_world covariance equals M @ Sigma_ndc @ M^T with the unprojection's linear part."""
    gen = torch.Generator().manual_seed(1)
    gaussians = _gaussians(32, gen)
    world = lift_to_world(gaussians, _INTRINSICS, _IMAGE_SHAPE)

    linear = get_unprojection_matrix(torch.eye(4), _INTRINSICS, _IMAGE_SHAPE)[:3, :3]
    sigma_ndc = compose_covariance_matrices(gaussians.quaternions, gaussians.singular_values)
    expected = linear @ sigma_ndc @ linear.transpose(-1, -2)
    assert torch.allclose(world.covars, expected, atol=1e-4, rtol=1e-3)


def test_means_lift_matches_unprojection() -> None:
    """World means equal the affine unprojection of the NDC-perspective means."""
    gen = torch.Generator().manual_seed(3)
    gaussians = _gaussians(32, gen)
    world = lift_to_world(gaussians, _INTRINSICS, _IMAGE_SHAPE)
    unproj = get_unprojection_matrix(torch.eye(4), _INTRINSICS, _IMAGE_SHAPE)
    expected = gaussians.mean_vectors @ unproj[:3, :3].transpose(-1, -2) + unproj[:3, 3]
    assert torch.allclose(world.means, expected, atol=1e-4, rtol=1e-3)


def test_lift_under_bf16_autocast() -> None:
    """The lift runs under a bf16 autocast step (linalg.inv / eigh pinned to fp32).

    Reproduces the iruka2 overfit crash (``linalg.inv: Low precision dtypes not supported``) on
    CPU: without the fp32 pinning, the autocast-bf16 matmul into inv/eigh would raise.
    """
    gen = torch.Generator().manual_seed(7)
    gaussians = _gaussians(16, gen)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        world = lift_for_render(gaussians, _INTRINSICS, _IMAGE_SHAPE, need_quats_scales=True)
    for name, tensor in (
        ("means", world.means),
        ("covars", world.covars),
        ("quats", world.quats),
        ("scales", world.scales),
    ):
        assert tensor.dtype == torch.float32, f"{name} not fp32: {tensor.dtype}"
        assert torch.isfinite(tensor).all(), f"{name} non-finite under bf16 autocast"


def test_lift_gradients_flow() -> None:
    """Gradients flow through the lift to means, quaternions and scales (no NaN, nonzero)."""
    gen = torch.Generator().manual_seed(2)
    gaussians = _gaussians(16, gen, requires_grad=True)
    world = lift_for_render(gaussians, _INTRINSICS, _IMAGE_SHAPE, need_quats_scales=True)
    loss = world.means.square().mean() + world.scales.square().mean() + world.quats.square().mean()
    loss.backward()
    for name, tensor in (
        ("means", gaussians.mean_vectors),
        ("quats", gaussians.quaternions),
        ("scales", gaussians.singular_values),
    ):
        assert tensor.grad is not None, f"no grad for {name}"
        assert torch.isfinite(tensor.grad).all(), f"non-finite grad for {name}"
        assert tensor.grad.abs().sum() > 0, f"zero grad for {name}"
