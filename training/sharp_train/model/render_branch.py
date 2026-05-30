"""Render one dual-branch Gaussian set through the (extended) gsplat renderer.

Reuses the inference :class:`sharp.utils.gsplat.GSplatRenderer` (extended in Step 2 with the
distortion / Unscented-Transform passthrough). Both branches render from ``quats`` + ``scales``
produced by the differentiable world lift (``lift_for_render(..., need_quats_scales=True)``):

- branch **P** renders classic pinhole (``with_ut=False``);
- branch **D** renders with the 3DGUT path (``with_ut=True, with_eval3d=True``) and the
  ``(xFoV, xi)``-derived ``camera_model`` + distortion coefficients.

Because ``compose(decompose(Sigma)) == Sigma`` (validated in ``test_world_lift``), rendering P
from the eigh-derived quats/scales is equivalent to rendering from ``covars`` directly, so a
single code path serves both branches.
"""

from __future__ import annotations

import torch
from sharp.utils.gaussians import Gaussians3D
from sharp.utils.gsplat import GSplatRenderer, RenderingOutputs

from .world_lift import WorldGaussians


def render_branch(
    renderer: GSplatRenderer,
    world: WorldGaussians,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    *,
    camera_model: str = "pinhole",
    radial_coeffs: torch.Tensor | None = None,
    tangential_coeffs: torch.Tensor | None = None,
    ftheta_coeffs=None,
    with_ut: bool = False,
    with_eval3d: bool = False,
) -> RenderingOutputs:
    """Render world-frame Gaussians for one branch.

    Args:
        renderer: A :class:`GSplatRenderer` (color space / background configured by the caller).
        world: World-frame Gaussians with ``quats``/``scales`` populated (UT path requires them).
        viewmats: World-to-camera extrinsics, shape [B, 4, 4].
        intrinsics: Camera intrinsics, shape [B, 4, 4] or [B, 3, 3].
        width: Output width in pixels.
        height: Output height in pixels.
        camera_model: gsplat camera model ("pinhole" for P; "fisheye"/"ftheta"/+radial for D).
        radial_coeffs: Per-camera radial distortion coefficients (branch D).
        tangential_coeffs: Per-camera tangential distortion coefficients (branch D).
        ftheta_coeffs: F-Theta distortion parameters (branch D).
        with_ut: Enable the Unscented Transform projection (required for distortion).
        with_eval3d: Evaluate the Gaussian response in 3D world space (3DGUT).

    Returns:
        The rendered color / depth / alpha.
    """
    if world.quats is None or world.scales is None:
        raise ValueError(
            "render_branch requires quats/scales; use lift_for_render(need_quats_scales=True)."
        )
    gaussians = Gaussians3D(
        mean_vectors=world.means,
        quaternions=world.quats,
        singular_values=world.scales,
        colors=world.colors,
        opacities=world.opacities,
    )
    return renderer(
        gaussians,
        viewmats,
        intrinsics,
        width,
        height,
        camera_model=camera_model,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        ftheta_coeffs=ftheta_coeffs,
        with_ut=with_ut,
        with_eval3d=with_eval3d,
    )
