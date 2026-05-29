"""xi=0 regression: the dual-branch predictor must reproduce single-branch SHARP exactly.

Both branches of :class:`DualBranchGaussianPredictor`, warm-started from a single-branch
checkpoint with ``ray_map_channels=0``, must produce Gaussians bit-identical (within a tiny
tolerance) to the original :class:`RGBGaussianPredictor`. This guards the inference path:
duplicating the Gaussian decoder is provably harmless at xi=0.

Resolution and checkpoint are configurable via env vars so the test is cheap to run on a
GPU node (iruka2) at full 1536 and still runnable on CPU at a reduced resolution:

    SHARP_TEST_RES   input square resolution (default 1536)
    SHARP_CKPT       path to the pretrained .pt (enables the "reproduces SHARP" test)
"""

from __future__ import annotations

import os

import pytest
import torch
from sharp.models import (
    DualPredictorParams,
    PredictorParams,
    create_dual_predictor,
    create_predictor,
    load_dual_from_single,
)

RES = int(os.environ.get("SHARP_TEST_RES", "1536"))
CKPT = os.environ.get(
    "SHARP_CKPT",
    "/data/uni0/users/kuno/3dfm_distortion/cache/torch/hub/checkpoints/sharp_2572gikvuh.pt",
)
TOL = 1e-5
ATTRS = ("mean_vectors", "singular_values", "quaternions", "colors", "opacities")


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _assert_branches_match(single_gaussians, dual_gaussians) -> None:
    for branch_name, branch in (
        ("pinhole", dual_gaussians.pinhole),
        ("distorted", dual_gaussians.distorted),
    ):
        for attr in ATTRS:
            diff = _max_abs_diff(getattr(single_gaussians, attr), getattr(branch, attr))
            assert diff < TOL, f"branch={branch_name} attr={attr} max|delta|={diff:.3e} >= {TOL}"


def test_dual_equals_single_random_init() -> None:
    """Wiring + warm-start: dual branches == single, regardless of (random) weights."""
    torch.manual_seed(0)
    single = create_predictor(PredictorParams()).eval()
    dual = create_dual_predictor(DualPredictorParams()).eval()
    load_dual_from_single(dual, single.state_dict(), ray_map_channels=0)

    image = torch.rand(1, 3, RES, RES)
    disparity_factor = torch.tensor([1.0])
    with torch.no_grad():
        single_g = single(image, disparity_factor)
        dual_g = dual(image, disparity_factor)
    _assert_branches_match(single_g, dual_g)


@pytest.mark.skipif(not os.path.exists(CKPT), reason="pretrained checkpoint not available")
def test_dual_reproduces_pretrained_sharp() -> None:
    """Anti-forgetting: dual branches == original SHARP using the pretrained checkpoint."""
    torch.manual_seed(0)
    state_dict = torch.load(CKPT, map_location="cpu", weights_only=True)

    single = create_predictor(PredictorParams()).eval()
    single.load_state_dict(state_dict, strict=True)

    dual = create_dual_predictor(DualPredictorParams()).eval()
    load_dual_from_single(dual, state_dict, ray_map_channels=0)

    image = torch.rand(1, 3, RES, RES)
    disparity_factor = torch.tensor([1.0])
    with torch.no_grad():
        single_g = single(image, disparity_factor)
        dual_g = dual(image, disparity_factor)
    _assert_branches_match(single_g, dual_g)
