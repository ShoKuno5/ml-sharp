"""Dual-branch Gaussian predictor for distorted-camera SHARP.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

This module is an *additive* extension of the inference model. It leaves
``RGBGaussianPredictor`` byte-identical (so the xi=0 single-branch regression is
preserved) and introduces a dual-branch variant that shares everything upstream of the
Gaussian decoder and duplicates only the weight-carrying, branch-specific modules
(``feature_model`` and ``prediction_head``; ``gaussian_composer`` is parameter-free).

Branch ``P`` is the pinhole / rectified branch and branch ``D`` is the input-aligned
(distorted) branch. Both consume the same shared trunk output and produce an independent
set of Gaussians in the canonical NDC-perspective frame; the renderer (extended
separately) applies the per-branch camera model.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import torch
from torch import nn

from sharp.models.monodepth import MonodepthWithEncodingAdaptor
from sharp.utils.gaussians import Gaussians3D

from .composer import GaussianComposer
from .params import PredictorParams
from .predictor import DepthAlignment


class DualGaussians(NamedTuple):
    """Output of :class:`DualBranchGaussianPredictor`.

    Attributes:
        pinhole: Gaussians from the pinhole/rectified branch (P).
        distorted: Gaussians from the input-aligned/distorted branch (D).
        aligned_monodepth: The shared (optionally GT-aligned) metric monodepth, exposed
            for the depth-supervision loss during training.
    """

    pinhole: Gaussians3D
    distorted: Gaussians3D
    aligned_monodepth: torch.Tensor


@dataclasses.dataclass
class DualPredictorParams:
    """Parameters for the dual-branch predictor.

    Wraps the standard :class:`PredictorParams` (which configures the shared trunk and
    the per-branch decoder/head/composer) and adds the distortion-specific knobs.
    """

    base: PredictorParams = dataclasses.field(default_factory=PredictorParams)
    # Number of analytic ray-map channels appended to the 5-channel feature input.
    # 0 reproduces the original SHARP feature input exactly (used by the xi=0 regression).
    ray_map_channels: int = 0
    # Ablation: a single ray-map-conditioned decoder shared by both branches (the branch
    # difference then comes only from the renderer camera model) vs two parallel decoders.
    single_conditional_decoder: bool = False


class DualBranchGaussianPredictor(nn.Module):
    """Predicts two parallel sets of 3D Gaussians from a single image.

    Shares ``monodepth_model``, ``depth_alignment`` and ``init_model`` across branches and
    runs two independent ``feature_model`` / ``prediction_head`` / ``gaussian_composer``
    stacks. The forward is identical to ``RGBGaussianPredictor.forward`` up to (and
    including) the initializer; only the decoder onward is duplicated.
    """

    def __init__(
        self,
        init_model: nn.Module,
        monodepth_model: MonodepthWithEncodingAdaptor,
        feature_model_p: nn.Module,
        feature_model_d: nn.Module,
        prediction_head_p: nn.Module,
        prediction_head_d: nn.Module,
        gaussian_composer_p: GaussianComposer,
        gaussian_composer_d: GaussianComposer,
        scale_map_estimator: nn.Module | None,
    ) -> None:
        """Initialize DualBranchGaussianPredictor.

        Args:
            init_model: Shared model mapping image and depth to base values.
            monodepth_model: Shared monodepth model with intermediate features.
            feature_model_p: Pinhole-branch image2image Gaussian decoder.
            feature_model_d: Distorted-branch image2image Gaussian decoder.
            prediction_head_p: Pinhole-branch delta-prediction head.
            prediction_head_d: Distorted-branch delta-prediction head.
            gaussian_composer_p: Pinhole-branch composer (parameter-free).
            gaussian_composer_d: Distorted-branch composer (parameter-free).
            scale_map_estimator: Shared module to align monodepth to ground-truth depth.
        """
        super().__init__()
        self.init_model = init_model
        self.monodepth_model = monodepth_model
        self.feature_model_p = feature_model_p
        self.feature_model_d = feature_model_d
        self.prediction_head_p = prediction_head_p
        self.prediction_head_d = prediction_head_d
        self.gaussian_composer_p = gaussian_composer_p
        self.gaussian_composer_d = gaussian_composer_d
        self.depth_alignment = DepthAlignment(scale_map_estimator)

    def _shared_trunk(
        self,
        image: torch.Tensor,
        disparity_factor: torch.Tensor,
        depth: torch.Tensor | None,
    ):
        """Run the shared trunk; mirrors RGBGaussianPredictor.forward (predictor.py:127-182)."""
        monodepth_output = self.monodepth_model(image)
        monodepth_disparity = monodepth_output.disparity

        disparity_factor = disparity_factor[:, None, None, None]
        monodepth = disparity_factor / monodepth_disparity.clamp(min=1e-4, max=1e4)

        monodepth, _ = self.depth_alignment(
            monodepth,
            depth,
            monodepth_output.decoder_features,
        )

        init_output = self.init_model(image, monodepth)
        return monodepth_output, monodepth, init_output

    def _run_branch(
        self,
        feature_model: nn.Module,
        prediction_head: nn.Module,
        gaussian_composer: GaussianComposer,
        init_output,
        output_features,
    ) -> Gaussians3D:
        """Run a single decoder branch; mirrors predictor.py:183-191."""
        image_features = feature_model(init_output.feature_input, encodings=output_features)
        delta_values = prediction_head(image_features)
        return gaussian_composer(
            delta=delta_values,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
        )

    def forward(
        self,
        image: torch.Tensor,
        disparity_factor: torch.Tensor,
        depth: torch.Tensor | None = None,
    ) -> DualGaussians:
        """Predict the pinhole and distorted Gaussian sets.

        Args:
            image: The image to process.
            disparity_factor: Factor to convert depth to disparities.
            depth: Ground-truth depth to align predicted depth to (training only).

        Returns:
            The two predicted Gaussian sets and the shared aligned monodepth.
        """
        monodepth_output, aligned_monodepth, init_output = self._shared_trunk(
            image, disparity_factor, depth
        )
        gaussians_p = self._run_branch(
            self.feature_model_p,
            self.prediction_head_p,
            self.gaussian_composer_p,
            init_output,
            monodepth_output.output_features,
        )
        gaussians_d = self._run_branch(
            self.feature_model_d,
            self.prediction_head_d,
            self.gaussian_composer_d,
            init_output,
            monodepth_output.output_features,
        )
        return DualGaussians(
            pinhole=gaussians_p,
            distorted=gaussians_d,
            aligned_monodepth=aligned_monodepth,
        )

    def internal_resolution(self) -> int:
        """Internal resolution."""
        return self.monodepth_model.internal_resolution()

    @property
    def output_resolution(self) -> int:
        """Output resolution of Gaussians."""
        return self.internal_resolution() // 2


def load_dual_from_single(
    dual_model: DualBranchGaussianPredictor,
    single_state_dict: dict[str, torch.Tensor],
    ray_map_channels: int = 0,
) -> None:
    """Warm-start a dual-branch model from a pretrained single-branch checkpoint.

    Deep-copies the pretrained ``feature_model.*`` and ``prediction_head.*`` weights into
    both branches and copies the shared trunk (``monodepth_model.*``, ``depth_alignment.*``;
    ``init_model`` is parameter-free) verbatim. With ``ray_map_channels == 0`` this yields
    branches that are bit-exact copies of the pretrained decoder, so the dual model
    reproduces the original SHARP Gaussians exactly.

    When ``ray_map_channels > 0`` the stem conv ``feature_model.image_encoder.conv.weight``
    (shape ``[out, 5, k, k]``) is widened to ``[out, 5 + R, k, k]`` with the extra input
    channels ZERO-initialized, so the appended ray-map channels contribute exactly zero at
    init and the bit-exact property is preserved.

    Args:
        dual_model: The dual-branch model to populate (modified in place).
        single_state_dict: State dict from a pretrained ``RGBGaussianPredictor``.
        ray_map_channels: Number of ray-map channels appended to the feature input.
    """
    single_branch = isinstance(dual_model.feature_model_p, nn.Module) and (
        dual_model.feature_model_p is dual_model.feature_model_d
    )
    stem_key = "feature_model.image_encoder.conv.weight"

    new_state_dict: dict[str, torch.Tensor] = {}
    for key, tensor in single_state_dict.items():
        if key.startswith("feature_model."):
            value = tensor
            if key == stem_key and ray_map_channels > 0:
                out_c, in_c, kh, kw = tensor.shape
                widened = tensor.new_zeros((out_c, in_c + ray_map_channels, kh, kw))
                widened[:, :in_c] = tensor
                value = widened
            suffix = key[len("feature_model.") :]
            if single_branch:
                new_state_dict[f"feature_model_p.{suffix}"] = value
            else:
                new_state_dict[f"feature_model_p.{suffix}"] = value.clone()
                new_state_dict[f"feature_model_d.{suffix}"] = value.clone()
        elif key.startswith("prediction_head."):
            suffix = key[len("prediction_head.") :]
            if single_branch:
                new_state_dict[f"prediction_head_p.{suffix}"] = tensor
            else:
                new_state_dict[f"prediction_head_p.{suffix}"] = tensor.clone()
                new_state_dict[f"prediction_head_d.{suffix}"] = tensor.clone()
        else:
            # Shared trunk (monodepth_model.*, depth_alignment.*) copied verbatim.
            new_state_dict[key] = tensor

    dual_model.load_state_dict(new_state_dict, strict=True)
