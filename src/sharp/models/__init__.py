"""Contains different Gaussian predictors.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import copy

from sharp.models.monodepth import (
    create_monodepth_adaptor,
    create_monodepth_dpt,
)

from .alignment import create_alignment
from .composer import GaussianComposer
from .dual_predictor import (
    DualBranchGaussianPredictor,
    DualPredictorParams,
    load_dual_from_single,
)
from .gaussian_decoder import create_gaussian_decoder
from .heads import DirectPredictionHead
from .initializer import create_initializer
from .params import PredictorParams
from .predictor import RGBGaussianPredictor


def create_predictor(params: PredictorParams) -> RGBGaussianPredictor:
    """Create gaussian predictor model specified by name."""
    if params.gaussian_decoder.stride < params.initializer.stride:
        raise ValueError(
            "We donot expected gaussian_decoder has higher resolution than initializer."
        )

    scale_factor = params.gaussian_decoder.stride // params.initializer.stride
    gaussian_composer = GaussianComposer(
        delta_factor=params.delta_factor,
        min_scale=params.min_scale,
        max_scale=params.max_scale,
        color_activation_type=params.color_activation_type,
        opacity_activation_type=params.opacity_activation_type,
        color_space=params.color_space,
        scale_factor=scale_factor,
        base_scale_on_predicted_mean=params.base_scale_on_predicted_mean,
    )
    if params.num_monodepth_layers > 1 and params.initializer.num_layers != 2:
        raise KeyError("We only support num_layers = 2 when num_monodepth_layers > 1.")

    monodepth_model = create_monodepth_dpt(params.monodepth)
    monodepth_adaptor = create_monodepth_adaptor(
        monodepth_model,
        params.monodepth_adaptor,
        params.num_monodepth_layers,
        params.sorting_monodepth,
    )

    if params.num_monodepth_layers == 2:
        monodepth_adaptor.replicate_head(params.num_monodepth_layers)

    gaussian_decoder = create_gaussian_decoder(
        params.gaussian_decoder,
        dims_depth_features=monodepth_adaptor.get_feature_dims(),
    )
    initializer = create_initializer(
        params.initializer,
    )
    prediction_head = DirectPredictionHead(
        feature_dim=gaussian_decoder.dim_out, num_layers=initializer.num_layers
    )
    decoder_dim = monodepth_model.decoder.dims_decoder[-1]
    return RGBGaussianPredictor(
        init_model=initializer,
        feature_model=gaussian_decoder,
        prediction_head=prediction_head,
        monodepth_model=monodepth_adaptor,
        gaussian_composer=gaussian_composer,
        scale_map_estimator=create_alignment(params.depth_alignment, depth_decoder_dim=decoder_dim),
    )


def create_dual_predictor(params: DualPredictorParams) -> DualBranchGaussianPredictor:
    """Create a dual-branch Gaussian predictor (shared trunk + duplicated decoder).

    Mirrors :func:`create_predictor` for the shared trunk (monodepth, alignment,
    initializer) and builds two independent ``feature_model`` / ``prediction_head`` /
    ``gaussian_composer`` stacks. ``create_gaussian_decoder`` mutates its params object
    in place, so each branch gets a deep-copied params instance.
    """
    base = params.base
    if base.gaussian_decoder.stride < base.initializer.stride:
        raise ValueError(
            "We donot expected gaussian_decoder has higher resolution than initializer."
        )
    scale_factor = base.gaussian_decoder.stride // base.initializer.stride

    def _make_composer() -> GaussianComposer:
        return GaussianComposer(
            delta_factor=base.delta_factor,
            min_scale=base.min_scale,
            max_scale=base.max_scale,
            color_activation_type=base.color_activation_type,
            opacity_activation_type=base.opacity_activation_type,
            color_space=base.color_space,
            scale_factor=scale_factor,
            base_scale_on_predicted_mean=base.base_scale_on_predicted_mean,
        )

    if base.num_monodepth_layers > 1 and base.initializer.num_layers != 2:
        raise KeyError("We only support num_layers = 2 when num_monodepth_layers > 1.")

    monodepth_model = create_monodepth_dpt(base.monodepth)
    monodepth_adaptor = create_monodepth_adaptor(
        monodepth_model,
        base.monodepth_adaptor,
        base.num_monodepth_layers,
        base.sorting_monodepth,
    )
    if base.num_monodepth_layers == 2:
        monodepth_adaptor.replicate_head(base.num_monodepth_layers)

    dims_depth_features = monodepth_adaptor.get_feature_dims()
    initializer = create_initializer(base.initializer)

    def _make_decoder():
        gaussian_decoder_params = copy.deepcopy(base.gaussian_decoder)
        # Widen the decoder input for the appended ray-map channels (0 by default).
        gaussian_decoder_params.dim_in = (
            base.gaussian_decoder.dim_in + params.ray_map_channels
        )
        return create_gaussian_decoder(
            gaussian_decoder_params, dims_depth_features=dims_depth_features
        )

    def _make_head(decoder):
        return DirectPredictionHead(
            feature_dim=decoder.dim_out, num_layers=initializer.num_layers
        )

    if params.single_conditional_decoder:
        shared_decoder = _make_decoder()
        shared_head = _make_head(shared_decoder)
        feature_model_p = feature_model_d = shared_decoder
        prediction_head_p = prediction_head_d = shared_head
    else:
        feature_model_p = _make_decoder()
        feature_model_d = _make_decoder()
        prediction_head_p = _make_head(feature_model_p)
        prediction_head_d = _make_head(feature_model_d)

    decoder_dim = monodepth_model.decoder.dims_decoder[-1]
    return DualBranchGaussianPredictor(
        init_model=initializer,
        monodepth_model=monodepth_adaptor,
        feature_model_p=feature_model_p,
        feature_model_d=feature_model_d,
        prediction_head_p=prediction_head_p,
        prediction_head_d=prediction_head_d,
        gaussian_composer_p=_make_composer(),
        gaussian_composer_d=_make_composer(),
        scale_map_estimator=create_alignment(
            base.depth_alignment, depth_decoder_dim=decoder_dim
        ),
    )


__all__ = [
    "PredictorParams",
    "create_predictor",
    "DualPredictorParams",
    "DualBranchGaussianPredictor",
    "create_dual_predictor",
    "load_dual_from_single",
]
