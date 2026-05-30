"""Model assembly + the render-supervised training step + a minimal fit loop.

The step wires: dual predictor -> differentiable world lift -> per-branch gsplat render
(P classic pinhole, D 3DGUT + distortion) -> symmetric loss. bf16 autocast and gradient
accumulation are supported. Multi-GPU is intended via ``torchrun`` + DDP (the model wraps
cleanly); the single-process path here is what the overfit sanity uses.

The world lift currently passes the (single) input intrinsics ``input_intrinsics[0]``, i.e. it
assumes one sample per GPU (the pilot config) or a shared input camera within the batch.
"""

from __future__ import annotations

import contextlib
import os

import torch
from sharp.models import (
    DualPredictorParams,
    PredictorParams,
    create_dual_predictor,
    load_dual_from_single,
)
from sharp.utils.gsplat import GSplatRenderer
from torch import nn

from sharp_train.config import TrainConfig
from sharp_train.data.batch import BatchSample
from sharp_train.losses import SymmetricLoss
from sharp_train.model.render_branch import render_branch
from sharp_train.model.world_lift import lift_for_render
from sharp_train.optim import build_param_groups, configure_stage


def build_model(config: TrainConfig, device: torch.device) -> nn.Module:
    """Build the dual-branch predictor, warm-start from the checkpoint, and stage-configure it."""
    params = DualPredictorParams(
        base=PredictorParams(),
        ray_map_channels=config.ray_map_channels,
        single_conditional_decoder=config.single_conditional_decoder,
    )
    params.base.gaussian_decoder.grad_checkpointing = config.grad_checkpointing
    params.base.monodepth.grad_checkpointing = config.grad_checkpointing

    model = create_dual_predictor(params)
    state_dict = torch.load(config.checkpoint_path, map_location="cpu", weights_only=True)
    load_dual_from_single(model, state_dict, ray_map_channels=config.ray_map_channels)
    configure_stage(model, config.stage)
    return model.to(device)


def build_renderer() -> GSplatRenderer:
    """Build the renderer matching the predictor's linearRGB Gaussian output."""
    return GSplatRenderer(
        color_space="linearRGB", background_color="black", low_pass_filter_eps=1e-2
    )


def train_step(
    model: nn.Module,
    renderer: GSplatRenderer,
    loss_fn: SymmetricLoss,
    batch: BatchSample,
    internal_resolution: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One render-supervised forward: returns ``(total_loss, term_dict)``."""
    dual = model(batch.image, batch.disparity_factor, depth=batch.gt_depth)
    internal_shape = (internal_resolution, internal_resolution)
    intrinsics = batch.input_intrinsics[0]  # [4, 4]; one sample per GPU (see module docstring)

    world_p = lift_for_render(dual.pinhole, intrinsics, internal_shape, need_quats_scales=True)
    world_d = lift_for_render(dual.distorted, intrinsics, internal_shape, need_quats_scales=True)

    render_p = render_branch(
        renderer,
        world_p,
        batch.target_viewmat,
        batch.target_intrinsics,
        batch.render_width,
        batch.render_height,
        camera_model="pinhole",
        with_ut=False,
    )
    render_d = render_branch(
        renderer,
        world_d,
        batch.target_viewmat,
        batch.target_intrinsics,
        batch.render_width,
        batch.render_height,
        camera_model=batch.camera_model,
        radial_coeffs=batch.radial_coeffs,
        tangential_coeffs=batch.tangential_coeffs,
        with_ut=True,
        with_eval3d=True,
    )

    total, terms = loss_fn(
        render_p_pin=render_p.color,
        render_d_dist=render_d.color,
        gt_pin=batch.gt_pinhole,
        gt_dist=batch.gt_distorted,
        mask_pin=batch.mask_pinhole,
        mask_dist=batch.mask_distorted,
        pred_depth=dual.aligned_monodepth,
        gt_depth=batch.gt_depth,
    )
    return total, terms


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def fit(
    model: nn.Module,
    renderer: GSplatRenderer,
    loss_fn: SymmetricLoss,
    batches,
    config: TrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
):
    """Run the fit loop over an iterable of :class:`BatchSample` (length == steps).

    Args:
        model: Dual-branch predictor (already staged).
        renderer: The gsplat renderer.
        loss_fn: The symmetric loss.
        batches: Iterable yielding ``BatchSample`` (e.g. a DataLoader or a repeated sample).
        config: Training configuration.
        device: Compute device.
        optimizer: Optional pre-built optimizer; if None, AdamW from staged param groups.

    Yields:
        ``(step, term_dict)`` after each optimizer step (for logging / asserting convergence).
    """
    if optimizer is None:
        groups = build_param_groups(model, config.lr_decoder, config.lr_trunk, config.weight_decay)
        optimizer = torch.optim.AdamW(groups)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for index, batch in enumerate(batches):
        batch = batch.to(device)
        with _autocast(device, config.bf16):
            total, terms = train_step(model, renderer, loss_fn, batch, config.internal_resolution)
        (total / config.grad_accum).backward()

        if (index + 1) % config.grad_accum == 0:
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), config.grad_clip
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            yield step, {k: float(v.detach()) for k, v in terms.items()}
            if step >= config.max_steps:
                break


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, out_dir: str):
    """Save model + optimizer state to ``out_dir/ckpt_<step>.pt``."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_{step}.pt")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path
    )
    return path
