"""Staged unfreezing and AdamW parameter groups for the dual-branch predictor.

Stage A trains only the duplicated Gaussian decoders + heads (``feature_model_{p,d}`` and
``prediction_head_{p,d}``; the appended ray-map stem channels live inside ``feature_model``).
Stage B additionally unfreezes the SHARP fine-tune scope — the monodepth image encoder, depth
decoder, depth head and the depth-alignment module — while keeping the heavy patch encoder
frozen. Because the patch encoder's parameters stay ``requires_grad=False`` and its input (the
image) does not require grad, autograd does not retain its activations, which is the main
memory lever; no explicit ``no_grad`` wrapper is needed.

This reuses SHARP's own ``set_requires_grad_`` / ``requires_grad_`` knobs and never mutates the
inference code.
"""

from __future__ import annotations

import torch
from torch import nn

STAGE_A = "A"
STAGE_B = "B"

_DECODER_PREFIXES = ("feature_model_p", "feature_model_d", "prediction_head_p", "prediction_head_d")


def configure_stage(model: nn.Module, stage: str) -> None:
    """Set ``requires_grad`` across the dual-branch model for the given stage.

    Args:
        model: A :class:`DualBranchGaussianPredictor`.
        stage: ``"A"`` (decoders + heads only) or ``"B"`` (also SHARP fine-tune scope).
    """
    if stage not in (STAGE_A, STAGE_B):
        raise ValueError(f"Unknown stage {stage!r}; expected 'A' or 'B'.")

    # Freeze everything, then selectively unfreeze.
    model.requires_grad_(False)

    for name in _DECODER_PREFIXES:
        module = getattr(model, name)
        module.requires_grad_(True)

    if stage == STAGE_B:
        monodepth = model.monodepth_model.monodepth_predictor
        # Keep the patch encoder frozen (memory); train the image encoder.
        monodepth.encoder.set_requires_grad_(patch_encoder=False, image_encoder=True)
        monodepth.decoder.requires_grad_(True)
        monodepth.head.requires_grad_(True)
        if model.depth_alignment.scale_map_estimator is not None:
            model.depth_alignment.scale_map_estimator.requires_grad_(True)


def _is_decoder_param(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _DECODER_PREFIXES)


def build_param_groups(
    model: nn.Module,
    lr_decoder: float = 2e-4,
    lr_trunk: float = 2e-5,
    weight_decay: float = 0.0,
) -> list[dict]:
    """Build AdamW parameter groups: a higher-LR decoder group and a lower-LR trunk group.

    Only parameters with ``requires_grad=True`` are included (call :func:`configure_stage`
    first). Aliased parameters (e.g. the single-conditional-decoder ablation, where both
    branches share one module) are de-duplicated by identity.

    Args:
        model: The configured dual-branch model.
        lr_decoder: Learning rate for the duplicated decoders + heads.
        lr_trunk: Learning rate for the (Stage B) unfrozen trunk modules.
        weight_decay: Weight decay applied to both groups.

    Returns:
        A list of parameter-group dicts suitable for ``torch.optim.AdamW``.
    """
    decoder_params: list[torch.nn.Parameter] = []
    trunk_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        (decoder_params if _is_decoder_param(name) else trunk_params).append(param)

    groups: list[dict] = []
    if decoder_params:
        groups.append({"params": decoder_params, "lr": lr_decoder, "weight_decay": weight_decay})
    if trunk_params:
        groups.append({"params": trunk_params, "lr": lr_trunk, "weight_decay": weight_decay})
    return groups


def count_trainable(model: nn.Module) -> int:
    """Total number of trainable scalar parameters (de-duplicated by identity)."""
    seen: set[int] = set()
    total = 0
    for param in model.parameters():
        if param.requires_grad and id(param) not in seen:
            seen.add(id(param))
            total += param.numel()
    return total
