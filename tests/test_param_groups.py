"""CPU tests for staged unfreezing + AdamW parameter groups (no forward pass needed).

Exact tensor counts (verified against the built model): decoders+heads = 222 tensors,
monodepth patch_encoder = 344 (frozen in both stages), monodepth image_encoder = 344,
depth_alignment = 156, total = 1149.
"""

from __future__ import annotations

import pytest
from sharp.models import DualPredictorParams, create_dual_predictor
from sharp_train.optim import build_param_groups, configure_stage, count_trainable

_DECODER_PREFIXES = (
    "feature_model_p",
    "feature_model_d",
    "prediction_head_p",
    "prediction_head_d",
)
_N_DECODER = 222


@pytest.fixture(scope="module")
def dual_model():
    """A dual-branch predictor built once for the module's stage tests."""
    return create_dual_predictor(DualPredictorParams())


def _trainable_names(model) -> set[str]:
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_stage_a_trains_only_decoders(dual_model) -> None:
    """Stage A: exactly the duplicated decoders + heads are trainable."""
    configure_stage(dual_model, "A")
    trainable = _trainable_names(dual_model)
    assert len(trainable) == _N_DECODER
    assert all(n.startswith(_DECODER_PREFIXES) for n in trainable)
    assert not any(n.startswith(("monodepth_model", "depth_alignment")) for n in trainable)

    groups = build_param_groups(dual_model)
    assert len(groups) == 1  # decoder group only; no trunk unfrozen


def test_stage_b_unfreezes_sharp_scope(dual_model) -> None:
    """Stage B: image-encoder body + depth alignment + decoders trainable; patch encoder frozen.

    ``set_requires_grad_`` intentionally re-freezes the encoders' unused ViT ``.head``
    (spn_encoder.py:176-177), so the image-encoder *body* (excluding ``.head``) is the
    trainable part.
    """
    configure_stage(dual_model, "A")
    n_a = count_trainable(dual_model)
    configure_stage(dual_model, "B")
    named = dict(dual_model.named_parameters())

    # Patch encoder stays fully frozen.
    assert all(not p.requires_grad for n, p in named.items() if "patch_encoder" in n)
    # Image-encoder body (excluding its unused ViT head) is unfrozen.
    image_body = [
        p
        for n, p in named.items()
        if "monodepth_model" in n and "image_encoder" in n and ".head." not in n
    ]
    assert len(image_body) > 0 and all(p.requires_grad for p in image_body)
    # Depth alignment + decoders unfrozen.
    assert all(p.requires_grad for n, p in named.items() if n.startswith("depth_alignment"))
    assert all(p.requires_grad for n, p in named.items() if n.startswith(_DECODER_PREFIXES))
    # Strictly more trainable than Stage A.
    assert count_trainable(dual_model) > n_a

    groups = build_param_groups(dual_model)
    assert len(groups) == 2  # decoder group + trunk group
    assert all(len(g["params"]) > 0 for g in groups)
    assert groups[0]["lr"] > groups[1]["lr"]  # decoder LR higher than trunk LR


def test_single_conditional_decoder_dedup() -> None:
    """The shared-decoder ablation de-duplicates aliased parameters in the groups."""
    model = create_dual_predictor(DualPredictorParams(single_conditional_decoder=True))
    configure_stage(model, "A")
    groups = build_param_groups(model)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))  # no parameter counted twice
