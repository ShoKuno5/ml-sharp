"""Training configuration dataclasses with plain-YAML load/dump (no new dependency).

Note on resolution: SHARP's sliding-pyramid patch encoder only accepts the 1536 internal
resolution (smaller inputs break the 384-tile arrangement), so ``internal_resolution`` is fixed
at 1536 for the pretrained model. The practical memory levers are therefore the render
resolution, gradient checkpointing, bf16, and keeping the patch encoder frozen (no activations).
"""

from __future__ import annotations

import dataclasses

import yaml

from sharp_train.losses import LossWeights


@dataclasses.dataclass
class TrainConfig:
    """End-to-end training configuration."""

    # Model / staging.
    checkpoint_path: str = (
        "/data/uni0/users/kuno/3dfm_distortion/cache/torch/hub/checkpoints/sharp_2572gikvuh.pt"
    )
    stage: str = "A"  # "A" (decoders only) or "B" (SHARP fine-tune scope)
    ray_map_channels: int = 0
    single_conditional_decoder: bool = False
    internal_resolution: int = 1536  # fixed by the patch encoder; see module docstring
    grad_checkpointing: bool = True

    # Data / render.
    render_size: int = 512
    focal_ratio: float = 1.0
    dummy_data: bool = True
    # ScanNet++ DSLR dataset (used when dummy_data is False).
    data_root: str = "/data/umiushi0/datasets/VGGT/ScanNet++"
    scannetpp_split: str = "train"
    pairs_per_scene: int = 50
    distorted_input_fraction: float = 0.7
    val_fraction: float = 0.1
    max_scenes: int | None = None
    num_workers: int = 4
    # DDP: set True if the staged graph leaves some trainable params without grad on a step.
    find_unused_parameters: bool = False
    resume: str = ""

    # Optimization.
    lr_decoder: float = 2.0e-4
    lr_trunk: float = 2.0e-5
    weight_decay: float = 0.0
    grad_accum: int = 1
    max_steps: int = 1000
    bf16: bool = True
    grad_clip: float = 1.0

    # Loss weights.
    loss: LossWeights = dataclasses.field(default_factory=LossWeights)

    # Logging / checkpointing.
    out_dir: str = "/data/uni0/users/kuno/3dfm_distortion/results/train"
    log_every: int = 10
    ckpt_every: int = 500
    seed: int = 0


def load_config(path: str) -> TrainConfig:
    """Load a :class:`TrainConfig` from a YAML file (``loss`` is a nested mapping)."""
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    loss_raw = raw.pop("loss", {}) or {}
    return TrainConfig(loss=LossWeights(**loss_raw), **raw)


def save_config(config: TrainConfig, path: str) -> None:
    """Dump a :class:`TrainConfig` to YAML."""
    data = dataclasses.asdict(config)
    with open(path, "w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
