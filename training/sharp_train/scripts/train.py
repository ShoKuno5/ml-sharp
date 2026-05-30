"""Distributed training entrypoint for dual-branch SHARP (ScanNet++ DSLR or dummy data).

Photometric-only pilot by default (pilot.yaml sets ``loss.lambda_depth=0``). Launch on iruka2:

    torchrun --nproc_per_node=N -m sharp_train.scripts.train --config .../pilot.yaml

Single-process (no torchrun env) also works without DDP. Staging / param groups are applied to the
*unwrapped* model (build_param_groups classifies by parameter name, which DDP would prefix), then
the model is wrapped in DDP for the forward.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from sharp_train.config import TrainConfig, load_config
from sharp_train.data import DummyDataset
from sharp_train.data.scannetpp import ScanNetppConfig, ScanNetppDslrDataset
from sharp_train.engine import build_model, build_renderer, fit
from sharp_train.engine.trainer import save_checkpoint
from sharp_train.losses import SymmetricLoss
from sharp_train.optim import build_param_groups


def _passthrough_collate(samples):
    """BatchSample already carries the leading B=1; DataLoader batch_size=1 just unwraps it."""
    return samples[0]


def _ddp_setup() -> tuple[bool, int, int, int]:
    """Initialize the process group from torchrun env; return (is_ddp, rank, world, local_rank)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def build_dataset(config: TrainConfig) -> torch.utils.data.Dataset:
    """Construct the dummy dataset or the ScanNet++ DSLR dataset from the config."""
    if config.dummy_data:
        return DummyDataset(length=1_000_000_000, render_size=config.render_size, seed=config.seed)
    return ScanNetppDslrDataset(
        ScanNetppConfig(
            root=config.data_root,
            split=config.scannetpp_split,
            internal_resolution=config.internal_resolution,
            render_size=config.render_size,
            distorted_input_fraction=config.distorted_input_fraction,
            pairs_per_scene=config.pairs_per_scene,
            val_fraction=config.val_fraction,
            max_scenes=config.max_scenes,
            seed=config.seed,
        )
    )


def cycle_batches(loader: DataLoader, sampler: DistributedSampler | None):
    """Yield batches forever, advancing the DistributedSampler epoch for fresh shuffles."""
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def main() -> None:
    """Run the (optionally distributed) training loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    is_ddp, rank, world, local_rank = _ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed + rank)

    dataset = build_dataset(config)
    sampler = (
        DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if is_ddp else None
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=(sampler is None and not config.dummy_data),
        num_workers=config.num_workers,
        collate_fn=_passthrough_collate,
        persistent_workers=config.num_workers > 0,
        pin_memory=False,
        drop_last=True,
    )

    # Build + stage + optimizer on the UNWRAPPED model (param-name classification), then DDP-wrap.
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(
        build_param_groups(model, config.lr_decoder, config.lr_trunk, config.weight_decay)
    )
    start_step = 0
    if config.resume and os.path.exists(config.resume):
        ckpt = torch.load(config.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))

    train_model = (
        DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=config.find_unused_parameters
        )
        if is_ddp
        else model
    )
    renderer = build_renderer()
    loss_fn = SymmetricLoss(config.loss).to(device)

    writer = None
    if rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(config.out_dir, exist_ok=True)
            writer = SummaryWriter(config.out_dir)
        except Exception:  # noqa: BLE001 - logging is best-effort
            writer = None
        print(
            f"[train] ddp={is_ddp} world={world} dataset_size={len(dataset)} "
            f"stage={config.stage} max_steps={config.max_steps} dummy={config.dummy_data}",
            flush=True,
        )

    batches = cycle_batches(loader, sampler)
    for step, terms in fit(train_model, renderer, loss_fn, batches, config, device, optimizer):
        global_step = start_step + step
        if rank == 0 and step % config.log_every == 0:
            shown = {k: terms[k] for k in ("total", "D_pp", "D_dd", "depth") if k in terms}
            msg = "  ".join(f"{k}={v:.4f}" for k, v in shown.items())
            print(f"step {global_step:6d}  {msg}", flush=True)
            if writer is not None:
                for key, value in terms.items():
                    writer.add_scalar(f"loss/{key}", value, global_step)
        if rank == 0 and config.ckpt_every > 0 and step % config.ckpt_every == 0:
            save_checkpoint(model, optimizer, global_step, config.out_dir)

    if rank == 0:
        save_checkpoint(model, optimizer, start_step + config.max_steps, config.out_dir)
        if writer is not None:
            writer.close()
        print("[train] done", flush=True)
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
