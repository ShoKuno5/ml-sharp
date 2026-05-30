"""Single-sample overfit sanity check (run on a CUDA GPU, e.g. iruka2).

Drives the full dual-branch pipeline (predict -> world lift -> render P + D -> symmetric loss)
on one fixed identity-pose self-reconstruction sample and asserts the loss decreases markedly.
Perceptual terms are off by default (no backbone downloads); pass --perceptual to enable them.

    PYTHONPATH=src:training python -m sharp_train.scripts.overfit --steps 300 --render-size 256
"""

from __future__ import annotations

import argparse
import dataclasses

import torch

from sharp_train.config import TrainConfig
from sharp_train.data import make_dummy_sample
from sharp_train.engine import build_model, build_renderer, fit
from sharp_train.losses import SymmetricLoss


def main() -> None:
    """Run the overfit sanity loop and report first/last loss."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--stage", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--perceptual", action="store_true", help="enable LPIPS + DISTS terms")
    parser.add_argument("--threshold", type=float, default=0.5, help="max(last/first) loss ratio")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainConfig(stage=args.stage, render_size=args.render_size, max_steps=args.steps)
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if not args.perceptual:
        config.loss = dataclasses.replace(config.loss, beta_lpips=0.0, gamma_dists=0.0)

    model = build_model(config, device)
    renderer = build_renderer()
    loss_fn = SymmetricLoss(config.loss).to(device)
    sample = make_dummy_sample(render_size=config.render_size, device="cpu")
    batches = (sample for _ in range(config.max_steps * config.grad_accum))

    first_loss = None
    last_loss = None
    for step, terms in fit(model, renderer, loss_fn, batches, config, device):
        if first_loss is None:
            first_loss = terms["total"]
        last_loss = terms["total"]
        if step % config.log_every == 0 or step == 1:
            print(f"step {step:5d}  total={terms['total']:.5f}")

    print(f"first={first_loss:.5f}  last={last_loss:.5f}")
    assert last_loss < args.threshold * first_loss, (
        f"overfit did not converge: last={last_loss:.5f} first={first_loss:.5f}"
    )
    print("OVERFIT OK")


if __name__ == "__main__":
    main()
