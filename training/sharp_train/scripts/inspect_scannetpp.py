"""Validate the ScanNet++ DSLR loader on real data (run on iruka2 where umiushi0 is mounted).

Builds the dataset over a few scenes, prints scene/pair counts + per-sample camera info, and saves
side-by-side grids (GT pinhole | GT distorted | distorted mask) so the pose/intrinsics/mask wiring
can be eyeballed against the real fisheye+undistorted pairs.

    PYTHONPATH=src:training python -m sharp_train.scripts.inspect_scannetpp --max-scenes 3 --num 6
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from sharp_train.data.scannetpp import ScanNetppConfig, ScanNetppDslrDataset


def _to_uint8(tensor) -> np.ndarray:
    array = tensor[0].detach().cpu().numpy()
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    array = np.transpose(array, (1, 2, 0))
    return (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _save_grid(sample, path: str) -> None:
    import cv2

    panels = [
        _to_uint8(sample.gt_pinhole),
        _to_uint8(sample.gt_distorted),
        _to_uint8(sample.mask_distorted),
    ]
    height = max(panel.shape[0] for panel in panels)
    resized = [
        cv2.resize(panel, (int(panel.shape[1] * height / panel.shape[0]), height))
        for panel in panels
    ]
    grid = np.concatenate(resized, axis=1)
    cv2.imwrite(path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def main() -> None:
    """Build the dataset over a few scenes and dump sample grids + stats."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/umiushi0/datasets/VGGT/ScanNet++")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-scenes", type=int, default=3)
    parser.add_argument("--num", type=int, default=6)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument(
        "--out", default="/data/uni0/users/kuno/3dfm_distortion/results/scannetpp_inspect"
    )
    args = parser.parse_args()

    config = ScanNetppConfig(
        root=args.root,
        split=args.split,
        max_scenes=args.max_scenes,
        render_size=args.render_size,
        pairs_per_scene=20,
    )
    dataset = ScanNetppDslrDataset(config)
    print(f"scenes_indexed={len(dataset.scenes)}  pairs={len(dataset)}")
    if len(dataset) == 0:
        print("No pairs built — check --root, split file, and per-scene transforms.json.")
        return

    os.makedirs(args.out, exist_ok=True)
    for i in range(min(args.num, len(dataset))):
        sample = dataset[i]
        radial = None if sample.radial_coeffs is None else sample.radial_coeffs.tolist()
        print(
            f"sample {i}: camera={sample.camera_model} "
            f"render={sample.render_width}x{sample.render_height} radial={radial}"
        )
        _save_grid(sample, os.path.join(args.out, f"sample_{i:03d}.png"))
    print("wrote grids to", args.out)


if __name__ == "__main__":
    main()
