"""CPU tests for the ScanNet++ DSLR loader: camera conventions, fisheye math, dataset smoke.

The real data lives on iruka2 (not the karei dev node), so these tests use synthetic fixtures and
verify the convention-critical math (OpenGL->OpenCV viewmat, fisheye project/unproject) directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sharp_train.data import BatchSample
from sharp_train.data.scannetpp import (
    ScanNetppConfig,
    ScanNetppDslrDataset,
    c2w_opengl_to_viewmat,
    fisheye_project,
    fisheye_unproject,
    intrinsics_4x4,
    relative_viewmat,
)


def _c2w(position, dtype=torch.float32) -> torch.Tensor:
    c2w = torch.eye(4, dtype=dtype)
    c2w[:3, 3] = torch.tensor(position, dtype=dtype)
    return c2w


def test_viewmat_maps_camera_center_to_origin() -> None:
    """Viewmat is world->camera: the camera center maps to the origin for any c2w."""
    torch.manual_seed(0)
    c2w = torch.eye(4)
    # A non-trivial rotation + translation.
    angle = 0.7
    c2w[:3, :3] = torch.tensor(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]]
    )
    c2w[:3, 3] = torch.tensor([1.3, -0.4, 2.0])
    viewmat = c2w_opengl_to_viewmat(c2w)
    center = torch.cat([c2w[:3, 3], torch.ones(1)])
    mapped = viewmat @ center
    assert torch.allclose(mapped[:3], torch.zeros(3), atol=1e-5)


def test_viewmat_in_front_is_positive_z() -> None:
    """A world point along the OpenGL look-at (-Z) lands in front of the OpenCV camera (z>0)."""
    c2w = torch.eye(4)  # camera at origin, OpenGL axes
    viewmat = c2w_opengl_to_viewmat(c2w)
    world_pt = torch.tensor([0.0, 0.0, -1.0, 1.0])  # OpenGL: in front
    cam = viewmat @ world_pt
    assert cam[2] > 0  # OpenCV: +z forward


def test_relative_viewmat_identity_when_input_is_target() -> None:
    """Relative pose with input == target is the identity."""
    c2w = _c2w([0.5, 1.0, -2.0])
    rel = relative_viewmat(c2w, c2w)
    assert torch.allclose(rel, torch.eye(4), atol=1e-5)


def test_fisheye_project_unproject_roundtrip() -> None:
    """Unprojecting a fisheye pixel then projecting the ray reproduces the pixel."""
    fx, fy, cx, cy = 616.95, 617.27, 876.0, 584.0
    k = np.array([0.0589, 0.0067, -1.06e-4, -1.71e-4])
    rng = np.random.default_rng(0)
    # Pixels within a generous in-FOV radius around the principal point.
    uv = np.stack(
        [cx + rng.uniform(-700, 700, 200), cy + rng.uniform(-450, 450, 200)], axis=-1
    )
    dirs = fisheye_unproject(uv, fx, fy, cx, cy, k)
    uv2 = fisheye_project(dirs, fx, fy, cx, cy, k)  # ray dir is a valid camera-space point
    assert np.allclose(uv, uv2, atol=1e-3)
    # Principal point maps to a forward ray.
    center_dir = fisheye_unproject(np.array([[cx, cy]]), fx, fy, cx, cy, k)[0]
    assert center_dir[2] > 0.999


def test_intrinsics_builder() -> None:
    """intrinsics_4x4 places fx,fy,cx,cy correctly."""
    k = intrinsics_4x4(100.0, 110.0, 50.0, 40.0)
    assert k[0, 0] == 100.0 and k[1, 1] == 110.0 and k[0, 2] == 50.0 and k[1, 2] == 40.0


def _write_synthetic_scene(root: Path, scene_id: str) -> None:
    """Create a minimal ScanNet++-like scene with two posed fisheye frames."""
    dslr = root / "data" / scene_id / "dslr"
    (dslr / "nerfstudio").mkdir(parents=True)
    for sub in ("resized_images", "resized_undistorted_images"):
        (dslr / sub).mkdir()
    w, h = 64, 48
    names = ["DSC1.JPG", "DSC2.JPG"]
    positions = [[0.0, 0.0, 0.0], [0.05, 0.0, 0.1]]  # baseline ~0.11, within (0.05,0.5)
    frames = []
    for name, pos in zip(names, positions):
        c2w = np.eye(4)
        c2w[:3, 3] = pos
        frames.append(
            {"file_path": f"images/{name}", "transform_matrix": c2w.tolist(), "is_bad": False}
        )
        img = np.random.default_rng(len(name)).integers(0, 255, (h, w, 3)).astype(np.uint8)
        cv2.imwrite(str(dslr / "resized_images" / name), img)
        cv2.imwrite(str(dslr / "resized_undistorted_images" / name), img)
    transforms = {
        "camera_model": "OPENCV_FISHEYE", "w": w, "h": h,
        "fl_x": 30.0, "fl_y": 30.0, "cx": 32.0, "cy": 24.0,
        "k1": 0.05, "k2": 0.0, "k3": 0.0, "k4": 0.0, "frames": frames,
    }
    (dslr / "nerfstudio" / "transforms.json").write_text(json.dumps(transforms))
    (dslr / "train_test_lists.json").write_text(json.dumps({"train": names, "test": []}))
    (root / "splits").mkdir(parents=True, exist_ok=True)
    (root / "splits" / "nvs_sem_train.txt").write_text(scene_id + "\n")


def test_dataset_yields_valid_batchsample(tmp_path) -> None:
    """A synthetic scene produces a shape-correct BatchSample for both branches."""
    _write_synthetic_scene(tmp_path, "scene0")
    config = ScanNetppConfig(
        root=str(tmp_path), split="train", internal_resolution=32, render_size=16,
        distorted_input_fraction=1.0, pairs_per_scene=2, baseline_range=(0.05, 0.5), seed=1,
    )
    dataset = ScanNetppDslrDataset(config)
    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, BatchSample)
    assert sample.image.shape == (1, 3, 32, 32)
    assert sample.camera_model == "fisheye"
    assert sample.radial_coeffs is not None and sample.radial_coeffs.shape == (1, 4)
    assert sample.target_viewmat.shape == (1, 4, 4)
    assert sample.target_intrinsics_pinhole is not None
    # Render targets share the (aspect-preserving) render resolution.
    assert sample.gt_distorted.shape == sample.gt_pinhole.shape
    assert sample.gt_distorted.shape[2] == sample.render_height
    assert sample.mask_distorted.shape[1] == 1
