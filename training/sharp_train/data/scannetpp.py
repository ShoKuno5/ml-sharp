"""ScanNet++ DSLR loader for dual-branch (P/D) NVS training.

ScanNet++ DSLR provides *real* OpenCV-fisheye pairs, so no synthetic warp is needed:
``resized_images/`` (distorted fisheye) and ``resized_undistorted_images/`` (rectified pinhole)
share filenames and registered poses. Branch D supervises against the fisheye image
(``camera_model="fisheye"``, ``radial_coeffs=[k1..k4]``); branch P against the undistorted image
(pinhole K). Poses come from ``dslr/nerfstudio/transforms.json`` (nerfstudio OpenGL camera-to-world
-> gsplat OpenCV world-to-camera). Scene-disjoint split: root ``splits/nvs_sem_{train,val}.txt``;
within-scene NVS roles: per-scene ``dslr/train_test_lists.json``.

Verified conventions (gsplat 1.5.3 + nerfstudio + ScanNet++ toolkit):
- transform_matrix is c2w in OpenGL (+x right, +y up, -z forward); gsplat wants world->cam in
  OpenCV (+x right, +y down, +z forward): ``viewmat = inv(c2w @ diag(1,-1,-1,1))``.
- OPENCV_FISHEYE k1..k4 map 1:1 to gsplat fisheye ``radial_coeffs`` (equidistant + odd poly).
- Undistorted images use a *different* pinhole K (cv2.fisheye.estimateNewCameraMatrixForUndistort
  Rectify, balance=0, principal point recentred), read from ``transforms_undistorted.json`` or
  recomputed here with cv2.
- Drop ``is_bad`` frames; masks (255=valid) -> valid_mask; depth is optional (start gt_depth=None).

Data lives on iruka2 (``/data/umiushi0/datasets/VGGT/ScanNet++``), not the karei dev node, so this
module is written against the documented layout and unit-tested with synthetic fixtures; real-data
loading is validated on iruka2.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from .batch import BatchSample

# ---------------------------------------------------------------------------------------------
# Camera conventions
# ---------------------------------------------------------------------------------------------

# OpenGL(camera) -> OpenCV(camera) basis flip; right-multiplied onto c2w (acts on camera axes).
_GL2CV = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def c2w_opengl_to_viewmat(c2w: torch.Tensor) -> torch.Tensor:
    """Convert a nerfstudio OpenGL camera-to-world to a gsplat OpenCV world-to-camera viewmat.

    Args:
        c2w: ``[..., 4, 4]`` camera-to-world (OpenGL: +x right, +y up, -z forward).

    Returns:
        ``[..., 4, 4]`` world-to-camera (OpenCV: +x right, +y down, +z forward).
    """
    c2w_cv = c2w @ _GL2CV.to(c2w)
    return torch.linalg.inv(c2w_cv)


def relative_viewmat(c2w_input: torch.Tensor, c2w_target: torch.Tensor) -> torch.Tensor:
    """World-to-camera viewmat of the target with the world frame set to the input camera.

    With world == input camera, the OpenGL->OpenCV flips cancel into the relative pose.
    """
    viewmat_input = c2w_opengl_to_viewmat(c2w_input)
    viewmat_target = c2w_opengl_to_viewmat(c2w_target)
    return viewmat_target @ torch.linalg.inv(viewmat_input)


def intrinsics_4x4(
    fx: float, fy: float, cx: float, cy: float, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build a 4x4 intrinsics matrix (OpenCV pixel units) with K in the top-left 3x3."""
    k = torch.eye(4, dtype=dtype)
    k[0, 0] = fx
    k[1, 1] = fy
    k[0, 2] = cx
    k[1, 2] = cy
    return k


def scale_intrinsics(k: torch.Tensor, scale_x: float, scale_y: float) -> torch.Tensor:
    """Scale a 4x4 intrinsics for a resized image (fx,cx by scale_x; fy,cy by scale_y)."""
    out = k.clone()
    out[0, 0] *= scale_x
    out[0, 2] *= scale_x
    out[1, 1] *= scale_y
    out[1, 2] *= scale_y
    return out


# ---------------------------------------------------------------------------------------------
# OpenCV-fisheye reference (equidistant + odd radial polynomial), matching gsplat's fisheye.
# Used for GT-depth ray generation (follow-on) and unit tests; the render itself uses gsplat.
# ---------------------------------------------------------------------------------------------


def fisheye_delta(theta: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Equidistant odd polynomial: delta = theta*(1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8)."""
    t2 = theta * theta
    poly = 1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))
    return theta * poly


def fisheye_project(
    points_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float, k: np.ndarray
) -> np.ndarray:
    """Project OpenCV-fisheye camera-space points ``[..., 3]`` to pixels ``[..., 2]``."""
    x, y, z = points_cam[..., 0], points_cam[..., 1], points_cam[..., 2]
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    delta = fisheye_delta(theta, k)
    scale = np.where(r > 1e-12, delta / np.where(r > 1e-12, r, 1.0), 0.0)
    u = fx * scale * x + cx
    v = fy * scale * y + cy
    return np.stack([u, v], axis=-1)


def fisheye_unproject(
    uv: np.ndarray, fx: float, fy: float, cx: float, cy: float, k: np.ndarray, iters: int = 12
) -> np.ndarray:
    """Unproject pixels ``[..., 2]`` to unit camera-ray directions ``[..., 3]`` (OpenCV frame).

    Inverts the equidistant odd polynomial with Newton iterations (matching gsplat's solver).
    """
    nx = (uv[..., 0] - cx) / fx
    ny = (uv[..., 1] - cy) / fy
    delta_obs = np.hypot(nx, ny)
    theta = delta_obs.copy()  # init theta0 = delta_obs
    for _ in range(iters):
        t2 = theta * theta
        f = theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))) - delta_obs
        fp = 1.0 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])))
        theta = theta - f / np.clip(fp, 1e-8, None)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    safe = np.where(delta_obs > 1e-12, delta_obs, 1.0)
    dx = np.where(delta_obs > 1e-12, sin_t * nx / safe, 0.0)
    dy = np.where(delta_obs > 1e-12, sin_t * ny / safe, 0.0)
    dz = np.where(delta_obs > 1e-12, cos_t, 1.0)
    return np.stack([dx, dy, dz], axis=-1)


def compute_undistort_pinhole_k(
    fx: float, fy: float, cx: float, cy: float, k: np.ndarray, width: int, height: int
) -> tuple[float, float, float, float]:
    """Recompute the ScanNet++ undistortion pinhole K (cv2 fisheye, balance=0, recentred).

    Mirrors scannetpp/dslr/undistort.py. Used only when ``transforms_undistorted.json`` is absent.
    """
    import cv2

    k_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(k, dtype=np.float64).reshape(4, 1)
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        k_mat, dist, (width, height), np.eye(3), balance=0.0
    )
    return float(new_k[0, 0]), float(new_k[1, 1]), width / 2.0, height / 2.0


# ---------------------------------------------------------------------------------------------
# Scene parsing
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass
class FrameMeta:
    """Per-frame metadata from transforms.json."""

    name: str  # image filename (e.g. DSC00001.JPG)
    c2w: np.ndarray  # [4,4] OpenGL camera-to-world
    is_bad: bool
    mask_name: str | None


@dataclasses.dataclass
class SceneCameras:
    """Per-scene fisheye + undistorted-pinhole intrinsics (native resolution)."""

    width: int
    height: int
    fisheye_fxfycxcy: tuple[float, float, float, float]
    radial_k: tuple[float, float, float, float]
    pinhole_fxfycxcy: tuple[float, float, float, float]


def _frames_from_transforms(transforms: dict) -> list[FrameMeta]:
    frames: list[FrameMeta] = []
    for entry in transforms.get("frames", []):
        name = Path(entry["file_path"]).name
        frames.append(
            FrameMeta(
                name=name,
                c2w=np.asarray(entry["transform_matrix"], dtype=np.float64),
                is_bad=bool(entry.get("is_bad", False)),
                mask_name=(Path(entry["mask_path"]).name if entry.get("mask_path") else None),
            )
        )
    return frames


def load_dslr_scene(
    scene_dir: Path,
) -> tuple[SceneCameras, dict[str, FrameMeta], list[str], list[str]]:
    """Parse a scene's ``dslr/`` directory.

    Returns:
        ``(cameras, frames_by_name, train_names, test_names)`` where ``frames_by_name`` maps the
        image filename to its :class:`FrameMeta`.
    """
    dslr = scene_dir / "dslr"
    transforms = json.loads((dslr / "nerfstudio" / "transforms.json").read_text())
    if transforms.get("camera_model") != "OPENCV_FISHEYE":
        got = transforms.get("camera_model")
        raise ValueError(f"{scene_dir.name}: expected OPENCV_FISHEYE camera_model, got {got}")

    width, height = int(transforms["w"]), int(transforms["h"])
    fisheye = (transforms["fl_x"], transforms["fl_y"], transforms["cx"], transforms["cy"])
    radial = (
        transforms.get("k1", 0.0),
        transforms.get("k2", 0.0),
        transforms.get("k3", 0.0),
        transforms.get("k4", 0.0),
    )

    # Pinhole K for the undistorted branch: prefer the toolkit's transforms_undistorted.json.
    undistorted_path = dslr / "nerfstudio" / "transforms_undistorted.json"
    if undistorted_path.exists():
        und = json.loads(undistorted_path.read_text())
        pinhole = (und["fl_x"], und["fl_y"], und["cx"], und["cy"])
    else:
        pinhole = compute_undistort_pinhole_k(*fisheye, np.asarray(radial), width, height)

    cameras = SceneCameras(
        width=width,
        height=height,
        fisheye_fxfycxcy=tuple(float(v) for v in fisheye),
        radial_k=tuple(float(v) for v in radial),
        pinhole_fxfycxcy=tuple(float(v) for v in pinhole),
    )

    frames = {f.name: f for f in _frames_from_transforms(transforms)}
    # test_frames (if present) are off-trajectory NVS targets; record them too.
    for f in _frames_from_transforms({"frames": transforms.get("test_frames", [])}):
        frames.setdefault(f.name, f)

    lists_path = dslr / "train_test_lists.json"
    if lists_path.exists():
        lists = json.loads(lists_path.read_text())
        train_names = [Path(n).name for n in lists.get("train", [])]
        test_names = [Path(n).name for n in lists.get("test", [])]
    else:
        train_names = list(frames.keys())
        test_names = []
    return cameras, frames, train_names, test_names


def read_scene_ids(root: Path, split: str) -> list[str]:
    """Read the scene-disjoint split list (``splits/nvs_sem_{train,val}.txt``)."""
    fname = {"train": "nvs_sem_train.txt", "val": "nvs_sem_val.txt"}.get(split, f"nvs_{split}.txt")
    path = root / "splits" / fname
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------------


def _camera_center(c2w: np.ndarray) -> np.ndarray:
    return c2w[:3, 3]


def _forward_axis(c2w: np.ndarray) -> np.ndarray:
    # OpenGL camera looks down -Z; world forward = -(third column).
    return -c2w[:3, 2]


def sample_pairs(
    frames: dict[str, FrameMeta],
    candidate_names: list[str],
    pairs_per_scene: int,
    baseline_range: tuple[float, float],
    max_angle_deg: float,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Sample (input, target) name pairs with a baseline + viewing-angle co-visibility filter."""
    cand = [n for n in candidate_names if n in frames and not frames[n].is_bad]
    pairs: list[tuple[str, str]] = []
    if len(cand) < 2:
        return pairs
    centers = {n: _camera_center(frames[n].c2w) for n in cand}
    fwd = {n: _forward_axis(frames[n].c2w) for n in cand}
    cos_thresh = math.cos(math.radians(max_angle_deg))
    lo, hi = baseline_range
    attempts = 0
    max_attempts = pairs_per_scene * 50
    while len(pairs) < pairs_per_scene and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(cand, 2)
        baseline = float(np.linalg.norm(centers[a] - centers[b]))
        if not (lo <= baseline <= hi):
            continue
        denom = np.linalg.norm(fwd[a]) * np.linalg.norm(fwd[b])
        cos_angle = float(np.dot(fwd[a], fwd[b]) / denom)
        if cos_angle < cos_thresh:
            continue
        pairs.append((a, b))
    return pairs


@dataclasses.dataclass
class ScanNetppConfig:
    """Configuration for :class:`ScanNetppDslrDataset`."""

    root: str
    split: str = "train"
    internal_resolution: int = 1536
    render_size: int = 512  # render height; width preserves aspect
    distorted_input_fraction: float = 0.7  # 1 - fraction is xi=0 (pinhole-input, anti-forgetting)
    pairs_per_scene: int = 50
    baseline_range: tuple[float, float] = (0.05, 0.5)
    max_angle_deg: float = 60.0
    max_scenes: int | None = None
    seed: int = 0


class ScanNetppDslrDataset(torch.utils.data.Dataset):
    """Dual-branch NVS pairs from ScanNet++ DSLR (real fisheye + undistorted)."""

    def __init__(self, config: ScanNetppConfig) -> None:
        """Index scenes from the split and pre-sample (input, target) pairs."""
        self.config = config
        self.root = Path(config.root)
        rng = random.Random(config.seed)
        scene_ids = read_scene_ids(self.root, config.split)
        if config.max_scenes is not None:
            scene_ids = scene_ids[: config.max_scenes]

        self.scenes: dict[str, tuple[SceneCameras, dict[str, FrameMeta]]] = {}
        # Each entry: (scene_id, input_name, target_name, distorted_input).
        self.index: list[tuple[str, str, str, bool]] = []
        for scene_id in scene_ids:
            scene_dir = self.root / "data" / scene_id
            if not (scene_dir / "dslr" / "nerfstudio" / "transforms.json").exists():
                continue
            cameras, frames, train_names, _ = load_dslr_scene(scene_dir)
            pairs = sample_pairs(
                frames, train_names, config.pairs_per_scene, config.baseline_range,
                config.max_angle_deg, rng,
            )
            if not pairs:
                continue
            self.scenes[scene_id] = (cameras, frames)
            for input_name, target_name in pairs:
                distorted = rng.random() < config.distorted_input_fraction
                self.index.append((scene_id, input_name, target_name, distorted))

    def __len__(self) -> int:
        return len(self.index)

    def _load_image(self, path: Path, width: int, height: int) -> torch.Tensor:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

    def _load_mask(self, path: Path | None, width: int, height: int) -> torch.Tensor:
        import cv2

        if path is None or not path.exists():
            return torch.ones(1, height, width)
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return (torch.from_numpy(mask).float() >= 128).float()[None]

    def __getitem__(self, idx: int) -> BatchSample:
        scene_id, input_name, target_name, distorted_input = self.index[idx]
        cameras, frames = self.scenes[scene_id]
        cfg = self.config
        w, h = cameras.width, cameras.height
        dslr = self.root / "data" / scene_id / "dslr"

        # Render resolution preserves aspect (uniform scale by render_size / height).
        render_scale = cfg.render_size / h
        render_h, render_w = cfg.render_size, int(round(w * render_scale))

        fx, fy, cx, cy = cameras.fisheye_fxfycxcy
        px, py, pcx, pcy = cameras.pinhole_fxfycxcy
        k = cameras.radial_k

        # Input image at the internal (square-stretched) resolution.
        res = cfg.internal_resolution
        input_dir = "resized_images" if distorted_input else "resized_undistorted_images"
        image = self._load_image(dslr / input_dir / input_name, res, res)
        if distorted_input:
            input_k = scale_intrinsics(intrinsics_4x4(fx, fy, cx, cy), res / w, res / h)
            disparity_factor = torch.tensor([fx / w])
            camera_model = "fisheye"
            radial = torch.tensor([list(k)], dtype=torch.float32)
        else:
            input_k = scale_intrinsics(intrinsics_4x4(px, py, pcx, pcy), res / w, res / h)
            disparity_factor = torch.tensor([px / w])
            camera_model = "pinhole"
            radial = None

        # Target GT images (always provide both branches' targets).
        gt_distorted = self._load_image(dslr / "resized_images" / target_name, render_w, render_h)
        gt_pinhole = self._load_image(
            dslr / "resized_undistorted_images" / target_name, render_w, render_h
        )
        tmeta = frames[target_name]
        stem = Path(target_name).stem
        mask_distorted = self._load_mask(
            dslr / "resized_anon_masks" / f"{stem}.png", render_w, render_h
        )
        mask_pinhole = self._load_mask(
            dslr / "resized_undistorted_masks" / f"{stem}.png", render_w, render_h
        )

        c2w_in = torch.from_numpy(frames[input_name].c2w).float()
        c2w_tg = torch.from_numpy(tmeta.c2w).float()
        target_viewmat = relative_viewmat(c2w_in, c2w_tg)[None]  # [1,4,4]

        target_k_d = scale_intrinsics(intrinsics_4x4(fx, fy, cx, cy), render_w / w, render_h / h)
        target_k_p = scale_intrinsics(intrinsics_4x4(px, py, pcx, pcy), render_w / w, render_h / h)

        return BatchSample(
            image=image[None],
            disparity_factor=disparity_factor,
            input_intrinsics=input_k[None],
            target_viewmat=target_viewmat,
            target_intrinsics=(target_k_d if distorted_input else target_k_p)[None],
            target_intrinsics_pinhole=target_k_p[None],
            render_width=render_w,
            render_height=render_h,
            gt_pinhole=gt_pinhole[None],
            gt_distorted=(gt_distorted if distorted_input else gt_pinhole)[None],
            mask_pinhole=mask_pinhole[None],
            mask_distorted=(mask_distorted if distorted_input else mask_pinhole)[None],
            camera_model=camera_model,
            radial_coeffs=radial,
            gt_depth=None,
        )
