# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class PointCloudFrameStats:
    near: float = 0.0
    far: float = 1.0
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0
    kept_points: int = 0


class PointCloudTemporalState:
    """Small per-export state used to reduce point cloud breathing.

    It does not assume true camera calibration. It only smooths the relative depth
    percentile range, object center and object scale between neighbouring frames.
    Call reset() on a detected scene cut.
    """

    def __init__(self) -> None:
        self.prev_near: float | None = None
        self.prev_far: float | None = None
        self.prev_center: np.ndarray | None = None
        self.prev_scale: float | None = None

    def reset(self) -> None:
        self.prev_near = None
        self.prev_far = None
        self.prev_center = None
        self.prev_scale = None

    def smooth_depth_range(self, near: float, far: float, smooth: float) -> tuple[float, float]:
        alpha = float(np.clip(smooth, 0.0, 0.98))
        near = float(near)
        far = float(far)
        if self.prev_near is None or self.prev_far is None or not np.isfinite([near, far]).all():
            self.prev_near, self.prev_far = near, far
            return near, far
        out_near = self.prev_near * alpha + near * (1.0 - alpha)
        out_far = self.prev_far * alpha + far * (1.0 - alpha)
        if out_far - out_near < 1e-6:
            out_near, out_far = near, far
        self.prev_near, self.prev_far = float(out_near), float(out_far)
        return float(out_near), float(out_far)

    def smooth_points(self, points: np.ndarray, center_smooth: float, scale_smooth: float) -> tuple[np.ndarray, PointCloudFrameStats]:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(pts) == 0:
            return pts, PointCloudFrameStats(kept_points=0)
        center_alpha = float(np.clip(center_smooth, 0.0, 0.98))
        scale_alpha = float(np.clip(scale_smooth, 0.0, 0.98))
        current_center = np.median(pts, axis=0).astype(np.float32)
        rel = pts - current_center[None, :]
        dist = np.linalg.norm(rel, axis=1)
        current_scale = float(np.percentile(dist, 90.0)) if len(dist) else 1.0
        current_scale = max(current_scale, 1e-6)

        if self.prev_center is None:
            target_center = current_center
        else:
            target_center = self.prev_center * center_alpha + current_center * (1.0 - center_alpha)
        if self.prev_scale is None:
            target_scale = current_scale
        else:
            target_scale = self.prev_scale * scale_alpha + current_scale * (1.0 - scale_alpha)
        target_scale = max(float(target_scale), 1e-6)
        pts = (rel * (target_scale / current_scale)) + target_center[None, :]
        self.prev_center = target_center.astype(np.float32)
        self.prev_scale = float(target_scale)
        return pts.astype(np.float32), PointCloudFrameStats(
            center=(float(target_center[0]), float(target_center[1]), float(target_center[2])),
            scale=float(target_scale),
            kept_points=int(len(pts)),
        )


def _clamp_percentile(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def percentile_depth_range(depth: np.ndarray, mask: np.ndarray | None, near_pct: float, far_pct: float) -> tuple[float, float]:
    arr = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(arr)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    if not finite.any():
        return 0.0, 1.0
    vals = arr[finite]
    p0 = float(np.percentile(vals, _clamp_percentile(near_pct, 0.0, 99.0)))
    p1 = float(np.percentile(vals, _clamp_percentile(far_pct, 1.0, 100.0)))
    lo, hi = (p0, p1) if p0 <= p1 else (p1, p0)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return float(lo), float(hi)


def depth_to_z_from_range(depth: np.ndarray, depth_range: tuple[float, float], z_near: float, z_far: float) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    lo, hi = float(depth_range[0]), float(depth_range[1])
    if hi - lo < 1e-6:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    zn = max(1e-4, float(z_near))
    zf = max(zn + 1e-4, float(z_far))
    return (zn + norm * (zf - zn)).astype(np.float32)


def robust_depth_to_z(
    depth: np.ndarray,
    mask: np.ndarray | None,
    near_pct: float,
    far_pct: float,
    z_near: float,
    z_far: float,
    temporal_state: PointCloudTemporalState | None = None,
    temporal_depth_smooth: float = 0.0,
    fixed_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Map depth to Blender Z.

    Legacy/model-depth mode may estimate a percentile range per frame and then
    smooth that range. Direct depth-video mode must pass ``fixed_range`` from
    the render pre-pass so the whole export uses one consistent depth scale.
    That preserves real in/out motion instead of remapping every frame into the
    same relief volume.
    """
    if fixed_range is not None:
        near, far = float(fixed_range[0]), float(fixed_range[1])
    else:
        near, far = percentile_depth_range(depth, mask, near_pct, far_pct)
        if temporal_state is not None and temporal_depth_smooth > 0:
            near, far = temporal_state.smooth_depth_range(near, far, temporal_depth_smooth)
    return depth_to_z_from_range(depth, (near, far), z_near, z_far)


def normalize_alpha(alpha: np.ndarray | None, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if alpha is None:
        return np.ones((h, w), dtype=np.float32)
    arr = np.asarray(alpha, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape[:2] != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    if float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)


def prepare_alpha_for_pointcloud(
    alpha: np.ndarray | None,
    shape_hw: tuple[int, int],
    threshold: float,
    erode_px: int = 1,
    dilate_px: int = 0,
    feather_px: int = 3,
    bbox_margin: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return processed alpha float map and binary sampling mask.

    Erosion removes the unreliable halo around alpha edges. Optional dilation can
    be used when the matte is too tight. Feather keeps confidence soft for PLY
    confidence values while the binary mask remains strict.
    """
    arr = normalize_alpha(alpha, shape_hw)
    mask_u8 = (arr >= float(threshold)).astype(np.uint8) * 255
    if erode_px > 0:
        k = int(erode_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)
    if dilate_px > 0:
        k = int(dilate_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    if bbox_margin > 0 and mask_u8.any():
        ys, xs = np.where(mask_u8 > 0)
        h, w = mask_u8.shape[:2]
        m = int(max(0, bbox_margin))
        x0, x1 = max(0, int(xs.min()) - m), min(w, int(xs.max()) + m + 1)
        y0, y1 = max(0, int(ys.min()) - m), min(h, int(ys.max()) + m + 1)
        bbox_mask = np.zeros_like(mask_u8)
        bbox_mask[y0:y1, x0:x1] = 255
        mask_u8 = cv2.bitwise_and(mask_u8, bbox_mask)
    if feather_px > 0 and mask_u8.any():
        k = int(feather_px) * 2 + 1
        soft = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0, (k, k), 0)
        processed_alpha = np.clip(np.maximum(arr * 0.35, soft), 0.0, 1.0).astype(np.float32)
    else:
        processed_alpha = (mask_u8.astype(np.float32) / 255.0).astype(np.float32)
    sample_mask = mask_u8 > 0
    return processed_alpha, sample_mask


def apply_grid_stride(mask: np.ndarray, stride: int) -> np.ndarray:
    step = max(1, int(stride))
    if step <= 1:
        return np.asarray(mask, dtype=bool)
    grid = np.zeros_like(mask, dtype=bool)
    grid[::step, ::step] = True
    return np.asarray(mask, dtype=bool) & grid


def remove_statistical_outliers(points: np.ndarray, sigma: float = 2.8) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) < 16:
        return np.ones(len(pts), dtype=bool)
    center = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - center[None, :], axis=1)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    if mad < 1e-8:
        std = float(np.std(dist))
        if std < 1e-8:
            return np.ones(len(pts), dtype=bool)
        limit = med + float(sigma) * std
    else:
        robust_std = 1.4826 * mad
        limit = med + float(sigma) * robust_std
    return np.asarray(dist <= limit, dtype=bool)


def voxel_downsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    source_id: np.ndarray,
    confidence: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = float(voxel_size)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    src = np.asarray(source_id, dtype=np.uint8).reshape(-1)
    conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
    n = min(len(pts), len(cols), len(src), len(conf))
    pts, cols, src, conf = pts[:n], cols[:n], src[:n], conf[:n]
    if n == 0 or size <= 1e-8:
        return pts, cols, src, conf
    keys = np.floor(pts / size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    out_n = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=out_n).astype(np.float32)
    pts_sum = np.vstack([np.bincount(inverse, weights=pts[:, axis], minlength=out_n) for axis in range(3)]).T
    cols_sum = np.vstack([np.bincount(inverse, weights=cols[:, axis].astype(np.float32), minlength=out_n) for axis in range(3)]).T
    conf_sum = np.bincount(inverse, weights=conf, minlength=out_n)
    pts_out = (pts_sum / counts[:, None]).astype(np.float32)
    cols_out = np.clip(cols_sum / counts[:, None] + 0.5, 0, 255).astype(np.uint8)
    conf_out = (conf_sum / counts).astype(np.float32)
    # Preserve semantic source after merge. For v7 normal relief this matters:
    # if any point in a voxel was actually displaced, keep source_id=3 instead
    # of losing that information to whichever point happened to come first.
    src_out = np.zeros(out_n, dtype=np.uint8)
    for source_value in np.unique(src):
        hit = np.bincount(inverse, weights=(src == source_value).astype(np.float32), minlength=out_n) > 0
        src_out[hit] = np.maximum(src_out[hit], np.uint8(source_value))
    return pts_out, cols_out, src_out.astype(np.uint8), conf_out


def deterministic_limit_indices(count: int, max_points: int, frame_index: int = 0) -> np.ndarray | None:
    count = int(count)
    max_points = int(max_points)
    if max_points <= 0 or count <= max_points:
        return None
    rng = np.random.default_rng(seed=1_000_003 + int(frame_index))
    return np.sort(rng.choice(count, size=max_points, replace=False))
