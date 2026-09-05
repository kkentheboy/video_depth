# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from common.media import resize_bgr_like_depth
from .camera import backproject_depth_to_xyz, estimate_intrinsics
from .pointcloud_filter import (
    PointCloudTemporalState,
    apply_grid_stride,
    deterministic_limit_indices,
    prepare_alpha_for_pointcloud,
    remove_statistical_outliers,
    robust_depth_to_z,
    voxel_downsample_points,
)


def _make_colors(rgb_bgr: np.ndarray, depth_z: np.ndarray, color_mode: str, sample_mask: np.ndarray | None = None) -> np.ndarray:
    mode = str(color_mode or "rgb").lower()
    h, w = depth_z.shape[:2]
    if mode == "fixed":
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[:, :] = (220, 220, 220)
        return out
    if mode == "depth_gray":
        z = np.asarray(depth_z, dtype=np.float32)
        if sample_mask is not None and np.asarray(sample_mask).any():
            vals = z[np.asarray(sample_mask, dtype=bool)]
        else:
            vals = z[np.isfinite(z)]
        if len(vals) == 0:
            gray = np.zeros((h, w), dtype=np.uint8)
        else:
            zmin, zmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            if zmax - zmin < 1e-6:
                gray = np.zeros((h, w), dtype=np.uint8)
            else:
                gray = (np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return np.dstack([gray, gray, gray])
    if mode == "source_debug":
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[:, :] = (80, 190, 255)
        return out
    bgr = resize_bgr_like_depth(rgb_bgr, (h, w))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_visible_depth_points(
    frame_bgr: np.ndarray,
    depth: np.ndarray,
    alpha: np.ndarray | None,
    *,
    frame_index: int = 0,
    stride: int = 2,
    max_points: int = 200_000,
    alpha_threshold: float = 0.20,
    near_percentile: float = 1.0,
    far_percentile: float = 99.0,
    color_mode: str = "rgb",
    coordinate_mode: str = "blender",
    z_near: float = 0.25,
    z_far: float = 2.0,
    alpha_erode_px: int = 1,
    alpha_dilate_px: int = 0,
    alpha_feather_px: int = 3,
    body_bbox_margin_px: int = 12,
    remove_outliers: bool = True,
    outlier_sigma: float = 2.8,
    voxel_downsample: bool = True,
    voxel_size: float = 0.006,
    temporal_state: PointCloudTemporalState | None = None,
    temporal_depth_smooth: float = 0.65,
    fixed_depth_range: tuple[float, float] | None = None,
    temporal_center_smooth: float = 0.80,
    temporal_scale_smooth: float = 0.85,
    normal_map: np.ndarray | None = None,
    normal_relief_enabled: bool = False,
    normal_relief_strength: float = 0.0,
    normal_relief_edge_fade_px: int = 10,
    normal_relief_min_alpha: float = 0.35,
    normal_relief_gamma: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build filtered visible-depth point data without writing a file.

    This is the shared V3 data path. Visible/fused USDA exporters consume the same point arrays so there is only one sampling
    result per frame.
    """
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim == 3:
        depth_arr = depth_arr[..., 0]
    h, w = depth_arr.shape[:2]

    alpha_arr, matte_mask = prepare_alpha_for_pointcloud(
        alpha,
        (h, w),
        threshold=float(alpha_threshold),
        erode_px=int(alpha_erode_px),
        dilate_px=int(alpha_dilate_px),
        feather_px=int(alpha_feather_px),
        bbox_margin=int(body_bbox_margin_px),
    )
    sample_mask = matte_mask & np.isfinite(depth_arr)
    sample_mask = apply_grid_stride(sample_mask, stride)
    if not sample_mask.any():
        empty_pts = np.zeros((0, 3), dtype=np.float32)
        empty_cols = np.zeros((0, 3), dtype=np.uint8)
        empty_src = np.zeros((0,), dtype=np.uint8)
        empty_conf = np.zeros((0,), dtype=np.float32)
        return empty_pts, empty_cols, empty_src, empty_conf

    z = robust_depth_to_z(
        depth_arr,
        sample_mask,
        near_percentile,
        far_percentile,
        z_near,
        z_far,
        temporal_state=temporal_state,
        temporal_depth_smooth=temporal_depth_smooth,
        fixed_range=fixed_depth_range,
    )
    xyz = backproject_depth_to_xyz(z, estimate_intrinsics(w, h), coordinate_mode=coordinate_mode)
    colors_img = _make_colors(frame_bgr, z, color_mode, sample_mask)
    pts = xyz[sample_mask]
    cols = colors_img[sample_mask]
    conf = np.clip(alpha_arr[sample_mask], 0.0, 1.0).astype(np.float32)
    source = np.zeros(len(pts), dtype=np.uint8)  # 0 = visible depth
    # Final workflow cleanup: Normal-relief auxiliary point generation is not part
    # of the required Blender USDA chain. Keep function arguments for backward
    # compatibility, but do not create source_id=3 points.

    if temporal_state is not None and len(pts) > 0 and (temporal_center_smooth > 0 or temporal_scale_smooth > 0):
        pts, _stats = temporal_state.smooth_points(pts, float(temporal_center_smooth), float(temporal_scale_smooth))

    if voxel_downsample and len(pts) > 0:
        pts, cols, source, conf = voxel_downsample_points(pts, cols, source, conf, float(voxel_size))

    limiter = deterministic_limit_indices(len(pts), int(max_points), frame_index)
    if limiter is not None:
        pts = pts[limiter]
        cols = cols[limiter]
        source = source[limiter]
        conf = conf[limiter]
    return pts.astype(np.float32), cols.astype(np.uint8), source.astype(np.uint8), conf.astype(np.float32)


def export_visible_depth_points_count(
    frame_bgr: np.ndarray,
    depth: np.ndarray,
    alpha: np.ndarray | None,
    output_path: Path,
    *,
    frame_index: int = 0,
    stride: int = 2,
    max_points: int = 200_000,
    alpha_threshold: float = 0.20,
    near_percentile: float = 1.0,
    far_percentile: float = 99.0,
    color_mode: str = "rgb",
    coordinate_mode: str = "blender",
    binary_ply: bool = True,
    z_near: float = 0.25,
    z_far: float = 2.0,
    alpha_erode_px: int = 1,
    alpha_dilate_px: int = 0,
    alpha_feather_px: int = 3,
    body_bbox_margin_px: int = 12,
    remove_outliers: bool = True,
    outlier_sigma: float = 2.8,
    voxel_downsample: bool = True,
    voxel_size: float = 0.006,
    temporal_state: PointCloudTemporalState | None = None,
    temporal_depth_smooth: float = 0.65,
    fixed_depth_range: tuple[float, float] | None = None,
    temporal_center_smooth: float = 0.80,
    temporal_scale_smooth: float = 0.85,
    normal_map: np.ndarray | None = None,
    normal_relief_enabled: bool = False,
    normal_relief_strength: float = 0.0,
    normal_relief_edge_fade_px: int = 10,
    normal_relief_min_alpha: float = 0.35,
    normal_relief_gamma: float = 1.6,
) -> int:
    pts, cols, source, conf = build_visible_depth_points(
        frame_bgr,
        depth,
        alpha,
        frame_index=frame_index,
        stride=stride,
        max_points=max_points,
        alpha_threshold=alpha_threshold,
        near_percentile=near_percentile,
        far_percentile=far_percentile,
        color_mode=color_mode,
        coordinate_mode=coordinate_mode,
        z_near=z_near,
        z_far=z_far,
        alpha_erode_px=alpha_erode_px,
        alpha_dilate_px=alpha_dilate_px,
        alpha_feather_px=alpha_feather_px,
        body_bbox_margin_px=body_bbox_margin_px,
        remove_outliers=remove_outliers,
        outlier_sigma=outlier_sigma,
        voxel_downsample=voxel_downsample,
        voxel_size=voxel_size,
        temporal_state=temporal_state,
        temporal_depth_smooth=temporal_depth_smooth,
        fixed_depth_range=fixed_depth_range,
        temporal_center_smooth=temporal_center_smooth,
        temporal_scale_smooth=temporal_scale_smooth,
        normal_map=normal_map,
        normal_relief_enabled=normal_relief_enabled,
        normal_relief_strength=normal_relief_strength,
        normal_relief_edge_fade_px=normal_relief_edge_fade_px,
        normal_relief_min_alpha=normal_relief_min_alpha,
        normal_relief_gamma=normal_relief_gamma,
    )
    # Per-frame debug output removed from final source; USDA is the only final point-cloud output.
    return int(len(pts))
