# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np

from .pointcloud_filter import PointCloudTemporalState
from .visible_pointcloud import build_visible_depth_points


def build_visible_frame_from_job(
    frame_bgr,
    depth: np.ndarray,
    alpha: np.ndarray | None,
    cfg,
    frame_index: int,
    temporal_state: PointCloudTemporalState | None = None,
    normal_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # noqa: ANN001
    return build_visible_depth_points(
        frame_bgr,
        depth,
        alpha,
        frame_index=frame_index,
        stride=int(getattr(cfg, "pointcloud_stride", 2)),
        max_points=int(getattr(cfg, "pointcloud_max_points", 200000)),
        alpha_threshold=float(getattr(cfg, "pointcloud_alpha_threshold", 0.20)),
        near_percentile=float(getattr(cfg, "pointcloud_depth_near_percentile", 1.0)),
        far_percentile=float(getattr(cfg, "pointcloud_depth_far_percentile", 99.0)),
        color_mode=str(getattr(cfg, "pointcloud_color_mode", "rgb")),
        coordinate_mode=str(getattr(cfg, "pointcloud_coordinate_mode", "blender")),
        z_near=float(getattr(cfg, "pointcloud_z_near", 0.25)),
        z_far=float(getattr(cfg, "pointcloud_z_far", 2.0)),
        alpha_erode_px=int(getattr(cfg, "pointcloud_alpha_erode_px", 1)),
        alpha_dilate_px=int(getattr(cfg, "pointcloud_alpha_dilate_px", 0)),
        alpha_feather_px=int(getattr(cfg, "pointcloud_alpha_feather_px", 3)),
        body_bbox_margin_px=int(getattr(cfg, "pointcloud_body_bbox_margin_px", 12)),
        remove_outliers=bool(getattr(cfg, "pointcloud_remove_outliers", True)),
        outlier_sigma=float(getattr(cfg, "pointcloud_outlier_sigma", 2.8)),
        voxel_downsample=bool(getattr(cfg, "pointcloud_voxel_downsample", True)),
        voxel_size=float(getattr(cfg, "pointcloud_voxel_size", 0.006)),
        temporal_state=temporal_state,
        temporal_depth_smooth=float(getattr(cfg, "pointcloud_temporal_depth_smooth", 0.65)),
        fixed_depth_range=getattr(cfg, "pointcloud_fixed_depth_range", None),
        temporal_center_smooth=float(getattr(cfg, "pointcloud_temporal_center_smooth", 0.80)),
        temporal_scale_smooth=float(getattr(cfg, "pointcloud_temporal_scale_smooth", 0.85)),
        normal_map=normal_map,
        normal_relief_enabled=bool(getattr(cfg, "pointcloud_normal_relief_enabled", False)),
        normal_relief_strength=float(getattr(cfg, "pointcloud_normal_relief_strength", 0.0)),
        normal_relief_edge_fade_px=int(getattr(cfg, "pointcloud_normal_relief_edge_fade_px", 10)),
        normal_relief_min_alpha=float(getattr(cfg, "pointcloud_normal_relief_min_alpha", 0.35)),
        normal_relief_gamma=float(getattr(cfg, "pointcloud_normal_relief_gamma", 1.6)),
    )


def write_visible_points_from_job(
    output_path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    source: np.ndarray,
    confidence: np.ndarray,
    cfg,
) -> int:  # noqa: ANN001
    # Per-frame debug output removed from final source; return count only.
    return int(len(points))


def export_visible_frame_from_job(
    frame_bgr,
    depth: np.ndarray,
    alpha: np.ndarray | None,
    output_path: Path,
    cfg,
    frame_index: int,
    temporal_state: PointCloudTemporalState | None = None,
    normal_map: np.ndarray | None = None,
) -> int:  # noqa: ANN001
    points, _colors, _source, _confidence = build_visible_frame_from_job(
        frame_bgr, depth, alpha, cfg, frame_index, temporal_state, normal_map
    )
    return int(len(points))
