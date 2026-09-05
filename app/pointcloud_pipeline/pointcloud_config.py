# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PointCloudExportConfig:
    enabled: bool = True
    stride: int = 2
    max_points: int = 200_000
    alpha_threshold: float = 0.20
    depth_near_percentile: float = 1.0
    depth_far_percentile: float = 99.0
    color_mode: str = "xyz"          # xyz | source_debug | rgb legacy
    coordinate_mode: str = "blender" # blender | opencv
    binary_ply: bool = True
    z_near: float = 0.25
    z_far: float = 2.00
    alpha_erode_px: int = 1
    alpha_dilate_px: int = 0
    alpha_feather_px: int = 3
    body_bbox_margin_px: int = 12
    remove_outliers: bool = True
    outlier_sigma: float = 2.8
    voxel_downsample: bool = True
    voxel_size: float = 0.006
    temporal_depth_smooth: float = 0.65
    temporal_center_smooth: float = 0.80
    temporal_scale_smooth: float = 0.85
    normal_relief_enabled: bool = False
    normal_relief_strength: float = 0.0
    normal_relief_edge_fade_px: int = 10
    normal_relief_min_alpha: float = 0.35
    normal_relief_gamma: float = 1.6
    template_sample_ratio: float = 0.45
    template_confidence: float = 0.55
    template_align_strength: float = 1.0
    resume_enabled: bool = True


def pointcloud_density_to_max_points(density: str, custom_max: int) -> int:
    text = str(density or "custom").strip().lower()
    if text in {"低", "low"}:
        return 50_000
    if text in {"中", "medium"}:
        return 120_000
    if text in {"高", "high"}:
        return 200_000
    return max(1_000, int(custom_max))
