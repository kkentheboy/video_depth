# -*- coding: utf-8 -*-
"""Human parsing / hair segmentation helpers for the image-driven Mesh workflow."""

from .human_parsing import SegmentationResult, check_segmentation_environment, run_human_parsing
from .segmentation_cache import (
    segmentation_cache_root,
    segmentation_frame_paths,
    segmentation_cache_summary,
    load_segmentation_frame,
    save_segmentation_frame,
    ensure_reference_segmentation,
    generate_segmentation_sequence_cache,
)
from .mesh_region_weights import build_mask_guided_region_weights, build_sequence_mask_guided_region_weights
from .mask_quality import classify_mask_quality, quality_to_meta
from .foreground import check_foreground_environment, read_alpha_foreground, constrain_by_foreground

__all__ = [
    "SegmentationResult",
    "check_segmentation_environment",
    "run_human_parsing",
    "segmentation_cache_root",
    "segmentation_frame_paths",
    "segmentation_cache_summary",
    "load_segmentation_frame",
    "save_segmentation_frame",
    "ensure_reference_segmentation",
    "generate_segmentation_sequence_cache",
    "build_mask_guided_region_weights",
    "build_sequence_mask_guided_region_weights",
    "classify_mask_quality",
    "quality_to_meta",
    "check_foreground_environment",
    "read_alpha_foreground",
    "constrain_by_foreground",
]
