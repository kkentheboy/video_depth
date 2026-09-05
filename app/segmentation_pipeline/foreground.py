# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _norm01(mask: Optional[np.ndarray], shape_hw: tuple[int, int] | None = None) -> Optional[np.ndarray]:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.size == 0:
        return None
    if float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    if shape_hw is not None and arr.shape[:2] != tuple(shape_hw):
        arr = cv2.resize(arr, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return arr


def read_alpha_foreground(video_path: str, frame_index: int, shape_hw: tuple[int, int]) -> Optional[np.ndarray]:
    """Read real alpha from the main video when available.

    This is the reliable foreground constraint. If the video has no real alpha,
    return None instead of guessing from RGB.
    """
    try:
        from depth_fusion_core import read_video_frame_alpha01
        alpha = read_video_frame_alpha01(str(video_path), int(frame_index), shape_hw)
        return _norm01(alpha, shape_hw)
    except Exception:
        return None


def constrain_by_foreground(mask: Optional[np.ndarray], foreground: Optional[np.ndarray], *, softness: float = 0.92) -> Optional[np.ndarray]:
    m = _norm01(mask, foreground.shape[:2] if foreground is not None else None)
    fg = _norm01(foreground, m.shape[:2] if m is not None else None)
    if m is None:
        return fg
    if fg is None:
        return m
    # Soft constraint: never expands the parser mask into background, but keeps a
    # little parser confidence around matte edges.
    return np.clip(m * (fg * float(softness) + (1.0 - float(softness)) * 0.15), 0.0, 1.0).astype(np.float32)


def check_foreground_environment(project_root: str | Path) -> dict:
    """Report foreground helpers. Current final flow uses real video Alpha when available."""
    root = Path(project_root) / "models" / "segmentation" / "rvm"
    candidates = [root / "rvm_mobilenetv3.onnx", root / "rvm_resnet50.onnx", root / "rvm_mobilenetv3.pth"]
    found = [p for p in candidates if p.exists()]
    return {
        "alpha_supported": True,
        "rvm_model_found": bool(found),
        "rvm_model_paths": [str(p) for p in found],
        "message": "Alpha 前景约束可用；未接额外视频抠像模型；非 Alpha 视频使用 parsing foreground 兜底",
    }
