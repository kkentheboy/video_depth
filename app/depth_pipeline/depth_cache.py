# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_depth_debug_cache(depth: np.ndarray, output_npy: Path, output_png16: Path | None = None) -> None:
    """Write reusable float depth cache plus optional 16bit debug depth image."""
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_npy), np.asarray(depth, dtype=np.float32))
    if output_png16 is None:
        return
    arr = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        png = np.zeros(arr.shape[:2], dtype=np.uint16)
    else:
        vals = arr[finite]
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        if hi - lo < 1e-6:
            png = np.zeros(arr.shape[:2], dtype=np.uint16)
        else:
            png = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
            png = (png * 65535.0 + 0.5).astype(np.uint16)
    output_png16.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_png16), png)
