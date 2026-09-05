# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np


def bbox_from_mask(mask: np.ndarray, scale: float = 1.8) -> tuple[int, int, int, int] | None:
    arr = np.asarray(mask, dtype=bool)
    if not arr.any():
        return None
    ys, xs = np.where(arr)
    h, w = arr.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    bw, bh = max(1.0, (x1 - x0 + 1) * float(scale)), max(1.0, (y1 - y0 + 1) * float(scale))
    return (max(0, int(cx - bw * 0.5)), max(0, int(cy - bh * 0.5)), min(w, int(cx + bw * 0.5)), min(h, int(cy + bh * 0.5)))
