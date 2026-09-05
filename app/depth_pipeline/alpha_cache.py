# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_alpha_debug_cache(alpha: np.ndarray, output_npy: Path, output_png8: Path | None = None) -> None:
    arr = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_npy), arr)
    if output_png8 is not None:
        output_png8.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_png8), (arr * 255.0 + 0.5).astype(np.uint8))
