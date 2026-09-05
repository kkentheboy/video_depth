# -*- coding: utf-8 -*-
from __future__ import annotations

import cv2
import numpy as np


def resize_bgr_like_depth(frame_bgr: np.ndarray, depth_hw: tuple[int, int]) -> np.ndarray:
    """Resize an RGB/BGR frame to match a depth map without changing aspect logic.

    The caller should pass the same preprocessed frame used by DA3 whenever
    possible. This helper only fixes residual size differences.
    """
    th, tw = depth_hw
    arr = np.asarray(frame_bgr)
    if arr.shape[:2] == (th, tw):
        return arr
    interpolation = cv2.INTER_AREA if arr.shape[0] > th or arr.shape[1] > tw else cv2.INTER_LINEAR
    return cv2.resize(arr, (tw, th), interpolation=interpolation)
