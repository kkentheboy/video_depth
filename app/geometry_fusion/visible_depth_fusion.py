# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class VisibleDepthPointSet:
    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    source_id: np.ndarray


def as_visible_pointset(points: np.ndarray, colors: np.ndarray, confidence: np.ndarray | None = None) -> VisibleDepthPointSet:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)[: len(pts)]
    if confidence is None:
        conf = np.ones(len(pts), dtype=np.float32)
    else:
        conf = np.asarray(confidence, dtype=np.float32).reshape(-1)[: len(pts)]
    return VisibleDepthPointSet(points=pts, colors=cols, confidence=conf, source_id=np.zeros(len(pts), dtype=np.uint8))
