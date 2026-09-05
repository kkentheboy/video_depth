# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def estimate_intrinsics(width: int, height: int, focal_scale: float = 0.85) -> CameraIntrinsics:
    """Estimate stable pseudo intrinsics for monocular relative depth.

    This is not camera calibration. It gives Blender a coherent, non-exploding
    point cloud volume until a real calibration or SMPL alignment stage exists.
    """
    w = max(1, int(width))
    h = max(1, int(height))
    f = max(w, h) * max(0.2, float(focal_scale))
    return CameraIntrinsics(width=w, height=h, fx=f, fy=f, cx=(w - 1) * 0.5, cy=(h - 1) * 0.5)


def backproject_depth_to_xyz(z: np.ndarray, intr: CameraIntrinsics, coordinate_mode: str = "blender") -> np.ndarray:
    h, w = z.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    zz = np.asarray(z, dtype=np.float32)
    x = (xx - float(intr.cx)) / max(float(intr.fx), 1e-6) * zz
    y = (yy - float(intr.cy)) / max(float(intr.fy), 1e-6) * zz
    if str(coordinate_mode).lower() == "opencv":
        return np.dstack([x, y, zz]).astype(np.float32)
    # Blender: X right, Y depth, Z up. OpenCV image Y points down, so flip it.
    return np.dstack([x, -zz, -y]).astype(np.float32)
