# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np


def sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    """Area-weighted triangle surface sampling for future fused mesh export."""
    v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(v) == 0 or len(f) == 0 or count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    tri = v[np.clip(f, 0, len(v) - 1)]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
    if not np.isfinite(area).all() or float(area.sum()) <= 1e-12:
        return np.zeros((0, 3), dtype=np.float32)
    prob = area / area.sum()
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(tri), size=int(count), replace=True, p=prob)
    t = tri[idx]
    r1 = np.sqrt(rng.random(int(count), dtype=np.float32))
    r2 = rng.random(int(count), dtype=np.float32)
    pts = (1 - r1)[:, None] * t[:, 0] + (r1 * (1 - r2))[:, None] * t[:, 1] + (r1 * r2)[:, None] * t[:, 2]
    return pts.astype(np.float32)
