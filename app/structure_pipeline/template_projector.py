# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np


def estimate_alignment_from_masks(alpha_mask: np.ndarray, template_mask: np.ndarray) -> dict:
    """Lightweight placeholder alignment stats for future SMPL/depth matching."""
    a = np.asarray(alpha_mask, dtype=bool)
    t = np.asarray(template_mask, dtype=bool)
    if not a.any() or not t.any():
        return {"available": False, "reason": "empty_mask"}
    ay, ax = np.where(a)
    ty, tx = np.where(t)
    ab = [int(ax.min()), int(ay.min()), int(ax.max()), int(ay.max())]
    tb = [int(tx.min()), int(ty.min()), int(tx.max()), int(ty.max())]
    aw, ah = max(1, ab[2] - ab[0] + 1), max(1, ab[3] - ab[1] + 1)
    tw, th = max(1, tb[2] - tb[0] + 1), max(1, tb[3] - tb[1] + 1)
    return {
        "available": True,
        "alpha_bbox": ab,
        "template_bbox": tb,
        "scale_xy": [float(aw / tw), float(ah / th)],
        "offset_xy": [float((ab[0] + ab[2] - tb[0] - tb[2]) * 0.5), float((ab[1] + ab[3] - tb[1] - tb[3]) * 0.5)],
    }
