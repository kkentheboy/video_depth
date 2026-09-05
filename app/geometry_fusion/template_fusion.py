# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np


def occlusion_mask_from_template(visible_mask: np.ndarray, template_mask: np.ndarray) -> np.ndarray:
    visible = np.asarray(visible_mask, dtype=bool)
    template = np.asarray(template_mask, dtype=bool)
    if visible.shape != template.shape:
        raise ValueError("visible_mask and template_mask must have the same shape")
    return template & ~visible
