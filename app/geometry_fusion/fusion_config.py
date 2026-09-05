# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FusionConfig:
    mode: str = "visible_depth"  # visible_depth | fused_body | fused_body_hand
    occlusion_fill_enabled: bool = False
    occlusion_fill_strength: float = 0.7
    normal_relief_strength: float = 0.0
