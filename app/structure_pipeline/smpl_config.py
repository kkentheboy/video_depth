# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StructureConfig:
    model: str = "none"  # none | 4dhumans | wham | smplx
    resolution: int = 720
    cache_enabled: bool = True
    confidence_threshold: float = 0.35
    device: str = "auto"
