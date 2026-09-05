# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path


def frame_stem(frame_index: int, one_based: bool = True) -> str:
    idx = int(frame_index) + (1 if one_based else 0)
    return f"frame_{idx:06d}"


def cache_frame_stem(frame_index: int) -> str:
    return f"{int(frame_index):08d}"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
