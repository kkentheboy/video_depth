# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np

from .hand_cache import HandFrame, save_hand_frame, load_hand_frame


def _normalize_faces(faces: np.ndarray | None) -> np.ndarray | None:
    if faces is None:
        return None
    arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if arr.size == 0:
        return arr.astype(np.int32)
    if int(arr.min()) >= 1:
        arr = arr - 1
    return arr.astype(np.int32)


def write_external_hand_cache(
    cache_root: str | Path,
    frame_index: int,
    *,
    left_vertices: np.ndarray | None = None,
    right_vertices: np.ndarray | None = None,
    left_faces: np.ndarray | None = None,
    right_faces: np.ndarray | None = None,
    joints: np.ndarray | None = None,
    confidence: float = 1.0,
    model_name: str = "external_hamer",
    coordinate_space: str = "structure",
    meta: dict | None = None,
) -> Path:
    """Adapter for locally deployed HaMeR runners.

    `coordinate_space` should be:
    - "structure" when hand vertices use the same space as SMPL/structure cache.
    - "pointcloud" when vertices are already aligned to exported point cloud space.
    """
    payload = dict(meta or {})
    payload["coordinate_space"] = str(coordinate_space)
    frame = HandFrame(
        frame_index=int(frame_index),
        left_vertices=np.asarray(left_vertices, dtype=np.float32).reshape(-1, 3) if left_vertices is not None else None,
        right_vertices=np.asarray(right_vertices, dtype=np.float32).reshape(-1, 3) if right_vertices is not None else None,
        left_faces=_normalize_faces(left_faces),
        right_faces=_normalize_faces(right_faces),
        joints=np.asarray(joints, dtype=np.float32).reshape(-1, 3) if joints is not None else None,
        confidence=float(confidence),
        model_name=str(model_name),
        meta=payload,
    )
    save_hand_frame(Path(cache_root), frame)
    return Path(cache_root) / "hand"


def validate_hand_cache(cache_root: str | Path, frame_index: int) -> dict:
    frame = load_hand_frame(Path(cache_root), int(frame_index))
    if frame is None:
        return {"ok": False, "reason": "missing_hand_frame", "frame_index": int(frame_index)}
    left = 0 if frame.left_vertices is None else int(len(frame.left_vertices))
    right = 0 if frame.right_vertices is None else int(len(frame.right_vertices))
    ok = frame.available and (left + right) > 0
    return {
        "ok": bool(ok),
        "reason": "ok" if ok else "invalid_hand_arrays_or_confidence",
        "frame_index": int(frame_index),
        "left_vertices": left,
        "right_vertices": right,
        "confidence": float(frame.confidence),
        "model_name": str(frame.model_name),
        "coordinate_space": str((frame.meta or {}).get("coordinate_space", "structure")),
    }
