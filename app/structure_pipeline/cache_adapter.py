# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np

from .structure_cache import StructureFrame, save_structure_frame, load_structure_frame


def normalize_faces(faces: np.ndarray) -> np.ndarray:
    arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if arr.size == 0:
        return arr.astype(np.int32)
    # Some SMPL exporters save OBJ-style 1-based faces. Convert safely.
    if int(arr.min()) >= 1:
        arr = arr - 1
    return arr.astype(np.int32)


def write_external_structure_cache(
    cache_root: str | Path,
    frame_index: int,
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray | None = None,
    *,
    camera: dict | None = None,
    confidence: float = 1.0,
    model_name: str = "external_structure",
) -> Path:
    """Adapter for locally deployed 4DHumans/WHAM/SMPL runners.

    Call this after your model inference. The point cloud pipeline will then read
    `cache/<digest>/structure/frame_XXXXXX_*` without knowing which model wrote it.
    """
    frame = StructureFrame(
        frame_index=int(frame_index),
        vertices=np.asarray(vertices, dtype=np.float32).reshape(-1, 3),
        faces=normalize_faces(faces),
        joints=np.asarray(joints, dtype=np.float32).reshape(-1, 3) if joints is not None else None,
        camera=dict(camera or {}),
        confidence=float(confidence),
        model_name=str(model_name),
    )
    save_structure_frame(Path(cache_root), frame)
    return Path(cache_root) / "structure"


def validate_structure_cache(cache_root: str | Path, frame_index: int) -> dict:
    frame = load_structure_frame(Path(cache_root), int(frame_index))
    if frame is None:
        return {"ok": False, "reason": "missing_structure_frame", "frame_index": int(frame_index)}
    vertices = np.asarray(frame.vertices) if frame.vertices is not None else np.zeros((0, 3))
    faces = np.asarray(frame.faces) if frame.faces is not None else np.zeros((0, 3))
    ok = frame.available and vertices.ndim == 2 and vertices.shape[1] == 3 and faces.ndim == 2 and faces.shape[1] == 3
    return {
        "ok": bool(ok),
        "reason": "ok" if ok else "invalid_structure_arrays_or_confidence",
        "frame_index": int(frame_index),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "confidence": float(frame.confidence),
        "model_name": str(frame.model_name),
    }
