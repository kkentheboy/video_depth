# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from hand_pipeline.hand_cache import HandFrame
from mesh_usd_exporter import deterministic_limit_indices
from .mesh_sampling import sample_mesh_surface
from .template_alignment import apply_template_alignment_to_points


def merge_hand_vertices(body_vertices: np.ndarray, hand_vertices: list[np.ndarray] | None) -> np.ndarray:
    body = np.asarray(body_vertices, dtype=np.float32).reshape(-1, 3)
    if not hand_vertices:
        return body
    pieces = [body] + [np.asarray(v, dtype=np.float32).reshape(-1, 3) for v in hand_vertices if v is not None]
    return np.concatenate(pieces, axis=0) if pieces else body


def _as_points_from_hand_piece(vertices: np.ndarray | None, faces: np.ndarray | None, sample_count: int, seed: int) -> np.ndarray:
    if vertices is None:
        return np.zeros((0, 3), dtype=np.float32)
    v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    if len(v) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if faces is not None and sample_count > 0:
        f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        pts = sample_mesh_surface(v, f, int(sample_count), seed=seed)
        if len(pts) > 0:
            return pts.astype(np.float32)
    # Fallback: HaMeR adapters may initially save only vertices.
    limiter = deterministic_limit_indices(len(v), max(1, int(sample_count)), seed)
    return v[limiter].astype(np.float32) if limiter is not None else v.astype(np.float32)


def build_hand_points(
    hand_frame: HandFrame | None,
    *,
    align_meta: dict | None = None,
    max_points: int = 30_000,
    frame_index: int = 0,
    require_alignment: bool = True,
) -> tuple[np.ndarray, dict]:
    """Return hand points in point-cloud space from an existing hand cache.

    No HaMeR model is launched here. If the cache is in SMPL/structure coordinate
    space, `align_meta` reuses the structure alignment transform. If the local
    hand adapter writes already aligned pointcloud-space vertices, put
    `"coordinate_space": "pointcloud"` in hand_meta.json and this function will
    keep them as-is.
    """
    meta = {
        "available": False,
        "reason": "hand_cache_missing_or_invalid",
        "hand_points": 0,
    }
    if hand_frame is None or not hand_frame.available:
        return np.zeros((0, 3), dtype=np.float32), meta

    per_side = max(1, int(max_points) // 2)
    left = _as_points_from_hand_piece(hand_frame.left_vertices, hand_frame.left_faces, per_side, 9100003 + int(frame_index))
    right = _as_points_from_hand_piece(hand_frame.right_vertices, hand_frame.right_faces, per_side, 9200003 + int(frame_index))
    pts = np.concatenate([p for p in [left, right] if len(p) > 0], axis=0) if len(left) + len(right) else np.zeros((0, 3), dtype=np.float32)
    if len(pts) == 0:
        meta["reason"] = "empty_hand_vertices"
        return pts, meta

    coordinate_space = str((hand_frame.meta or {}).get("coordinate_space", "structure")).lower()
    align_result = {"available": False, "reason": "not_needed"}
    if coordinate_space not in {"pointcloud", "world", "aligned"}:
        aligned, align_result = apply_template_alignment_to_points(pts, align_meta)
        if not bool(align_result.get("available", False)) and require_alignment:
            meta.update({"reason": "hand_alignment_missing", "alignment": align_result})
            return np.zeros((0, 3), dtype=np.float32), meta
        pts = aligned

    limiter = deterministic_limit_indices(len(pts), int(max_points), 9300003 + int(frame_index))
    if limiter is not None:
        pts = pts[limiter]
    meta.update({
        "available": True,
        "reason": "ok",
        "hand_points": int(len(pts)),
        "model_name": str(hand_frame.model_name),
        "confidence": float(hand_frame.confidence),
        "coordinate_space": coordinate_space,
        "alignment": align_result,
    })
    return pts.astype(np.float32), meta
