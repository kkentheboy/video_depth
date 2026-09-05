# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _bbox(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0:
        z = np.zeros(3, dtype=np.float32)
        return z, z
    return np.nanmin(pts, axis=0).astype(np.float32), np.nanmax(pts, axis=0).astype(np.float32)


def align_template_to_visible_points(
    template_vertices: np.ndarray,
    visible_points: np.ndarray,
    *,
    strength: float = 1.0,
    camera: dict | None = None,
    cache_root: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Coarsely align an external SMPL-like template mesh to visible point space.

    This is deliberately model-agnostic. It does not pretend to solve true SMPL
    camera fitting. It only makes cached structure vertices usable as a same-space
    fallback by matching bbox center and a robust uniform scale to the visible
    depth point cloud.
    """
    tv = np.asarray(template_vertices, dtype=np.float32).reshape(-1, 3)
    vp = np.asarray(visible_points, dtype=np.float32).reshape(-1, 3)
    if len(tv) == 0 or len(vp) == 0:
        return tv.copy(), {"available": False, "reason": "empty_template_or_visible"}

    t_min, t_max = _bbox(tv)
    v_min, v_max = _bbox(vp)
    t_center = (t_min + t_max) * 0.5
    v_center = np.median(vp, axis=0).astype(np.float32)
    t_extent = np.maximum(t_max - t_min, 1e-6)
    v_extent = np.maximum(v_max - v_min, 1e-6)

    strength = float(np.clip(strength, 0.0, 1.0))
    cam = dict(camera or {})
    coord = str(cam.get("coordinate_space", "")).lower()
    has_world = bool(cam.get("has_world_trajectory", False)) or "world" in coord or "wham" in coord

    # WHAM-style world-coordinate structure should keep its cross-frame motion.
    # Initialize one global visible<->world transform, then reuse it for all
    # frames. This avoids the old per-frame bbox/median rescale that made the
    # template breathe and discarded world trajectory.
    if has_world and cache_root is not None:
        align_path = Path(cache_root) / "structure" / "global_world_alignment.json"
        if align_path.exists():
            try:
                payload = json.loads(align_path.read_text(encoding="utf-8"))
                scale = float(payload.get("scale", 1.0))
                src_center = np.asarray(payload.get("template_center"), dtype=np.float32).reshape(3)
                dst_center = np.asarray(payload.get("target_center"), dtype=np.float32).reshape(3)
                aligned = (tv - src_center[None, :]) * max(scale, 1e-8) + dst_center[None, :]
                if strength < 1.0:
                    aligned = tv * (1.0 - strength) + aligned * strength
                payload.update({"available": True, "method": "global_world_anchor_reuse", "strength": float(strength)})
                return aligned.astype(np.float32), payload
            except Exception:
                pass

        scale = float(np.median(v_extent / t_extent))
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        aligned = (tv - t_center[None, :]) * scale + v_center[None, :]
        if strength < 1.0:
            aligned = tv * (1.0 - strength) + aligned * strength
        meta = {
            "available": True,
            "method": "global_world_anchor_init",
            "scale": float(scale),
            "strength": float(strength),
            "template_center": [float(x) for x in t_center],
            "target_center": [float(x) for x in v_center],
            "coordinate_space": coord or "world",
        }
        try:
            align_path.parent.mkdir(parents=True, exist_ok=True)
            align_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return aligned.astype(np.float32), meta

    # Default 4DHumans/HMR2 fallback: per-frame visible alignment. These models
    # improve body completeness but do not provide a stable world trajectory here.
    scale = float(np.median(v_extent / t_extent))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    aligned = (tv - t_center[None, :]) * scale + v_center[None, :]
    if strength < 1.0:
        aligned = tv * (1.0 - strength) + aligned * strength
    meta = {
        "available": True,
        "method": "bbox_median_scale_to_visible_points",
        "scale": float(scale),
        "strength": float(strength),
        "template_bbox_min": [float(x) for x in t_min],
        "template_bbox_max": [float(x) for x in t_max],
        "visible_bbox_min": [float(x) for x in v_min],
        "visible_bbox_max": [float(x) for x in v_max],
        "template_center": [float(x) for x in t_center],
        "target_center": [float(x) for x in v_center],
    }
    return aligned.astype(np.float32), meta


def apply_template_alignment_to_points(
    points: np.ndarray,
    align_meta: dict | None,
    *,
    strength: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply a saved coarse template alignment to another same-space point set.

    This is used for hand vertices when the hand cache is already in the same
    model coordinate space as the structure mesh. It intentionally does not try
    to solve HaMeR crop-space alignment. Crop-space hand output must be converted
    by the local HaMeR adapter before being saved to cache.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    meta = dict(align_meta or {})
    if len(pts) == 0 or not bool(meta.get("available", False)):
        return pts.copy(), {"available": False, "reason": "alignment_missing_or_empty_points"}
    try:
        scale = float(meta.get("scale", 1.0))
        src_center = np.asarray(meta.get("template_center"), dtype=np.float32).reshape(3)
        dst_center = np.asarray(meta.get("target_center"), dtype=np.float32).reshape(3)
    except Exception:
        return pts.copy(), {"available": False, "reason": "alignment_meta_incomplete"}
    s = float(meta.get("strength", 1.0) if strength is None else strength)
    s = float(np.clip(s, 0.0, 1.0))
    aligned = (pts - src_center[None, :]) * max(scale, 1e-8) + dst_center[None, :]
    if s < 1.0:
        aligned = pts * (1.0 - s) + aligned * s
    return aligned.astype(np.float32), {"available": True, "method": "reuse_structure_alignment", "scale": float(scale), "strength": float(s)}


def save_alignment_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
