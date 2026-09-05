# -*- coding: utf-8 -*-
from __future__ import annotations

import cv2
import numpy as np


def _as_mask(mask: np.ndarray | None, shape_hw: tuple[int, int] | None = None) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.size <= 1:
        return None
    if float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if shape_hw is not None and arr.shape[:2] != tuple(shape_hw):
        arr = cv2.resize(arr, (int(shape_hw[1]), int(shape_hw[0])), interpolation=cv2.INTER_NEAREST)
    return arr.astype(np.float32)


def _soft_expand_mask(mask: np.ndarray | None, radius: int) -> np.ndarray | None:
    arr = _as_mask(mask)
    if arr is None:
        return None
    r = max(0, int(radius))
    if r <= 0:
        return arr.astype(np.float32)
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    hard = cv2.dilate((arr > 0.35).astype(np.float32), kernel, iterations=1)
    soft = cv2.GaussianBlur(np.maximum(arr, hard * 0.72).astype(np.float32), (0, 0), sigmaX=max(0.8, r * 0.45))
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0.35)
    if len(xs) < 16 or len(ys) < 16:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _first_camera_vector(camera: dict | None) -> np.ndarray | None:
    if not isinstance(camera, dict):
        return None
    for key in ("pred_cam", "cam", "camera", "weak_perspective_camera"):
        val = camera.get(key)
        if val is None:
            continue
        arr = np.asarray(val, dtype=np.float32).reshape(-1)
        if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
            return arr[:3].astype(np.float32)
    return None


def _try_project_vertices_with_camera(vertices: np.ndarray, camera: dict | None, shape_hw: tuple[int, int], foreground: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Project vertices with cached weak-perspective camera when it is usable.

    4D/WHAM caches do not always contain the crop-to-full-frame transform, so this
    is intentionally conservative. If the projected points do not cover a sane
    part of the foreground box, the caller falls back to the old bbox projection.
    """
    pts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0 or not isinstance(camera, dict):
        return None
    h, w = shape_hw
    cam = _first_camera_vector(camera)
    if cam is None:
        return None
    s, tx, ty = float(cam[0]), float(cam[1]), float(cam[2])
    if not np.isfinite(s) or abs(s) < 1e-6 or abs(s) > 100.0:
        return None
    # HMR/4D style weak-perspective camera: normalized image coordinates.
    u = (s * pts[:, 0] + tx + 1.0) * 0.5 * max(1, w - 1)
    v = (-s * pts[:, 1] + ty + 1.0) * 0.5 * max(1, h - 1)
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        return None
    px = np.rint(u).astype(np.int32)
    py = np.rint(v).astype(np.int32)
    in_frame = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if float(np.mean(in_frame)) < 0.20:
        return None
    if foreground is not None:
        box = _mask_bbox(foreground)
        if box is not None:
            x0, y0, x1, y1 = box
            pad_x = max(2, int((x1 - x0 + 1) * 0.08))
            pad_y = max(2, int((y1 - y0 + 1) * 0.08))
            x0, x1 = max(0, x0 - pad_x), min(w - 1, x1 + pad_x)
            y0, y1 = max(0, y0 - pad_y), min(h - 1, y1 + pad_y)
            in_box = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
            # A valid camera projection should land a meaningful portion of body
            # vertices inside the visible subject bbox. Otherwise crop transforms
            # are probably missing; fallback is safer than sampling wrong masks.
            if float(np.mean(in_box)) < 0.16:
                return None
    return np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)


def _project_vertices_to_mask(vertices: np.ndarray, foreground: np.ndarray | None, shape_hw: tuple[int, int], camera: dict | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    pts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    h, w = shape_hw
    projected = _try_project_vertices_with_camera(pts, camera, (h, w), foreground)
    if projected is not None:
        return projected[0], projected[1], "weak_perspective_camera"
    # Fallback: use body local X/Y and fit to the foreground bbox. This is less
    # accurate for side views, but deterministic and safe when camera payload lacks
    # a crop/full-frame transform.
    x = pts[:, 0]
    y = -pts[:, 1]
    xmin, xmax = float(np.nanpercentile(x, 1)), float(np.nanpercentile(x, 99))
    ymin, ymax = float(np.nanpercentile(y, 1)), float(np.nanpercentile(y, 99))
    if abs(xmax - xmin) < 1e-6:
        xmax = xmin + 1.0
    if abs(ymax - ymin) < 1e-6:
        ymax = ymin + 1.0
    if foreground is not None:
        box = _mask_bbox(foreground)
    else:
        box = None
    if box is None:
        x0, y0, x1, y1 = 0, 0, w - 1, h - 1
    else:
        x0, y0, x1, y1 = box
        # Slight margin so border garment/hair samples are not clipped away.
        pad_x = max(2, int((x1 - x0 + 1) * 0.04))
        pad_y = max(2, int((y1 - y0 + 1) * 0.04))
        x0, x1 = max(0, x0 - pad_x), min(w - 1, x1 + pad_x)
        y0, y1 = max(0, y0 - pad_y), min(h - 1, y1 + pad_y)
    px = ((x - xmin) / (xmax - xmin) * max(1, (x1 - x0)) + x0).round().astype(np.int32)
    py = ((y - ymin) / (ymax - ymin) * max(1, (y1 - y0)) + y0).round().astype(np.int32)
    return np.clip(px, 0, w - 1), np.clip(py, 0, h - 1), "local_xy_bbox"


def build_mask_guided_region_weights(
    vertices: np.ndarray,
    base_region: dict[str, np.ndarray],
    segmentation: dict | None,
    *,
    camera: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return stable garment/hair weights for dense vertices.

    The current single-view pipeline cannot know hidden clothing geometry. The
    mask is therefore used as a visible-region constraint, while the built-in
    topology/height prior remains the fallback for clipped, occluded or missing
    regions.
    """
    n = len(np.asarray(vertices).reshape(-1, 3))
    base_g = np.asarray(base_region.get("garment", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    base_h = np.asarray(base_region.get("hair", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if len(base_g) != n:
        base_g = np.zeros((n,), dtype=np.float32)
    if len(base_h) != n:
        base_h = np.zeros((n,), dtype=np.float32)
    meta = {"source": "geometry_fallback", "mask_used": False, "confidence": 0.0}
    if not segmentation:
        return base_g.astype(np.float32), base_h.astype(np.float32), meta
    fg = _as_mask(segmentation.get("foreground"))
    garment = _as_mask(segmentation.get("garment"), fg.shape if fg is not None else None)
    hair = _as_mask(segmentation.get("hair"), fg.shape if fg is not None else None)
    if garment is not None:
        garment = _soft_expand_mask(garment, 3)
    if hair is not None:
        hair = _soft_expand_mask(hair, 5)
    if fg is None and garment is not None:
        fg = (garment > 0.35).astype(np.float32)
    if fg is None and hair is not None:
        fg = (hair > 0.35).astype(np.float32)
    if fg is None:
        return base_g.astype(np.float32), base_h.astype(np.float32), meta
    h, w = fg.shape[:2]
    px, py, projection_method = _project_vertices_to_mask(vertices, fg, (h, w), camera=camera)
    sample_g = garment[py, px] if garment is not None else np.zeros((n,), dtype=np.float32)
    sample_h = hair[py, px] if hair is not None else np.zeros((n,), dtype=np.float32)
    sample_fg = fg[py, px]
    # Base prior is weakened but not removed. This handles clipped / occluded / leaked masks.
    g = np.maximum(base_g * 0.30, sample_g * np.maximum(sample_fg, 0.65))
    h_weight = np.maximum(base_h * 0.25, sample_h * np.maximum(sample_fg, 0.65))
    g = np.clip(g, 0.0, 1.0).astype(np.float32)
    h_weight = np.clip(h_weight, 0.0, 1.0).astype(np.float32)
    meta_payload = segmentation.get("meta", {}) if isinstance(segmentation.get("meta", {}), dict) else {}
    meta = {
        "source": "human_parsing_mask+soft_silhouette_expand",
        "mask_used": True,
        "confidence": float(meta_payload.get("confidence", 0.0) or 0.0),
        "provider": str(meta_payload.get("provider", "unknown")),
        "garment_mask_vertex_ratio": float(np.mean(sample_g > 0.35)) if len(sample_g) else 0.0,
        "hair_mask_vertex_ratio": float(np.mean(sample_h > 0.35)) if len(sample_h) else 0.0,
        "mask_soft_expand_px": {"garment": 3, "hair": 5},
        "projection_method": projection_method,
    }
    return g, h_weight, meta


def build_sequence_mask_guided_region_weights(
    vertices: np.ndarray,
    base_region: dict[str, np.ndarray],
    cache_root: str,
    total_frames: int,
    *,
    camera: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Aggregate per-frame parsing masks into stable dense-vertex weights.

    The output weights are fixed to dense vertex IDs, so garment/hair layers do
    not slide with motion. Per-frame masks are used as evidence; clipped or
    unstable frames get lower weight and never delete the geometric prior.
    """
    from .segmentation_cache import load_segmentation_frame, segmentation_cache_summary

    pts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    n = len(pts)
    base_g = np.asarray(base_region.get("garment", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    base_h = np.asarray(base_region.get("hair", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if len(base_g) != n:
        base_g = np.zeros((n,), dtype=np.float32)
    if len(base_h) != n:
        base_h = np.zeros((n,), dtype=np.float32)
    total = max(1, int(total_frames or 1))
    g_acc = np.zeros((n,), dtype=np.float32)
    h_acc = np.zeros((n,), dtype=np.float32)
    w_acc = 0.0
    used = 0
    status_count: dict[str, int] = {}
    # Cap evidence frames for responsiveness. For long videos use a two-stage
    # stratified sample so side-view/partial/occluded ranges are not fully washed
    # out by early trusted front-view frames.
    max_evidence = 120
    if total <= max_evidence:
        candidate_indices = list(range(total))
    else:
        candidate_indices = sorted(set(np.linspace(0, total - 1, min(total, max_evidence * 2)).astype(int).tolist()))

    candidates: list[tuple[int, str, float, dict]] = []
    for frame_index in candidate_indices:
        seg = load_segmentation_frame(cache_root, frame_index)
        if not seg:
            continue
        meta = seg.get("meta", {}) if isinstance(seg, dict) else {}
        q = meta.get("quality", {}) if isinstance(meta, dict) else {}
        status = str(q.get("status", meta.get("quality_status", "unknown")))
        status_count[status] = status_count.get(status, 0) + 1
        if status == "unknown":
            weight = 0.0
        elif status == "trusted":
            weight = 1.00
        elif status == "partial":
            weight = 0.72
        elif status == "clipped":
            weight = 0.55
        elif status == "unstable":
            weight = 0.35
        elif status == "occluded":
            # Occluded masks should not erase a known garment/hair layer.
            weight = 0.42
        else:
            weight = 0.50
        if weight <= 0.0:
            continue
        candidates.append((int(frame_index), status, float(weight), seg))

    if len(candidates) > max_evidence:
        buckets: dict[str, list[tuple[int, str, float, dict]]] = {}
        for item in candidates:
            buckets.setdefault(item[1], []).append(item)
        priority = ["trusted", "partial", "clipped", "occluded", "unstable"] + sorted(k for k in buckets if k not in {"trusted", "partial", "clipped", "occluded", "unstable"})
        selected: list[tuple[int, str, float, dict]] = []
        # Reserve representation for every non-empty quality state first.
        non_empty = [k for k in priority if buckets.get(k)]
        base_quota = max(1, max_evidence // max(1, len(non_empty)))
        for status in non_empty:
            bucket = buckets.get(status, [])
            take = min(len(bucket), base_quota)
            if take <= 0:
                continue
            pick = np.linspace(0, len(bucket) - 1, take).astype(int).tolist() if len(bucket) > take else list(range(len(bucket)))
            selected.extend(bucket[i] for i in pick)
        if len(selected) < max_evidence:
            seen = {item[0] for item in selected}
            for item in candidates:
                if item[0] in seen:
                    continue
                selected.append(item)
                if len(selected) >= max_evidence:
                    break
        candidates = sorted(selected[:max_evidence], key=lambda item: item[0])

    for frame_index, status, weight, seg in candidates:
        g, h, _m = build_mask_guided_region_weights(pts, base_region, seg, camera=camera)
        g_acc += np.asarray(g, dtype=np.float32).reshape(-1) * weight
        h_acc += np.asarray(h, dtype=np.float32).reshape(-1) * weight
        w_acc += weight
        used += 1
    if used <= 0 or w_acc <= 1e-6:
        return base_g.astype(np.float32), base_h.astype(np.float32), {
            "source": "geometry_fallback",
            "mask_used": False,
            "cached_frames_used": 0,
            "status_count": status_count,
            "summary": segmentation_cache_summary(cache_root),
        }
    g_avg = g_acc / max(w_acc, 1e-6)
    h_avg = h_acc / max(w_acc, 1e-6)
    # Never allow segmentation to erase stable prior completely. This handles
    # crop/occlusion and keeps the exported mesh animation stable.
    g_final = np.maximum(base_g * 0.24, g_avg).astype(np.float32)
    h_final = np.maximum(base_h * 0.22, h_avg).astype(np.float32)
    return np.clip(g_final, 0.0, 1.0), np.clip(h_final, 0.0, 1.0), {
        "source": "per_frame_human_parsing_cache+temporal_quality+geometry_prior",
        "mask_used": True,
        "cached_frames_used": int(used),
        "candidate_frames_checked": int(len(candidate_indices)),
        "status_count": status_count,
        "summary": segmentation_cache_summary(cache_root),
        "garment_mask_vertex_ratio": float(np.mean(g_avg > 0.35)) if len(g_avg) else 0.0,
        "hair_mask_vertex_ratio": float(np.mean(h_avg > 0.35)) if len(h_avg) else 0.0,
    }
