# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(slots=True)
class DenseMeshTemplate:
    """Fixed per-face barycentric subdivision template.

    The template is generated once from the base SMPL topology and reused for
    every frame. Dense vertex N is always `(base_face_id, barycentric)`, so ID is
    stable even when the body deforms.
    """

    base_face_indices: np.ndarray
    barycentric: np.ndarray
    faces: np.ndarray
    segments: int
    face_base_indices: Optional[np.ndarray] = None
    method: str = "shared_edge_barycentric_fixed_topology"


def _as_faces(faces: np.ndarray) -> np.ndarray:
    arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    return arr.astype(np.int64, copy=False)


def build_dense_mesh_template(faces: np.ndarray, *, segments: int = 2, max_base_faces: Optional[int] = None) -> DenseMeshTemplate:
    """Build a stable dense triangle mesh template with shared edge vertices.

    The template is generated once from the fixed base topology and reused for
    every frame. Dense vertex IDs are stable. Unlike the old per-face template,
    vertices on shared base edges are de-duplicated, so Blender imports one
    connected mesh instead of many tiny face islands.
    """
    base_faces = _as_faces(faces)
    if max_base_faces is not None and max_base_faces > 0 and len(base_faces) > max_base_faces:
        base_faces = base_faces[: int(max_base_faces)]
    s = int(np.clip(int(segments or 1), 1, 5))
    denom = float(max(1, s))
    face_ids: list[int] = []
    bary: list[tuple[float, float, float]] = []
    out_faces: list[tuple[int, int, int]] = []
    out_face_base_ids: list[int] = []
    global_index: dict[tuple, int] = {}

    def _vertex_key(face_vertices: np.ndarray, weights: tuple[int, int, int], face_id: int) -> tuple:
        nonzero = [idx for idx, w in enumerate(weights) if int(w) != 0]
        if len(nonzero) == 1:
            return ("v", int(face_vertices[nonzero[0]]))
        if len(nonzero) == 2:
            a_i, b_i = nonzero
            va, vb = int(face_vertices[a_i]), int(face_vertices[b_i])
            wa, wb = int(weights[a_i]), int(weights[b_i])
            lo, hi = (va, vb) if va <= vb else (vb, va)
            weight_lo = wa if va == lo else wb
            return ("e", lo, hi, int(weight_lo), int(s))
        return ("f", int(face_id), int(weights[0]), int(weights[1]), int(weights[2]), int(s))

    def _get_index(face_vertices: np.ndarray, weights: tuple[int, int, int], face_id: int) -> int:
        key = _vertex_key(face_vertices, weights, face_id)
        found = global_index.get(key)
        if found is not None:
            return int(found)
        idx = len(face_ids)
        global_index[key] = idx
        face_ids.append(int(face_id))
        bary.append((weights[0] / denom, weights[1] / denom, weights[2] / denom))
        return idx

    for face_id, face_vertices in enumerate(base_faces):
        local: dict[tuple[int, int], int] = {}
        for i in range(s + 1):
            for j in range(s + 1 - i):
                k = s - i - j
                local[(i, j)] = _get_index(face_vertices, (i, j, k), face_id)
        for i in range(s):
            for j in range(s - i):
                a = local[(i, j)]
                b = local[(i + 1, j)]
                c = local[(i, j + 1)]
                out_faces.append((a, b, c))
                out_face_base_ids.append(int(face_id))
                if i + j + 1 < s:
                    d = local[(i + 1, j)]
                    e = local[(i + 1, j + 1)]
                    f = local[(i, j + 1)]
                    out_faces.append((d, e, f))
                    out_face_base_ids.append(int(face_id))
    return DenseMeshTemplate(
        base_face_indices=np.asarray(face_ids, dtype=np.int64),
        barycentric=np.asarray(bary, dtype=np.float32).reshape(-1, 3),
        faces=np.asarray(out_faces, dtype=np.int64).reshape(-1, 3),
        segments=s,
        face_base_indices=np.asarray(out_face_base_ids, dtype=np.int64).reshape(-1),
        method="shared_edge_barycentric_fixed_topology",
    )

def evaluate_dense_vertices(vertices: np.ndarray, faces: np.ndarray, template: DenseMeshTemplate) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    f = _as_faces(faces)
    if len(v) == 0 or len(f) == 0 or len(template.base_face_indices) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    base_idx = np.clip(np.asarray(template.base_face_indices, dtype=np.int64), 0, len(f) - 1)
    tri_indices = np.clip(f[base_idx], 0, max(0, len(v) - 1))
    tri = v[tri_indices]
    b = np.asarray(template.barycentric, dtype=np.float32).reshape(-1, 3)
    pts = tri[:, 0] * b[:, 0:1] + tri[:, 1] * b[:, 1:2] + tri[:, 2] * b[:, 2:3]
    return np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def evaluate_dense_normals(vertices: np.ndarray, faces: np.ndarray, template: DenseMeshTemplate) -> np.ndarray:
    """Evaluate smooth per-dense-vertex normals from the dense topology.

    The old implementation reused the base face normal for every point inside a
    base triangle. With shared dense vertices this would make shell offsets
    depend on the first face that created an edge vertex. This version averages
    adjacent dense face normals, which is also the same convention Blender will
    use when it computes normals from the exported faces.
    """
    pts = evaluate_dense_vertices(vertices, faces, template)
    f = np.asarray(template.faces, dtype=np.int64).reshape(-1, 3)
    if len(pts) == 0 or len(f) == 0:
        return np.zeros((len(pts), 3), dtype=np.float32)
    valid = (f >= 0).all(axis=1) & (f < len(pts)).all(axis=1)
    f = f[valid]
    if len(f) == 0:
        return np.zeros((len(pts), 3), dtype=np.float32)
    tri = pts[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flen = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(flen, 1e-8)
    # Topology is fixed, so avoid np.add.at's slow unbuffered writes.
    # bincount is deterministic and noticeably faster on dense templates.
    idx = f.reshape(-1)
    vals = np.repeat(fn, 3, axis=0)
    accum = np.zeros_like(pts, dtype=np.float32)
    if len(idx):
        for c in range(3):
            accum[:, c] = np.bincount(idx, weights=vals[:, c], minlength=len(pts))[: len(pts)]
    nlen = np.linalg.norm(accum, axis=1, keepdims=True)
    out = accum / np.maximum(nlen, 1e-8)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def validate_dense_template_winding(vertices: np.ndarray, faces: np.ndarray, template: DenseMeshTemplate) -> dict[str, float | int | bool]:
    """Check that dense sub-faces preserve the base face winding.

    USD/Blender derive mesh normals from face winding when explicit normals are
    not authored. This validation does not try to decide whether the original
    SMPL/WHAM faces point outward; it verifies that the dense topology does not
    flip any sub-face relative to its source base face.
    """
    base_v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    base_f = _as_faces(faces)
    dense_v = evaluate_dense_vertices(base_v, base_f, template)
    dense_f = np.asarray(template.faces, dtype=np.int64).reshape(-1, 3)
    if len(base_v) == 0 or len(base_f) == 0 or len(dense_v) == 0 or len(dense_f) == 0:
        return {"ok": True, "checked_faces": 0, "flipped_faces": 0, "min_dot": 1.0}
    base_tri = base_v[np.clip(base_f, 0, max(0, len(base_v) - 1))]
    base_n = np.cross(base_tri[:, 1] - base_tri[:, 0], base_tri[:, 2] - base_tri[:, 0])
    base_len = np.linalg.norm(base_n, axis=1, keepdims=True)
    base_n = base_n / np.maximum(base_len, 1e-8)
    valid_dense = (dense_f >= 0).all(axis=1) & (dense_f < len(dense_v)).all(axis=1)
    dense_f_valid = dense_f[valid_dense]
    if len(dense_f_valid) == 0:
        return {"ok": True, "checked_faces": 0, "flipped_faces": 0, "min_dot": 1.0}
    dense_tri = dense_v[dense_f_valid]
    dense_n = np.cross(dense_tri[:, 1] - dense_tri[:, 0], dense_tri[:, 2] - dense_tri[:, 0])
    dense_len = np.linalg.norm(dense_n, axis=1, keepdims=True)
    dense_n = dense_n / np.maximum(dense_len, 1e-8)
    if template.face_base_indices is not None and len(template.face_base_indices) == len(dense_f):
        map_ids = np.asarray(template.face_base_indices, dtype=np.int64)[valid_dense]
    else:
        map_ids = np.asarray(template.base_face_indices, dtype=np.int64)[np.clip(dense_f_valid[:, 0], 0, len(template.base_face_indices) - 1)]
    map_ids = np.clip(map_ids, 0, len(base_n) - 1)
    dot = np.sum(dense_n * base_n[map_ids], axis=1)
    finite = np.isfinite(dot)
    dot = dot[finite]
    if len(dot) == 0:
        return {"ok": True, "checked_faces": 0, "flipped_faces": 0, "min_dot": 1.0}
    flipped = int(np.count_nonzero(dot < -1e-5))
    return {
        "ok": bool(flipped == 0),
        "checked_faces": int(len(dot)),
        "flipped_faces": int(flipped),
        "min_dot": float(np.min(dot)),
    }

def _vertical_axis(points: np.ndarray) -> int:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return 1
    lo = np.nanpercentile(pts, 2.0, axis=0)
    hi = np.nanpercentile(pts, 98.0, axis=0)
    extent = np.nan_to_num(hi - lo, nan=0.0, posinf=0.0, neginf=0.0)
    # Project convention inside this app is Y-Up.  Do not let a wide arm pose,
    # root-stabilized trajectory spread, or WHAM/4D coordinate noise silently move
    # garment/hair height bands onto X/Z.  Only fall back to auto-detect when Y is
    # effectively degenerate compared with the largest axis.
    max_extent = float(np.max(extent)) if extent.size else 0.0
    y_extent = float(extent[1]) if extent.size >= 2 else 0.0
    if y_extent > max(1e-5, max_extent * 0.42):
        return 1
    return int(np.argmax(extent))


def _height01(points: np.ndarray, axis: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0,), dtype=np.float32)
    lo = float(np.nanpercentile(pts[:, axis], 2.0))
    hi = float(np.nanpercentile(pts[:, axis], 98.0))
    return np.clip((pts[:, axis] - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _smooth_band(h: np.ndarray, start: float, full: float, fade: float, end: float) -> np.ndarray:
    """Soft trapezoid weight on normalized body height."""
    up = _smoothstep01((h - float(start)) / max(float(full) - float(start), 1e-6))
    down = 1.0 - _smoothstep01((h - float(fade)) / max(float(end) - float(fade), 1e-6))
    return np.clip(up * down, 0.0, 1.0).astype(np.float32)


def soft_region_weights(points: np.ndarray) -> dict[str, np.ndarray]:
    """Conservative heuristic layer weights without semantic reconstruction.

    This is intentionally mild. It should never pretend to recover real cloth or
    hair; it only marks a stable region where a small outline shell may be added.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0:
        z = np.zeros((0,), dtype=np.float32)
        return {"garment": z, "hair": z, "face_hand_protect": z}
    axis = _vertical_axis(pts)
    h = _height01(pts, axis)

    # Keep garment away from feet and the upper head/face area. The previous
    # 0.12..0.86 band overlapped heavily with the hair cap and created a bloated
    # shell. This narrower band is less visually aggressive and more honest.
    garment = _smooth_band(h, 0.16, 0.28, 0.68, 0.80)
    hair = _smoothstep01((h - 0.86) / 0.10)
    protect = np.maximum(_smoothstep01((0.14 - h) / 0.12), _smoothstep01((h - 0.78) / 0.16))
    garment = garment * (1.0 - 0.85 * protect)
    hair = hair * (1.0 - np.clip(garment, 0.0, 1.0))
    return {
        "garment": np.clip(garment, 0.0, 1.0).astype(np.float32),
        "hair": np.clip(hair, 0.0, 1.0).astype(np.float32),
        "face_hand_protect": np.clip(protect, 0.0, 1.0).astype(np.float32),
    }


def soft_clamp(values: np.ndarray, limit: float) -> np.ndarray:
    lim = max(1e-8, float(limit))
    return (np.tanh(np.asarray(values, dtype=np.float32) / lim) * lim).astype(np.float32)


def slope_limit_by_faces(values: np.ndarray, faces: np.ndarray, *, max_delta: float, iterations: int = 2) -> np.ndarray:
    """Limit abrupt height jumps across triangle edges using vectorized aggregation.

    The older implementation looped over every face in Python for every frame.
    This version computes a per-vertex neighbour/face mean with np.bincount, then
    clamps in one vectorized pass per iteration. It is approximate but stable and
    much faster for dense meshes.
    """
    vals = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    f = _as_faces(faces)
    if len(vals) == 0 or len(f) == 0:
        return vals.astype(np.float32)
    valid = (f >= 0).all(axis=1) & (f < len(vals)).all(axis=1)
    f = f[valid]
    if len(f) == 0:
        return vals.astype(np.float32)
    md = max(1e-8, float(max_delta))
    flat = f.reshape(-1)
    for _ in range(max(0, int(iterations))):
        face_mean = np.mean(vals[f], axis=1).astype(np.float32)
        accum = np.bincount(flat, weights=np.repeat(face_mean, 3), minlength=len(vals)).astype(np.float32)
        count = np.bincount(flat, minlength=len(vals)).astype(np.float32)
        mean = vals.copy()
        mask = count > 0
        mean[mask] = accum[mask] / np.maximum(count[mask], 1.0)
        vals = np.clip(vals, mean - md, mean + md).astype(np.float32)
    return vals.astype(np.float32)


def apply_shell_offset(points: np.ndarray, normals: np.ndarray, weights: np.ndarray, *, offset: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    if len(pts) == 0 or len(nrm) != len(pts) or len(w) != len(pts):
        return pts.astype(np.float32)
    return (pts + nrm * (w * float(offset))[:, None]).astype(np.float32)


def body_height(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return 0.0
    axis = _vertical_axis(pts)
    lo = float(np.nanpercentile(pts[:, axis], 2.0))
    hi = float(np.nanpercentile(pts[:, axis], 98.0))
    return max(0.0, hi - lo)


def conservative_shell_offsets(
    points: np.ndarray,
    garment_weights: np.ndarray,
    hair_weights: np.ndarray,
    *,
    garment_offset: float,
    hair_offset: float,
) -> np.ndarray:
    """Return per-point shell offsets in mesh units.

    Offsets are clamped by body height to avoid broken proportions when the
    imported structure is not in meter scale. Garment and hair use max(), not
    sum(), so overlapping heuristic regions do not double-inflate the surface.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    gw = np.asarray(garment_weights, dtype=np.float32).reshape(-1)
    hw = np.asarray(hair_weights, dtype=np.float32).reshape(-1)
    if len(pts) == 0 or len(gw) != len(pts) or len(hw) != len(pts):
        return np.zeros((len(pts),), dtype=np.float32)
    h = body_height(pts)
    if h <= 1e-6:
        return np.zeros((len(pts),), dtype=np.float32)
    # Stage-2 caps: still bounded, but no longer so tiny that Blender imports
    # look identical to the body. Real silhouette is handled by side-wall layers;
    # this cap only separates the shell surface from the skin.
    g = min(max(0.0, float(garment_offset)), h * 0.018)
    hr = min(max(0.0, float(hair_offset)), h * 0.032)
    offsets = np.maximum(np.clip(gw, 0.0, 1.0) * g, np.clip(hw, 0.0, 1.0) * hr)
    return np.nan_to_num(offsets, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def apply_shell_offsets(points: np.ndarray, normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    off = np.asarray(offsets, dtype=np.float32).reshape(-1)
    if len(pts) == 0 or len(nrm) != len(pts) or len(off) != len(pts):
        return pts.astype(np.float32)
    return (pts + nrm * off[:, None]).astype(np.float32)


def raster_signal_from_mask(mask: Optional[np.ndarray], shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask is None:
        return np.ones((h, w), dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.size == 0:
        return np.ones((h, w), dtype=np.float32)
    if float(np.nanmax(m)) > 1.5:
        m = m / 255.0
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(np.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
