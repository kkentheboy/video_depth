# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(slots=True)
class SurfaceSampleSpec:
    """Stable mesh surface sampling recipe.

    `face_indices` and `barycentric` are generated once from the first valid
    mesh and reused for every frame. This makes point N stay on the same body
    surface location across time instead of being randomly resampled per frame.
    """

    face_indices: np.ndarray
    barycentric: np.ndarray
    seed: int


@dataclass(slots=True)
class RootStabilizeResult:
    vertices: np.ndarray
    roots: np.ndarray
    reference_root: np.ndarray
    max_root_jump: float
    median_root_jump: float
    method: str


@dataclass(slots=True)
class StructureTemporalSmoothResult:
    vertices: np.ndarray
    spikes_fixed: int
    median_motion_rms: float
    max_motion_rms: float
    method: str


def _as_vertices(vertices: np.ndarray) -> np.ndarray:
    return np.asarray(vertices, dtype=np.float32).reshape(-1, 3)


def _as_faces(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0:
        return f
    return np.clip(f, 0, max(0, int(vertex_count) - 1)).astype(np.int64)


def make_surface_sample_spec(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int = 9100003) -> SurfaceSampleSpec:
    """Create a reusable area-weighted sampling spec for a stable-topology mesh."""
    v = _as_vertices(vertices)
    f = _as_faces(faces, len(v))
    count = int(count)
    if len(v) == 0 or len(f) == 0 or count <= 0:
        return SurfaceSampleSpec(
            face_indices=np.zeros((0,), dtype=np.int64),
            barycentric=np.zeros((0, 3), dtype=np.float32),
            seed=int(seed),
        )
    tri = v[f]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
    if not np.isfinite(area).all() or float(area.sum()) <= 1e-12:
        return SurfaceSampleSpec(
            face_indices=np.zeros((0,), dtype=np.int64),
            barycentric=np.zeros((0, 3), dtype=np.float32),
            seed=int(seed),
        )
    prob = area / max(float(area.sum()), 1e-12)
    rng = np.random.default_rng(int(seed))
    face_indices = rng.choice(len(f), size=count, replace=True, p=prob).astype(np.int64)
    r1 = np.sqrt(rng.random(count, dtype=np.float32))
    r2 = rng.random(count, dtype=np.float32)
    bary = np.stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1).astype(np.float32)
    return SurfaceSampleSpec(face_indices=face_indices, barycentric=bary, seed=int(seed))


def sample_mesh_surface_with_spec(vertices: np.ndarray, faces: np.ndarray, spec: SurfaceSampleSpec) -> np.ndarray:
    v = _as_vertices(vertices)
    f = _as_faces(faces, len(v))
    if len(v) == 0 or len(f) == 0 or len(spec.face_indices) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    idx = np.clip(np.asarray(spec.face_indices, dtype=np.int64), 0, len(f) - 1)
    tri = v[f[idx]]
    bary = np.asarray(spec.barycentric, dtype=np.float32).reshape(-1, 3)
    if len(bary) != len(tri):
        n = min(len(bary), len(tri))
        tri = tri[:n]
        bary = bary[:n]
    pts = tri[:, 0] * bary[:, 0:1] + tri[:, 1] * bary[:, 1:2] + tri[:, 2] * bary[:, 2:3]
    return np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def sample_mesh_normals_with_spec(vertices: np.ndarray, faces: np.ndarray, spec: SurfaceSampleSpec) -> np.ndarray:
    v = _as_vertices(vertices)
    f = _as_faces(faces, len(v))
    if len(v) == 0 or len(f) == 0 or len(spec.face_indices) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    idx = np.clip(np.asarray(spec.face_indices, dtype=np.int64), 0, len(f) - 1)
    tri = v[f[idx]]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(length, 1e-8)
    return np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def robust_geometry_center(points: np.ndarray, low_percentile: float = 2.0, high_percentile: float = 98.0) -> np.ndarray:
    """Return a stable geometric center for export-origin placement.

    The center is based on robust percentiles instead of raw min/max, so one bad
    spike from the solver or depth detail does not move the Blender object origin.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(3, dtype=np.float32)
    pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    lo_p = float(np.clip(low_percentile, 0.0, 49.0))
    hi_p = float(np.clip(high_percentile, 51.0, 100.0))
    lo = np.nanpercentile(pts, lo_p, axis=0)
    hi = np.nanpercentile(pts, hi_p, axis=0)
    return ((lo + hi) * 0.5).astype(np.float32)


def mesh_root_from_vertices(vertices: np.ndarray, joints: Optional[np.ndarray] = None) -> np.ndarray:
    """Return a stable body root. Prefer SMPL pelvis joint; fallback to robust bbox center."""
    if joints is not None:
        try:
            j = np.asarray(joints, dtype=np.float32).reshape(-1, 3)
            if len(j) >= 1 and np.isfinite(j[0]).all():
                return j[0].astype(np.float32)
            if len(j) >= 3 and np.isfinite(j[:3]).all():
                return np.mean(j[:3], axis=0).astype(np.float32)
        except Exception:
            pass
    v = _as_vertices(vertices)
    if len(v) == 0:
        return np.zeros(3, dtype=np.float32)
    lo = np.nanpercentile(v, 5.0, axis=0)
    hi = np.nanpercentile(v, 95.0, axis=0)
    return ((lo + hi) * 0.5).astype(np.float32)


def stabilize_vertices_by_root(vertices_seq: list[np.ndarray], joints_seq: list[Optional[np.ndarray]]) -> RootStabilizeResult:
    """Lock per-frame mesh origin to the first frame body root.

    This removes model-origin drift while preserving local body motion and pose.
    It intentionally does not force torso rotation alignment by default, because
    that can erase real turns from WHAM/4D output.
    """
    if not vertices_seq:
        empty = np.zeros((0, 0, 3), dtype=np.float32)
        return RootStabilizeResult(empty, np.zeros((0, 3), dtype=np.float32), np.zeros(3, dtype=np.float32), 0.0, 0.0, "empty")
    roots = []
    for v, j in zip(vertices_seq, joints_seq):
        roots.append(mesh_root_from_vertices(v, j))
    root_arr = np.asarray(roots, dtype=np.float32).reshape(-1, 3)
    ref = root_arr[0].copy()
    stable = []
    for v, root in zip(vertices_seq, root_arr):
        stable.append((_as_vertices(v) - root[None, :] + ref[None, :]).astype(np.float32))
    jumps = np.linalg.norm(np.diff(root_arr, axis=0), axis=1) if len(root_arr) > 1 else np.zeros((0,), dtype=np.float32)
    return RootStabilizeResult(
        vertices=np.stack(stable, axis=0).astype(np.float32),
        roots=root_arr.astype(np.float32),
        reference_root=ref.astype(np.float32),
        max_root_jump=float(np.max(jumps)) if len(jumps) else 0.0,
        median_root_jump=float(np.median(jumps)) if len(jumps) else 0.0,
        method="pelvis_or_bbox_translation_lock",
    )




def _stack_vertices_seq(vertices_seq: list[np.ndarray]) -> np.ndarray:
    if not vertices_seq:
        return np.zeros((0, 0, 3), dtype=np.float32)
    return np.stack([_as_vertices(v) for v in vertices_seq], axis=0).astype(np.float32)


def _temporal_median3(arr: np.ndarray) -> np.ndarray:
    seq = np.asarray(arr, dtype=np.float32)
    if seq.ndim != 3 or len(seq) < 3:
        return seq.copy()
    pad = np.concatenate([seq[:1], seq, seq[-1:]], axis=0)
    out = []
    for i in range(1, len(pad) - 1):
        out.append(np.median(pad[i - 1:i + 2], axis=0))
    return np.stack(out, axis=0).astype(np.float32)


def _frame_rms_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return 0.0
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=2)))) if diff.ndim == 3 else float(np.sqrt(np.mean(diff * diff)))


def smooth_structure_vertices_temporal(
    vertices_seq: list[np.ndarray],
    confidences: list[float],
    *,
    smooth_amount: float = 0.68,
    spike_guard: float = 0.55,
) -> StructureTemporalSmoothResult:
    """Reduce frame-to-frame twitching while preserving real motion.

    The structure solvers often contain high-frequency pose jitter: shoulders,
    elbows and torso can twitch even after root-lock. This routine works in the
    root-stabilized local space and applies three conservative steps:
    1) one-frame spike suppression against neighbour average;
    2) light temporal median blend;
    3) adaptive EMA smoothing that follows larger real motions faster.
    """
    arr = _stack_vertices_seq(vertices_seq)
    total = int(arr.shape[0])
    if total <= 1 or arr.size == 0 or float(smooth_amount) <= 1e-6:
        return StructureTemporalSmoothResult(arr.astype(np.float32), 0, 0.0, 0.0, 'disabled_or_too_short')
    smooth_amount = float(np.clip(smooth_amount, 0.0, 0.95))
    spike_guard = float(np.clip(spike_guard, 0.0, 0.95))
    conf = np.asarray(confidences, dtype=np.float32).reshape(-1)
    if len(conf) < total:
        conf = np.pad(conf, (0, total - len(conf)), constant_values=1.0)
    elif len(conf) > total:
        conf = conf[:total]
    out = arr.copy()

    # Step A: suppress isolated one-frame spikes.
    # Use double buffering: read from a stable source array, write to a new array.
    # Otherwise frame t-1 already modified by the loop can bias frame t and create
    # asymmetric drift.
    spikes_fixed = 0
    if total >= 3:
        src = out.copy()
        bridge_vals = []
        for t in range(1, total - 1):
            bridge = 0.5 * (src[t - 1] + src[t + 1])
            bridge_vals.append(_frame_rms_delta(src[t:t+1], bridge[None, ...]))
        med_bridge = float(np.median(bridge_vals)) if bridge_vals else 0.0
        threshold = max(1e-5, med_bridge * (2.0 + 1.8 * spike_guard))
        blend = 0.45 + 0.35 * spike_guard
        dst = src.copy()
        for t in range(1, total - 1):
            bridge = 0.5 * (src[t - 1] + src[t + 1])
            bridge_err = _frame_rms_delta(src[t:t+1], bridge[None, ...])
            conf_t = float(np.clip(conf[t], 0.0, 1.0))
            # High-confidence frames are more likely to be real fast motion, not
            # solver spikes. Raise the trigger threshold and reduce correction
            # strength for them. Low-confidence frames remain aggressively guarded.
            threshold_t = threshold * (0.85 + 0.65 * conf_t)
            blend_t = float(np.clip(blend * (1.10 - 0.55 * conf_t), 0.12, 0.85))
            if bridge_err > threshold_t:
                dst[t] = (src[t] * (1.0 - blend_t) + bridge * blend_t).astype(np.float32)
                spikes_fixed += 1
        out = dst.astype(np.float32)

    # Step B: light temporal median blend, keeps motion but damps single-frame shakes.
    med = _temporal_median3(out)
    med_blend = 0.18 + 0.18 * smooth_amount
    out = (out * (1.0 - med_blend) + med * med_blend).astype(np.float32)

    # Step C: adaptive EMA in local vertex space.
    motion = []
    for t in range(1, total):
        motion.append(_frame_rms_delta(out[t:t+1], out[t-1:t]))
    median_motion = float(np.median(motion)) if motion else 0.0
    max_motion = float(np.max(motion)) if motion else 0.0
    ema = out.copy()
    base_alpha = float(np.clip(1.0 - 0.55 * smooth_amount, 0.48, 0.82))
    for t in range(1, total):
        step = _frame_rms_delta(out[t:t+1], ema[t - 1:t])
        ratio = step / max(median_motion, 1e-6) if median_motion > 1e-6 else 1.0
        conf_t = float(np.clip(conf[t], 0.0, 1.0))
        follow_boost = min(0.16, 0.06 * ratio)
        conf_boost = max(-0.10, (conf_t - 0.5) * 0.16)
        alpha = float(np.clip(base_alpha + follow_boost + conf_boost, 0.48, 0.90))
        ema[t] = (ema[t - 1] * (1.0 - alpha) + out[t] * alpha).astype(np.float32)

    return StructureTemporalSmoothResult(
        vertices=ema.astype(np.float32),
        spikes_fixed=int(spikes_fixed),
        median_motion_rms=float(median_motion),
        max_motion_rms=float(max_motion),
        method='root_local_spike_guard_median3_adaptive_ema',
    )


def _bbox_from_alpha(alpha: Optional[np.ndarray], shape_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape_hw
    if alpha is not None:
        a = np.asarray(alpha, dtype=np.float32)
        if a.ndim == 3:
            a = a[:, :, 0]
        if a.shape[:2] != (h, w):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        yy, xx = np.where(a > 0.05)
        if len(xx) >= 64:
            margin = max(2, int(round(min(h, w) * 0.025)))
            x0 = max(0, int(xx.min()) - margin)
            x1 = min(w - 1, int(xx.max()) + margin)
            y0 = max(0, int(yy.min()) - margin)
            y1 = min(h - 1, int(yy.max()) + margin)
            if x1 > x0 and y1 > y0:
                return x0, y0, x1, y1
    return 0, 0, max(0, w - 1), max(0, h - 1)


def _depth_high_frequency(depth: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    d = np.nan_to_num(d, nan=0.0, posinf=1.0, neginf=0.0)
    d = np.clip(d, 0.0, 1.0)
    h, w = d.shape[:2]
    # Preserve clothing-scale wrinkles, suppress global body volume.
    sigma = max(2.0, min(h, w) / 90.0)
    base = cv2.GaussianBlur(d, (0, 0), sigmaX=sigma, sigmaY=sigma)
    detail = d - base
    if alpha is not None:
        a = np.asarray(alpha, dtype=np.float32)
        if a.ndim == 3:
            a = a[:, :, 0]
        if a.shape[:2] != (h, w):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        a = np.clip(a, 0.0, 1.0)
        # Avoid pushing unstable matte edges too hard.
        edge_fade = cv2.GaussianBlur((a > 0.08).astype(np.float32), (0, 0), sigmaX=max(1.0, min(h, w) / 220.0))
        detail *= np.clip(a * edge_fade, 0.0, 1.0)
    return np.nan_to_num(detail, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)




def _normal_relief_signal(normal_map: Optional[np.ndarray], shape_hw: tuple[int, int], alpha: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return a signed high-frequency relief signal from a normal map.

    Normal is allowed to lead local cloth/hair detail, but not body volume.
    The signal is normalized inside alpha when available and has a zero-ish
    mean after broad blur subtraction.
    """
    if normal_map is None:
        return None
    h, w = shape_hw
    nm = np.asarray(normal_map, dtype=np.float32)
    if nm.ndim != 3 or nm.shape[2] < 3 or h < 4 or w < 4:
        return None
    nm = np.nan_to_num(nm[:, :, :3], nan=0.0, posinf=1.0, neginf=-1.0)
    if float(np.nanmax(nm)) > 1.5:
        nm = nm / 127.5 - 1.0
    elif float(np.nanmin(nm)) >= 0.0 and float(np.nanmax(nm)) <= 1.0:
        nm = nm * 2.0 - 1.0
    nm = np.clip(nm, -1.0, 1.0)
    if nm.shape[:2] != (h, w):
        nm = cv2.resize(nm, (w, h), interpolation=cv2.INTER_LINEAR)
    length = np.linalg.norm(nm, axis=2, keepdims=True)
    nm = nm / np.maximum(length, 1e-6)
    nx = nm[:, :, 0].astype(np.float32, copy=False)
    ny = nm[:, :, 1].astype(np.float32, copy=False)
    nz = nm[:, :, 2].astype(np.float32, copy=False)
    tangent = np.clip(1.0 - np.abs(nz), 0.0, 1.0)
    # A soft directional signal: signed, broad enough for cloth folds, not a
    # pure edge detector. This is later high-passed and clamped.
    relief = 0.38 * nx - 0.30 * ny + 0.22 * tangent
    relief = cv2.GaussianBlur(relief.astype(np.float32), (0, 0), sigmaX=0.75, sigmaY=0.75)
    base_sigma = max(4.0, min(h, w) / 55.0)
    base = cv2.GaussianBlur(relief.astype(np.float32), (0, 0), sigmaX=base_sigma, sigmaY=base_sigma)
    signal = relief - base
    if alpha is not None:
        a = np.asarray(alpha, dtype=np.float32)
        if a.ndim == 3:
            a = a[:, :, 0]
        if a.shape[:2] != (h, w):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        a = np.clip(a, 0.0, 1.0)
        # Erode-like edge fade: normal at matte borders is often unstable.
        inside = (a > 0.12).astype(np.float32)
        edge_fade = cv2.GaussianBlur(inside, (0, 0), sigmaX=max(1.0, min(h, w) / 180.0))
        signal *= np.clip(a * edge_fade, 0.0, 1.0)
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _sample_image_signal(signal: np.ndarray, points: np.ndarray, alpha: Optional[np.ndarray]) -> tuple[np.ndarray, dict]:
    """Sample an image-space signal at stable mesh surface points.

    This keeps the existing conservative bbox projection. It is not a camera solve;
    it is a robust detail lookup for point-cloud art output.
    """
    sig = np.asarray(signal, dtype=np.float32)
    if sig.ndim == 3:
        sig = sig[:, :, 0]
    h, w = sig.shape[:2]
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0 or h < 4 or w < 4:
        return np.zeros((len(pts),), dtype=np.float32), {"alpha_bbox": [0, 0, max(0, w - 1), max(0, h - 1)], "vertical_axis": 1}
    x0, y0, x1, y1 = _bbox_from_alpha(alpha, (h, w))
    lo = np.nanpercentile(pts, 2.0, axis=0)
    hi = np.nanpercentile(pts, 98.0, axis=0)
    extent = np.maximum(hi - lo, 1e-6)
    vertical_axis = 1 if extent[1] >= extent[2] * 0.45 else 2
    u = (pts[:, 0] - lo[0]) / extent[0]
    v = (hi[vertical_axis] - pts[:, vertical_axis]) / extent[vertical_axis]
    px = x0 + u * max(1, x1 - x0)
    py = y0 + v * max(1, y1 - y0)
    px_i = np.clip(np.rint(px).astype(np.int64), 0, w - 1)
    py_i = np.clip(np.rint(py).astype(np.int64), 0, h - 1)
    return sig[py_i, px_i].astype(np.float32), {
        "alpha_bbox": [int(x0), int(y0), int(x1), int(y1)],
        "vertical_axis": int(vertical_axis),
    }


def _robust_unit(values: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(arr) == 0:
        return arr, 0.0
    scale = float(np.nanpercentile(np.abs(arr), 95.0))
    if not np.isfinite(scale) or scale <= 1e-6:
        return np.zeros_like(arr, dtype=np.float32), 0.0
    return np.clip(arr / scale, -1.0, 1.0).astype(np.float32), scale


def apply_normal_depth_detail_displacement(
    points: np.ndarray,
    normals: np.ndarray,
    depth: Optional[np.ndarray],
    alpha: Optional[np.ndarray],
    normal_map: Optional[np.ndarray],
    *,
    normal_strength: float,
    depth_constraint_weight: float = 0.25,
    clip_distance: float = 0.025,
) -> tuple[np.ndarray, dict]:
    """Normal-led clothing detail, with depth used only as a constraint.

    The previous depth-only layer used high-frequency depth residual directly.
    That can still carry body-volume information and stretch limbs/torso. This
    function lets a normal map decide the signed local relief, while depth only
    gates the allowed amplitude and helps reject flat/noisy areas.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0 or len(nrm) != len(pts) or normal_strength <= 1e-8:
        return pts.astype(np.float32), {"enabled": False, "reason": "missing_points_or_zero_normal_strength"}
    if depth is None:
        return pts.astype(np.float32), {"enabled": False, "reason": "missing_required_depth_constraint"}
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    h, w = d.shape[:2]
    if h < 4 or w < 4:
        return pts.astype(np.float32), {"enabled": False, "reason": "invalid_depth_constraint"}
    normal_signal = _normal_relief_signal(normal_map, (h, w), alpha)
    if normal_signal is None:
        return pts.astype(np.float32), {"enabled": False, "reason": "missing_or_invalid_normal_map"}
    depth_signal = _depth_high_frequency(d, alpha)
    normal_samples, map_meta = _sample_image_signal(normal_signal, pts, alpha)
    depth_samples, _ = _sample_image_signal(depth_signal, pts, alpha)
    normal_unit, normal_scale = _robust_unit(normal_samples)
    depth_unit, depth_scale = _robust_unit(depth_samples)
    if normal_scale <= 1e-6:
        return pts.astype(np.float32), {"enabled": False, "reason": "flat_normal_detail"}
    dw = float(np.clip(depth_constraint_weight, 0.0, 1.0))
    # Normal supplies sign and main shape. Depth only changes confidence/amplitude;
    # it never flips the normal detail direction or creates broad body volume.
    amplitude_gate = np.clip((1.0 - dw) + dw * np.abs(depth_unit), 0.20, 1.0).astype(np.float32)
    signed_detail = normal_unit * amplitude_gate
    displacement = np.clip(signed_detail * float(normal_strength), -float(clip_distance), float(clip_distance)).astype(np.float32)
    out = pts + nrm * displacement[:, None]
    return out.astype(np.float32), {
        "enabled": True,
        "method": "normal_led_depth_constrained_detail_v2",
        "normal_strength": float(normal_strength),
        "depth_constraint_weight": float(dw),
        "clip_distance": float(clip_distance),
        "normal_scale_p95": float(normal_scale),
        "depth_scale_p95": float(depth_scale),
        "displacement_min": float(np.min(displacement)) if len(displacement) else 0.0,
        "displacement_max": float(np.max(displacement)) if len(displacement) else 0.0,
        "displacement_mean_abs": float(np.mean(np.abs(displacement))) if len(displacement) else 0.0,
        **map_meta,
    }

def apply_depth_detail_displacement(
    points: np.ndarray,
    normals: np.ndarray,
    depth: Optional[np.ndarray],
    alpha: Optional[np.ndarray],
    *,
    strength: float,
    clip_distance: float = 0.055,
) -> tuple[np.ndarray, dict]:
    """Push stable mesh points along normals with high-frequency depth residual.

    This is intentionally a small detail layer. It does not use depth as the main
    body geometry source, so it will not fight the 4D/WHAM base mesh.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0 or depth is None or strength <= 1e-8:
        return pts.astype(np.float32), {"enabled": False, "reason": "missing_depth_or_zero_strength"}
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    h, w = d.shape[:2]
    if h < 4 or w < 4 or len(nrm) != len(pts):
        return pts.astype(np.float32), {"enabled": False, "reason": "invalid_depth_or_normals"}
    detail = _depth_high_frequency(d, alpha)
    x0, y0, x1, y1 = _bbox_from_alpha(alpha, (h, w))
    lo = np.nanpercentile(pts, 2.0, axis=0)
    hi = np.nanpercentile(pts, 98.0, axis=0)
    extent = np.maximum(hi - lo, 1e-6)
    # Image x maps to mesh x. Image y maps to vertical mesh y when available;
    # fallback to z if the cached model uses y as depth. This keeps the feature
    # useful for both local 4DHumans and WHAM coordinate variants.
    vertical_axis = 1 if extent[1] >= extent[2] * 0.45 else 2
    u = (pts[:, 0] - lo[0]) / extent[0]
    v = (hi[vertical_axis] - pts[:, vertical_axis]) / extent[vertical_axis]
    px = x0 + u * max(1, x1 - x0)
    py = y0 + v * max(1, y1 - y0)
    px_i = np.clip(np.rint(px).astype(np.int64), 0, w - 1)
    py_i = np.clip(np.rint(py).astype(np.int64), 0, h - 1)
    residual = detail[py_i, px_i].astype(np.float32)
    # Robustly normalize residual before applying a metric-ish clamp.
    scale = float(np.nanpercentile(np.abs(residual), 95.0))
    if not np.isfinite(scale) or scale <= 1e-6:
        return pts.astype(np.float32), {"enabled": False, "reason": "flat_depth_detail"}
    residual = np.clip(residual / scale, -1.0, 1.0)
    displacement = np.clip(residual * float(strength), -float(clip_distance), float(clip_distance)).astype(np.float32)
    out = pts + nrm * displacement[:, None]
    return out.astype(np.float32), {
        "enabled": True,
        "method": "depth_high_frequency_normal_displacement",
        "strength": float(strength),
        "clip_distance": float(clip_distance),
        "alpha_bbox": [int(x0), int(y0), int(x1), int(y1)],
        "vertical_axis": int(vertical_axis),
        "displacement_min": float(np.min(displacement)) if len(displacement) else 0.0,
        "displacement_max": float(np.max(displacement)) if len(displacement) else 0.0,
        "displacement_mean_abs": float(np.mean(np.abs(displacement))) if len(displacement) else 0.0,
    }


def validate_structure_sequence(vertices_seq: list[np.ndarray], faces_seq: list[np.ndarray], confidences: list[float]) -> dict:
    """Lightweight sequence quality gate before export."""
    total = len(vertices_seq)
    if total <= 0:
        return {"ok": False, "reason": "empty_sequence"}
    v_counts = [int(len(_as_vertices(v))) for v in vertices_seq]
    f_counts = [int(len(np.asarray(f).reshape(-1, 3))) for f in faces_seq]
    if min(v_counts) <= 0 or min(f_counts) <= 0:
        return {"ok": False, "reason": "empty_vertices_or_faces", "vertex_counts": v_counts[:8], "face_counts": f_counts[:8]}
    if len(set(v_counts)) != 1:
        return {"ok": False, "reason": "vertex_count_changed", "unique_vertex_counts": sorted(set(v_counts))[:12]}
    if len(set(f_counts)) != 1:
        return {"ok": False, "reason": "face_count_changed", "unique_face_counts": sorted(set(f_counts))[:12]}
    # Fixed-topology export writes the first frame's faces once and then only
    # time-samples vertex positions. Same face count is not enough: if a runner
    # reorders faces per frame, Blender will display broken animated surfaces.
    base_faces = np.asarray(faces_seq[0], dtype=np.int64).reshape(-1, 3)
    changed_faces: list[int] = []
    for idx, faces in enumerate(faces_seq[1:], start=1):
        curr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if curr.shape != base_faces.shape or not np.array_equal(curr, base_faces):
            changed_faces.append(int(idx + 1))
            if len(changed_faces) >= 20:
                break
    if changed_faces:
        return {
            "ok": False,
            "reason": "face_topology_changed",
            "frames": changed_faces,
            "note": "fixed-topology export requires identical face indices/order for every frame",
        }
    conf = np.asarray(confidences, dtype=np.float32)
    low_conf = np.where(conf <= 0.0)[0]
    if len(low_conf):
        return {"ok": False, "reason": "non_positive_confidence", "frames": [int(i + 1) for i in low_conf[:20]]}
    centers = []
    extents = []
    for v in vertices_seq:
        vv = _as_vertices(v)
        lo = np.nanpercentile(vv, 2.0, axis=0)
        hi = np.nanpercentile(vv, 98.0, axis=0)
        centers.append((lo + hi) * 0.5)
        extents.append(np.maximum(hi - lo, 1e-6))
    c = np.asarray(centers, dtype=np.float32)
    e = np.asarray(extents, dtype=np.float32)
    jumps = np.linalg.norm(np.diff(c, axis=0), axis=1) if total > 1 else np.zeros((0,), dtype=np.float32)
    med_extent = float(np.median(np.linalg.norm(e, axis=1)))
    max_jump = float(np.max(jumps)) if len(jumps) else 0.0
    median_jump = float(np.median(jumps)) if len(jumps) else 0.0
    # This is not a hard fail; root stabilizer can handle origin drift. Report it.
    drift_warning = bool(med_extent > 1e-6 and max_jump > med_extent * 0.45)
    return {
        "ok": True,
        "reason": "ok",
        "frames": int(total),
        "vertices": int(v_counts[0]),
        "faces": int(f_counts[0]),
        "confidence_min": float(np.min(conf)),
        "confidence_median": float(np.median(conf)),
        "max_center_jump": max_jump,
        "median_center_jump": median_jump,
        "median_extent_norm": med_extent,
        "drift_warning": drift_warning,
    }
