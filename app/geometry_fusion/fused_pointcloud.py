# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hand_pipeline.hand_cache import HandFrame
from pointcloud_pipeline.pointcloud_filter import deterministic_limit_indices
from structure_pipeline.structure_cache import StructureFrame, structure_frame_paths
from .hand_fusion import build_hand_points
from .mesh_sampling import sample_mesh_surface
from .template_alignment import align_template_to_visible_points, save_alignment_json


def _median_color(colors: np.ndarray) -> np.ndarray:
    cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(cols) == 0:
        return np.array([210, 210, 210], dtype=np.uint8)
    return np.median(cols.astype(np.float32), axis=0).clip(0, 255).astype(np.uint8)


def _nearest_visible_colors(template_points: np.ndarray, visible_points: np.ndarray, visible_colors: np.ndarray, *, max_candidates: int = 4096) -> np.ndarray:
    tp = np.asarray(template_points, dtype=np.float32).reshape(-1, 3)
    vp = np.asarray(visible_points, dtype=np.float32).reshape(-1, 3)
    vc = np.asarray(visible_colors, dtype=np.uint8).reshape(-1, 3)
    if len(tp) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if len(vp) == 0 or len(vc) == 0:
        return np.tile(_median_color(vc), (len(tp), 1)).astype(np.uint8)
    n = min(len(vp), len(vc))
    vp, vc = vp[:n], vc[:n]
    if n > int(max_candidates):
        rng = np.random.default_rng(2000003 + len(tp) + n)
        idx = np.sort(rng.choice(n, size=int(max_candidates), replace=False))
        vp, vc = vp[idx], vc[idx]
    out = np.empty((len(tp), 3), dtype=np.uint8)
    chunk = 1024
    for start in range(0, len(tp), chunk):
        sub = tp[start:start + chunk]
        d2 = ((sub[:, None, :] - vp[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argmin(d2, axis=1)
        out[start:start + len(sub)] = vc[nearest]
    return out


def _source_debug_colors(source_id: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    src = np.asarray(source_id, dtype=np.uint8).reshape(-1)
    out = np.zeros((len(src), 3), dtype=np.uint8)
    # 0 visible, 1 body template, 2 hand, 3 normal relief/future
    palette = {
        0: (80, 190, 255),
        1: (255, 150, 60),
        2: (220, 80, 255),
        3: (110, 255, 150),
    }
    for sid, color in palette.items():
        out[src == sid] = color
    if fallback is not None:
        fb = np.asarray(fallback, dtype=np.uint8).reshape(-1, 3)
        mask = ~np.isin(src, list(palette.keys()))
        out[mask] = fb[: len(out)][mask]
    return out


def _save_fused_cache(cache_root: Path | None, frame_index: int, points: np.ndarray, source_id: np.ndarray, confidence: np.ndarray, meta: dict) -> None:
    if cache_root is None:
        return
    try:
        root = Path(cache_root) / "fused"
        root.mkdir(parents=True, exist_ok=True)
        stem = f"frame_{int(frame_index):06d}"
        np.save(root / f"{stem}_fused_vertices.npy", np.asarray(points, dtype=np.float32))
        np.save(root / f"{stem}_fused_source.npy", np.asarray(source_id, dtype=np.uint8))
        np.save(root / f"{stem}_fused_confidence.npy", np.asarray(confidence, dtype=np.float32))
        with open(root / f"{stem}_fused_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def build_fused_body_points(
    visible_points: np.ndarray,
    visible_colors: np.ndarray,
    visible_source: np.ndarray,
    visible_confidence: np.ndarray,
    structure_frame: StructureFrame | None,
    hand_frame: HandFrame | None = None,
    *,
    frame_index: int = 0,
    cache_root: Path | None = None,
    max_points: int = 200_000,
    template_sample_ratio: float = 0.45,
    template_confidence: float = 0.55,
    align_strength: float = 1.0,
    color_mode: str = "xyz",
    hand_enabled: bool = True,
    hand_sample_ratio: float = 0.12,
    hand_confidence: float = 0.70,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Merge visible depth points, cached structure mesh and cached hand points.

    No model is launched here. Missing structure/hand cache is a controlled
    downgrade, not an export failure. PLY source_id stays stable:
    0=visible depth, 1=structure template, 2=hand cache, 3=reserved normal relief.
    """
    vp = np.asarray(visible_points, dtype=np.float32).reshape(-1, 3)
    vc = np.asarray(visible_colors, dtype=np.uint8).reshape(-1, 3)[: len(vp)]
    vs = np.asarray(visible_source, dtype=np.uint8).reshape(-1)[: len(vp)]
    vf = np.asarray(visible_confidence, dtype=np.float32).reshape(-1)[: len(vp)]
    max_points = max(1, int(max_points))
    meta = {
        "frame_index": int(frame_index),
        "mode": "fused_body_hand" if hand_enabled else "fused_body",
        "visible_points": int(len(vp)),
        "template_points": 0,
        "hand_points": 0,
        "final_points": int(len(vp)),
        "downgraded": False,
        "downgrade_reasons": [],
    }

    pieces_pts = [vp]
    pieces_cols = [vc]
    pieces_src = [vs]
    pieces_conf = [vf]
    align_meta: dict | None = None

    if structure_frame is None or not structure_frame.available:
        meta["downgraded"] = True
        meta["downgrade_reasons"].append("structure_cache_missing_or_invalid")
    else:
        vertices = np.asarray(structure_frame.vertices, dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(structure_frame.faces, dtype=np.int64).reshape(-1, 3)
        if len(vertices) == 0 or len(faces) == 0:
            meta["downgraded"] = True
            meta["downgrade_reasons"].append("empty_structure_mesh")
        else:
            aligned_vertices, align_meta = align_template_to_visible_points(
                vertices,
                vp,
                strength=float(align_strength),
                camera=structure_frame.camera,
                cache_root=Path(cache_root) if cache_root is not None else None,
            )
            template_count = int(max(0, min(max_points, round(max_points * float(template_sample_ratio)))))
            if template_count > 0:
                template_points = sample_mesh_surface(aligned_vertices, faces, template_count, seed=7000003 + int(frame_index))
                template_source = np.full(len(template_points), 1, dtype=np.uint8)
                color_key = str(color_mode or "xyz").lower()
                if color_key == "source_debug":
                    template_colors = _source_debug_colors(template_source)
                elif color_key in {"rgb", "visible_rgb"}:
                    template_colors = _nearest_visible_colors(template_points, vp, vc)
                else:
                    template_colors = np.zeros((len(template_points), 3), dtype=np.uint8)
                template_conf = np.full(len(template_points), float(template_confidence), dtype=np.float32)
                pieces_pts.append(template_points.astype(np.float32))
                pieces_cols.append(template_colors.astype(np.uint8))
                pieces_src.append(template_source)
                pieces_conf.append(template_conf)
                meta.update({
                    "structure_model": str(structure_frame.model_name),
                    "structure_confidence": float(structure_frame.confidence),
                    "template_points": int(len(template_points)),
                    "alignment": align_meta,
                })
            else:
                meta["downgraded"] = True
                meta["downgrade_reasons"].append("template_sample_count_zero")

    if hand_enabled:
        hand_budget = int(max(0, min(max_points, round(max_points * float(hand_sample_ratio)))))
        hand_points, hand_meta = build_hand_points(
            hand_frame,
            align_meta=align_meta,
            max_points=hand_budget,
            frame_index=frame_index,
            require_alignment=True,
        )
        meta["hand"] = hand_meta
        if len(hand_points) > 0 and bool(hand_meta.get("available", False)):
            hand_source = np.full(len(hand_points), 2, dtype=np.uint8)
            color_key = str(color_mode or "xyz").lower()
            if color_key == "source_debug":
                hand_colors = _source_debug_colors(hand_source)
            elif color_key in {"rgb", "visible_rgb"}:
                hand_colors = _nearest_visible_colors(hand_points, vp, vc)
            else:
                hand_colors = np.zeros((len(hand_points), 3), dtype=np.uint8)
            hand_conf = np.full(len(hand_points), float(hand_confidence) * float(hand_meta.get("confidence", 1.0)), dtype=np.float32)
            pieces_pts.append(hand_points.astype(np.float32))
            pieces_cols.append(hand_colors.astype(np.uint8))
            pieces_src.append(hand_source)
            pieces_conf.append(hand_conf)
            meta["hand_points"] = int(len(hand_points))
        elif hand_frame is not None:
            meta["downgraded"] = True
            meta["downgrade_reasons"].append(str(hand_meta.get("reason", "hand_cache_unusable")))

    pts = np.concatenate(pieces_pts, axis=0).astype(np.float32) if pieces_pts else np.zeros((0, 3), dtype=np.float32)
    cols = np.concatenate(pieces_cols, axis=0).astype(np.uint8) if pieces_cols else np.zeros((0, 3), dtype=np.uint8)
    src = np.concatenate(pieces_src, axis=0).astype(np.uint8) if pieces_src else np.zeros((0,), dtype=np.uint8)
    conf = np.concatenate(pieces_conf, axis=0).astype(np.float32) if pieces_conf else np.zeros((0,), dtype=np.float32)

    limiter = deterministic_limit_indices(len(pts), max_points, frame_index + 880000)
    if limiter is not None:
        pts, cols, src, conf = pts[limiter], cols[limiter], src[limiter], conf[limiter]
    if str(color_mode or "xyz").lower() == "source_debug":
        cols = _source_debug_colors(src, cols)

    meta["final_points"] = int(len(pts))
    unique, counts = np.unique(src, return_counts=True) if len(src) else (np.array([], dtype=np.uint8), np.array([], dtype=np.int64))
    meta["source_counts"] = {str(int(k)): int(v) for k, v in zip(unique, counts)}

    if cache_root is not None:
        try:
            if align_meta is not None:
                paths = structure_frame_paths(Path(cache_root), int(frame_index))
                save_alignment_json(paths["alignment"], meta)
            _save_fused_cache(Path(cache_root), int(frame_index), pts, src, conf, meta)
        except Exception:
            pass
    return pts, cols, src, conf, meta


def build_fused_body_arrays(
    output_path: Path,
    visible_points: np.ndarray,
    visible_colors: np.ndarray,
    visible_source: np.ndarray,
    visible_confidence: np.ndarray,
    structure_frame: StructureFrame | None,
    hand_frame: HandFrame | None = None,
    *,
    frame_index: int = 0,
    cache_root: Path | None = None,
    max_points: int = 200_000,
    binary_ply: bool = True,
    template_sample_ratio: float = 0.45,
    template_confidence: float = 0.55,
    align_strength: float = 1.0,
    color_mode: str = "xyz",
    hand_enabled: bool = True,
    hand_sample_ratio: float = 0.12,
    hand_confidence: float = 0.70,
) -> tuple[int, dict, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    pts, cols, src, conf, meta = build_fused_body_points(
        visible_points,
        visible_colors,
        visible_source,
        visible_confidence,
        structure_frame,
        hand_frame,
        frame_index=frame_index,
        cache_root=cache_root,
        max_points=max_points,
        template_sample_ratio=template_sample_ratio,
        template_confidence=template_confidence,
        align_strength=align_strength,
        color_mode=color_mode,
        hand_enabled=hand_enabled,
        hand_sample_ratio=hand_sample_ratio,
        hand_confidence=hand_confidence,
    )
    # Per-frame debug output removed from final source; return arrays for USDA writer.
    return int(len(pts)), meta, (pts, cols, src, conf)
