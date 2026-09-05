# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .foreground import constrain_by_foreground, read_alpha_foreground
from .human_parsing import SegmentationResult, run_human_parsing
from .mask_quality import classify_mask_quality, quality_to_meta


def segmentation_cache_root(cache_root: str | Path) -> Path:
    return Path(cache_root) / "segmentation"


def segmentation_frame_paths(cache_root: str | Path, frame_index: int) -> dict[str, Path]:
    root = segmentation_cache_root(cache_root)
    stem = f"frame_{int(frame_index):06d}"
    return {
        "root": root,
        "npz": root / f"{stem}_parsing_masks.npz",
        "meta": root / f"{stem}_parsing_meta.json",
        "preview": root / f"{stem}_parsing_preview.png",
    }


def segmentation_summary_path(cache_root: str | Path) -> Path:
    return segmentation_cache_root(cache_root) / "segmentation_summary.json"


def _norm(mask: np.ndarray | None, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    if mask is None:
        if shape_hw:
            return np.zeros(shape_hw, dtype=np.float32)
        return np.zeros((1, 1), dtype=np.float32)
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.size == 0:
        if shape_hw:
            return np.zeros(shape_hw, dtype=np.float32)
        return np.zeros((1, 1), dtype=np.float32)
    if float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    if shape_hw and arr.shape[:2] != shape_hw:
        arr = cv2.resize(arr, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return arr


def _save_preview(paths: dict[str, Path], result: SegmentationResult, quality_meta: dict) -> None:
    try:
        fg = _norm(result.foreground)
        h, w = fg.shape[:2]
        preview = np.zeros((h, w, 3), dtype=np.uint8)
        preview[:] = (28, 28, 28)
        preview[fg > 0.5] = (60, 60, 60)
        garment = _norm(result.garment, (h, w))
        hair = _norm(result.hair, (h, w))
        skin = _norm(result.skin, (h, w))
        preview[skin > 0.5] = (60, 120, 70)
        preview[garment > 0.5] = (190, 110, 50)
        preview[hair > 0.5] = (160, 80, 180)
        status = str(quality_meta.get("status", "unknown"))
        cv2.putText(preview, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.imwrite(str(paths["preview"]), preview)
    except Exception:
        pass


def save_segmentation_frame(cache_root: str | Path, frame_index: int, result: SegmentationResult, quality_meta: Optional[dict] = None) -> dict:
    paths = segmentation_frame_paths(cache_root, frame_index)
    paths["root"].mkdir(parents=True, exist_ok=True)
    fg_shape = None
    if result.foreground is not None:
        fg_shape = np.asarray(result.foreground).shape[:2]
    np.savez_compressed(
        paths["npz"],
        hair=_norm(result.hair, fg_shape),
        garment=_norm(result.garment, fg_shape),
        skin=_norm(result.skin, fg_shape),
        foreground=_norm(result.foreground, fg_shape),
        labels=np.asarray(result.labels if result.labels is not None else np.zeros((1, 1), dtype=np.uint8)),
    )
    quality_meta = quality_meta or {}
    meta = {
        "ok": bool(result.ok),
        "provider": str(result.provider),
        "confidence": float(result.confidence),
        "reason": str(result.reason),
        "quality": quality_meta,
        "quality_status": str(quality_meta.get("status", "unknown")),
        "meta": result.meta or {},
    }
    paths["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_preview(paths, result, quality_meta)
    return meta


def load_segmentation_frame(cache_root: str | Path, frame_index: int) -> Optional[dict]:
    paths = segmentation_frame_paths(cache_root, frame_index)
    if not paths["npz"].exists():
        return None
    try:
        data = np.load(paths["npz"])
        meta = {}
        if paths["meta"].exists():
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        return {
            "hair": np.asarray(data["hair"], dtype=np.float32),
            "garment": np.asarray(data["garment"], dtype=np.float32),
            "skin": np.asarray(data["skin"], dtype=np.float32),
            "foreground": np.asarray(data["foreground"], dtype=np.float32),
            "labels": np.asarray(data["labels"]),
            "meta": meta,
        }
    except Exception:
        return None


def _blend_missing_with_previous(result: SegmentationResult, previous: Optional[SegmentationResult], status: str) -> SegmentationResult:
    if previous is None or status == "trusted":
        return result
    # Missing / clipped / unstable masks should not delete geometry. Use previous
    # masks as temporal memory with lower weight, then save the filled result.
    if result.foreground is None:
        return result
    shape = np.asarray(result.foreground).shape[:2]
    prev_fg = _norm(previous.foreground, shape)
    prev_g = _norm(previous.garment, shape)
    prev_h = _norm(previous.hair, shape)
    cur_fg = _norm(result.foreground, shape)
    cur_g = _norm(result.garment, shape)
    cur_h = _norm(result.hair, shape)
    memory = 0.82 if status == "occluded" else (0.72 if status in {"clipped", "unstable", "partial"} else 0.55)
    result.foreground = np.maximum(cur_fg, prev_fg * memory).astype(np.float32)
    result.garment = np.maximum(cur_g, prev_g * memory).astype(np.float32)
    result.hair = np.maximum(cur_h, prev_h * memory).astype(np.float32)
    meta = dict(result.meta or {})
    meta["temporal_fill"] = True
    meta["fill_status"] = status
    result.meta = meta
    return result


def generate_segmentation_sequence_cache(
    cfg,
    cache_root: str | Path,
    *,
    project_root: str | Path,
    log: Callable[[str], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Generate per-frame human parsing cache for the whole input video.

    This is the image-driven stage. It does not generate mesh; it creates stable
    garment/hair/foreground masks plus quality metadata for every frame. Export
    then uses these cached masks to build stable dense-vertex region weights.
    """
    try:
        from depth_fusion_core import probe_video, read_video_frame_bgr
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"无法导入视频读取函数：{exc}") from exc
    input_path = str(getattr(cfg, "input_path", "") or "")
    if not input_path:
        raise RuntimeError("未选择主视频，不能生成分割缓存。")
    info = probe_video(input_path)
    total = max(1, int(info.frame_count or 1))
    provider = str(getattr(cfg, "segmentation_provider", "Auto") or "Auto")
    if provider.lower() == "off":
        raise RuntimeError("分割模型已关闭，无法生成分割缓存。")
    root = segmentation_cache_root(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    prev_result: Optional[SegmentationResult] = None
    prev_quality: Optional[dict] = None
    trusted = clipped = unstable = partial = occluded = unknown = ok_count = 0
    for frame_index in range(total):
        if progress:
            progress(frame_index, total)
        frame = read_video_frame_bgr(input_path, frame_index)
        if frame is None:
            unknown += 1
            if log:
                log(f"分割缓存：frame {frame_index+1}/{total} 读取失败，跳过。")
            continue
        result = run_human_parsing(frame, project_root=project_root, provider=provider, log=log)
        if not result.ok:
            # Stop early: no model/deps. Do not silently write fake masks.
            raise RuntimeError(result.reason)
        h, w = frame.shape[:2]
        alpha_fg = read_alpha_foreground(input_path, frame_index, (h, w))
        if alpha_fg is not None:
            result.foreground = constrain_by_foreground(result.foreground, alpha_fg, softness=0.96)
            result.garment = constrain_by_foreground(result.garment, result.foreground, softness=0.96)
            result.hair = constrain_by_foreground(result.hair, result.foreground, softness=0.96)
            meta = dict(result.meta or {})
            meta["foreground_constraint"] = "real_alpha"
            result.meta = meta
        quality = classify_mask_quality(
            foreground=result.foreground,
            garment=result.garment,
            hair=result.hair,
            prev_meta=prev_quality,
            parser_confidence=float(result.confidence),
        )
        qmeta = quality_to_meta(quality)
        result = _blend_missing_with_previous(result, prev_result, quality.status)
        if quality.status == "trusted":
            trusted += 1
        elif quality.status == "clipped":
            clipped += 1
        elif quality.status == "unstable":
            unstable += 1
        elif quality.status == "occluded":
            occluded += 1
        elif quality.status == "partial":
            partial += 1
        else:
            unknown += 1
        ok_count += 1
        save_segmentation_frame(cache_root, frame_index, result, qmeta)
        prev_result = result
        prev_quality = qmeta
        if log and (frame_index == 0 or (frame_index + 1) % 25 == 0 or frame_index + 1 == total):
            log(f"分割缓存：{frame_index+1}/{total} status={quality.status} conf={quality.confidence:.2f}")
    summary = {
        "ok": bool(ok_count > 0),
        "total_frames": int(total),
        "cached_frames": int(ok_count),
        "provider": provider,
        "trusted": int(trusted),
        "clipped": int(clipped),
        "unstable": int(unstable),
        "partial": int(partial),
        "occluded": int(occluded),
        "unknown": int(unknown),
        "message": f"逐帧分割缓存完成：{ok_count}/{total} 帧，trusted={trusted}, clipped={clipped}, unstable={unstable}, occluded={occluded}",
    }
    segmentation_summary_path(cache_root).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress:
        progress(total, total)
    return summary


def segmentation_cache_summary(cache_root: str | Path) -> dict:
    p = segmentation_summary_path(cache_root)
    if not p.exists():
        return {"ok": False, "message": "未生成逐帧分割缓存"}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"ok": False, "message": "分割缓存摘要不可读"}


def ensure_reference_segmentation(
    cfg,
    cache_root: str | Path,
    frame_index: int,
    *,
    project_root: str | Path,
    log: Callable[[str], None] | None = None,
) -> Optional[dict]:
    # Backward-compatible exact-frame helper. It does not use frame+1 fallback.
    if not bool(getattr(cfg, "segmentation_enabled", False)):
        return None
    cached = load_segmentation_frame(cache_root, frame_index)
    if cached is not None:
        return cached
    try:
        from depth_fusion_core import read_video_frame_bgr
        frame_bgr = read_video_frame_bgr(getattr(cfg, "input_path", ""), int(frame_index))
    except Exception:
        frame_bgr = None
    if frame_bgr is None:
        if log:
            log("分割缓存：无法读取当前帧，本次 Garment/Hair 将保持 Body Only。")
        return None
    result = run_human_parsing(
        frame_bgr,
        project_root=project_root,
        provider=str(getattr(cfg, "segmentation_provider", "Auto") or "Auto"),
        log=log,
    )
    if not result.ok:
        if log:
            log("画面分割不可用，本次 Garment/Hair 将保持 Body Only：" + result.reason)
        return None
    h, w = frame_bgr.shape[:2]
    alpha_fg = read_alpha_foreground(str(getattr(cfg, "input_path", "") or ""), int(frame_index), (h, w))
    if alpha_fg is not None:
        result.foreground = constrain_by_foreground(result.foreground, alpha_fg, softness=0.96)
        result.garment = constrain_by_foreground(result.garment, result.foreground, softness=0.96)
        result.hair = constrain_by_foreground(result.hair, result.foreground, softness=0.96)
    quality = classify_mask_quality(
        foreground=result.foreground,
        garment=result.garment,
        hair=result.hair,
        parser_confidence=float(result.confidence),
    )
    meta = save_segmentation_frame(cache_root, frame_index, result, quality_to_meta(quality))
    if log:
        log(f"画面分割缓存已生成：frame={frame_index}, provider={meta.get('provider')}, status={meta.get('quality_status')}")
    return load_segmentation_frame(cache_root, frame_index)
