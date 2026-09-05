# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import json
import math
import subprocess
import shutil
import sys

import cv2

from depth_fusion_core import (
    APP_NAME, APP_STYLESHEET, AUX_MODEL_SPECS, BUILTIN_PRESETS, CACHE_FORMAT_VERSION, CacheEntry, Callable,
    DEFAULT_MATANYONE_MODEL_PATH, DEFAULT_MATTING_MASK_DIR, ENCODER_MODES, ExternalDepthFusionState,
    FfmpegRawVideoWriter, JobConfig, MAX_SAFE_LONG_SIDE_HINT, MODEL_IDS, NORMALIZE_MODES, Optional,
    PROJECT_CACHE_DIR, PROJECT_DIR, PROJECT_HF_HOME, PROJECT_HF_HUB, PROJECT_LOG_DIR, PROJECT_MODELS_DIR,
    Path, PngSequenceWriter, QApplication, QCheckBox, QColor, QComboBox, QCursor, QDialog, QDragEnterEvent,
    QDropEvent, QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QImage, QLabel, QLineEdit,
    QLinearGradient, QMainWindow, QMessageBox, QObject, QPainter, QPainterPath, QPen, QPixmap,
    QPlainTextEdit, QPointF, QProgressBar, QPushButton, QRectF, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QStackedWidget, QThread, QTimer, QVBoxLayout, QWidget, Qt, RESOURCES_DIR, RotatingFileHandler,
    SAFE_DEFAULT_LONG_SIDE, SAFE_DEFAULT_PROCESS_RES, Signal, TONE_RANGE_BASE_CENTERS,
    TONE_RANGE_MIN_GAPS, TONE_RANGE_WIDTHS, VIDEO_EXTS,
    VideoInfo, apply_anti_banding, apply_curve_lut, apply_depth_output_grade, apply_depth_smoothing,
    apply_detail_boost, apply_input_adjustments_bgr, apply_input_adjustments_from_cfg, apply_levels,
    apply_normal_depth_fusion, apply_subject_background_fill, apply_tone_ranges, atexit, bgr_to_pixmap,
    build_alignment_subject_mask, build_configured_subject_mask, build_curve_lut, build_sliding_ranges,
    clear_all_cache, clear_cache_entry, clear_cache_older_than, clear_memory_model_cache,
    compute_mask_depth_stats, contextlib, cuda_total_memory_gb, current_event_log_path,
    curve_points_are_identity, cv2, dataclass, default_output_path, default_structure_output_dir,
    depth_affine_to_reference, depth_percentile_range, describe_real_alpha_source, directory_size_bytes,
    ensure_aux_model_file, ensure_builtin_preset_files, estimate_vram_gb, even_int, event_exception,
    event_log, extract_subject_alpha_from_cutout_frame, extract_subject_alpha_from_depth_frame,
    extract_valid_alpha_from_external_depth_frame, fast_bilateral_float, ffmpeg_has_encoder, format_bytes,
    format_seconds, frame_alpha_path, frame_cache_paths, frame_cache_root, frame_refined_depth_path,
    fuse_base_depth_with_external_reference, gc, get_cached_aux_model, get_cached_model, hashlib, importlib,
    init_event_logger, install_global_event_hooks,
    io, is_direct_depth_video_workflow, is_likely_cutout_frame, is_local_model_ready, is_scene_cut, json,
    lightweight_environment_report, list_cache_entries, load_alpha_mask_for_depth, local_aux_model_file,
    logging, make_base_gray_for_levels, make_diagnostic_preview_bgr, map_aux_frame_index,
    model_cache_dir_from_id, mux_original_audio, normal_guided_depth_refine, normalize_curve_points,
    normalize_depth, np, os, partial_output_path, partial_png_sequence_dir, png_sequence_output_dir,
    postprocess_subject_mask, prepare_da3_input_frame, probe_video, qInstallMessageHandler, queue,
    quiet_third_party_output, read_external_depth_reference, read_external_depth_reference_with_alpha,
    read_external_subject_mask, read_masked_depth_video_frame, read_video_frame_alpha01,
    read_video_frame_bgr, refine_depth_with_human_crop, remove_partial_output, render_depth_frame,
    render_depth_frame_16bit, render_depth_gray_float, resolve_device_mode, robust_global_range,
    save_npy_safely, scaled_size_from_long_side, scene_cut_score, scene_cut_signature, short_error_message,
    shutil, stabilize_subject_mask_temporal, structure_cache_root, subject_bbox_from_mask, subprocess, sys,
    temporal_align_depth_by_mask, temporal_ema_masked, threading, time, tone_range_reference_bands,
    traceback, trim_cuda_allocator_cache, try_load_npy, update_anchor_stats, verbose_third_party_output,
    video_has_real_alpha, warmup_three_models,
)
# Private helpers are not part of the explicit public import block, so import
# the ones this module needs directly.
from depth_fusion_core import _resize_mask_like, _external_depth_gray01
from depth_pipeline.cache_state import (
    alpha_cache_signature,
    cache_entry_matches,
    depth_cache_signature,
    normal_cache_signature,
    record_cache_error,
    record_cache_frame,
    summarize_cache_validation,
    validate_geometry_cache,
    write_pipeline_state,
)
from common.cache import frame_stem
from common.paths import pointcloud_export_root
from geometry_fusion.mesh_sampling import sample_mesh_surface
from geometry_fusion.structure_detail import (
    make_surface_sample_spec,
    sample_mesh_normals_with_spec,
    sample_mesh_surface_with_spec,
    robust_geometry_center,
    stabilize_vertices_by_root,
    smooth_structure_vertices_temporal,
    validate_structure_sequence,
)
from geometry_fusion.stable_dense_mesh import (
    apply_shell_offsets,
    build_dense_mesh_template,
    conservative_shell_offsets,
    evaluate_dense_normals,
    evaluate_dense_vertices,
    soft_region_weights,
    validate_dense_template_winding,
)
from structure_pipeline.structure_cache import load_structure_frame
from mesh_usd_exporter import UsdLayeredMeshSequenceWriter, UsdMeshSequenceWriter, UsdPointSequenceWriter, deterministic_limit_indices
from segmentation_pipeline.segmentation_cache import ensure_reference_segmentation, generate_segmentation_sequence_cache, segmentation_cache_summary
from segmentation_pipeline.mesh_region_weights import build_mask_guided_region_weights, build_sequence_mask_guided_region_weights


def pointcloud_config_signature(cfg: JobConfig) -> str:
    payload = json.dumps(getattr(cfg, "__dict__", {}), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

def write_export_state(root: Path, **data) -> None:
    try:
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        with (root / "export_state.json").open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass

def cleanup_stale_output_frames(*a, **k): return None
def clean_output_dir_frames(*a, **k): return None
def existing_frame_is_complete(*a, **k): return False
def validate_pointcloud_outputs(*a, **k): return {}
def write_alpha_debug_cache(alpha: np.ndarray, output_npy: Path, output_png8: Path | None = None) -> None:
    save_npy_safely(Path(output_npy), np.asarray(alpha, dtype=np.float32))

def voxel_downsample_points(points, *a, **k): return points
def remove_statistical_outliers(points, *a, **k): return points
class PointCloudTemporalState: pass

def build_visible_frame_from_job(*a, **k):
    raise RuntimeError("当前清理版已删除旧可见深度点云流程。")
def write_visible_points_from_job(*a, **k):
    raise RuntimeError("当前清理版已删除旧可见深度点云流程。")
def build_fused_body_arrays(*a, **k):
    raise RuntimeError("当前清理版已删除旧融合点云流程。")
def load_hand_frame(*a, **k): return None


def _resize_gray01_for_worker(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 3:
        out = out[..., 0]
    if out.size == 0:
        return np.zeros(shape_hw, dtype=np.float32)
    if float(np.nanmax(out)) > 1.5:
        out = out / 255.0
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0)
    th, tw = shape_hw
    if out.shape[:2] != (th, tw):
        out = cv2.resize(out, (tw, th), interpolation=cv2.INTER_LINEAR)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class _SequentialBgrFrameReader:
    """Fast sequential frame reader for per-frame reference videos.

    The previous helper opened VideoCapture and seeked for every frame. That is
    robust, but very slow during export. This reader keeps one capture alive and
    only seeks when the requested index jumps backwards or far ahead.
    """

    def __init__(self, path_text: str) -> None:
        self.path = str(path_text or "").strip()
        self.suffix = Path(self.path).suffix.lower() if self.path else ""
        self._cap: Optional[cv2.VideoCapture] = None
        self._image: Optional[np.ndarray] = None
        self._last_index = -1
        self._last_frame: Optional[np.ndarray] = None
        self._is_image = self.suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        if self._is_image and self.path:
            self._image = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)

    def _open(self) -> Optional[cv2.VideoCapture]:
        if not self.path or not Path(self.path).is_file():
            return None
        if self._cap is None:
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                cap.release()
                return None
            self._cap = cap
            self._last_index = -1
            self._last_frame = None
        return self._cap

    def read(self, frame_index: int) -> Optional[np.ndarray]:
        if self._is_image:
            return None if self._image is None else self._image.copy()
        target = max(0, int(frame_index))
        if self._last_frame is not None and target == self._last_index:
            return self._last_frame.copy()
        cap = self._open()
        if cap is None:
            return read_video_frame_bgr(self.path, target)
        if target < self._last_index or target > self._last_index + 8:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self._last_index = target - 1
            self._last_frame = None
        frame = None
        ok = False
        while self._last_index < target:
            ok, frame = cap.read()
            self._last_index += 1
            if not ok or frame is None:
                frame = None
                break
        if self._last_index == target and frame is None:
            ok, frame = cap.read()
            if ok and frame is not None:
                self._last_index = target
        if frame is None:
            return read_video_frame_bgr(self.path, target)
        self._last_frame = frame.copy()
        return frame

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None


class _SequentialRgbaFrameReader:
    """Single-process ffmpeg RGBA reader for alpha-capable videos.

    This avoids launching one ffmpeg process per frame when reading alpha masks.
    """

    def __init__(self, path_text: str) -> None:
        self.path = str(path_text or "").strip()
        self.suffix = Path(self.path).suffix.lower() if self.path else ""
        self._image: Optional[np.ndarray] = None
        self._proc: Optional[subprocess.Popen] = None
        self._last_index = -1
        self._last_rgba: Optional[np.ndarray] = None
        self.width = 0
        self.height = 0
        self.stride = 0
        self.available = False
        self._is_image = self.suffix in {".png", ".tif", ".tiff", ".webp"}
        if not self.path or not Path(self.path).is_file():
            return
        if self._is_image:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] >= 4:
                self._image = img
                self.available = True
            return
        try:
            has_alpha = bool(video_has_real_alpha(self.path))
            self.available = bool(has_alpha and shutil.which("ffmpeg"))
            if self.available:
                info = probe_video(self.path)
                self.width = int(info.width)
                self.height = int(info.height)
                self.stride = self.width * self.height * 4
        except Exception:
            self.available = False

    def _open(self) -> bool:
        if not self.available or self._is_image:
            return False
        if self._proc is not None and self._proc.stdout is not None:
            return True
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg or self.width <= 0 or self.height <= 0:
            return False
        self._proc = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", self.path, "-an", "-sn", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._last_index = -1
        self._last_rgba = None
        return self._proc.stdout is not None

    def _restart(self) -> bool:
        self.close()
        return self._open()

    def _read_next_rgba(self) -> Optional[np.ndarray]:
        if self._proc is None or self._proc.stdout is None or self.stride <= 0:
            return None
        buf = self._proc.stdout.read(self.stride)
        if len(buf) < self.stride:
            return None
        self._last_index += 1
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width, 4)).copy()
        self._last_rgba = arr
        return arr

    def read_rgba(self, frame_index: int) -> Optional[np.ndarray]:
        target = max(0, int(frame_index))
        if self._is_image:
            return None if self._image is None else self._image.copy()
        if not self.available:
            return None
        if self._last_rgba is not None and target == self._last_index:
            return self._last_rgba.copy()
        if target < self._last_index:
            if not self._restart():
                return None
        elif not self._open():
            return None
        while self._last_index < target:
            arr = self._read_next_rgba()
            if arr is None:
                return None
        return None if self._last_rgba is None else self._last_rgba.copy()

    def read_alpha01(self, frame_index: int, shape_hw: tuple[int, int]) -> Optional[np.ndarray]:
        rgba = self.read_rgba(frame_index)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
            return None
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        coverage = float(np.mean(alpha > 0.01)) if alpha.size else 0.0
        if coverage <= 0.0005 or coverage >= 0.9995:
            return None
        return _resize_gray01_for_worker(alpha, shape_hw)

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdout is not None:
                    self._proc.stdout.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=1)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        self._last_index = -1
        self._last_rgba = None


def _load_frame_cache_0based_first(loader, cache_root: Path, frame_index: int):  # noqa: ANN001, ANN201
    idx = int(frame_index)
    return loader(cache_root, idx)


# 防止补丁应用顺序导致 depth_fusion_core 的公开别名没有进入本模块。
# 预览/原图/外部参考都走这个稳定读取入口，不能再在 worker 里直接依赖一个
# 可能未导入的全局名。
try:
    read_video_frame_bgr  # type: ignore[name-defined]
except NameError:
    def read_video_frame_bgr(path_text: str, frame_index: int):  # noqa: ANN201
        path = str(path_text or "").strip()
        if not path or not Path(path).is_file():
            return None
        if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img

        def _open_cap():  # noqa: ANN202
            cap_obj = cv2.VideoCapture(path)
            if not cap_obj.isOpened():
                cap_obj.release()
                return None
            return cap_obj

        idx = max(0, int(frame_index))
        cap = _open_cap()
        if cap is None:
            return None
        try:
            from depth_fusion_core import probe_video
            total = probe_video(path).frame_count
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if total > 0:
                idx = min(idx, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
            if fps > 1e-3:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, idx * 1000.0 / fps))
                ok, frame = cap.read()
                if ok and frame is not None:
                    return frame
        finally:
            cap.release()

        for back in (8, 24, 60, 120, 240):
            start = max(0, idx - back)
            cap = _open_cap()
            if cap is None:
                return None
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                frame = None
                ok = False
                for _ in range(start, idx + 1):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                if ok and frame is not None:
                    return frame
            finally:
                cap.release()
        return None


def _mask_to_hw(mask: Optional[np.ndarray], shape_hw: tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    return _resize_mask_like(mask, shape_hw, cv2.INTER_LINEAR)


def _resolve_subject_masks(cfg: JobConfig, cache_root: Optional[Path], depth: np.ndarray, frame_index: int) -> tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    shape_hw = np.asarray(depth).shape[:2]
    if is_direct_depth_video_workflow(cfg):
        _direct_depth, direct_alpha = read_masked_depth_video_frame(cfg, frame_index, shape_hw, require_alpha=True)
        if direct_alpha is None:
            zero = np.zeros(shape_hw, dtype=np.float32)
            return None, zero, zero
        raw = _mask_to_hw(direct_alpha, shape_hw)
        align = build_configured_subject_mask(depth, raw, cfg)
        # Final matte is exactly the depth video's alpha, not alpha*alpha and
        # not an intersection with the source video's alpha.
        return raw, align, raw

    raw_subject_mask = read_external_subject_mask(cfg, frame_index, shape_hw)
    strict_cutout = bool(getattr(cfg, "input_cutout_mask_enabled", False))
    ref_alpha = None
    if getattr(cfg, "external_depth_enabled", False):
        try:
            _ref_depth, ref_alpha = read_external_depth_reference_with_alpha(cfg, frame_index, shape_hw)
            ref_alpha = _mask_to_hw(ref_alpha, shape_hw) if ref_alpha is not None else None
        except Exception:
            ref_alpha = None

    def _final_matte(main_mask: np.ndarray) -> np.ndarray:
        matte = np.clip(np.asarray(main_mask, dtype=np.float32), 0.0, 1.0)
        if ref_alpha is not None and float(np.nanmax(ref_alpha)) > 0.02:
            matte = np.clip(matte * np.clip(ref_alpha, 0.0, 1.0), 0.0, 1.0)
        return matte.astype(np.float32)

    if raw_subject_mask is not None:
        raw_subject_mask = _mask_to_hw(raw_subject_mask, shape_hw)
        align_mask = build_configured_subject_mask(depth, raw_subject_mask, cfg)
        render_matte = _final_matte(raw_subject_mask)
        return raw_subject_mask, align_mask, render_matte
    if strict_cutout:
        zero = np.zeros(shape_hw, dtype=np.float32)
        return None, zero, zero
    raw_subject_mask = load_alpha_mask_for_depth(cache_root, frame_index, shape_hw) if (cache_root is not None and getattr(cfg, "matting_enabled", False)) else None
    align_mask = build_configured_subject_mask(depth, raw_subject_mask, cfg)
    if raw_subject_mask is not None:
        render_matte = _final_matte(raw_subject_mask)
    else:
        render_matte = _final_matte(align_mask)
    return raw_subject_mask, align_mask, render_matte


def _depth_cache_path_for_validation(root: Path, frame_index: int) -> Path:
    return frame_cache_paths(root, int(frame_index))[0]


def _normal_cache_path_for_validation(root: Path, frame_index: int) -> Path:
    return frame_cache_paths(root, int(frame_index))[1]


def _alpha_cache_path_for_validation(root: Path, frame_index: int) -> Path:
    return frame_alpha_path(root, int(frame_index))


def _log_cache_validation(log: Callable[[str], None], validation: dict) -> None:
    try:
        summary = summarize_cache_validation(validation)
        if summary:
            log(f"缓存校验: {summary}")
    except Exception:
        pass


def ensure_matanyone_alpha_cache(cfg: JobConfig, cache_root: Path, total_frames: int, log: Callable[[str], None]) -> None:
    """Ensure alpha cache exists without pretending to run an unavailable matte model.

    The original UI option is named MatAnyone, but the packaged project may not
    contain a working MatAnyone runner. For batch export we still need a stable
    alpha cache for point clouds, so this function writes a reusable alpha cache
    from the best available source:
      1) external mask/cutout video when configured;
      2) existing alpha cache with matching signature;
      3) 旧深度 depth subject-mask fallback.
    """
    sig = alpha_cache_signature(cfg)
    total = max(0, int(total_frames))
    wrote = 0
    reused = 0
    failed = 0
    if total <= 0:
        return
    log("Alpha 缓存: 校验/生成人物 alpha。")
    for frame_no in range(total):
        alpha_path = frame_alpha_path(cache_root, frame_no)
        if cache_entry_matches(cache_root, "alpha", frame_no, alpha_path, sig, allow_legacy=False):
            arr = try_load_npy(alpha_path)
            if arr is not None:
                reused += 1
                continue
        try:
            depth_path, _normal_path = frame_cache_paths(cache_root, frame_no)
            depth = try_load_npy(depth_path)
            if depth is None:
                record_cache_error(cache_root, "alpha", frame_no, sig, "missing depth cache")
                failed += 1
                continue
            shape_hw = np.asarray(depth).shape[:2]
            alpha = read_external_subject_mask(cfg, frame_no, shape_hw)
            if alpha is None:
                alpha = build_configured_subject_mask(np.asarray(depth, dtype=np.float32), None, cfg)
            alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
            write_alpha_debug_cache(alpha, alpha_path, None)
            record_cache_frame(cache_root, "alpha", frame_no, alpha_path, sig, alpha, status="ok", extra={"source": "external_mask_or_depth_fallback"})
            wrote += 1
        except Exception as exc:  # noqa: BLE001
            record_cache_error(cache_root, "alpha", frame_no, sig, short_error_message(str(exc)))
            failed += 1
    log(f"Alpha 缓存完成: 复用 {reused} 帧，写入 {wrote} 帧，失败 {failed} 帧。")


def is_structure_xyz_export_config(cfg: JobConfig) -> bool:
    """Return True when the job should use the Mesh/Shell exporter.

    Kept outside the worker class so UI routing does not need to instantiate the
    legacy depth worker just to decide which exporter to run.
    """
    mode = str(getattr(cfg, "pointcloud_mode", "") or "").strip()
    if mode not in {"fused_body", "fused_body_hand", "structure_xyz"}:
        return False
    return bool(
        getattr(cfg, "mesh_export_enabled", False)
        or getattr(cfg, "detail_mesh_export_enabled", False)
        or getattr(cfg, "pointcloud_usd_sequence", False)
    )


def _subset_mesh_layer_from_weights(
    faces: np.ndarray,
    vertex_weights: np.ndarray,
    *,
    threshold: float = 0.18,
    fallback_top_fraction: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a stable sub-mesh from per-vertex garment/hair weights.

    Returns remapped faces and the source vertex indices needed to write the
    animated layer. Topology is fixed because weights are computed once on the
    canonical dense vertex IDs.
    """
    face_arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    weights = np.asarray(vertex_weights, dtype=np.float32).reshape(-1)
    if len(face_arr) == 0 or len(weights) == 0:
        return np.zeros((0, 3), dtype=np.int32), np.zeros((0,), dtype=np.int64), {"faces": 0, "vertices": 0, "threshold": float(threshold)}
    valid = (face_arr >= 0).all(axis=1) & (face_arr < len(weights)).all(axis=1)
    face_arr = face_arr[valid]
    if len(face_arr) == 0:
        return np.zeros((0, 3), dtype=np.int32), np.zeros((0,), dtype=np.int64), {"faces": 0, "vertices": 0, "threshold": float(threshold)}
    fw = np.mean(weights[face_arr], axis=1)
    mask = fw >= float(threshold)
    if not np.any(mask) and float(np.nanmax(fw)) > 1e-6:
        frac = float(np.clip(fallback_top_fraction, 0.005, 0.20))
        keep_n = max(1, int(round(len(fw) * frac)))
        cutoff = np.partition(fw, max(0, len(fw) - keep_n))[max(0, len(fw) - keep_n)]
        mask = fw >= cutoff
    selected = face_arr[mask]
    if len(selected) == 0:
        return np.zeros((0, 3), dtype=np.int32), np.zeros((0,), dtype=np.int64), {"faces": 0, "vertices": 0, "threshold": float(threshold)}
    used = np.unique(selected.reshape(-1)).astype(np.int64)
    remap = np.full((len(weights),), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    remapped = remap[selected].astype(np.int32)
    return remapped.reshape(-1, 3), used, {
        "faces": int(len(remapped)),
        "vertices": int(len(used)),
        "threshold": float(threshold),
        "mean_face_weight": float(np.mean(fw[mask])) if np.any(mask) else 0.0,
        "max_face_weight": float(np.max(fw)) if len(fw) else 0.0,
    }



def _boundary_edges_from_faces(faces: np.ndarray) -> list[tuple[int, int]]:
    """Return oriented boundary edges from a remapped triangle layer."""
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0:
        return []
    counts: dict[tuple[int, int], int] = {}
    oriented: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b, c in f:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u <= v else (v, u)
            counts[key] = counts.get(key, 0) + 1
            oriented.setdefault(key, (u, v))
    return [oriented[k] for k, c in counts.items() if c == 1]


def _build_silhouette_side_layer(shell_faces: np.ndarray, shell_used_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a stable side-wall layer for the visible garment/hair boundary.

    The shell surface and side wall are exported as separate prims. The side wall
    duplicates each boundary vertex once; frame writing computes the duplicate
    positions by expanding the boundary away from the local layer center. This
    gives Blender a visible outline instead of a body-region patch that lies on
    top of the skin.
    """
    faces = np.asarray(shell_faces, dtype=np.int64).reshape(-1, 3)
    used = np.asarray(shell_used_indices, dtype=np.int64).reshape(-1)
    edges = _boundary_edges_from_faces(faces)
    if len(faces) == 0 or len(used) == 0 or not edges:
        return np.zeros((0, 3), dtype=np.int32), np.zeros((0,), dtype=np.int64), {"faces": 0, "vertices": 0, "boundary_edges": 0}
    boundary_local = np.asarray(sorted({idx for e in edges for idx in e}), dtype=np.int64)
    boundary_set = {int(v): i for i, v in enumerate(boundary_local.tolist())}
    side_faces: list[tuple[int, int, int]] = []
    n = len(boundary_local)
    for a, b in edges:
        if int(a) not in boundary_set or int(b) not in boundary_set:
            continue
        ai = boundary_set[int(a)]
        bi = boundary_set[int(b)]
        ao = n + ai
        bo = n + bi
        side_faces.append((ai, bi, bo))
        side_faces.append((ai, bo, ao))
    if not side_faces:
        return np.zeros((0, 3), dtype=np.int32), np.zeros((0,), dtype=np.int64), {"faces": 0, "vertices": 0, "boundary_edges": len(edges)}
    boundary_source = used[np.clip(boundary_local, 0, max(0, len(used) - 1))].astype(np.int64)
    return np.asarray(side_faces, dtype=np.int32).reshape(-1, 3), boundary_source, {
        "faces": int(len(side_faces)),
        "vertices": int(n * 2),
        "boundary_edges": int(len(edges)),
        "boundary_vertices": int(n),
    }


def _body_height_for_layer(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return 0.0
    lo = np.nanpercentile(pts, 2.0, axis=0)
    hi = np.nanpercentile(pts, 98.0, axis=0)
    return float(max(1e-6, np.max(hi - lo)))


def _vertical_axis_for_layer(points: np.ndarray) -> int:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return 1
    lo = np.nanpercentile(pts, 2.0, axis=0)
    hi = np.nanpercentile(pts, 98.0, axis=0)
    return int(np.argmax(hi - lo))


def _make_silhouette_side_vertices(
    shell_vertices_all: np.ndarray,
    normals_all: np.ndarray,
    boundary_source_indices: np.ndarray,
    *,
    expand_ratio: float,
    normal_ratio: float,
) -> np.ndarray:
    """Return [inner_boundary, outer_boundary] vertices for a side-wall prim."""
    all_pts = np.asarray(shell_vertices_all, dtype=np.float32).reshape(-1, 3)
    all_n = np.asarray(normals_all, dtype=np.float32).reshape(-1, 3)
    src = np.asarray(boundary_source_indices, dtype=np.int64).reshape(-1)
    if len(all_pts) == 0 or len(src) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    src = np.clip(src, 0, max(0, len(all_pts) - 1))
    inner = all_pts[src].astype(np.float32)
    nrm = all_n[src].astype(np.float32) if len(all_n) == len(all_pts) else np.zeros_like(inner)
    h = _body_height_for_layer(all_pts)
    center = np.nanmedian(inner, axis=0).astype(np.float32)
    vertical_axis = _vertical_axis_for_layer(all_pts)
    radial = inner - center[None, :]
    radial[:, vertical_axis] = 0.0
    radial_len = np.linalg.norm(radial, axis=1, keepdims=True)
    fallback = nrm.copy()
    fallback[:, vertical_axis] *= 0.25
    fallback_len = np.linalg.norm(fallback, axis=1, keepdims=True)
    fallback = fallback / np.maximum(fallback_len, 1e-8)
    radial = np.where(radial_len > h * 0.012, radial / np.maximum(radial_len, 1e-8), fallback)
    direction = radial * 0.78 + fallback * 0.22
    dlen = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / np.maximum(dlen, 1e-8)
    expand = max(0.0, float(expand_ratio)) * h
    normal_boost = max(0.0, float(normal_ratio)) * h
    outer = inner + direction * expand + fallback * normal_boost
    return np.concatenate([inner, outer.astype(np.float32)], axis=0).astype(np.float32)


class DepthWorker(QObject):
    progress = Signal(int, int)
    log = Signal(str)
    stage_signal = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, cfg: JobConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _log(self, text: str) -> None:
        event_log(text, channel="EXPORT")
        self.log.emit(text)

    def _stage(self, text: str) -> None:
        self.stage_signal.emit(text)
        self._log(text)

    def run(self) -> None:
        try:
            self._run_impl()
            self.finished.emit(self.cfg.output_path)
        except Exception as exc:  # noqa: BLE001
            event_exception("导出任务失败", exc, output_path=getattr(self.cfg, "output_path", ""))
            try:
                if getattr(self.cfg, "cache_enabled", True):
                    write_pipeline_state(frame_cache_root(self.cfg), self.cfg, status="failed", extra={"error": short_error_message(str(exc))})
            except Exception:
                pass
            remove_partial_output(self.cfg.output_path)
            if "out of memory" in str(exc).lower() or ("cuda" in str(exc).lower() and "memory" in str(exc).lower()):
                clear_memory_model_cache(self._log)
            tb = traceback.format_exc()
            self.failed.emit(f"{exc}\n\n{tb}")

    def _is_structure_xyz_workflow(self) -> bool:
        # Kept for backward compatibility. New UI code routes mesh jobs to
        # MeshExportWorker directly; legacy calls still dispatch safely here.
        return is_structure_xyz_export_config(self.cfg)

    def _run_structure_xyz_pointcloud_impl(self) -> None:
        """Export structure-cache mesh samples as animated XYZ point cloud.

        Target pipeline: structure cache supplies stable skinned body geometry;
        dense mesh and garment/hair shells are generated from fixed topology.
        External Depth/法线 are not part of the main workflow.
        """
        cap = cv2.VideoCapture(self.cfg.input_path)
        if not cap.isOpened():
            raise RuntimeError("无法打开输入视频。")
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            if fps <= 1e-3:
                fps = 25.0
            from depth_fusion_core import probe_video
            total = probe_video(self.cfg.input_path).frame_count
        finally:
            cap.release()
        if total <= 0:
            total = 1

        cache_root = structure_cache_root(self.cfg)
        structure_dir = cache_root / "structure"
        if not structure_dir.exists() or not any(structure_dir.glob("frame_*_smpl_vertices.npy")):
            raise RuntimeError(
                "缺少 structure cache。当前主流程不再用深度视频降级；请先在 3D 构建页点击“生成结构缓存”。"
            )

        pointcloud_root = pointcloud_export_root(self.cfg.input_path, self.cfg.output_path)
        pointcloud_root.mkdir(parents=True, exist_ok=True)
        pointcloud_meta_dir = pointcloud_root / "frame_meta"
        pointcloud_meta_dir.mkdir(parents=True, exist_ok=True)
        pointcloud_config_sig = pointcloud_config_signature(self.cfg)
        usd_path = pointcloud_root / "pointcloud_combined.usda"
        mesh_low_path = pointcloud_root / "mesh_body_stable_low.usda"
        mesh_garment_path = pointcloud_root / "mesh_garment.usda"
        mesh_hair_path = pointcloud_root / "mesh_hair.usda"
        mesh_combined_path = pointcloud_root / "mesh_combined.usda"
        # Remove legacy single detail file too, so users do not confuse it with the final split outputs.
        mesh_detail_path = pointcloud_root / "mesh_detail_shell.usda"
        for _old_path in (usd_path, mesh_low_path, mesh_garment_path, mesh_hair_path, mesh_combined_path, mesh_detail_path):
            try:
                if _old_path.exists():
                    _old_path.unlink()
            except Exception:
                pass

        write_export_state(
            pointcloud_root,
            status="running",
            config_signature=pointcloud_config_sig,
            input_path=self.cfg.input_path,
            mode="structure_xyz",
            resume_active=False,
        )
        self._stage("阶段: 导出结构 XYZ 点云")
        self._log("主流程：4D/WHAM 提供基础人体，Dense Mesh + Garment/Hair Shell 提供外层结构，最后可选导出稳定点云；不使用外部 Depth/法线。")
        self._log(f"结构缓存: {structure_dir}")
        self._log(f"USDA 点云输出: {usd_path}")
        self._log(f"低模 Body Mesh 输出: {mesh_low_path}")
        self._log(f"Garment Mesh 输出: {mesh_garment_path}")
        self._log(f"Hair Mesh 输出: {mesh_hair_path}")
        self._log(f"Combined Mesh 输出: {mesh_combined_path}")

        max_points = int(getattr(self.cfg, "pointcloud_usd_max_points", 120000) or 120000)
        pointcloud_export_enabled = bool(getattr(self.cfg, "pointcloud_usd_sequence", True))
        mesh_export_enabled = bool(getattr(self.cfg, "mesh_export_enabled", True))
        detail_mesh_export_enabled = bool(getattr(self.cfg, "detail_mesh_export_enabled", True))
        if not (pointcloud_export_enabled or mesh_export_enabled or detail_mesh_export_enabled):
            raise RuntimeError("没有选择任何输出内容。请至少勾选低模 Mesh、细节 Mesh 或稳定点云之一。")
        writer: Optional[UsdPointSequenceWriter] = None
        if pointcloud_export_enabled:
            writer = UsdPointSequenceWriter(
                usd_path,
                fps=float(fps or 24.0),
                start_frame=1,
                end_frame=max(1, int(total)),
                point_width=float(getattr(self.cfg, "pointcloud_usd_point_width", 0.008)),
                max_points_per_frame=max_points,
                label="structure_xyz_pointcloud",
                include_colors=False,
                include_source_id=False,
                include_confidence=False,
            )
        low_mesh_writer: Optional[UsdMeshSequenceWriter] = None
        detail_mesh_writer: Optional[UsdLayeredMeshSequenceWriter] = None  # combined layered mesh writer
        garment_mesh_writer: Optional[UsdLayeredMeshSequenceWriter] = None
        hair_mesh_writer: Optional[UsdLayeredMeshSequenceWriter] = None
        garment_vertex_indices: Optional[np.ndarray] = None
        hair_vertex_indices: Optional[np.ndarray] = None
        garment_layer_faces: Optional[np.ndarray] = None
        hair_layer_faces: Optional[np.ndarray] = None
        garment_silhouette_faces: Optional[np.ndarray] = None
        hair_silhouette_faces: Optional[np.ndarray] = None
        garment_silhouette_indices: Optional[np.ndarray] = None
        hair_silhouette_indices: Optional[np.ndarray] = None
        dense_template = None
        dense_point_indices: Optional[np.ndarray] = None
        metas: list[dict] = []
        errors = 0
        try:
            # Step 1: preload and validate structure cache. The structure mesh is
            # the base skinned body; shell layers are generated from fixed topology.
            self._stage("阶段: 1/4 读取并质检结构缓存")
            self._log("正在预加载结构网格，并检查帧完整性...")
            all_vertices: list[np.ndarray] = []
            all_faces: list[np.ndarray] = []
            all_joints: list[Optional[np.ndarray]] = []
            all_confidences: list[float] = []
            all_models: list[str] = []
            all_cameras: list[dict] = []
            missing_indices: list[int] = []

            for frame_index in range(total):
                if self._cancel:
                    raise RuntimeError("任务已取消。")
                structure_frame = _load_frame_cache_0based_first(load_structure_frame, cache_root, frame_index)
                if structure_frame is None or not structure_frame.available:
                    errors += 1
                    missing_indices.append(int(frame_index))
                    continue
                vertices = np.asarray(structure_frame.vertices, dtype=np.float32).reshape(-1, 3)
                faces = np.asarray(structure_frame.faces, dtype=np.int64).reshape(-1, 3)
                joints = np.asarray(structure_frame.joints, dtype=np.float32).reshape(-1, 3) if structure_frame.joints is not None else None
                all_vertices.append(vertices)
                all_faces.append(faces)
                all_joints.append(joints)
                all_confidences.append(float(structure_frame.confidence or 1.0))
                all_models.append(str(structure_frame.model_name))
                all_cameras.append(dict(structure_frame.camera or {}))
                self.progress.emit(min(frame_index + 1, total), max(1, total))

            if missing_indices:
                preview = ", ".join(str(i + 1) for i in missing_indices[:12])
                more = "..." if len(missing_indices) > 12 else ""
                raise RuntimeError(
                    f"结构缓存缺帧：共 {len(missing_indices)} / {total} 帧缺少 SMPL 网格。"
                    f"缺失帧(1-based)：{preview}{more}。请重新生成结构缓存，导出阶段不再用最近帧定格填补，避免生成假动画。"
                )
            if len(all_vertices) != int(total):
                raise RuntimeError(f"结构缓存帧数不完整：读取 {len(all_vertices)} / {total}。")

            quality = validate_structure_sequence(all_vertices, all_faces, all_confidences)
            if not bool(quality.get("ok", False)):
                raise RuntimeError(f"结构缓存质检失败：{quality.get('reason', 'unknown')}。请重新生成结构缓存。")
            if bool(quality.get("drift_warning", False)):
                self._log(
                    "检测到结构中心存在明显跳变；将启用身体根节点锁定，避免 4D/WHAM 原点漂移带动点云抖动。"
                )

            # Step 2: lock model-origin drift. This keeps local body pose, but
            # removes per-frame root/origin wandering from 4D estimators.
            self._stage("阶段: 2/4 Root 稳定 + 固定采样准备")
            self.progress.emit(0, 0)
            self._log("正在执行 Root Stabilizer：锁定 pelvis/body root，修正原点漂移...")
            stable_result = stabilize_vertices_by_root(all_vertices, all_joints)
            stable_vertices = stable_result.vertices
            self._log(
                f"Root Stabilizer: {stable_result.method}; "
                f"root_jump median={stable_result.median_root_jump:.5f}, max={stable_result.max_root_jump:.5f}"
            )

            temporal_body_smooth = max(0.0, min(0.95, float(getattr(self.cfg, "pointcloud_temporal_center_smooth", 0.0))))
            temporal_spike_guard = max(0.0, min(0.95, float(getattr(self.cfg, "pointcloud_temporal_scale_smooth", 0.0))))
            if temporal_body_smooth > 1e-6:
                self._log("正在执行结构时序去抖：抑制单帧抽搐，并对 root 锁定后的局部人体做轻度时序平滑...")
                smooth_result = smooth_structure_vertices_temporal(
                    [stable_vertices[i] for i in range(len(stable_vertices))],
                    all_confidences,
                    smooth_amount=temporal_body_smooth,
                    spike_guard=temporal_spike_guard,
                )
                stable_vertices = smooth_result.vertices
                self._log(
                    f"结构去抖完成：{smooth_result.method}; "
                    f"spikes_fixed={smooth_result.spikes_fixed}, "
                    f"motion_rms median={smooth_result.median_motion_rms:.5f}, max={smooth_result.max_motion_rms:.5f}"
                )

            hand_enabled = bool(getattr(self.cfg, "pointcloud_hand_enabled", False))
            hand_ratio = max(0.0, min(0.5, float(getattr(self.cfg, "pointcloud_hand_sample_ratio", 0.12))))
            hand_count_target = int(max_points * hand_ratio) if hand_enabled else 0
            if hand_enabled and hand_count_target >= max_points:
                hand_count_target = max(0, max_points - 100)
            body_count_target = max(1, max_points - hand_count_target)

            # Step 3: fixed surface sampling. Generate once, then reuse the same
            # face+barycentric locations every frame. This fixes temporal shimmer.
            self._log("正在生成固定表面采样点：后续每帧复用同一组 face/barycentric，避免随机闪烁...")
            sample_spec = make_surface_sample_spec(stable_vertices[0], all_faces[0], body_count_target, seed=9100003)
            if len(sample_spec.face_indices) <= 0:
                raise RuntimeError("结构网格无法生成表面采样点，请检查 faces/vertices 是否有效。")

            # Blender object origin should sit at the geometry center, not at the
            # solver/pelvis root. Keep a fixed origin from the first stabilized
            # body frame, subtract it from every frame's local points, and write it
            # back as the parent Xform translate in USD. This makes imported rotate/
            # scale operations pivot around the visual center of the point object.
            export_origin = robust_geometry_center(sample_mesh_surface_with_spec(stable_vertices[0], all_faces[0], sample_spec))
            if writer is not None:
                writer.xform_translate = export_origin.astype(np.float32)
            self._log(
                "Blender 原点已设置到首帧几何中心："
                f"({export_origin[0]:.5f}, {export_origin[1]:.5f}, {export_origin[2]:.5f})；"
                "导出的点坐标会转为相对该原点。"
            )

            dense_segments = int(np.clip(int(getattr(self.cfg, "mesh_dense_segments", 2) or 2), 1, 3))
            garment_shell_enabled = bool(getattr(self.cfg, "garment_shell_enabled", False))
            hair_shell_enabled = bool(getattr(self.cfg, "hair_shell_enabled", False))
            garment_shell_offset = float(np.clip(float(getattr(self.cfg, "garment_shell_offset", 0.006)), 0.0, 0.025))
            hair_shell_offset = float(np.clip(float(getattr(self.cfg, "hair_shell_offset", 0.010)), 0.0, 0.040))
            fixed_garment_w: Optional[np.ndarray] = None
            fixed_hair_w: Optional[np.ndarray] = None

            if mesh_export_enabled:
                low_mesh_writer = UsdMeshSequenceWriter(
                    mesh_low_path,
                    faces=all_faces[0],
                    fps=float(fps or 24.0),
                    start_frame=1,
                    end_frame=max(1, int(total)),
                    label="stable_body_mesh",
                    mesh_name="body_low_fixed_topology",
                    xform_translate=export_origin.astype(np.float32),
                )
                self._log("低模稳定 Mesh 已启用：直接导出 root 锁定 + 时序去抖后的 SMPL/结构网格，vertex ID 保持稳定。")

            if detail_mesh_export_enabled or pointcloud_export_enabled:
                dense_template = build_dense_mesh_template(all_faces[0], segments=dense_segments)
                winding_check = validate_dense_template_winding(stable_vertices[0], all_faces[0], dense_template)
                if not bool(winding_check.get("ok", True)):
                    raise RuntimeError(
                        "Dense Mesh 子面朝向校验失败："
                        f"flipped={winding_check.get('flipped_faces', 0)}, "
                        f"min_dot={winding_check.get('min_dot', 0.0):.6f}。"
                        "为避免 Blender 中法向错误，已停止导出。"
                    )
                self._log(
                    f"固定拓扑 dense mesh 已生成：segments={dense_segments}, "
                    f"vertices={len(dense_template.base_face_indices)}, faces={len(dense_template.faces)}；"
                    "共享边界顶点，后续每帧复用同一套 face_id/barycentric，点 ID 不会乱走。"
                )
                self._log(
                    "Dense Mesh 面朝向校验通过："
                    f"checked={winding_check.get('checked_faces', 0)}, "
                    f"flipped={winding_check.get('flipped_faces', 0)}, "
                    f"min_dot={winding_check.get('min_dot', 1.0):.6f}；"
                    "USD 不写显式 normals，Blender 会按 face winding 自动计算法向。"
                )
                ref_dense_vertices = evaluate_dense_vertices(stable_vertices[0], all_faces[0], dense_template)
                fixed_region = soft_region_weights(ref_dense_vertices)
                mask_region_meta = {"mask_used": False, "source": "body_only_no_segmentation"}
                region_cache_path = Path(cache_root) / "region_weights.npz"
                if bool(getattr(self.cfg, "segmentation_enabled", True)):
                    mask_garment_w, mask_hair_w, mask_region_meta = build_sequence_mask_guided_region_weights(
                        ref_dense_vertices,
                        fixed_region,
                        str(cache_root),
                        int(total),
                        camera=(all_cameras[0] if all_cameras else None),
                    )
                    if not bool(mask_region_meta.get("mask_used", False)):
                        segmentation_ref = ensure_reference_segmentation(
                            self.cfg,
                            cache_root,
                            0,
                            project_root=PROJECT_DIR,
                            log=self._log,
                        )
                        mask_garment_w, mask_hair_w, mask_region_meta = build_mask_guided_region_weights(
                            ref_dense_vertices,
                            fixed_region,
                            segmentation_ref,
                            camera=(all_cameras[0] if all_cameras else None),
                        )
                else:
                    mask_garment_w = np.zeros((len(ref_dense_vertices),), dtype=np.float32)
                    mask_hair_w = np.zeros((len(ref_dense_vertices),), dtype=np.float32)
                    mask_region_meta = {"mask_used": False, "source": "segmentation_disabled"}

                if bool(mask_region_meta.get("mask_used", False)):
                    fixed_garment_w = np.asarray(mask_garment_w, dtype=np.float32).reshape(-1)
                    fixed_hair_w = np.asarray(mask_hair_w, dtype=np.float32).reshape(-1)
                    try:
                        np.savez_compressed(
                            region_cache_path,
                            garment=fixed_garment_w,
                            hair=fixed_hair_w,
                            meta=json.dumps(mask_region_meta, ensure_ascii=False, default=str),
                        )
                    except Exception as exc:
                        self._log(f"Region weight cache 写入失败，可继续导出：{short_error_message(str(exc))}")
                    self._log(
                        "Garment/Hair 区域权重：已使用 Human Parsing / Hair mask；"
                        f"provider={mask_region_meta.get('provider', 'unknown')}, "
                        f"garment_ratio={mask_region_meta.get('garment_mask_vertex_ratio', 0.0):.3f}, "
                        f"hair_ratio={mask_region_meta.get('hair_mask_vertex_ratio', 0.0):.3f}。"
                    )
                    self._log(f"Region weights 已缓存：{region_cache_path}")
                else:
                    fixed_garment_w = np.zeros((len(ref_dense_vertices),), dtype=np.float32)
                    fixed_hair_w = np.zeros((len(ref_dense_vertices),), dtype=np.float32)
                    self._log("未找到可用 Human Parsing 分割缓存/结果：本次明确按 Body Only 导出，不再生成假的 Garment/Hair。")
                self._log("Garment/Hair 区域权重已固定到 dense vertex ID；预览和导出共用同一类 region weight。")
                has_mask_region = bool(mask_region_meta.get("mask_used", False))
                has_garment_region = has_mask_region and bool(garment_shell_enabled) and float(np.max(fixed_garment_w)) > 1e-6
                has_hair_region = has_mask_region and bool(hair_shell_enabled) and float(np.max(fixed_hair_w)) > 1e-6
                has_real_region = bool(has_garment_region or has_hair_region)
                if detail_mesh_export_enabled:
                    garment_layer_faces = np.zeros((0, 3), dtype=np.int32)
                    hair_layer_faces = np.zeros((0, 3), dtype=np.int32)
                    garment_vertex_indices = np.zeros((0,), dtype=np.int64)
                    hair_vertex_indices = np.zeros((0,), dtype=np.int64)
                    garment_silhouette_faces = np.zeros((0, 3), dtype=np.int32)
                    hair_silhouette_faces = np.zeros((0, 3), dtype=np.int32)
                    garment_silhouette_indices = np.zeros((0,), dtype=np.int64)
                    hair_silhouette_indices = np.zeros((0,), dtype=np.int64)
                    garment_layer_meta = {"faces": 0, "vertices": 0, "disabled": not bool(garment_shell_enabled)}
                    hair_layer_meta = {"faces": 0, "vertices": 0, "disabled": not bool(hair_shell_enabled)}
                    garment_silhouette_meta = {"faces": 0, "vertices": 0}
                    hair_silhouette_meta = {"faces": 0, "vertices": 0}
                    if has_garment_region:
                        garment_layer_faces, garment_vertex_indices, garment_layer_meta = _subset_mesh_layer_from_weights(
                            dense_template.faces,
                            fixed_garment_w,
                            threshold=0.18,
                            fallback_top_fraction=0.0,
                        )
                        garment_silhouette_faces, garment_silhouette_indices, garment_silhouette_meta = _build_silhouette_side_layer(
                            garment_layer_faces,
                            garment_vertex_indices if garment_vertex_indices is not None else np.zeros((0,), dtype=np.int64),
                        )
                    if has_hair_region:
                        hair_layer_faces, hair_vertex_indices, hair_layer_meta = _subset_mesh_layer_from_weights(
                            dense_template.faces,
                            fixed_hair_w,
                            threshold=0.15,
                            fallback_top_fraction=0.0,
                        )
                        hair_silhouette_faces, hair_silhouette_indices, hair_silhouette_meta = _build_silhouette_side_layer(
                            hair_layer_faces,
                            hair_vertex_indices if hair_vertex_indices is not None else np.zeros((0,), dtype=np.int64),
                        )

                    layers = {
                        "Body": {"faces": dense_template.faces, "color": (0.62, 0.65, 0.70)},
                    }
                    if has_garment_region and garment_layer_faces is not None and len(garment_layer_faces):
                        layers["GarmentShell"] = {"faces": garment_layer_faces, "color": (0.18, 0.48, 0.95)}
                    if has_garment_region and garment_silhouette_faces is not None and len(garment_silhouette_faces):
                        layers["GarmentSilhouette"] = {"faces": garment_silhouette_faces, "color": (0.10, 0.36, 0.82)}
                    if has_hair_region and hair_layer_faces is not None and len(hair_layer_faces):
                        layers["HairShell"] = {"faces": hair_layer_faces, "color": (0.90, 0.42, 0.16)}
                    if has_hair_region and hair_silhouette_faces is not None and len(hair_silhouette_faces):
                        layers["HairSilhouette"] = {"faces": hair_silhouette_faces, "color": (0.72, 0.25, 0.08)}
                    detail_mesh_writer = UsdLayeredMeshSequenceWriter(
                        mesh_combined_path,
                        layers=layers,
                        fps=float(fps or 24.0),
                        start_frame=1,
                        end_frame=max(1, int(total)),
                        label="combined_layered_mesh_sequence",
                        xform_translate=export_origin.astype(np.float32),
                        note="Body plus parsed Garment/Hair shell and silhouette side-wall layers. Missing parsing exports Body only.",
                    )
                    if has_garment_region and garment_layer_faces is not None and len(garment_layer_faces):
                        garment_layers = {"GarmentShell": {"faces": garment_layer_faces, "color": (0.18, 0.48, 0.95)}}
                        if garment_silhouette_faces is not None and len(garment_silhouette_faces):
                            garment_layers["GarmentSilhouette"] = {"faces": garment_silhouette_faces, "color": (0.10, 0.36, 0.82)}
                        garment_mesh_writer = UsdLayeredMeshSequenceWriter(
                            mesh_garment_path,
                            layers=garment_layers,
                            fps=float(fps or 24.0),
                            start_frame=1,
                            end_frame=max(1, int(total)),
                            label="garment_layered_sequence",
                            xform_translate=export_origin.astype(np.float32),
                            note="Garment shell plus expanded silhouette side-wall from human parsing mask.",
                        )
                    if has_hair_region and hair_layer_faces is not None and len(hair_layer_faces):
                        hair_layers = {"HairShell": {"faces": hair_layer_faces, "color": (0.90, 0.42, 0.16)}}
                        if hair_silhouette_faces is not None and len(hair_silhouette_faces):
                            hair_layers["HairSilhouette"] = {"faces": hair_silhouette_faces, "color": (0.72, 0.25, 0.08)}
                        hair_mesh_writer = UsdLayeredMeshSequenceWriter(
                            mesh_hair_path,
                            layers=hair_layers,
                            fps=float(fps or 24.0),
                            start_frame=1,
                            end_frame=max(1, int(total)),
                            label="hair_layered_sequence",
                            xform_translate=export_origin.astype(np.float32),
                            note="Hair shell plus expanded silhouette side-wall from human parsing mask.",
                        )
                    self._log(
                        "分层 Combined Mesh 导出已启用："
                        f"body faces={len(dense_template.faces)}, "
                        f"garment shell={garment_layer_meta.get('faces', 0)}, "
                        f"garment silhouette={garment_silhouette_meta.get('faces', 0)}, "
                        f"hair shell={hair_layer_meta.get('faces', 0)}, "
                        f"hair silhouette={hair_silhouette_meta.get('faces', 0)}。"
                    )

            if hand_enabled:
                self._log(f"结构+手部模式：body={body_count_target} 点，hand_target={hand_count_target} 点，避免导出末端随机裁切。")

            self._stage("阶段: 3/4 准备 Dense Mesh / Shell")
            self.progress.emit(0, 0)
            self._log(
                f"Shell 参数：dense_segments={dense_segments}, "
                f"garment={'开' if garment_shell_enabled else '关'}({garment_shell_offset:.3f}m), "
                f"hair={'开' if hair_shell_enabled else '关'}({hair_shell_offset:.3f}m)。"
            )

            hand_surface_specs: dict[str, object] = {}
            hand_vertex_indices: dict[str, np.ndarray] = {}
            if hand_enabled and hand_count_target > 0:
                per_side_target = max(1, hand_count_target // 2)
                for spec_frame_index in range(total):
                    hand_frame0 = _load_frame_cache_0based_first(load_hand_frame, cache_root, spec_frame_index)
                    if hand_frame0 is None or not hand_frame0.available:
                        continue
                    for side, hv, hf, seed_base in (
                        ("left", hand_frame0.left_vertices, hand_frame0.left_faces, 9710003),
                        ("right", hand_frame0.right_vertices, hand_frame0.right_faces, 9720003),
                    ):
                        if hv is None:
                            continue
                        hv_arr0 = np.asarray(hv, dtype=np.float32).reshape(-1, 3)
                        if hf is not None:
                            spec = make_surface_sample_spec(hv_arr0, np.asarray(hf, dtype=np.int64).reshape(-1, 3), per_side_target, seed=seed_base)
                            if len(spec.face_indices) > 0:
                                hand_surface_specs[side] = spec
                                continue
                        idx = deterministic_limit_indices(len(hv_arr0), per_side_target, seed_base)
                        if idx is None:
                            idx = np.arange(len(hv_arr0), dtype=np.int64)
                        hand_vertex_indices[side] = np.asarray(idx, dtype=np.int64)
                    if hand_surface_specs or hand_vertex_indices:
                        self._log("手部固定采样已生成：后续帧复用同一组 hand face/barycentric 或顶点索引，避免手部闪烁。")
                        break
                if not hand_surface_specs and not hand_vertex_indices:
                    self._log("手部模式未找到 hand cache：导出将只写身体结构点。")

            # Step 4: write point cloud sequence.
            self._stage("阶段: 4/4 写入最终 Mesh / 点云")
            self.progress.emit(0, max(1, total))
            self._log("开始写入稳定 Mesh / 细节 Shell / 稳定点云...")
            temporal_detail_smooth = max(0.0, min(0.95, float(getattr(self.cfg, "pointcloud_temporal_depth_smooth", 0.0))))
            prev_body_points_temporal: Optional[np.ndarray] = None
            prev_hand_points_temporal: Optional[np.ndarray] = None
            if temporal_detail_smooth > 1e-6:
                self._log(f"最终点云时序平滑已启用：body/detail smooth={temporal_detail_smooth:.2f}；将减少衣服细节闪烁和抽搐感。")

            for frame_index in range(total):
                if self._cancel:
                    raise RuntimeError("任务已取消。")

                vertices = stable_vertices[frame_index]
                # validate_structure_sequence() has already enforced exact fixed topology.
                # Reuse frame 0 faces so mesh preview/export and dense template always share one canonical topology.
                faces = all_faces[0]
                if low_mesh_writer is not None:
                    low_mesh_writer.add_frame(frame_index, (np.asarray(vertices, dtype=np.float32).reshape(-1, 3) - export_origin[None, :]).astype(np.float32))

                detail_vertices = None
                base_detail_vertices = None
                detail_normals = None
                detail_region_meta = {"enabled": False}
                if dense_template is not None:
                    detail_vertices = evaluate_dense_vertices(vertices, faces, dense_template)
                    base_detail_vertices = np.asarray(detail_vertices, dtype=np.float32).copy()
                    detail_normals = evaluate_dense_normals(vertices, faces, dense_template)
                    if fixed_garment_w is not None and len(fixed_garment_w) == len(detail_vertices):
                        garment_region_w = fixed_garment_w
                    else:
                        garment_region_w = np.zeros((len(detail_vertices),), dtype=np.float32)
                    if fixed_hair_w is not None and len(fixed_hair_w) == len(detail_vertices):
                        hair_region_w = fixed_hair_w
                    else:
                        hair_region_w = np.zeros((len(detail_vertices),), dtype=np.float32)
                    garment_w = garment_region_w if garment_shell_enabled else np.zeros((len(detail_vertices),), dtype=np.float32)
                    hair_w = hair_region_w if hair_shell_enabled else np.zeros((len(detail_vertices),), dtype=np.float32)
                    shell_offsets = conservative_shell_offsets(
                        detail_vertices,
                        garment_w,
                        hair_w,
                        garment_offset=garment_shell_offset,
                        hair_offset=hair_shell_offset,
                    )
                    shell_w = (shell_offsets > 1e-7).astype(np.float32)
                    detail_vertices = apply_shell_offsets(detail_vertices, detail_normals, shell_offsets)
                    detail_region_meta = {
                        "enabled": True,
                        "dense_segments": int(dense_template.segments),
                        "dense_vertices": int(len(detail_vertices)),
                        "dense_faces": int(len(dense_template.faces)),
                        "garment_shell": bool(garment_shell_enabled),
                        "hair_shell": bool(hair_shell_enabled),
                        "garment_offset": float(garment_shell_offset),
                        "hair_offset": float(hair_shell_offset),
                    }
                    points = detail_vertices.copy()
                    normals = detail_normals.copy()
                else:
                    points = sample_mesh_surface_with_spec(vertices, faces, sample_spec)
                    normals = sample_mesh_normals_with_spec(vertices, faces, sample_spec)
                    shell_w = np.ones((len(points),), dtype=np.float32)
                detail_meta = {"enabled": False, "reason": "external_depth_normal_removed_from_main_flow"}

                # External depth/normal detail displacement has been removed from the Mesh main flow.

                if temporal_detail_smooth > 1e-6 and len(points):
                    curr_body_points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
                    if prev_body_points_temporal is not None and len(prev_body_points_temporal) == len(curr_body_points):
                        base_alpha = float(np.clip(1.0 - 0.55 * temporal_detail_smooth, 0.52, 0.86))
                        conf_t = float(np.clip(all_confidences[frame_index], 0.0, 1.0))
                        alpha = float(np.clip(base_alpha + max(-0.08, (conf_t - 0.5) * 0.14), 0.50, 0.90))
                        curr_body_points = (prev_body_points_temporal * (1.0 - alpha) + curr_body_points * alpha).astype(np.float32)
                    prev_body_points_temporal = curr_body_points.copy()
                    points = curr_body_points
                    detail_meta["temporal_smooth"] = True
                    detail_meta["temporal_smooth_strength"] = float(temporal_detail_smooth)

                if detail_mesh_writer is not None and dense_template is not None and detail_vertices is not None:
                    # Write the same stabilized dense mesh that feeds point export.
                    # Earlier builds wrote this before temporal smoothing, which made
                    # the mesh exit twitchier than the point-cloud exit.
                    mesh_vertices_to_write = np.asarray(points if len(points) == len(detail_vertices) else detail_vertices, dtype=np.float32).reshape(-1, 3)
                    local_mesh_vertices = (mesh_vertices_to_write - export_origin[None, :]).astype(np.float32)
                    if base_detail_vertices is not None and len(base_detail_vertices) == len(local_mesh_vertices):
                        local_body_vertices = (np.asarray(base_detail_vertices, dtype=np.float32).reshape(-1, 3) - export_origin[None, :]).astype(np.float32)
                    else:
                        local_body_vertices = local_mesh_vertices
                    layer_payload = {"Body": local_body_vertices}
                    garment_shell_local = np.zeros((0, 3), dtype=np.float32)
                    hair_shell_local = np.zeros((0, 3), dtype=np.float32)
                    garment_sil_local = np.zeros((0, 3), dtype=np.float32)
                    hair_sil_local = np.zeros((0, 3), dtype=np.float32)
                    if garment_vertex_indices is not None and len(garment_vertex_indices):
                        garment_shell_world = mesh_vertices_to_write[np.asarray(garment_vertex_indices, dtype=np.int64)]
                        garment_shell_local = (garment_shell_world - export_origin[None, :]).astype(np.float32)
                        layer_payload["GarmentShell"] = garment_shell_local
                    if garment_silhouette_indices is not None and len(garment_silhouette_indices):
                        garment_sil_world = _make_silhouette_side_vertices(
                            mesh_vertices_to_write,
                            detail_normals if detail_normals is not None else np.zeros_like(mesh_vertices_to_write),
                            garment_silhouette_indices,
                            expand_ratio=float(getattr(self.cfg, "garment_silhouette_expand_ratio", 0.026)),
                            normal_ratio=float(getattr(self.cfg, "garment_silhouette_normal_ratio", 0.006)),
                        )
                        garment_sil_local = (garment_sil_world - export_origin[None, :]).astype(np.float32) if len(garment_sil_world) else np.zeros((0, 3), dtype=np.float32)
                        layer_payload["GarmentSilhouette"] = garment_sil_local
                    if hair_vertex_indices is not None and len(hair_vertex_indices):
                        hair_shell_world = mesh_vertices_to_write[np.asarray(hair_vertex_indices, dtype=np.int64)]
                        hair_shell_local = (hair_shell_world - export_origin[None, :]).astype(np.float32)
                        layer_payload["HairShell"] = hair_shell_local
                    if hair_silhouette_indices is not None and len(hair_silhouette_indices):
                        hair_sil_world = _make_silhouette_side_vertices(
                            mesh_vertices_to_write,
                            detail_normals if detail_normals is not None else np.zeros_like(mesh_vertices_to_write),
                            hair_silhouette_indices,
                            expand_ratio=float(getattr(self.cfg, "hair_silhouette_expand_ratio", 0.040)),
                            normal_ratio=float(getattr(self.cfg, "hair_silhouette_normal_ratio", 0.010)),
                        )
                        hair_sil_local = (hair_sil_world - export_origin[None, :]).astype(np.float32) if len(hair_sil_world) else np.zeros((0, 3), dtype=np.float32)
                        layer_payload["HairSilhouette"] = hair_sil_local
                    detail_mesh_writer.add_frame(frame_index, layer_payload)
                    if garment_mesh_writer is not None:
                        gp = {"GarmentShell": garment_shell_local}
                        if len(garment_sil_local):
                            gp["GarmentSilhouette"] = garment_sil_local
                        garment_mesh_writer.add_frame(frame_index, gp)
                    if hair_mesh_writer is not None:
                        hp = {"HairShell": hair_shell_local}
                        if len(hair_sil_local):
                            hp["HairSilhouette"] = hair_sil_local
                        hair_mesh_writer.add_frame(frame_index, hp)

                if pointcloud_export_enabled and dense_template is not None and detail_vertices is not None and len(points) == len(detail_vertices):
                    if dense_point_indices is None:
                        dense_point_indices = deterministic_limit_indices(len(points), max_points, 9300003)
                        if dense_point_indices is None:
                            dense_point_indices = np.arange(len(points), dtype=np.int64)
                    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)[dense_point_indices]
                    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)[dense_point_indices]

                hand_meta = {"enabled": False, "reason": "disabled"}
                if hand_enabled:
                    hand_frame = _load_frame_cache_0based_first(load_hand_frame, cache_root, frame_index)
                    hand_pieces = []
                    if hand_frame is not None and hand_frame.available:
                        coord_space = str((hand_frame.meta or {}).get("coordinate_space", "structure")).lower()
                        for side, hv, hf, _seed_base in (
                            ("left", hand_frame.left_vertices, hand_frame.left_faces, 9710003),
                            ("right", hand_frame.right_vertices, hand_frame.right_faces, 9720003),
                        ):
                            if hv is None:
                                continue
                            hv_arr = np.asarray(hv, dtype=np.float32).reshape(-1, 3)
                            if coord_space not in {"pointcloud", "aligned"}:
                                hv_arr = hv_arr - stable_result.roots[frame_index][None, :] + stable_result.reference_root[None, :]
                            hp = np.zeros((0, 3), dtype=np.float32)
                            spec = hand_surface_specs.get(side)
                            if spec is not None and hf is not None:
                                hp = sample_mesh_surface_with_spec(hv_arr, np.asarray(hf, dtype=np.int64).reshape(-1, 3), spec)
                            elif side in hand_vertex_indices:
                                idx = np.clip(hand_vertex_indices[side], 0, max(0, len(hv_arr) - 1))
                                hp = hv_arr[idx] if len(hv_arr) else np.zeros((0, 3), dtype=np.float32)
                            if len(hp):
                                hand_pieces.append(hp.astype(np.float32))
                        if hand_pieces:
                            hand_points = np.concatenate(hand_pieces, axis=0).astype(np.float32)
                            if temporal_detail_smooth > 1e-6 and prev_hand_points_temporal is not None and len(prev_hand_points_temporal) == len(hand_points):
                                hand_alpha = float(np.clip(1.0 - 0.45 * temporal_detail_smooth, 0.58, 0.90))
                                hand_points = (prev_hand_points_temporal * (1.0 - hand_alpha) + hand_points * hand_alpha).astype(np.float32)
                            prev_hand_points_temporal = hand_points.copy()
                            points = np.concatenate([points, hand_points], axis=0) if len(points) else hand_points
                            hand_meta = {
                                "enabled": True,
                                "reason": "ok_fixed_sampling",
                                "points": int(len(hand_points)),
                                "model_name": str(hand_frame.model_name),
                                "confidence": float(hand_frame.confidence),
                                "coordinate_space": coord_space,
                                "temporal_smooth": bool(temporal_detail_smooth > 1e-6),
                            }
                        else:
                            hand_meta = {"enabled": False, "reason": "empty_hand_vertices_or_spec"}
                    else:
                        hand_meta = {"enabled": False, "reason": "hand_cache_missing"}

                # Store frame geometry in local coordinates around the fixed
                # geometry-center origin. The parent USD Xform carries the world
                # offset, so Blender imports with the object origin at the visual
                # center instead of a drifting solver root/pelvis.
                if len(points):
                    points = (np.asarray(points, dtype=np.float32).reshape(-1, 3) - export_origin[None, :]).astype(np.float32)

                source = np.ones((len(points),), dtype=np.uint8)
                confidence = np.full((len(points),), all_confidences[frame_index], dtype=np.float32)
                colors = np.zeros((len(points), 3), dtype=np.uint8)

                # Structure XYZ requires stable point identity. Voxel/downsample and
                # statistical outlier filters change point order/count per frame, so
                # they are intentionally skipped in this workflow. The fixed sampler
                # already keeps the body below max_points; only apply a constant-seed
                # safety cap if optional hand points push the array over the limit.
                if pointcloud_export_enabled and len(points) > max_points:
                    limiter = deterministic_limit_indices(len(points), max_points, 9300003)
                    if limiter is not None:
                        points = points[limiter]
                        source = source[limiter]
                        confidence = confidence[limiter]

                written = writer.add_frame(frame_index, points) if writer is not None else 0
                meta = {
                    "frame_index": int(frame_index),
                    "status": "structure_xyz_detail" if bool(detail_meta.get("enabled", False)) else "structure_xyz",
                    "points": int(written),
                    "structure_model": all_models[frame_index],
                    "structure_confidence": all_confidences[frame_index],
                    "root_stabilizer": {
                        "method": stable_result.method,
                        "reference_root": [float(x) for x in stable_result.reference_root],
                        "current_root_before_stabilize": [float(x) for x in stable_result.roots[frame_index]],
                    },
                    "blender_origin": {
                        "mode": "first_frame_robust_geometry_center",
                        "xform_translate": [float(x) for x in export_origin],
                        "points_are_local_to_origin": True,
                    },
                    "fixed_surface_sampling": {
                        "seed": int(sample_spec.seed),
                        "samples": int(len(sample_spec.face_indices)),
                    },
                    "shell_detail": detail_region_meta,
                    "external_depth_normal": detail_meta,
                    "hand_fusion": hand_meta,
                }
                if frame_index == 0:
                    meta["structure_quality"] = quality
                metas.append(meta)
                with open(pointcloud_meta_dir / f"frame_{frame_index:06d}.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
                if frame_index == 0 or (frame_index + 1) % 30 == 0:
                    if bool(detail_meta.get("enabled", False)):
                        self._log(
                            f"结构细节点云: 第 {frame_index + 1} 帧写入 {written} 点，"
                            f"detail_mean_abs={float(detail_meta.get('displacement_mean_abs', 0.0)):.5f}"
                        )
                    else:
                        self._log(f"结构点云: 第 {frame_index + 1} 帧写入 {written} 点")
                self.progress.emit(frame_index + 1, total)
        finally:
            try:
                if 'fast_depth_reader' in locals() and fast_depth_reader is not None:
                    fast_depth_reader.close()
                if 'fast_normal_reader' in locals() and fast_normal_reader is not None:
                    fast_normal_reader.close()
            except Exception:
                pass
            for _mesh_writer, _name in ((low_mesh_writer, "Body Mesh"), (garment_mesh_writer, "Garment Mesh"), (hair_mesh_writer, "Hair Mesh"), (detail_mesh_writer, "Combined Mesh")):
                try:
                    if _mesh_writer is not None:
                        _mesh_writer.close()
                except Exception as exc:  # noqa: BLE001
                    self._log(f"{_name} 收尾失败: {short_error_message(str(exc))}")
            try:
                if writer is not None:
                    writer.close()
            except Exception as exc:  # noqa: BLE001
                self._log(f"USDA 点云收尾失败: {short_error_message(str(exc))}")

        manifest_status = "red" if errors >= total else ("yellow" if errors > 0 else "green")
        manifest = {
            "input_path": self.cfg.input_path,
            "config_signature": pointcloud_config_sig,
            "frame_count": int(total),
            "fps": float(fps),
            "format": "USDA animated fixed-topology mesh / optional XYZ point cloud",
            "primary_format": "stable_mesh_shell_plus_optional_pointcloud",
            "usd_structure_path": str(usd_path) if usd_path.exists() else "",
            "mesh_low_path": str(mesh_low_path) if mesh_low_path.exists() else "",
            "mesh_garment_path": str(mesh_garment_path) if mesh_garment_path.exists() else "",
            "mesh_hair_path": str(mesh_hair_path) if mesh_hair_path.exists() else "",
            "mesh_combined_path": str(mesh_combined_path) if mesh_combined_path.exists() else "",
            "frame_meta_dir": str(pointcloud_meta_dir),
            "mode": "stable_mesh_shell_pointcloud",
            "color": "none",
            "fields": ["stable_vertex_id", "fixed_faces", "points"],
            "mesh_exports": {
                "low_mesh": bool(mesh_low_path.exists()),
                "garment_mesh": bool(mesh_garment_path.exists()),
                "hair_mesh": bool(mesh_hair_path.exists()),
                "combined_mesh": bool(mesh_combined_path.exists()),
                "pointcloud": bool(usd_path.exists()) if pointcloud_export_enabled else False,
                "dense_segments": int(getattr(dense_template, "segments", 0) or 0) if 'dense_template' in locals() and dense_template is not None else 0,
                "vertex_id_stable": True,
                "per_frame_remesh": False,
            },
            "status": manifest_status,
            "errors": int(errors),
            "geometry_cache": {"cache_root": str(cache_root), "structure_dir": str(structure_dir)},
            "blender_origin": {
                "mode": "first_frame_robust_geometry_center",
                "xform_translate": [float(x) for x in export_origin] if 'export_origin' in locals() else [0.0, 0.0, 0.0],
                "points_are_local_to_origin": True,
            },
            "frames": metas,
            "note": "Main output is split fixed-topology mesh animation: body, garment, hair and combined. Dense mesh uses fixed face_id/barycentric IDs; no per-frame remesh. UV/normals are not authored; Blender computes normals from face winding.",
        }
        with open(pointcloud_root / "pointcloud_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        write_export_state(
            pointcloud_root,
            status="completed" if manifest_status != "red" else "failed",
            config_signature=pointcloud_config_sig,
            frame_count=int(total),
            manifest_status=manifest_status,
            export_errors=int(errors),
        )
        if manifest_status == "red":
            raise RuntimeError("所有帧都缺少 structure cache，未生成有效点云。")
        self.cfg.output_path = str(mesh_combined_path if mesh_combined_path.exists() else (mesh_low_path if mesh_low_path.exists() else usd_path))
        self._log("Body / Garment / Hair / Combined Mesh 导出完成。")

    def _run_impl(self) -> None:
        if self._is_structure_xyz_workflow():
            self._run_structure_xyz_pointcloud_impl()
            return
        raise RuntimeError("当前清理版只支持图像驱动网格主流程；旧深度/法线/可见点云流程已删除。")

class MeshExportWorker(DepthWorker):
    """Dedicated worker for the current Mesh/Shell export path.

    This keeps the active mesh workflow out of the old 旧深度/depth-video branch at
    the UI routing level. The implementation reuses the already-tested structure
    exporter body, but _run_impl cannot fall through to legacy depth processing.
    """

    def _run_impl(self) -> None:
        self._run_structure_xyz_pointcloud_impl()


class DepthExportWorker(DepthWorker):
    """Explicit name for the legacy depth-video exporter.

    The original DepthWorker remains as a compatibility wrapper; UI code should
    instantiate DepthExportWorker only for old 旧深度/direct-depth workflows.
    """

    pass


class StructureCacheWorker(QObject):
    log = Signal(str)
    progress = Signal(str)
    progress_value = Signal(str, int, int)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        input_path: str,
        cache_root: str,
        model: str,
        project_root: str,
        max_side: int = 720,
        *,
        start_frame: int = 0,
        end_frame: int = -1,
    ) -> None:
        super().__init__()
        self.input_path = str(input_path)
        self.cache_root = str(cache_root)
        self.model = str(model or "4dhumans")
        self.project_root = str(project_root)
        self.max_side = max(256, min(1024, int(max_side or 720)))
        self.start_frame = max(0, int(start_frame or 0))
        self.end_frame = int(end_frame if end_frame is not None else -1)
        if self.end_frame >= 0 and self.end_frame < self.start_frame:
            self.end_frame = self.start_frame
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _log(self, text: str) -> None:
        event_log(text, channel="STRUCTURE")
        self.log.emit(text)

    def run(self) -> None:
        tmp_cache_root: Optional[Path] = None
        try:
            project = Path(self.project_root)
            cache_root = Path(self.cache_root)
            cache_root.mkdir(parents=True, exist_ok=True)
            tmp_cache_root = cache_root / "__structure_tmp__"
            if tmp_cache_root.exists():
                shutil.rmtree(tmp_cache_root, ignore_errors=True)
            tmp_cache_root.mkdir(parents=True, exist_ok=True)
            model_key = str(self.model or "4dhumans").strip().lower()
            self._log(f"启动结构缓存生成: model={model_key}, input={self.input_path}, cache_root={cache_root}, range={self.start_frame}-{self.end_frame if self.end_frame >= 0 else 'end'}")
            self._log("结构缓存会先写入临时目录；第三方模型成功后再替换旧缓存，失败时保留旧缓存。")
            self._log("12G显存策略：结构模型单独运行，最长边限制到 1024，完成后释放 GPU。")

            if model_key == "wham":
                from structure_pipeline.wham_runner import WhamRunner
                runner = WhamRunner()
            else:
                from structure_pipeline.fourdhumans_runner import FourDHumansRunner
                runner = FourDHumansRunner()

            def _runner_log(text: str) -> None:
                if self._cancel:
                    raise RuntimeError("结构缓存生成已取消")
                line = str(text or "").strip()
                if not line:
                    return
                if line.startswith("[Progress]"):
                    payload = line.replace("[Progress]", "", 1).strip()
                    self.progress.emit(payload)
                    try:
                        import re as _re
                        m = _re.search(r"^(.*?)\s+(-?\d+)\s*/\s*(-?\d+)", payload)
                        if m:
                            stage = m.group(1).strip() or "结构缓存"
                            self.progress_value.emit(stage, int(m.group(2)), int(m.group(3)))
                    except Exception:
                        pass
                else:
                    self._log(line)

            result = runner.run_video_to_cache(
                self.input_path,
                tmp_cache_root,
                project,
                start_frame=int(self.start_frame),
                end_frame=int(self.end_frame),
                max_side=int(self.max_side),
                log=_runner_log,
            )
            self._log("结构缓存结果: " + json.dumps(result, ensure_ascii=False, default=str))
            if self._cancel:
                raise RuntimeError("结构缓存生成已取消")
            if not isinstance(result, dict) or not bool(result.get("ok", False)):
                reason = str(result.get("reason") if isinstance(result, dict) else "unknown")
                missing = result.get("missing_modules") if isinstance(result, dict) else None
                if reason == "missing_python_modules" and missing:
                    raise RuntimeError("结构缓存生成失败：缺少 Python 依赖：" + "、".join(str(x) for x in missing))
                if reason == "missing_home_env":
                    raise RuntimeError("结构缓存生成失败：Windows HOME 环境变量未设置。")
                if reason == "repo_missing":
                    raise RuntimeError("结构缓存生成失败：外部模型仓库不存在：" + str(result.get("repo", "")))
                if reason == "no_importable_outputs":
                    raise RuntimeError("结构缓存生成失败：第三方模型已运行，但没有生成可导入的 mesh/npz/pkl 输出。")
                raise RuntimeError("结构缓存生成失败：" + reason)
            new_structure_dir = tmp_cache_root / "structure"
            if not new_structure_dir.exists():
                raise RuntimeError("结构缓存生成失败：临时目录没有生成 structure 数据。")
            final_structure_dir = cache_root / "structure"
            backup_structure_dir = cache_root / "structure_backup_previous"
            if backup_structure_dir.exists():
                shutil.rmtree(backup_structure_dir, ignore_errors=True)
            if final_structure_dir.exists():
                shutil.move(str(final_structure_dir), str(backup_structure_dir))
            shutil.move(str(new_structure_dir), str(final_structure_dir))
            shutil.rmtree(tmp_cache_root, ignore_errors=True)
            self.finished.emit("结构缓存生成完成: " + self.cache_root)
        except Exception as exc:  # noqa: BLE001
            event_exception("结构缓存生成失败", exc, model=self.model, input_path=self.input_path)
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class ModelPreloadWorker(QObject):
    log = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, model_id: str, device_mode: str, load_normal: bool = False) -> None:
        super().__init__()
        self.model_id = model_id
        self.device_mode = device_mode
        self.load_normal = bool(load_normal)

    def _log(self, text: str) -> None:
        event_log(text, channel="PRELOAD")
        self.log.emit(text)

    def run(self) -> None:
        try:
            self._log("当前清理版无需预热旧模型；请使用环境页检查 4DHumans / FASHN。")
            self.finished.emit("网格主流程无需旧模型预热。")
        except Exception as exc:  # noqa: BLE001
            event_exception("模型预热状态检查失败", exc)
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


def _mesh_preview_available_indices(cache_root: Path) -> list[int]:
    """Return actually readable structure-frame indices for preview.

    Do not trust file presence alone. Some partially generated caches contain
    vertices/faces files with confidence 0, missing camera metadata, or arrays
    that cannot be loaded. Preview is diagnostic, so it must skip those frames
    and choose the nearest readable one instead of surfacing a red traceback.
    """
    root = Path(cache_root) / "structure"
    if not root.exists():
        return []
    out: list[int] = []
    for vp in root.glob("frame_*_smpl_vertices.npy"):
        name = vp.name
        try:
            idx_text = name.replace("frame_", "", 1).replace("_smpl_vertices.npy", "")
            idx = int(idx_text)
        except Exception:
            continue
        if not (root / f"frame_{idx:06d}_smpl_faces.npy").exists():
            continue
        try:
            fr = load_structure_frame(cache_root, idx)
            if fr is not None and fr.available:
                out.append(idx)
        except Exception:
            continue
    return sorted(set(out))


def _mesh_preview_nearest_index(cache_root: Path, frame_index: int) -> tuple[int | None, list[int]]:
    indices = _mesh_preview_available_indices(cache_root)
    if not indices:
        return None, indices
    target = max(0, int(frame_index))
    if target in indices:
        return target, indices
    nearest = min(indices, key=lambda i: (abs(i - target), i))
    return int(nearest), indices


def _mesh_preview_load_frame(cache_root: Path, frame_index: int):  # noqa: ANN202
    idx = int(frame_index)
    frame = load_structure_frame(cache_root, idx)
    if frame is None or not frame.available:
        raise RuntimeError(f"第 {idx} 帧 structure cache 不可读。请重新生成结构缓存，或换一帧预览。")
    return frame


def _mesh_preview_message_bgr(title: str, line1: str, line2: str = "") -> np.ndarray:
    canvas = np.zeros((620, 720, 3), dtype=np.uint8)
    canvas[:] = (18, 22, 28)
    cv2.putText(canvas, str(title or "Mesh Preview")[:42], (32, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (235, 240, 248), 2, cv2.LINE_AA)
    cv2.putText(canvas, str(line1 or "No preview available")[:76], (32, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 220, 235), 2, cv2.LINE_AA)
    if line2:
        cv2.putText(canvas, str(line2)[:92], (32, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 172, 190), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (66, 230), (654, 552), (70, 80, 95), 1, cv2.LINE_AA)
    return canvas


def _mesh_preview_effective_max_points(cfg: JobConfig) -> int:
    density = str(getattr(cfg, "pointcloud_density", "中") or "中")
    if density == "低":
        return 50000
    if density == "中":
        return 120000
    if density == "高":
        return 200000
    return int(max(1000, min(int(getattr(cfg, "pointcloud_max_points", 120000) or 120000), 800000)))


def _mesh_preview_axes_and_origin(vertices: np.ndarray, faces: np.ndarray, max_points: int) -> tuple[np.ndarray, int, int, dict]:
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(verts) == 0:
        raise RuntimeError("Mesh 顶点为空，无法预览。")
    finite = np.isfinite(verts).all(axis=1)
    if not bool(np.any(finite)):
        raise RuntimeError("Mesh 顶点包含无效数值，无法预览。")
    clean = np.nan_to_num(verts, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.nanpercentile(clean[finite], 2.0, axis=0)
    hi = np.nanpercentile(clean[finite], 98.0, axis=0)
    extent = np.maximum(hi - lo, 1e-6)
    # Fixed front-like preview axes. Auto-guessing by largest extent made sitting/side
    # poses flip the preview orientation and created false "broken mesh" signals.
    horizontal_axis = 0
    vertical_axis = 1
    if len(f) > 0:
        origin_sample_count = min(8000, max(2000, int(max_points // 24)))
        try:
            origin_spec = make_surface_sample_spec(clean, f, origin_sample_count, seed=9100003)
            origin = robust_geometry_center(sample_mesh_surface_with_spec(clean, f, origin_spec))
        except Exception:
            origin = np.nanmedian(clean[finite], axis=0).astype(np.float32)
    else:
        origin = np.nanmedian(clean[finite], axis=0).astype(np.float32)
    meta = {
        "horizontal_axis": horizontal_axis,
        "vertical_axis": vertical_axis,
        "extent": extent,
        "origin": origin.astype(np.float32),
    }
    return origin.astype(np.float32), horizontal_axis, vertical_axis, meta


def _mesh_preview_project_points(points: np.ndarray, horizontal_axis: int, vertical_axis: int, origin: np.ndarray, canvas_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, float]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    canvas_h, canvas_w = int(canvas_shape[0]), int(canvas_shape[1])
    centred = pts - np.asarray(origin, dtype=np.float32).reshape(1, 3)
    finite = np.isfinite(centred).all(axis=1)
    if not bool(np.any(finite)):
        raise RuntimeError("预览点包含无效数值。")
    lo = np.nanpercentile(centred[finite], 2.0, axis=0)
    hi = np.nanpercentile(centred[finite], 98.0, axis=0)
    span_x = max(float(hi[horizontal_axis] - lo[horizontal_axis]), 1e-6)
    span_y = max(float(hi[vertical_axis] - lo[vertical_axis]), 1e-6)
    draw_w, draw_h = canvas_w * 0.78, canvas_h * 0.72
    scale = min(draw_w / span_x, draw_h / span_y) * 0.92
    x = np.clip((canvas_w * 0.50 + centred[:, horizontal_axis] * scale).astype(np.int32), 0, canvas_w - 1)
    y = np.clip((canvas_h * 0.52 - centred[:, vertical_axis] * scale).astype(np.int32), 0, canvas_h - 1)
    return x, y, float(scale)


def _mesh_preview_rotated_points(points: np.ndarray, origin: np.ndarray, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return pts
    yaw = math.radians(float(yaw_deg or 0.0))
    pitch = math.radians(float(pitch_deg or 0.0))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # Y-up yaw, then X-axis pitch. Preview projection stays x/y, depth is z.
    ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    rot = (rx @ ry).astype(np.float32)
    org = np.asarray(origin, dtype=np.float32).reshape(1, 3)
    return ((pts - org) @ rot.T + org).astype(np.float32)

def _mesh_preview_mesh_bgr(vertices: np.ndarray, faces: np.ndarray, title: str, detail: str = "", max_points: int = 120000, fill_color: tuple[int, int, int] = (92, 101, 116), yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    face_arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    canvas = np.zeros((620, 720, 3), dtype=np.uint8)
    canvas[:] = (18, 22, 28)
    if len(verts) == 0:
        cv2.putText(canvas, title[:42] if title else "Mesh", (32, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (235, 240, 248), 2, cv2.LINE_AA)
        cv2.putText(canvas, "empty layer", (32, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (210, 220, 235), 2, cv2.LINE_AA)
        if detail:
            cv2.putText(canvas, str(detail)[:86], (32, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 172, 190), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Generate segmentation cache or enable layer shell.", (32, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 172, 190), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (66, 230), (654, 552), (70, 80, 95), 1, cv2.LINE_AA)
        return canvas
    origin, horizontal_axis, vertical_axis, _meta = _mesh_preview_axes_and_origin(verts, face_arr, max_points)
    draw_verts = _mesh_preview_rotated_points(verts, origin, yaw_deg=yaw_deg, pitch_deg=pitch_deg)
    x, y, _scale = _mesh_preview_project_points(draw_verts, horizontal_axis, vertical_axis, origin, canvas.shape)
    valid_faces = face_arr[(face_arr >= 0).all(axis=1) & (face_arr < len(verts)).all(axis=1)] if len(face_arr) else face_arr
    if len(valid_faces) > 26000:
        idx = np.linspace(0, len(valid_faces) - 1, 26000).astype(np.int64)
        valid_faces = valid_faces[idx]
    depth_axis = ({0, 1, 2} - {int(horizontal_axis), int(vertical_axis)}).pop()
    if len(valid_faces):
        px = x[valid_faces]
        py = y[valid_faces]
        signed_area = ((px[:, 1] - px[:, 0]) * (py[:, 2] - py[:, 0]) - (py[:, 1] - py[:, 0]) * (px[:, 2] - px[:, 0])).astype(np.float32)
        area_abs = np.abs(signed_area)
        keep_area = area_abs > 0.35
        pos_count = int(np.count_nonzero(signed_area[keep_area] > 0))
        neg_count = int(np.count_nonzero(signed_area[keep_area] < 0))
        if pos_count or neg_count:
            front_sign = 1.0 if pos_count >= neg_count else -1.0
            keep_face = keep_area & ((signed_area * front_sign) > 0.0)
            if int(np.count_nonzero(keep_face)) >= max(32, int(0.12 * len(valid_faces))):
                valid_faces = valid_faces[keep_face]
        if len(valid_faces) > 18000:
            valid_faces = valid_faces[np.linspace(0, len(valid_faces) - 1, 18000).astype(np.int64)]
        depths = np.nanmean((draw_verts[valid_faces] - origin[None, None, :])[:, :, depth_axis], axis=1)
        valid_faces = valid_faces[np.argsort(depths)]
        polys = np.stack([x[valid_faces], y[valid_faces]], axis=2).astype(np.int32)
        if len(polys):
            cv2.fillPoly(canvas, list(polys), tuple(int(x) for x in fill_color), cv2.LINE_AA)
            edge_faces = valid_faces
            if len(edge_faces) > 6000:
                edge_faces = edge_faces[np.linspace(0, len(edge_faces) - 1, 6000).astype(np.int64)]
            edge_polys = np.stack([x[edge_faces], y[edge_faces]], axis=2).astype(np.int32)
            cv2.polylines(canvas, list(edge_polys), True, (42, 52, 68), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (66, 74), (654, 552), (70, 80, 95), 1, cv2.LINE_AA)
    cv2.putText(canvas, title[:42], (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 240, 248), 2, cv2.LINE_AA)
    axis_names = ("x", "y", "z")
    cv2.putText(canvas, f"mesh verts={len(verts)} faces={len(face_arr)} view yaw={float(yaw_deg or 0):.0f} pitch={float(pitch_deg or 0):.0f}", (26, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 190, 205), 1, cv2.LINE_AA)
    if detail:
        cv2.putText(canvas, detail[:80], (26, 608), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 172, 190), 1, cv2.LINE_AA)
    return canvas


def _mesh_preview_pointcloud_bgr(points: np.ndarray, origin_vertices: np.ndarray, origin_faces: np.ndarray, title: str, detail: str = "", max_points: int = 120000, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    origin, horizontal_axis, vertical_axis, _meta = _mesh_preview_axes_and_origin(origin_vertices, origin_faces, max_points)
    canvas = np.zeros((620, 720, 3), dtype=np.uint8)
    canvas[:] = (18, 22, 28)
    if len(pts) == 0:
        cv2.putText(canvas, "empty pointcloud", (32, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 220, 235), 2, cv2.LINE_AA)
        return canvas
    draw_pts = _mesh_preview_rotated_points(pts, origin, yaw_deg=yaw_deg, pitch_deg=pitch_deg)
    x, y, _scale = _mesh_preview_project_points(draw_pts, horizontal_axis, vertical_axis, origin, canvas.shape)
    if len(x) > 22000:
        idx = np.linspace(0, len(x) - 1, 22000).astype(np.int64)
        x, y = x[idx], y[idx]
    canvas[y, x] = (230, 236, 245)
    cv2.rectangle(canvas, (66, 74), (654, 552), (70, 80, 95), 1, cv2.LINE_AA)
    cv2.putText(canvas, title[:42], (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 240, 248), 2, cv2.LINE_AA)
    axis_names = ("x", "y", "z")
    cv2.putText(canvas, f"points={len(pts)} view yaw={float(yaw_deg or 0):.0f} pitch={float(pitch_deg or 0):.0f}", (26, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 190, 205), 1, cv2.LINE_AA)
    if detail:
        cv2.putText(canvas, detail[:80], (26, 608), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 172, 190), 1, cv2.LINE_AA)
    return canvas


class SegmentationCacheWorker(QObject):
    progress = Signal(str)
    progress_value = Signal(str, int, int)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, cfg: JobConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _log(self, text: str) -> None:
        event_log(text, channel="SEGMENTATION_CACHE")
        self.progress.emit(str(text))

    def run(self) -> None:
        try:
            cache_root = structure_cache_root(self.cfg)
            def _progress(done: int, total: int) -> None:
                if self._cancel:
                    raise RuntimeError("分割缓存任务已取消。")
                self.progress_value.emit("segmentation", int(done), int(total))
            summary = generate_segmentation_sequence_cache(
                self.cfg,
                cache_root,
                project_root=PROJECT_DIR,
                log=self._log,
                progress=_progress,
            )
            self.finished.emit(summary)
        except Exception as exc:  # noqa: BLE001
            event_exception("逐帧分割缓存生成失败", exc, input_path=getattr(self.cfg, "input_path", ""))
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class MeshPreviewWorker(QObject):
    log = Signal(str)
    finished = Signal(str, object, str, float, int)
    failed = Signal(str, str, int)

    def __init__(self, cfg: JobConfig, frame_index: int, mode: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.frame_index = max(0, int(frame_index))
        self.mode = str(mode or "stable")

    def _log(self, text: str) -> None:
        event_log(text, channel="MESH_PREVIEW")
        self.log.emit(text)

    def run(self) -> None:
        started = time.time()
        try:
            bgr, status = self._run_impl()
            self.finished.emit(self.mode, bgr, status, time.time() - started, self.frame_index)
        except RuntimeError as exc:
            msg = str(exc)
            expected_missing_cache = (
                "结构缓存没有可读帧" in msg
                or "structure cache 不可读" in msg
                or "missing_structure_frame" in msg
            )
            if expected_missing_cache:
                bgr = _mesh_preview_message_bgr(
                    "Mesh Preview",
                    "No readable structure cache for this frame.",
                    "Generate structure cache, or move the preview frame into the processed range.",
                )
                self.finished.emit(self.mode, bgr, "当前帧没有可读结构缓存；未执行模型预览。", time.time() - started, self.frame_index)
                return
            event_exception("Mesh 预览任务失败", exc, frame_index=self.frame_index, input_path=getattr(self.cfg, "input_path", ""), mode=self.mode)
            self.failed.emit(self.mode, f"{exc}\n\n{traceback.format_exc()}", self.frame_index)
        except Exception as exc:  # noqa: BLE001
            event_exception("Mesh 预览任务失败", exc, frame_index=self.frame_index, input_path=getattr(self.cfg, "input_path", ""), mode=self.mode)
            self.failed.emit(self.mode, f"{exc}\n\n{traceback.format_exc()}", self.frame_index)

    def _load_local_stabilized_preview(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        cache_root = structure_cache_root(self.cfg)
        requested = max(0, int(self.frame_index))
        target, available_indices = _mesh_preview_nearest_index(cache_root, requested)
        if target is None:
            raise RuntimeError("当前结构缓存没有可读帧。请先生成结构缓存。")

        # Use frame 0 as root reference when present; otherwise use the first cached
        # frame. This matters for projects generated from a selected range where
        # frame_000000 may not exist, or for partially imported third-party results.
        reference_idx = 0 if 0 in available_indices else int(available_indices[0])
        frame0 = _mesh_preview_load_frame(cache_root, reference_idx)
        f0 = np.asarray(frame0.faces, dtype=np.int64).reshape(-1, 3)
        v0 = np.asarray(frame0.vertices, dtype=np.float32).reshape(-1, 3)
        j0 = np.asarray(frame0.joints, dtype=np.float32).reshape(-1, 3) if frame0.joints is not None else None

        radius = int(np.clip(int(getattr(self.cfg, "preview_temporal_radius", 3) or 3), 1, 8))
        candidate_indices = [i for i in available_indices if abs(int(i) - int(target)) <= radius]
        if target not in candidate_indices:
            candidate_indices.append(int(target))
        indices = sorted(set(int(i) for i in candidate_indices))

        frames = []
        kept_indices: list[int] = []
        for idx in indices:
            try:
                fr = _mesh_preview_load_frame(cache_root, idx)
            except Exception:
                continue
            ft = np.asarray(fr.faces, dtype=np.int64).reshape(-1, 3)
            if f0.shape != ft.shape or not np.array_equal(f0, ft):
                raise RuntimeError("当前 structure cache 拓扑不固定：faces 在帧间发生变化，不能安全预览/导出固定拓扑 Mesh。请重新生成结构缓存。")
            frames.append(fr)
            kept_indices.append(idx)
        if target not in kept_indices:
            # Last-resort tolerance: use the nearest frame that was successfully
            # loaded in this preview window. Do not try to load target again; it
            # may be present on disk but marked unavailable/corrupt.
            if kept_indices:
                nearest_kept = min(kept_indices, key=lambda i: (abs(int(i) - int(target)), int(i)))
                target = int(nearest_kept)
            else:
                # If the local window failed, try the nearest globally readable
                # frame from the pre-filtered list.
                target = int(min(available_indices, key=lambda i: (abs(int(i) - int(requested)), int(i))))
                try:
                    fr = _mesh_preview_load_frame(cache_root, target)
                    ft = np.asarray(fr.faces, dtype=np.int64).reshape(-1, 3)
                    if f0.shape != ft.shape or not np.array_equal(f0, ft):
                        raise RuntimeError("当前 structure cache 拓扑不固定：faces 在帧间发生变化，不能安全预览/导出固定拓扑 Mesh。请重新生成结构缓存。")
                    frames = [fr]
                    kept_indices = [int(target)]
                except Exception:
                    raise RuntimeError("当前结构缓存没有可读帧。请重新生成结构缓存。")

        verts = [np.asarray(fr.vertices, dtype=np.float32).reshape(-1, 3) for fr in frames]
        joints = [np.asarray(fr.joints, dtype=np.float32).reshape(-1, 3) if fr.joints is not None else None for fr in frames]
        confidences = [float(getattr(fr, "confidence", 1.0) or 1.0) for fr in frames]

        # Match export root locking: use the reference frame as root, but only
        # smooth the local preview window so preview remains responsive.
        root_result = stabilize_vertices_by_root([v0] + verts, [j0] + joints)
        stable_window = np.asarray(root_result.vertices[1:], dtype=np.float32)
        smooth_amount = max(0.0, min(0.95, float(getattr(self.cfg, "pointcloud_temporal_center_smooth", 0.0))))
        spike_guard = max(0.0, min(0.95, float(getattr(self.cfg, "pointcloud_temporal_scale_smooth", 0.0))))
        smooth_method = "root_only"
        if smooth_amount > 1e-6 and len(stable_window) >= 3:
            smoothed = smooth_structure_vertices_temporal(
                [stable_window[i] for i in range(len(stable_window))],
                confidences,
                smooth_amount=smooth_amount,
                spike_guard=spike_guard,
            )
            stable_window = smoothed.vertices
            smooth_method = f"local_temporal_window_r{radius}:{smoothed.method}"
        pos = kept_indices.index(target) if target in kept_indices else 0
        svt = np.asarray(stable_window[pos], dtype=np.float32).reshape(-1, 3)
        sv0 = np.asarray(root_result.vertices[0], dtype=np.float32).reshape(-1, 3)
        camera_payload = dict(getattr(frames[pos], "camera", {}) or {}) if frames else {}
        fallback_used = int(requested) != int(target)
        meta = {
            "method": smooth_method,
            "window_frames": kept_indices,
            "requested_frame": int(requested),
            "target_frame": int(target),
            "preview_frame": int(target),
            "reference_frame": int(reference_idx),
            "preview_temporal_radius": radius,
            "available_first_frame": int(available_indices[0]),
            "available_last_frame": int(available_indices[-1]),
            "frame_fallback_used": bool(fallback_used),
            "camera": camera_payload,
        }
        return svt.astype(np.float32), sv0.astype(np.float32), f0.astype(np.int64), meta

    def _preview_structure_geometry(self, mode: str = "body") -> tuple[np.ndarray, np.ndarray, dict]:
        mode = str(mode or "body").lower()
        if mode in {"stable", "low"}:
            mode = "body"
        if mode == "detail":
            mode = "combined"
        svt, sv0, f0, preview_meta = self._load_local_stabilized_preview()
        if mode == "body":
            meta_body = {"method": preview_meta.get("method", "stable_low_mesh_preview"), "mesh_type": "body"}
            for key in ("requested_frame", "preview_frame", "target_frame", "available_first_frame", "available_last_frame", "frame_fallback_used"):
                if key in preview_meta:
                    meta_body[key] = preview_meta[key]
            return svt.astype(np.float32), f0.astype(np.int64), meta_body

        segments = int(np.clip(int(getattr(self.cfg, "mesh_dense_segments", 2) or 2), 1, 3))
        tmpl = build_dense_mesh_template(f0, segments=segments)
        winding_check = validate_dense_template_winding(svt, f0, tmpl)
        if not bool(winding_check.get("ok", True)):
            raise RuntimeError("细节 Mesh 面朝向校验失败，已停止预览。请重新生成结构缓存或降低细分等级。")
        pts0 = evaluate_dense_vertices(sv0, f0, tmpl)
        pts = evaluate_dense_vertices(svt, f0, tmpl)
        nrm = evaluate_dense_normals(svt, f0, tmpl)
        base_region = soft_region_weights(pts0)
        cache_root = structure_cache_root(self.cfg)
        region_cache_path = Path(cache_root) / "region_weights.npz"
        mask_region_meta = {"mask_used": False, "source": "body_only_no_segmentation"}
        garment_region_w = np.zeros((len(pts0),), dtype=np.float32)
        hair_region_w = np.zeros((len(pts0),), dtype=np.float32)
        if region_cache_path.exists():
            try:
                data = np.load(region_cache_path, allow_pickle=False)
                if "garment" not in data.files or "hair" not in data.files:
                    raise RuntimeError("region_weights.npz 缺少 garment/hair")
                g = np.asarray(data["garment"], dtype=np.float32).reshape(-1)
                h = np.asarray(data["hair"], dtype=np.float32).reshape(-1)
                if len(g) == len(pts0) and len(h) == len(pts0):
                    garment_region_w, hair_region_w = g, h
                    mask_region_meta = {"mask_used": True, "source": "region_weights_cache"}
            except Exception as exc:
                self._log(f"Region weight cache 读取失败，改用分割缓存重算：{short_error_message(str(exc))}")
        if not bool(mask_region_meta.get("mask_used", False)) and bool(getattr(self.cfg, "segmentation_enabled", True)):
            mask_garment_w, mask_hair_w, mask_region_meta = build_sequence_mask_guided_region_weights(
                pts0,
                base_region,
                str(cache_root),
                int(getattr(self.cfg, "_preview_total_frames", 0) or 0) or int(probe_video(getattr(self.cfg, "input_path", "")).frame_count or 1),
                camera=preview_meta.get("camera"),
            )
            if not bool(mask_region_meta.get("mask_used", False)):
                segmentation_ref = ensure_reference_segmentation(
                    self.cfg,
                    cache_root,
                    self.frame_index,
                    project_root=PROJECT_DIR,
                    log=self._log,
                )
                mask_garment_w, mask_hair_w, mask_region_meta = build_mask_guided_region_weights(pts0, base_region, segmentation_ref, camera=preview_meta.get("camera"))
            if bool(mask_region_meta.get("mask_used", False)):
                garment_region_w = np.asarray(mask_garment_w, dtype=np.float32).reshape(-1)
                hair_region_w = np.asarray(mask_hair_w, dtype=np.float32).reshape(-1)
                try:
                    np.savez_compressed(region_cache_path, garment=garment_region_w, hair=hair_region_w, meta=json.dumps(mask_region_meta, ensure_ascii=False, default=str))
                except Exception:
                    pass
        if not bool(mask_region_meta.get("mask_used", False)):
            garment_region_w = np.zeros((len(pts0),), dtype=np.float32)
            hair_region_w = np.zeros((len(pts0),), dtype=np.float32)
        garment_enabled = bool(getattr(self.cfg, "garment_shell_enabled", False))
        hair_enabled = bool(getattr(self.cfg, "hair_shell_enabled", False))
        has_garment_region = bool(mask_region_meta.get("mask_used", False)) and garment_enabled and float(np.max(garment_region_w)) > 1e-6
        has_hair_region = bool(mask_region_meta.get("mask_used", False)) and hair_enabled and float(np.max(hair_region_w)) > 1e-6
        garment_w = garment_region_w if has_garment_region else np.zeros((len(pts),), dtype=np.float32)
        hair_w = hair_region_w if has_hair_region else np.zeros((len(pts),), dtype=np.float32)
        shell_offsets = conservative_shell_offsets(
            pts,
            garment_w,
            hair_w,
            garment_offset=float(getattr(self.cfg, "garment_shell_offset", 0.006)),
            hair_offset=float(getattr(self.cfg, "hair_shell_offset", 0.010)),
        )
        combined_pts = apply_shell_offsets(pts, nrm, shell_offsets)
        face_arr = np.asarray(tmpl.faces, dtype=np.int64).reshape(-1, 3)
        meta = {
            "method": f"dense_mesh_layer_preview+{preview_meta.get('method', 'root_only')}",
            "mesh_type": mode,
            "winding_min_dot": float(winding_check.get("min_dot", 1.0)),
            "dense_segments": segments,
            "dense_vertices": int(len(combined_pts)),
            "garment_shell": garment_enabled,
            "hair_shell": hair_enabled,
            "segmentation_mask_used": bool(mask_region_meta.get("mask_used", False)),
            "segmentation_source": str(mask_region_meta.get("source", "geometry_fallback")),
            "garment_region_active": bool(has_garment_region),
            "hair_region_active": bool(has_hair_region),
            "mean_shell_offset": float(np.mean(np.abs(shell_offsets))) if len(shell_offsets) else 0.0,
            "requested_frame": preview_meta.get("requested_frame", int(self.frame_index)),
            "preview_frame": preview_meta.get("preview_frame", int(self.frame_index)),
            "target_frame": preview_meta.get("target_frame", int(self.frame_index)),
            "available_first_frame": preview_meta.get("available_first_frame", 0),
            "available_last_frame": preview_meta.get("available_last_frame", 0),
            "frame_fallback_used": bool(preview_meta.get("frame_fallback_used", False)),
        }
        if mode == "garment":
            if not has_garment_region:
                meta.update({"layer": "garment", "layer_faces": 0, "layer_vertices": 0, "disabled": not garment_enabled})
                return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64), meta
            layer_faces, idx, layer_meta = _subset_mesh_layer_from_weights(face_arr, garment_region_w, threshold=0.18, fallback_top_fraction=0.0)
            side_faces, side_idx, side_meta = _build_silhouette_side_layer(layer_faces, idx)
            shell_pts = combined_pts[np.asarray(idx, dtype=np.int64)] if len(idx) else np.zeros((0, 3), dtype=np.float32)
            side_pts = _make_silhouette_side_vertices(
                combined_pts,
                nrm,
                side_idx,
                expand_ratio=float(getattr(self.cfg, "garment_silhouette_expand_ratio", 0.026)),
                normal_ratio=float(getattr(self.cfg, "garment_silhouette_normal_ratio", 0.006)),
            )
            if len(side_pts) and len(side_faces):
                vertices_out = np.concatenate([shell_pts, side_pts], axis=0).astype(np.float32)
                faces_out = np.concatenate([layer_faces.astype(np.int64), side_faces.astype(np.int64) + len(shell_pts)], axis=0)
            else:
                vertices_out = shell_pts.astype(np.float32)
                faces_out = layer_faces.astype(np.int64)
            meta.update({
                "layer": "garment",
                "layer_faces": int(layer_meta.get("faces", 0)),
                "layer_vertices": int(layer_meta.get("vertices", 0)),
                "silhouette_faces": int(side_meta.get("faces", 0)),
                "silhouette_vertices": int(side_meta.get("vertices", 0)),
            })
            return vertices_out, faces_out, meta
        if mode == "hair":
            if not has_hair_region:
                meta.update({"layer": "hair", "layer_faces": 0, "layer_vertices": 0, "disabled": not hair_enabled})
                return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64), meta
            layer_faces, idx, layer_meta = _subset_mesh_layer_from_weights(face_arr, hair_region_w, threshold=0.15, fallback_top_fraction=0.0)
            side_faces, side_idx, side_meta = _build_silhouette_side_layer(layer_faces, idx)
            shell_pts = combined_pts[np.asarray(idx, dtype=np.int64)] if len(idx) else np.zeros((0, 3), dtype=np.float32)
            side_pts = _make_silhouette_side_vertices(
                combined_pts,
                nrm,
                side_idx,
                expand_ratio=float(getattr(self.cfg, "hair_silhouette_expand_ratio", 0.040)),
                normal_ratio=float(getattr(self.cfg, "hair_silhouette_normal_ratio", 0.010)),
            )
            if len(side_pts) and len(side_faces):
                vertices_out = np.concatenate([shell_pts, side_pts], axis=0).astype(np.float32)
                faces_out = np.concatenate([layer_faces.astype(np.int64), side_faces.astype(np.int64) + len(shell_pts)], axis=0)
            else:
                vertices_out = shell_pts.astype(np.float32)
                faces_out = layer_faces.astype(np.int64)
            meta.update({
                "layer": "hair",
                "layer_faces": int(layer_meta.get("faces", 0)),
                "layer_vertices": int(layer_meta.get("vertices", 0)),
                "silhouette_faces": int(side_meta.get("faces", 0)),
                "silhouette_vertices": int(side_meta.get("vertices", 0)),
            })
            return vertices_out, faces_out, meta
        return combined_pts.astype(np.float32), face_arr.astype(np.int64), meta

    def _run_impl(self) -> tuple[np.ndarray, str]:
        max_points = _mesh_preview_effective_max_points(self.cfg)
        yaw = float(getattr(self.cfg, "mesh_preview_yaw", 0.0) or 0.0)
        pitch = float(getattr(self.cfg, "mesh_preview_pitch", 0.0) or 0.0)
        if self.mode == "pointcloud":
            vertices, faces, meta = self._preview_structure_geometry("combined")
            sample_count = min(int(max_points), 24000)
            sample_count = max(2000, sample_count)
            spec = make_surface_sample_spec(vertices, faces, sample_count, seed=9100003)
            points = sample_mesh_surface_with_spec(vertices, faces, spec)
            bgr = _mesh_preview_pointcloud_bgr(points, vertices, faces, "最终点云", "from detail mesh preview", max_points=max_points, yaw_deg=yaw, pitch_deg=pitch)
            return bgr, f"点云预览完成：{len(points)} 点。"
        if self.mode in {"garment", "hair", "detail", "combined"}:
            out_mode = "combined" if self.mode in {"detail", "combined"} else self.mode
            vertices, faces, meta = self._preview_structure_geometry(out_mode)
            shell = float(meta.get("mean_shell_offset", 0.0)) if isinstance(meta, dict) else 0.0
            seg_note = "mask" if isinstance(meta, dict) and bool(meta.get("segmentation_mask_used", False)) else "geo"
            title_map = {"garment": "Garment Mesh", "hair": "Hair Mesh", "combined": "Combined Mesh"}
            color_map = {"garment": (190, 120, 35), "hair": (42, 105, 210), "combined": (92, 101, 116)}
            detail = f"mode={out_mode} segments={meta.get('dense_segments', '-') if isinstance(meta, dict) else '-'} shell_mean={shell:.5f} region={seg_note}"
            fallback_note = ""
            if isinstance(meta, dict) and bool(meta.get("frame_fallback_used", False)):
                fallback_note = f"第 {meta.get('requested_frame', self.frame_index)} 帧未在缓存内，已显示第 {meta.get('preview_frame', '-')} 帧。"
                detail = fallback_note
            if out_mode in {"garment", "hair"} and len(vertices) == 0:
                disabled = bool(meta.get("disabled", False)) if isinstance(meta, dict) else False
                if disabled:
                    detail = "layer shell disabled; current output is Body Only"
                elif isinstance(meta, dict) and not bool(meta.get("segmentation_mask_used", False)):
                    detail = "no segmentation cache; current output is Body Only"
                else:
                    detail = "segmentation exists but no usable region on this frame"
            bgr = _mesh_preview_mesh_bgr(vertices, faces, title_map.get(out_mode, "Combined Mesh"), detail, max_points=max_points, fill_color=color_map.get(out_mode, (92, 101, 116)), yaw_deg=yaw, pitch_deg=pitch)
            if out_mode in {"garment", "hair"} and len(vertices) == 0:
                return bgr, f"{title_map.get(out_mode, 'Mesh')} 没有可用分割区域；当前是 Body Only。"
            status = f"{title_map.get(out_mode, 'Combined Mesh')} 预览完成。"
            if fallback_note:
                status += " " + fallback_note
            return bgr, status
        vertices, faces, _meta = self._preview_structure_geometry("body")
        detail = "局部时序预览；导出会用完整序列平滑"
        status = "Body Mesh 预览完成（已应用局部时序平滑，完整序列以导出为准）。"
        if isinstance(_meta, dict) and bool(_meta.get("frame_fallback_used", False)):
            note = f"第 {_meta.get('requested_frame', self.frame_index)} 帧未在缓存内，已显示第 {_meta.get('preview_frame', '-')} 帧。"
            detail = note
            status += " " + note
        bgr = _mesh_preview_mesh_bgr(vertices, faces, "Body Mesh", detail, max_points=max_points, yaw_deg=yaw, pitch_deg=pitch)
        return bgr, status


class PreviewWorker(QObject):
    log = Signal(str)
    finished = Signal(object, object, object, object, float, int)
    failed = Signal(str)

    def __init__(self, cfg: JobConfig, frame_index: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.frame_index = max(0, int(frame_index))

    def run(self) -> None:
        msg = "当前清理版已删除旧深度预览；请使用 Body / Garment / Hair / Combined 网格预览。"
        event_log(msg, channel="PREVIEW")
        self.failed.emit(msg)


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def collect_unused_model_candidates() -> list[Path]:
    candidates: list[Path] = []
    for p in [
        PROJECT_HF_HUB / "models--old-depth--old-depth-cache",
        PROJECT_HF_HUB / "models--facebook--old-normal-cache-0.3b-torchscript",
        PROJECT_DIR / "models" / "video_depth_anything",
        PROJECT_DIR / "vendor" / "old-depth-vendor",
    ]:
        if p.exists():
            candidates.append(p)
    if PROJECT_HF_HUB.exists():
        candidates.extend(PROJECT_HF_HUB.rglob("*.incomplete"))
    return candidates

def delete_unused_models(candidates: list[Path]) -> tuple[list[str], list[str]]:
    removed: list[str] = []
    errors: list[str] = []
    for item in candidates:
        try:
            if item.is_dir():
                safe_rmtree(item)
            elif item.exists():
                item.unlink()
            if not item.exists():
                removed.append(str(item))
            else:
                errors.append(f"删除失败: {item}")
        except OSError as exc:
            errors.append(f"删除失败: {item} -> {exc}")
    return removed, errors

class LocalModelManagerDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("本地模型管理")
        self.resize(780, 540)
        layout = QVBoxLayout(self)
        note = QLabel("当前主线只需要 4DHumans/SMPL/FASHN。这里仅列出可清理的旧深度/旧法线缓存。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.text_box = QPlainTextEdit()
        self.text_box.setReadOnly(True)
        layout.addWidget(self.text_box, 1)
        row = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新")
        self.open_btn = QPushButton("打开 models")
        self.clear_mem_btn = QPushButton("清空内存缓存")
        self.delete_btn = QPushButton("删除旧缓存")
        self.close_btn = QPushButton("关闭")
        for btn in [self.refresh_btn, self.open_btn, self.clear_mem_btn, self.delete_btn]:
            row.addWidget(btn)
        row.addStretch(1)
        row.addWidget(self.close_btn)
        layout.addLayout(row)
        self.refresh_btn.clicked.connect(self.refresh)
        self.open_btn.clicked.connect(self.open_models_dir)
        self.clear_mem_btn.clicked.connect(self.clear_memory_cache)
        self.delete_btn.clicked.connect(self.delete_unused)
        self.close_btn.clicked.connect(self.accept)
        self.refresh()

    def refresh(self) -> None:
        lines = [f"项目目录: {PROJECT_DIR}", f"模型目录: {PROJECT_MODELS_DIR}", ""]
        candidates = collect_unused_model_candidates()
        if not candidates:
            lines.append("未发现旧深度/旧法线缓存。")
        total = 0
        for p in candidates:
            size = directory_size_bytes(p)
            total += size
            lines.append(f"[可删] {p.name}")
            lines.append(f"  大小: {format_bytes(size)}")
            lines.append(f"  路径: {p}")
            lines.append("")
        lines.append(f"合计: {format_bytes(total)}")
        self.text_box.setPlainText("\n".join(lines))

    def open_models_dir(self) -> None:
        PROJECT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(PROJECT_MODELS_DIR))  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"无法打开目录: {exc}")

    def clear_memory_cache(self) -> None:
        clear_memory_model_cache()
        QMessageBox.information(self, APP_NAME, "已清空内存模型缓存。")

    def delete_unused(self) -> None:
        candidates = collect_unused_model_candidates()
        if not candidates:
            QMessageBox.information(self, APP_NAME, "没有发现可删除旧缓存。")
            return
        preview = "\n".join(str(p) for p in candidates[:20])
        if QMessageBox.question(
            self,
            APP_NAME,
            "确认删除这些旧缓存？\n\n" + preview,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        removed, errors = delete_unused_models(candidates)
        clear_memory_model_cache()
        QMessageBox.information(self, APP_NAME, "删除完成。\n" + "\n".join(removed + errors))
        self.refresh()




class OriginalFrameWorker(QObject):
    finished = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, input_path: str, frame_index: int, output_width: int, output_height: int) -> None:
        super().__init__()
        self.input_path = input_path
        self.frame_index = max(0, int(frame_index))
        self.output_width = int(output_width)
        self.output_height = int(output_height)

    def run(self) -> None:
        cap = None
        try:
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                raise RuntimeError("无法打开视频。")
            cap.release()
            frame_bgr = read_video_frame_bgr(self.input_path, self.frame_index)
            if frame_bgr is None:
                raise RuntimeError("无法读取原始帧：当前视频随机定位失败。")
            frame_bgr = cv2.resize(
                frame_bgr,
                (self.output_width, self.output_height),
                interpolation=cv2.INTER_AREA,
            )
            self.finished.emit(self.frame_index, frame_bgr)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self.frame_index, str(exc))
        finally:
            if cap is not None:
                cap.release()












class _BaseRebuildWorker(QObject):
    """Runs make_base_gray_for_levels in a background thread to avoid blocking the UI."""

    finished = Signal(object, object, object)  # base_gray, hist_gray, key
    failed = Signal(str, object)  # error, key

    def __init__(
        self,
        depth: np.ndarray,
        subject_mask: Optional[np.ndarray],
        normal_map: Optional[np.ndarray],
        invert: bool,
        black_pct: float,
        white_pct: float,
        gamma: float,
        detail_boost: int,
        normal_strength: int,
        normal_refine: int,
        depth_smooth: int,
        edge_preserve: int,
        target_shape: tuple[int, int],
        key: tuple,
    ) -> None:
        super().__init__()
        self._depth = depth
        self._subject_mask = subject_mask
        self._normal_map = normal_map
        self._invert = invert
        self._black_pct = black_pct
        self._white_pct = white_pct
        self._gamma = gamma
        self._detail_boost = detail_boost
        self._normal_strength = normal_strength
        self._normal_refine = normal_refine
        self._depth_smooth = depth_smooth
        self._edge_preserve = edge_preserve
        self._target_shape = target_shape
        self._key = key

    def run(self) -> None:
        try:
            base_gray = make_base_gray_for_levels(
                self._depth,
                self._invert,
                self._black_pct,
                self._white_pct,
                self._gamma,
                self._detail_boost,
                self._normal_strength,
                self._normal_refine,
                self._depth_smooth,
                self._edge_preserve,
                subject_mask=self._subject_mask,
                normal_map=self._normal_map,
            )
            th, tw = self._target_shape
            if base_gray.shape[:2] != (th, tw):
                hist_gray = cv2.resize(base_gray, (tw, th), interpolation=cv2.INTER_CUBIC)
            else:
                hist_gray = base_gray
            self.finished.emit(base_gray, hist_gray, self._key)
        except Exception as exc:  # noqa: BLE001
            event_exception("融合底图重建失败", exc, key=self._key)
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}", self._key)


