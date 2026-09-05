# -*- coding: utf-8 -*-
from __future__ import annotations

import atexit
import contextlib
import gc
import hashlib
import importlib
import io
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Optional, Any

import cv2
import numpy as np

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, QSize, Signal, qInstallMessageHandler
from PySide6.QtGui import QColor, QCursor, QDragEnterEvent, QDropEvent, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

APP_NAME = "视频人体网格重建"
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DATA_DIR = PROJECT_DIR / "data"
RESOURCES_DIR = PROJECT_DATA_DIR / "resources"
PROJECT_MODELS_DIR = PROJECT_DATA_DIR / "models"
PROJECT_CACHE_DIR = PROJECT_DATA_DIR / "cache"
PROJECT_LOG_DIR = PROJECT_DATA_DIR / "logs"
PROJECT_HF_HOME = PROJECT_MODELS_DIR / "huggingface"
PROJECT_HF_HUB = PROJECT_HF_HOME / "hub"
PROJECT_LOG_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

APP_STYLESHEET = """
/* ── Base ── */
QWidget {
    background: #09090b;
    color: #e4e4e7;
    font-size: 13px;
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
}

/* ── Buttons ── */
QPushButton {
    background: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 7px;
    padding: 7px 14px;
    color: #e4e4e7;
    font-size: 13px;
}
QPushButton:hover {
    background: #3f3f46;
    border-color: #52525b;
}
QPushButton:pressed {
    background: #52525b;
}
QPushButton:disabled {
    color: #52525b;
    background: #18181b;
    border-color: #27272a;
}

/* ── Input controls ── */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #0f0f12;
    border: 1px solid #3f3f46;
    border-radius: 6px;
    padding: 6px 8px;
    color: #e4e4e7;
    selection-background-color: #2563eb;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #3b82f6;
}

QLineEdit#dropZone {
    background: #0d0d10;
    border: 1.5px dashed #3f3f46;
    border-radius: 8px;
    color: #52525b;
    font-size: 13px;
    padding: 10px;
}
QLineEdit#dropZone:hover {
    border-color: #2563eb;
    background: #0f0f14;
}
QLineEdit#dropZone[dragging="true"] {
    border-color: #3b82f6;
    background: #0f1420;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #18181b;
    border: 1px solid #3f3f46;
    selection-background-color: #2563eb;
    color: #e4e4e7;
}

/* ── Text areas ── */
QPlainTextEdit {
    background: #0a0a0c;
    border: 1px solid #27272a;
    border-radius: 6px;
    padding: 6px;
    color: #a1a1aa;
    font-family: 'Consolas', 'Cascadia Code', 'Courier New', monospace;
    font-size: 11px;
    line-height: 1.4;
}

/* ── Frames & containers ── */
QFrame, QGroupBox, QScrollArea {
    border: none;
    border-radius: 0px;
    background: transparent;
}

/* ── Progress bars ── */
QProgressBar {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 5px;
    text-align: center;
    color: #a1a1aa;
    font-size: 11px;
    min-height: 18px;
    max-height: 22px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #3b82f6);
    border-radius: 4px;
}

/* ── Checkboxes ── */
QCheckBox {
    spacing: 6px;
    color: #d4d4d8;
    font-size: 12px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #52525b;
    background: #18181b;
}
QCheckBox::indicator:checked {
    background: #18181b;
    border-color: #3b82f6;
    image: url(app/resources/checkmark.svg);
}
QCheckBox::indicator:disabled {
    background: #27272a;
    border-color: #3f3f46;
}

/* ── Scroll bars ── */
QScrollBar:vertical {
    background: transparent;
    width: 7px;
    margin: 2px 0;
}
QScrollBar::handle:vertical {
    background: #3f3f46;
    min-height: 30px;
    border-radius: 3px;
}
QScrollBar::handle:vertical:hover {
    background: #52525b;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
    height: 0px;
}
QScrollBar:horizontal {
    background: transparent;
    height: 7px;
    margin: 0 2px;
}
QScrollBar::handle:horizontal {
    background: #3f3f46;
    min-width: 30px;
    border-radius: 3px;
}
QScrollBar::handle:horizontal:hover {
    background: #52525b;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
    width: 0px;
}

/* ── Sliders ── */
QSlider::groove:horizontal {
    height: 4px;
    background: #27272a;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #3b82f6;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    border: 2px solid #1d4ed8;
}
QSlider::handle:horizontal:hover {
    background: #60a5fa;
}

/* ── Tooltips ── */
QToolTip {
    background: #27272a;
    color: #e4e4e7;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}
"""

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
CACHE_FORMAT_VERSION = 15
SAFE_DEFAULT_LONG_SIDE = 1024
SAFE_DEFAULT_PROCESS_RES = 1024
MAX_SAFE_LONG_SIDE_HINT = 2048
MODEL_IDS = {"图像驱动网格主流程": "image-driven-mesh"}
AUX_MODEL_SPECS: dict[str, dict] = {}
BUILTIN_PRESETS: dict[str, dict] = {}
ENCODER_MODES = ["USDA Mesh"]
NORMALIZE_MODES = ["不使用"]
DEFAULT_MATANYONE_MODEL_PATH = PROJECT_MODELS_DIR / "matanyone" / "matanyone.pth"
DEFAULT_MATTING_MASK_DIR = PROJECT_DIR / "data" / "masks"
LEGACY_AUX_INPUT_H = 0
LEGACY_AUX_INPUT_W = 0
LEGACY_AUX_MEAN = (0.0, 0.0, 0.0)
LEGACY_AUX_STD = (1.0, 1.0, 1.0)
TONE_RANGE_BASE_CENTERS = [0.1, 0.3, 0.5, 0.7, 0.9]
TONE_RANGE_MIN_GAPS = [0.02, 0.02, 0.02, 0.02]
TONE_RANGE_WIDTHS = [0.2, 0.2, 0.2, 0.2, 0.2]

_event_listeners: list[Callable[[str], None]] = []
_stdio_installed = False

def current_event_log_path() -> Path:
    PROJECT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECT_LOG_DIR / "events.log"

def add_event_listener(cb: Callable[[str], None]) -> None:
    if cb not in _event_listeners:
        _event_listeners.append(cb)

def remove_event_listener(cb: Callable[[str], None]) -> None:
    if cb in _event_listeners:
        _event_listeners.remove(cb)

def event_log(text: str, *, channel: str = "APP") -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] [{channel}] {text}"
    try:
        with current_event_log_path().open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    if os.environ.get("DEPTH_FUSION_CONSOLE_EVENTS", "").lower() in {"1", "true", "yes", "on"}:
        try:
            print(line, flush=True)
        except Exception:
            pass
    for cb in list(_event_listeners):
        try:
            cb(line)
        except Exception:
            pass

def event_exception(title: str, exc: BaseException, **extra: Any) -> None:
    detail = f"{title}: {exc}"
    if extra:
        detail += " " + json.dumps(extra, ensure_ascii=False, default=str)
    event_log(detail, channel="ERROR")
    event_log(traceback.format_exc(), channel="ERROR")

def init_event_logger() -> None:
    current_event_log_path().parent.mkdir(parents=True, exist_ok=True)
    event_log("事件日志初始化", channel="APP")

def install_global_event_hooks() -> None:
    def excepthook(tp, val, tb):
        event_log("未捕获异常: " + "".join(traceback.format_exception(tp, val, tb)), channel="ERROR")
        sys.__excepthook__(tp, val, tb)
    sys.excepthook = excepthook

def install_stdio_event_tee() -> None:
    global _stdio_installed
    if _stdio_installed:
        return
    _stdio_installed = True
    # Keep stdout/stderr intact. The bat already uses unbuffered mode; event_log writes file.

def verbose_third_party_output() -> bool:
    return os.environ.get("DEPTH_FUSION_VERBOSE_THIRD_PARTY", "").lower() in {"1", "true", "yes", "on"}

def quiet_third_party_output():
    return contextlib.nullcontext()

@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    has_alpha: bool = False

@dataclass
class CacheEntry:
    path: str = ""
    signature: str = ""
    status: str = ""

@dataclass
class ExternalDepthFusionState:
    enabled: bool = False

@dataclass(init=False)
class JobConfig:
    def __init__(self, **kwargs: Any) -> None:
        defaults = dict(
            input_path="", output_path="", output_width=0, output_height=0, fps=25.0,
            model_id="image-driven-mesh", device_mode="auto", process_res=1024,
            encoder_mode="USDA Mesh", normalize_mode="不使用", cache_enabled=True,
            pointcloud_mode="structure_xyz", pointcloud_usd_sequence=False,
            mesh_export_enabled=True, garment_mesh_export_enabled=True, hair_mesh_export_enabled=True,
            combined_mesh_export_enabled=True, detail_mesh_export_enabled=True,
            garment_shell_enabled=True, hair_shell_enabled=True, dense_mesh_level=1,
            garment_shell_offset=0.020, hair_shell_offset=0.035,
            garment_silhouette_expand_ratio=0.026, hair_silhouette_expand_ratio=0.040,
            garment_silhouette_normal_ratio=0.006, hair_silhouette_normal_ratio=0.010,
            pointcloud_max_points=60000, pointcloud_stride=1,
            segmentation_enabled=True, segmentation_provider="FASHN Human Parser",
            segmentation_fallback_geometry=False,
            processing_start_frame=0, processing_end_frame=-1,
            mesh_preview_yaw=0.0, mesh_preview_pitch=0.0,
            project_dir="",
        )
        defaults.update(kwargs)
        self.__dict__.update(defaults)

class FfmpegRawVideoWriter:
    def __init__(self, *a, **k):
        raise RuntimeError("当前清理版只输出 USDA 网格，不再输出视频深度。")

class PngSequenceWriter:
    def __init__(self, *a, **k):
        raise RuntimeError("当前清理版只输出 USDA 网格，不再输出深度 PNG 序列。")

def short_error_message(text: str, limit: int = 320) -> str:
    s = str(text).replace("\n", " ").strip()
    return s if len(s) <= limit else s[:limit] + "..."

def format_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:.1f}s"
    return f"{sec/60:.1f}min"

def format_bytes(n: int | float) -> str:
    n = float(n or 0)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024

def directory_size_bytes(path: Path) -> int:
    try:
        p = Path(path)
        if not p.exists():
            return 0
        if p.is_file():
            return p.stat().st_size
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except Exception:
        return 0

def even_int(v: int) -> int:
    v = int(round(v))
    return v if v % 2 == 0 else v + 1

def scaled_size_from_long_side(w: int, h: int, long_side: int) -> tuple[int, int]:
    w, h, long_side = int(w), int(h), int(long_side)
    if w <= 0 or h <= 0 or long_side <= 0:
        return max(2, w), max(2, h)
    scale = long_side / max(w, h)
    return even_int(max(2, w * scale)), even_int(max(2, h * scale))

def probe_video(path: str | Path) -> VideoInfo:
    """Fast video probe used by UI/project opening.

    Do not decode/sample alpha here. The previous version called ffmpeg while
    opening a project, so the main window stayed hidden until alpha probing
    timed out. Alpha is handled later by the actual frame/preprocess readers.
    """
    p = str(path)
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {p}")
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return VideoInfo(path=p, width=w, height=h, fps=fps if fps > 0 else 25.0, frame_count=max(0, total), has_alpha=False)

def _resize_alpha_like(alpha: np.ndarray | None, shape_hw: tuple[int, int] | None) -> np.ndarray | None:
    if alpha is None:
        return None
    arr = np.asarray(alpha, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.size <= 1:
        return None
    if float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    coverage = float(np.mean(arr > 0.01)) if arr.size else 0.0
    # All-opaque/all-transparent alpha is not a useful subject matte.
    if not (0.0005 < coverage < 0.9995):
        return None
    if shape_hw is not None and arr.shape[:2] != tuple(shape_hw):
        th, tw = int(shape_hw[0]), int(shape_hw[1])
        arr = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _composite_bgr_on_black(frame_bgr: np.ndarray, alpha01: np.ndarray | None) -> np.ndarray:
    frame = np.asarray(frame_bgr)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return frame_bgr
    alpha = _resize_alpha_like(alpha01, frame.shape[:2])
    if alpha is None:
        return frame[..., :3].copy()
    out = frame[..., :3].astype(np.float32) * alpha[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _read_video_frame_bgr_raw(path: str | Path, frame_index: int = 0) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def read_video_frame_alpha01(
    path: str | Path,
    frame_index: int = 0,
    shape_hw: tuple[int, int] | None = None,
    *,
    allow_slow_video_alpha: bool = False,
) -> np.ndarray | None:
    """Read useful real alpha as float [0, 1].

    Still images are cheap and are checked directly. Video alpha sampling through
    ffmpeg is disabled by default because launching ffmpeg during project open
    made the UI appear only after a long event-log stall. Long-running pipeline
    stages should use the sequential alpha reader instead of per-frame ffmpeg.
    """
    p = Path(str(path))
    if not p.is_file():
        return None
    suffix = p.suffix.lower()
    if suffix in {".png", ".tif", ".tiff", ".webp"}:
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if arr is not None and arr.ndim == 3 and arr.shape[2] >= 4:
            return _resize_alpha_like(arr[..., 3].astype(np.float32) / 255.0, shape_hw)
        return None

    if suffix not in VIDEO_EXTS or not allow_slow_video_alpha:
        return None
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return None
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        return None
    # Accurate enough for random access/debug checks. Sequential paths use the
    # dedicated stream reader in structure_runner/depth_fusion_workers.
    t = max(0, int(frame_index))
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", str(p), "-vf", f"select=eq(n\\,{t})",
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=8, check=False)
        raw = proc.stdout or b""
        stride = w * h * 4
        if len(raw) < stride:
            return None
        rgba = np.frombuffer(raw[:stride], dtype=np.uint8).reshape((h, w, 4))
        return _resize_alpha_like(rgba[..., 3].astype(np.float32) / 255.0, shape_hw)
    except Exception:
        return None


def video_has_real_alpha(path: str | Path) -> bool:
    """Fast alpha check.

    This intentionally does not launch ffmpeg for video files. It is used by UI
    validation and project loading, so it must stay instant.
    """
    return read_video_frame_alpha01(path, 0, allow_slow_video_alpha=False) is not None


def describe_real_alpha_source(path: str | Path, frame_index: int = 0) -> tuple[bool, str]:
    p = Path(str(path or ""))
    if p.suffix.lower() in VIDEO_EXTS:
        return False, "视频 Alpha 不在打开项目时检测；预处理阶段按实际帧处理"
    ok = bool(read_video_frame_alpha01(p, int(frame_index), allow_slow_video_alpha=False) is not None)
    return ok, ("检测到 Alpha；透明区会自动合成黑色" if ok else "无可用 Alpha；按普通视频处理")


def read_video_frame_bgr(path: str | Path, frame_index: int = 0, *, allow_slow_video_alpha: bool = False) -> np.ndarray | None:
    """Read a BGR frame for the current pipeline.

    If a useful alpha channel exists, transparent pixels are composited to black
    immediately. This keeps 4D/WHAM/FASHN/preview seeing the same input.
    """
    p = Path(str(path))
    suffix = p.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if arr is None:
            return None
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] >= 4:
            return _composite_bgr_on_black(arr[..., :3], arr[..., 3].astype(np.float32) / 255.0)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[..., :3].copy()
        return None
    frame = _read_video_frame_bgr_raw(p, frame_index)
    if frame is None:
        return None
    alpha = read_video_frame_alpha01(p, int(frame_index), frame.shape[:2], allow_slow_video_alpha=allow_slow_video_alpha)
    return _composite_bgr_on_black(frame, alpha)

def bgr_to_pixmap(img: np.ndarray) -> QPixmap:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    h, w = arr.shape[:2]
    arr = np.ascontiguousarray(arr)
    qimg = QImage(arr.data, w, h, arr.strides[0], QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)

def get_project_cache_dir(cfg: JobConfig) -> Path:
    pdir = getattr(cfg, "project_dir", "")
    if pdir:
        return Path(pdir) / "cache"
    return PROJECT_CACHE_DIR

def _input_cache_key(path_text: str | Path) -> str:
    p = Path(str(path_text or ""))
    stem = p.stem or "input"
    stem = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in stem).strip("._-") or "input"
    try:
        st = p.stat()
        raw = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        raw = str(path_text or stem)
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"{stem}_{digest}"


def _safe_cache_part(text: str, fallback: str = "item") -> str:
    out = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(text or "")).strip("._-")
    return out or fallback


def structure_cache_root(cfg: JobConfig) -> Path:
    """Cache root for one input + one structure solver + one processing range.

    Earlier builds keyed the structure cache only by input filename/hash. Running
    WHAM after 4DHumans, or changing the in/out range, could overwrite/reuse the
    wrong cache. The workflow now treats 4DHumans and WHAM as two separate
    schemes and stores them independently.
    """
    input_key = _input_cache_key(getattr(cfg, "input_path", "input"))
    model = _safe_cache_part(str(getattr(cfg, "structure_model", "4dhumans") or "4dhumans").lower(), "4dhumans")
    start = max(0, int(getattr(cfg, "processing_start_frame", 0) or 0))
    end_raw = int(getattr(cfg, "processing_end_frame", -1) if getattr(cfg, "processing_end_frame", -1) is not None else -1)
    end_part = "end" if end_raw < 0 else f"{max(start, end_raw):06d}"
    range_key = f"f{start:06d}_e{end_part}"
    return get_project_cache_dir(cfg) / "structure" / input_key / model / range_key

def default_structure_output_dir(input_path: str | Path) -> str:
    p = Path(input_path)
    return str(p.with_name(p.stem + "_mesh_output"))

def default_output_path(input_path: str | Path, *a, **k) -> str:
    return default_structure_output_dir(input_path)

def pointcloud_export_root(input_path: str | Path, output_path: str | Path) -> Path:
    out = Path(output_path)
    return out if out.suffix == "" else out.parent

def partial_output_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(p.name + ".partial")

def remove_partial_output(path: str | Path) -> None:
    p = partial_output_path(path)
    if p.exists():
        try:
            if p.is_dir(): shutil.rmtree(p)
            else: p.unlink()
        except Exception: pass

def png_sequence_output_dir(path: str | Path) -> Path:
    p = Path(path)
    return p if p.suffix == "" else p.with_suffix("")

def partial_png_sequence_dir(path: str | Path) -> Path:
    return Path(str(path) + ".partial")

def frame_cache_root(cfg: JobConfig) -> Path:
    return get_project_cache_dir(cfg) / "frames" / (Path(str(getattr(cfg, "input_path", "input"))).stem or "input")

def frame_cache_paths(root: Path, idx: int) -> tuple[Path, Path]:
    stem = f"frame_{int(idx):06d}"
    return Path(root) / f"{stem}_mesh.npy", Path(root) / f"{stem}_aux.npy"

def frame_alpha_path(root: Path, idx: int) -> Path:
    return Path(root) / f"frame_{int(idx):06d}_alpha.npy"

def frame_refined_depth_path(root: Path, idx: int) -> Path:
    return Path(root) / f"frame_{int(idx):06d}_refined.npy"

def save_npy_safely(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp, arr)
    real_tmp = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npy")
    real_tmp.replace(path)

def try_load_npy(path: Path) -> np.ndarray | None:
    try:
        return np.load(path)
    except Exception:
        return None

def clear_memory_model_cache(log: Callable[[str], None] | None = None) -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if log: log("已清空内存模型缓存")

def trim_cuda_allocator_cache() -> None:
    clear_memory_model_cache()

def cuda_total_memory_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
    except Exception:
        pass
    return 0.0

def resolve_device_mode(mode: str = "auto") -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() and str(mode).lower() != "cpu" else "cpu"
    except Exception:
        return "cpu"

def ffmpeg_has_encoder(name: str) -> bool:
    return True

def estimate_vram_gb(*a, **k) -> float: return 0.0

def local_aux_model_file(*a, **k) -> Path: return PROJECT_MODELS_DIR / "unused"
def model_cache_dir_from_id(model_id: str) -> Path: return PROJECT_HF_HUB / str(model_id).replace("/", "--")
def ensure_aux_model_file(*a, **k): raise RuntimeError("当前清理版不包含旧辅助模型下载。")
def get_cached_aux_model(*a, **k): raise RuntimeError("当前清理版不包含旧辅助模型。")
def get_cached_model(*a, **k): raise RuntimeError("当前清理版不包含旧模型推理。")
def warmup_three_models(*a, **k): return None
def is_local_model_ready(*a, **k) -> bool: return False

def lightweight_environment_report() -> str:
    py = sys.executable
    lines = [f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} / {py}"]
    try:
        import torch
        cuda = "CUDA 可用" if torch.cuda.is_available() else "CUDA 不可用"
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        lines.append(f"Torch/CUDA: {torch.__version__} / {cuda} / {gpu}")
    except Exception as exc:
        lines.append(f"Torch/CUDA: 未检测到 ({short_error_message(str(exc))})")
    lines.append("当前清理版主线：4DHumans/WHAM structure cache + FASHN 分割 + USDA 网格导出。")
    return "\n".join(lines)

# Simple image/math helpers kept for UI compatibility.
def normalize_curve_points(points=None): return points or [(0.0, 0.0), (1.0, 1.0)]
def build_curve_lut(points=None): return np.arange(256, dtype=np.uint8)
def tone_range_reference_bands(*a, **k): return []
def build_sliding_ranges(*a, **k): return []
def curve_points_are_identity(*a, **k): return True
def apply_curve_lut(img, *a, **k): return img
def apply_levels(img, *a, **k): return img
def apply_tone_ranges(img, *a, **k): return img
def apply_input_adjustments_bgr(img, *a, **k): return img
def apply_input_adjustments_from_cfg(img, *a, **k): return img
def apply_subject_background_fill(img, *a, **k): return img
def apply_anti_banding(img, *a, **k): return img
def apply_depth_smoothing(img, *a, **k): return img
def apply_detail_boost(img, *a, **k): return img
def apply_depth_output_grade(img, *a, **k): return img
def apply_normal_depth_fusion(img, *a, **k): return img
def normal_guided_depth_refine(img, *a, **k): return img
def normalize_depth(arr, *a, **k):
    x=np.asarray(arr,dtype=np.float32); mn=float(np.nanmin(x)) if x.size else 0; mx=float(np.nanmax(x)) if x.size else 1
    return np.zeros_like(x) if mx-mn<1e-8 else (x-mn)/(mx-mn)
def render_depth_frame(arr, *a, **k): return (normalize_depth(arr)*255).astype(np.uint8)
def render_depth_frame_16bit(arr, *a, **k): return (normalize_depth(arr)*65535).astype(np.uint16)
def render_depth_gray_float(arr, *a, **k): return normalize_depth(arr)
def make_base_gray_for_levels(arr,*a,**k): return render_depth_frame(arr)
def robust_global_range(arr,*a,**k): return (0.0,1.0)
def depth_percentile_range(arr,*a,**k): return (0.0,1.0)
def compute_mask_depth_stats(*a, **k): return {}
def make_diagnostic_preview_bgr(*a, **k): return np.zeros((240,320,3),np.uint8)
def fast_bilateral_float(arr,*a,**k): return arr

def _resize_mask_like(mask, shape_hw):
    if mask is None: return None
    arr=np.asarray(mask,dtype=np.float32)
    if arr.ndim==3: arr=arr[...,0]
    h,w=shape_hw
    if arr.shape[:2]!=(h,w): arr=cv2.resize(arr,(w,h),interpolation=cv2.INTER_LINEAR)
    return np.clip(arr,0,1)
def _external_depth_gray01(*a, **k): return None

def subject_bbox_from_mask(mask): return None
def build_configured_subject_mask(*a, **k): return None
def build_alignment_subject_mask(*a, **k): return None
def postprocess_subject_mask(mask,*a,**k): return mask
def stabilize_subject_mask_temporal(mask,*a,**k): return mask
def temporal_ema_masked(x,*a,**k): return x
def temporal_align_depth_by_mask(x,*a,**k): return x
def depth_affine_to_reference(x,*a,**k): return x
def fuse_base_depth_with_external_reference(x,*a,**k): return x
def extract_subject_alpha_from_cutout_frame(*a, **k): return None
def extract_subject_alpha_from_depth_frame(*a, **k): return None
def extract_valid_alpha_from_external_depth_frame(*a, **k): return None
def load_alpha_mask_for_depth(*a, **k): return None
def read_external_subject_mask(*a, **k): return None
def read_external_depth_reference(*a, **k): return (None, None)
def read_external_depth_reference_with_alpha(*a, **k): return (None, None)
def read_masked_depth_video_frame(*a, **k): return (None, None)
def prepare_da3_input_frame(frame,*a,**k): return frame
def refine_depth_with_human_crop(x,*a,**k): return x
def map_aux_frame_index(idx,*a,**k): return idx
def is_direct_depth_video_workflow(*a,**k): return False
def is_likely_cutout_frame(*a,**k): return False
def is_scene_cut(*a,**k): return False
def scene_cut_score(*a,**k): return 0.0
def scene_cut_signature(*a,**k): return ""
def update_anchor_stats(*a,**k): return None
def mux_original_audio(*a,**k): return None
def list_cache_entries(*a, **k): return []
def clear_all_cache(*a, **k): return None
def clear_cache_entry(*a, **k): return None
def clear_cache_older_than(*a, **k): return None
def ensure_builtin_preset_files(*a, **k): return None
