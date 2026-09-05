# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Optional

import cv2
import numpy as np

from depth_fusion_core import probe_video, read_video_frame_bgr, video_has_real_alpha


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

