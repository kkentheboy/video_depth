# -*- coding: utf-8 -*-
from __future__ import annotations

import traceback
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from depth_fusion_core import (
    JobConfig, event_exception, event_log, make_base_gray_for_levels, read_video_frame_bgr,
)


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

