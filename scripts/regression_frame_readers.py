#!/usr/bin/env python3
"""Numeric regression for image-backed sequential frame readers."""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

# The image-backed paths under test do not require the full Qt-heavy core module.
# Supply only the three video fallbacks imported by frame_readers so this
# regression stays inside the numeric-stack Gate.
fake_core = types.ModuleType("depth_fusion_core")
fake_core.probe_video = lambda _path: None
fake_core.read_video_frame_bgr = lambda _path, _index: None
fake_core.video_has_real_alpha = lambda _path: False
sys.modules["depth_fusion_core"] = fake_core

from depth_pipeline.frame_readers import (  # noqa: E402
    _SequentialBgrFrameReader,
    _SequentialRgbaFrameReader,
    _resize_gray01_for_worker,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    raw = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    resized = _resize_gray01_for_worker(raw, (4, 6))
    require(resized.shape == (4, 6), f"resize shape mismatch: {resized.shape}")
    require(resized.dtype == np.float32, f"resize dtype mismatch: {resized.dtype}")
    require(float(resized.min()) >= 0.0 and float(resized.max()) <= 1.0, "resize range escaped 0..1")
    require(float(resized.max()) > 0.95, "uint8 normalization lost white point")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        bgr = np.zeros((5, 7, 3), dtype=np.uint8)
        bgr[..., 0] = 17
        bgr[..., 1] = np.arange(7, dtype=np.uint8)[None, :] * 20
        bgr[..., 2] = np.arange(5, dtype=np.uint8)[:, None] * 30
        bgr_path = root / "frame.png"
        require(bool(cv2.imwrite(str(bgr_path), bgr)), "failed to write BGR fixture")

        bgr_reader = _SequentialBgrFrameReader(str(bgr_path))
        first = bgr_reader.read(0)
        require(first is not None, "BGR image reader returned None")
        require(np.array_equal(first, bgr), "BGR image reader changed fixture bytes")
        first[0, 0, :] = 255
        second = bgr_reader.read(99)
        require(second is not None, "BGR image reader second read returned None")
        require(np.array_equal(second, bgr), "BGR reader did not return an isolated copy")
        bgr_reader.close()

        bgra = np.zeros((4, 6, 4), dtype=np.uint8)
        bgra[..., :3] = (12, 34, 56)
        bgra[:, :3, 3] = 0
        bgra[:, 3:, 3] = 255
        rgba_path = root / "alpha.png"
        require(bool(cv2.imwrite(str(rgba_path), bgra)), "failed to write BGRA fixture")

        rgba_reader = _SequentialRgbaFrameReader(str(rgba_path))
        require(rgba_reader.available, "RGBA image reader did not mark alpha image available")
        rgba = rgba_reader.read_rgba(12)
        require(rgba is not None, "RGBA image reader returned None")
        require(np.array_equal(rgba, bgra), "RGBA image reader changed fixture bytes")

        alpha = rgba_reader.read_alpha01(0, (8, 12))
        require(alpha is not None, "alpha extraction unexpectedly rejected mixed alpha")
        require(alpha.shape == (8, 12), f"alpha resize shape mismatch: {alpha.shape}")
        require(alpha.dtype == np.float32, f"alpha dtype mismatch: {alpha.dtype}")
        require(float(alpha.min()) <= 0.01 and float(alpha.max()) >= 0.99, "alpha endpoints changed")
        require(0.40 <= float(alpha.mean()) <= 0.60, f"alpha coverage changed: {float(alpha.mean()):.4f}")
        rgba_reader.close()

    print("video_depth frame reader numeric round trip: PASS")


if __name__ == "__main__":
    main()
