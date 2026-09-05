#!/usr/bin/env python3
"""Dependency-free regression for encoder mode/display mapping extraction."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from common.encoder_display import (  # noqa: E402
    ENCODER_DISPLAY_MODE,
    ENCODER_MODE_DISPLAY,
    encoder_display_name,
    encoder_internal_name,
)

EXPECTED = {
    "FFmpeg H.264": "黑白 MP4",
    "FFmpeg H.264 10-bit": "黑白 MP4 10-bit",
    "FFmpeg H.265 10-bit": "黑白 H.265 10-bit",
    "FFmpeg H.264 NVENC": "黑白 MP4 NVENC",
    "OpenCV mp4v": "兼容 MP4",
    "PNG序列 16-bit": "16bit PNG 序列",
}


def main() -> None:
    if ENCODER_MODE_DISPLAY != EXPECTED:
        raise SystemExit(f"encoder display mapping changed: {ENCODER_MODE_DISPLAY!r}")
    if ENCODER_DISPLAY_MODE != {display: mode for mode, display in EXPECTED.items()}:
        raise SystemExit("encoder reverse mapping is not the exact inverse")

    for mode, display in EXPECTED.items():
        if encoder_display_name(mode) != display:
            raise SystemExit(f"display lookup failed for {mode!r}")
        if encoder_internal_name(display) != mode:
            raise SystemExit(f"reverse lookup failed for {display!r}")
        if encoder_internal_name(mode) != mode:
            raise SystemExit(f"internal mode passthrough failed for {mode!r}")

    unknown = "future encoder"
    if encoder_display_name(unknown) != unknown or encoder_internal_name(unknown) != unknown:
        raise SystemExit("unknown encoder values must remain forward-compatible passthroughs")

    ui_source = (APP / "depth_fusion_ui.py").read_text(encoding="utf-8")
    required_import = "from common.encoder_display import encoder_display_name, encoder_internal_name"
    if required_import not in ui_source:
        raise SystemExit("depth_fusion_ui.py is not using common.encoder_display")
    if "ENCODER_MODE_DISPLAY: dict[str, str]" in ui_source:
        raise SystemExit("encoder mapping was reintroduced into depth_fusion_ui.py")

    print(f"video_depth encoder display contract: PASS modes={len(EXPECTED)}")


if __name__ == "__main__":
    main()
