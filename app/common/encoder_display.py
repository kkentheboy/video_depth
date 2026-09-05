"""Stable encoder mode/display-name mapping shared by the UI."""

from __future__ import annotations

ENCODER_MODE_DISPLAY: dict[str, str] = {
    "FFmpeg H.264": "黑白 MP4",
    "FFmpeg H.264 10-bit": "黑白 MP4 10-bit",
    "FFmpeg H.265 10-bit": "黑白 H.265 10-bit",
    "FFmpeg H.264 NVENC": "黑白 MP4 NVENC",
    "OpenCV mp4v": "兼容 MP4",
    "PNG序列 16-bit": "16bit PNG 序列",
}
ENCODER_DISPLAY_MODE: dict[str, str] = {
    display: mode for mode, display in ENCODER_MODE_DISPLAY.items()
}


def encoder_display_name(mode: str) -> str:
    """Return the localized display label while preserving unknown modes."""
    text = str(mode)
    return ENCODER_MODE_DISPLAY.get(text, text)


def encoder_internal_name(display_or_mode: str) -> str:
    """Return the internal encoder mode while preserving unknown values."""
    text = str(display_or_mode)
    return ENCODER_DISPLAY_MODE.get(text, text)
