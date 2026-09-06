# -*- coding: utf-8 -*-
from __future__ import annotations

from depth_fusion_core import QTimer, format_seconds


class PreviewFrameMixin:
    def _preview_frame_control_pairs(self) -> tuple[tuple[object | None, object | None], ...]:
        return (
            (getattr(self, "preview_frame_slider", None), getattr(self, "preview_frame_spin", None)),
            (getattr(self, "input_preview_frame_slider", None), getattr(self, "input_preview_frame_spin", None)),
            (getattr(self, "structure_preview_frame_slider", None), getattr(self, "structure_preview_frame_spin", None)),
        )

    def _all_preview_frame_controls(self) -> list[object]:
        controls: list[object] = []
        for slider, spin in self._preview_frame_control_pairs():
            if slider is not None:
                controls.append(slider)
            if spin is not None:
                controls.append(spin)
        return controls

    def set_preview_frame_range(self, frame_count: int) -> None:
        max_frame = max(0, int(frame_count) - 1)
        default_frame = max_frame // 2 if max_frame > 0 else 0
        page_step = max(1, int((self.video_info.fps if self.video_info else 24) or 24))
        for ctrl in self._all_preview_frame_controls():
            try:
                ctrl.blockSignals(True)
                ctrl.setRange(0, max_frame)
                ctrl.setValue(default_frame)
                if hasattr(ctrl, "setPageStep"):
                    ctrl.setPageStep(page_step)
            finally:
                try:
                    ctrl.blockSignals(False)
                except Exception:
                    pass
        if hasattr(self, "export_frame_slider"):
            self.export_frame_slider.blockSignals(True)
            self.export_frame_slider.setRange(0, max_frame)
            self.export_frame_slider.setValue(default_frame)
            self.export_frame_slider.setPageStep(page_step)
            self.export_frame_slider.blockSignals(False)
        self._set_processing_frame_range(0, max_frame, reset_values=True)
        self.update_preview_frame_label(default_frame)

    def update_preview_frame_label(self, frame_index: int) -> None:
        if not self.video_info:
            text = "第 0 帧 / 0:00"
            self.preview_frame_label.setText(text)
            if hasattr(self, "export_frame_label"):
                self.export_frame_label.setText(text)
            return
        fps = max(1e-3, float(self.video_info.fps or 24.0))
        seconds = frame_index / fps
        total = max(0, self.video_info.frame_count - 1)
        text = f"第 {frame_index}/{total} 帧 / {format_seconds(seconds)}"
        self.preview_frame_label.setText(text)
        if hasattr(self, "export_frame_label"):
            self.export_frame_label.setText(text)
        if hasattr(self, "export_frame_slider") and self.export_frame_slider.value() != int(frame_index):
            self.export_frame_slider.blockSignals(True)
            self.export_frame_slider.setValue(int(frame_index))
            self.export_frame_slider.blockSignals(False)

    def _apply_preview_frame_value(self, value: int, *, refresh_mesh: bool = True) -> None:
        max_frame = max(0, int(self.video_info.frame_count) - 1) if self.video_info else 0
        value = max(0, min(int(value), max_frame))
        for ctrl in self._all_preview_frame_controls():
            try:
                ctrl.blockSignals(True)
                ctrl.setValue(value)
            finally:
                try:
                    ctrl.blockSignals(False)
                except Exception:
                    pass
        self.update_preview_frame_label(value)
        self.show_original_frame_immediately(value)
        if hasattr(self, "preview_status_label"):
            self.preview_status_label.setText("已切换帧，正在读取原视频预览...")
        QTimer.singleShot(80, self._refresh_reference_preview_tiles)
        if refresh_mesh:
            self._schedule_active_mesh_preview_refresh()

    def on_preview_frame_slider_changed(self, value: int) -> None:
        self._apply_preview_frame_value(int(value))

    def on_preview_frame_spin_changed(self) -> None:
        self._apply_preview_frame_value(int(self.preview_frame_spin.value()))

    def toggle_preview_playback(self) -> None:
        if not self.current_input or not self.video_info:
            return
        self._preview_playing = not bool(getattr(self, "_preview_playing", False))
        if self._preview_playing:
            fps = max(1.0, min(60.0, float(self.video_info.fps or 24.0)))
            self.preview_play_timer.setInterval(max(16, int(round(1000.0 / fps))))
            self.preview_play_timer.start()
            if hasattr(self, "input_preview_play_btn"):
                self.input_preview_play_btn.setText("暂停")
        else:
            self.preview_play_timer.stop()
            if hasattr(self, "input_preview_play_btn"):
                self.input_preview_play_btn.setText("播放")

    def _advance_preview_playback(self) -> None:
        if not self.current_input or not self.video_info:
            self._preview_playing = False
            self.preview_play_timer.stop()
            return
        start, end = self._processing_range_values()
        if end < start:
            start, end = 0, max(0, int(self.video_info.frame_count) - 1)
        cur = int(self.preview_frame_spin.value()) if hasattr(self, "preview_frame_spin") else start
        nxt = cur + 1
        if nxt > end:
            nxt = start
        self._apply_preview_frame_value(nxt, refresh_mesh=False)

