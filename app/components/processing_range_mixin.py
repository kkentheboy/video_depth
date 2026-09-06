# -*- coding: utf-8 -*-
from __future__ import annotations


class ProcessingRangeMixin:
    def _set_processing_frame_range(self, start_min: int, end_max: int, *, reset_values: bool = False) -> None:
        max_frame = max(0, int(end_max))
        widgets = [
            getattr(self, "processing_start_slider", None), getattr(self, "processing_end_slider", None),
            getattr(self, "processing_start_spin", None), getattr(self, "processing_end_spin", None),
        ]
        for widget in widgets:
            if widget is not None:
                widget.blockSignals(True)
                widget.setRange(0, max_frame)
        if reset_values:
            if getattr(self, "processing_start_slider", None) is not None:
                self.processing_start_slider.setValue(0)
            if getattr(self, "processing_start_spin", None) is not None:
                self.processing_start_spin.setValue(0)
            if getattr(self, "processing_end_slider", None) is not None:
                self.processing_end_slider.setValue(max_frame)
            if getattr(self, "processing_end_spin", None) is not None:
                self.processing_end_spin.setValue(max_frame)
        for widget in widgets:
            if widget is not None:
                widget.blockSignals(False)
        self._refresh_processing_range_label()

    def _processing_range_values(self) -> tuple[int, int]:
        if not self.video_info:
            return 0, -1
        max_frame = max(0, int(self.video_info.frame_count) - 1)
        start = int(getattr(self, "processing_start_spin", self.preview_frame_spin).value()) if hasattr(self, "processing_start_spin") else 0
        end = int(getattr(self, "processing_end_spin", self.preview_frame_spin).value()) if hasattr(self, "processing_end_spin") else max_frame
        start = max(0, min(start, max_frame))
        end = max(start, min(end, max_frame))
        return start, end

    def _refresh_processing_range_label(self) -> None:
        start, end = self._processing_range_values()
        total = max(0, int(self.video_info.frame_count) if self.video_info else 0)
        frames = max(0, end - start + 1) if end >= start else 0
        pct = 100 if total <= 0 else int(round(frames * 100 / max(1, total)))
        text = f"帧 {start} — {end}  /  共 {total} 帧  ({frames} 帧)"
        if hasattr(self, "processing_range_label"):
            self.processing_range_label.setText(text)
        if hasattr(self, "processing_range_progress"):
            self.processing_range_progress.setValue(max(0, min(100, pct)))
            self.processing_range_progress.setFormat(f"{max(0, min(100, pct))}%")

    def _set_processing_values(self, start: int | None = None, end: int | None = None) -> None:
        if not self.video_info:
            return
        max_frame = max(0, int(self.video_info.frame_count) - 1)
        cur_start, cur_end = self._processing_range_values()
        new_start = cur_start if start is None else max(0, min(int(start), max_frame))
        new_end = cur_end if end is None else max(0, min(int(end), max_frame))
        if new_start > new_end:
            if start is not None and end is None:
                new_end = new_start
            elif end is not None and start is None:
                new_start = new_end
            else:
                new_start, new_end = min(new_start, new_end), max(new_start, new_end)
        pairs = (
            (getattr(self, "processing_start_slider", None), new_start),
            (getattr(self, "processing_start_spin", None), new_start),
            (getattr(self, "processing_end_slider", None), new_end),
            (getattr(self, "processing_end_spin", None), new_end),
        )
        for widget, value in pairs:
            if widget is not None:
                widget.blockSignals(True)
                widget.setValue(int(value))
                widget.blockSignals(False)
        self._refresh_processing_range_label()
        if hasattr(self, "refresh_workflow_action_gates"):
            self.refresh_workflow_action_gates()

    def _on_processing_start_slider_changed(self, value: int) -> None:
        self._set_processing_values(start=int(value))
        self._apply_preview_frame_value(int(value), refresh_mesh=False)

    def _on_processing_end_slider_changed(self, value: int) -> None:
        self._set_processing_values(end=int(value))
        self._apply_preview_frame_value(int(value), refresh_mesh=False)

    def _on_processing_start_spin_changed(self, value: int) -> None:
        self._set_processing_values(start=int(value))
        self._apply_preview_frame_value(int(value), refresh_mesh=False)

    def _on_processing_end_spin_changed(self, value: int) -> None:
        self._set_processing_values(end=int(value))
        self._apply_preview_frame_value(int(value), refresh_mesh=False)
