# -*- coding: utf-8 -*-
from __future__ import annotations

import os

from depth_fusion_core import (
    Optional, Path, default_output_path, default_structure_output_dir,
    png_sequence_output_dir, scaled_size_from_long_side,
)
from common.encoder_display import encoder_display_name, encoder_internal_name


class OutputRoutingMixin:
        def _set_encoder_combo_value(self, mode_or_display: str) -> None:
            display = encoder_display_name(encoder_internal_name(mode_or_display))
            if display in [self.encoder_combo.itemText(i) for i in range(self.encoder_combo.count())]:
                self.encoder_combo.setCurrentText(display)

        def _current_encoder_mode(self) -> str:
            return encoder_internal_name(self.encoder_combo.currentText())

        def _is_structure_output_mode(self) -> bool:
            """Current main workflow outputs Mesh/Shell/point-cloud folders, not depth videos."""
            return self._pointcloud_mode() != "structure_xyz"

        def _is_png_sequence_mode(self, encoder_mode: Optional[str] = None) -> bool:
            if self._is_structure_output_mode():
                return False
            mode = encoder_internal_name(encoder_mode if encoder_mode is not None else self.encoder_combo.currentText())
            return mode == "PNG序列 16-bit"

        def _default_output_path_for_encoder(self, out_w: int, out_h: int, encoder_mode: Optional[str] = None) -> str:
            if not self.current_input:
                return ""
            if self._is_structure_output_mode():
                return os.path.normpath(default_structure_output_dir(self.current_input))
            base = Path(default_output_path(self.current_input, out_w, out_h))
            if self._is_png_sequence_mode(encoder_mode):
                return str(png_sequence_output_dir(base))
            return str(base)

        def _coerce_output_path_for_encoder(self, path_text: str, out_w: int, out_h: int, encoder_mode: Optional[str] = None) -> str:
            mode = encoder_mode if encoder_mode is not None else self.encoder_combo.currentText()
            text = (path_text or "").strip()
            if not text:
                return os.path.normpath(self._default_output_path_for_encoder(out_w, out_h, mode))
            path = Path(text)
            if self._is_structure_output_mode():
                if path.suffix:
                    path = path.with_suffix("")
                return os.path.normpath(str(path))
            if self._is_png_sequence_mode(mode):
                if path.suffix:
                    path = png_sequence_output_dir(path)
                return os.path.normpath(str(path))
            if not path.suffix:
                name = path.name[:-6] if path.name.endswith("_png16") else path.name
                path = path.with_name(name + ".mp4")
            elif path.suffix.lower() != ".mp4":
                path = path.with_suffix(".mp4")
            return os.path.normpath(str(path))

        def on_encoder_changed(self, _text: str = "") -> None:
            if not self.current_input or not self.video_info:
                return
            out_w, out_h = scaled_size_from_long_side(
                self.video_info.width,
                self.video_info.height,
                self.long_side_spin.value(),
            )
            current = self.output_path_edit.text().strip()
            if not self._manual_output_path:
                self.output_path_edit.setText(self._default_output_path_for_encoder(out_w, out_h))
            elif current:
                self.output_path_edit.setText(self._coerce_output_path_for_encoder(current, out_w, out_h))
            if self._is_structure_output_mode():
                mode_note = "当前主流程会输出到 Mesh / 点云文件夹。"
            else:
                mode_note = "PNG 序列会输出到当前显示的文件夹。" if self._is_png_sequence_mode() else "视频模式会输出为 .mp4 文件。"
            self.preview_status_label.setText(mode_note)
            self.output_open_btn.setEnabled(bool(self.output_path_edit.text().strip()))

        def refresh_output_size(self) -> None:
            if not self.video_info:
                self.out_size_label.setText("输出: -")
                self.output_path_edit.clear()
                self.output_open_btn.setEnabled(False)
                return
            out_w, out_h = scaled_size_from_long_side(
                self.video_info.width,
                self.video_info.height,
                self.long_side_spin.value(),
            )
            self.out_size_label.setText(f"{out_w}x{out_h}")
            if self.current_input and not self._manual_output_path:
                self.output_path_edit.setText(self._default_output_path_for_encoder(out_w, out_h))
            self.output_open_btn.setEnabled(bool(self.output_path_edit.text().strip()))

        def on_output_geometry_changed(self) -> None:
            """Refresh output size and invalidate preview caches when output geometry changes."""
            self.refresh_output_size()
            if not self.current_input or not self.video_info:
                return
            if self._has_background_processing():
                return
            self.preview_depth = None
            self.preview_subject_mask = None
            self.preview_normal_map = None
            self.preview_depth_version += 1
            self.preview_base_gray_cache = None
            self.preview_hist_gray_cache = None
            self.preview_base_key = None
            self.preview_depth_render_bgr = None
            self.preview_depth_label.clearImage("等待 Mesh 预览")
            self.preview_big_btn.setEnabled(False)
            self._refresh_reference_preview_tiles()
            self.preview_status_label.setText("输出尺寸已变化，请重新预览 Mesh / 点云。")
            self.show_original_frame_immediately(int(self.preview_frame_spin.value()))

