# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from depth_fusion_core import APP_NAME, Path, QFileDialog, QMessageBox, scaled_size_from_long_side

class FilePathActionsMixin:
    def pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            "",
            "Video Files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.wmv);;All Files (*.*)",
        )
        if path:
            self.load_video(path)

    def pick_output_path(self) -> None:
        if not self.current_input or not self.video_info:
            QMessageBox.warning(self, APP_NAME, "请先导入视频。")
            return
        out_w, out_h = scaled_size_from_long_side(
            self.video_info.width,
            self.video_info.height,
            self.long_side_spin.value(),
        )
        current = self.output_path_edit.text().strip() or self._default_output_path_for_encoder(out_w, out_h)
        if self._is_structure_output_mode():
            start_dir = str(Path(current).parent if current else Path(self.current_input).parent)
            path = QFileDialog.getExistingDirectory(self, "选择 Mesh / 点云输出文件夹", start_dir)
        else:
            if self._is_png_sequence_mode():
                title = "选择 PNG 序列输出目录名"
                file_filter = "PNG Sequence Folder (*);;All Files (*.*)"
            else:
                title = "选择输出 MP4"
                file_filter = "MP4 Video (*.mp4);;All Files (*.*)"
            path, _ = QFileDialog.getSaveFileName(self, title, current, file_filter)
        if path:
            self._manual_output_path = True
            self.output_path_edit.setText(self._coerce_output_path_for_encoder(path, out_w, out_h))
            self.output_open_btn.setEnabled(True)

    def open_output_dir(self) -> None:
        path = self.output_path_edit.text().strip()
        if not path:
            try:
                self.preview_status_label.setText("还没有输出路径。请先选择输出位置。")
            except Exception:
                pass
            QMessageBox.warning(self, APP_NAME, "还没有输出路径。请先选择输出位置。")
            return
        path_obj = Path(path)
        folder = path_obj if self._is_structure_output_mode() or self._is_png_sequence_mode() or not path_obj.suffix else path_obj.parent
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(folder))  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"无法打开目录: {exc}")
