# -*- coding: utf-8 -*-
from __future__ import annotations

from depth_fusion_core import APP_STYLESHEET, QCursor, QPushButton, QWidget, Qt


class UiPresentationStateMixin:
    def _apply_style(self) -> None:
        # The current card-based UI sets object names while building panels.
        # Do not overwrite primaryButton / secondaryButton / navButton names here,
        # otherwise their page-specific QSS stops matching after _apply_style().
        for btn in (
            self.model_manager_btn, self.cache_manager_btn, self.log_dir_btn,
            self.external_mask_pick_btn, self.external_depth_pick_btn,
            self.preview_btn, self.preview_big_btn,
            self.preset_human_btn, self.preset_neutral_btn, self.preset_displacement_btn,
            self.preset_high_png_btn, self.preset_low_mem_btn, self.preset_import_btn,
            self.preset_export_btn,
        ):
            if not btn.objectName():
                btn.setObjectName("secondaryAction")
        if not self.path_edit.objectName():
            self.path_edit.setObjectName("pathEdit")
        self.preview_status_label.setObjectName("previewStatusLabel")
        self.info_label.setObjectName("infoLabel")

        self.setStyleSheet(APP_STYLESHEET)
        self._install_button_cursor_policy()

    def _install_button_cursor_policy(self) -> None:
        for btn in self.findChildren(QPushButton):
            btn.installEventFilter(self)
            self._sync_button_cursor(btn)

    def _sync_button_cursor(self, btn: QPushButton) -> None:
        if btn.isEnabled():
            btn.setCursor(QCursor(Qt.PointingHandCursor))
        else:
            btn.setCursor(QCursor(Qt.ArrowCursor))

    def _refresh_widget_style(self, widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _set_depth_preview_busy(self, busy: bool) -> None:
        if hasattr(self.preview_depth_label, "setOverlayText"):
            self.preview_depth_label.setOverlayText("计算中..." if busy else "")
        if hasattr(self, "preview_depth_status_line"):
            self.preview_depth_status_line.setProperty("busy", "1" if busy else "0")
            self._refresh_widget_style(self.preview_depth_status_line)

