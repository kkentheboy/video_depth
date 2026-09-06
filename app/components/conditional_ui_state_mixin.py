# -*- coding: utf-8 -*-
from __future__ import annotations


class ConditionalUiStateMixin:
    def _on_density_mode_changed(self) -> None:
        is_custom = False
        pointcloud_on = bool(getattr(self, "pointcloud_usd_check", None) is None or self.pointcloud_usd_check.isChecked())
        for name in ("pointcloud_stride_row", "pointcloud_max_points_row"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setVisible(bool(pointcloud_on and is_custom))
        for name in ("pointcloud_stride_spin", "pointcloud_max_points_spin"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setVisible(bool(is_custom))
                widget.setEnabled(bool(is_custom))

    def _update_conditional_visibility(self) -> None:
        """Keep low-frequency controls out of sight until their mode makes them useful."""
        source_mode = self._current_source_mode() if hasattr(self, "_current_source_mode") else self._source_mode_from_current_controls()
        matting_on = source_mode == "matanyone"
        external_mask_on = source_mode == "external_mask"
        if hasattr(self, "matting_paths_widget"):
            self.matting_paths_widget.setVisible(matting_on)
        if hasattr(self, "external_mask_paths_widget"):
            self.external_mask_paths_widget.setVisible(external_mask_on)
        bg_is_gray = bool(hasattr(self, "background_mode_combo") and self.background_mode_combo.currentText() == "背景灰")
        if hasattr(self, "background_gray_spin"):
            self.background_gray_spin.setEnabled(bg_is_gray)
        if hasattr(self, "background_gray_row"):
            self.background_gray_row.setVisible(bg_is_gray)
        if hasattr(self, "pointcloud_density_combo"):
            self._on_density_mode_changed()

