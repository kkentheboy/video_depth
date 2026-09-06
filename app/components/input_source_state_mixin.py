# -*- coding: utf-8 -*-
from __future__ import annotations


class InputSourceStateMixin:
    def on_matting_controls_changed(self) -> None:
        pass

    def _update_matting_status_label(self) -> None:
        pass

    def _source_mode_from_current_controls(self) -> str:
        """The structure-XYZ workflow uses the alpha channel embedded in the main video."""
        return "cutout_video"

    def _current_source_mode(self) -> str:
        return "cutout_video"

    def _sync_source_mode_radios(self) -> None:
        if not hasattr(self, "source_cutout_radio"):
            return
        for radio in (self.source_cutout_radio, self.source_matanyone_radio, self.source_external_mask_radio):
            radio.blockSignals(True)
            radio.setChecked(radio is self.source_cutout_radio)
            radio.blockSignals(False)
        self._update_conditional_visibility()

    def _apply_source_mode(self, mode: str) -> None:
        """Force the single-input workflow: main video alpha only."""
        for widget, checked in (
            (self.input_cutout_mask_check, True),
        ):
            widget.blockSignals(True)
            widget.setChecked(bool(checked))
            widget.blockSignals(False)
        self._sync_source_mode_radios()
        self._update_matting_status_label()
        self._update_conditional_visibility()
        self.on_external_media_changed()
