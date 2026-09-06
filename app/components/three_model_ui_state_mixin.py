# -*- coding: utf-8 -*-
from __future__ import annotations


class ThreeModelUiStateMixin:
    def _effective_normal_strength(self) -> int:
        return 0

    def _effective_normal_refine(self) -> int:
        return 0

    def _update_three_model_status(self) -> None:
        if hasattr(self, "three_model_state_label"):
            solver = self.structure_solver_combo.currentText() if hasattr(self, "structure_solver_combo") else "4DHumans"
            self.three_model_state_label.setText(f"主线：{solver} → Root稳定/时序去抖 → Dense Mesh → Garment/Hair Shell → Mesh / 可选点云")
        self._update_matting_status_label()
        self._update_external_media_status_label()
        self.refresh_3d_model_status()
        try:
            self.normal_strength_spin.setEnabled(False)
            self.normal_refine_spin.setEnabled(False)
        except Exception:
            pass

