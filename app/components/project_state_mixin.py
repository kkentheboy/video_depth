# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from depth_fusion_core import APP_NAME, BUILTIN_PRESETS, NORMALIZE_MODES, PROJECT_DIR, Path, QFileDialog, QMessageBox, json
from common.encoder_display import encoder_internal_name

class ProjectStateMixin:
    def _preset_payload(self) -> dict:
        return {
            "version": 3,
            "input_path": self.current_input if hasattr(self, "current_input") and self.current_input else "",
            "output_long_side": int(self.long_side_spin.value()),
            "batch_size": int(self.batch_spin.value()),
            "process_res": int(self.process_res_spin.value()),
            "processing_range": {
                "start": int(self._processing_range_values()[0]) if hasattr(self, "_processing_range_values") else 0,
                "end": int(self._processing_range_values()[1]) if hasattr(self, "_processing_range_values") else -1,
            },
            "encoder_mode": encoder_internal_name(self.encoder_combo.currentText()),
            "normalize_mode": self.normalize_mode_combo.currentText(),
            "invert": self.invert_check.isChecked(),
            "cache_enabled": self.cache_enable_check.isChecked(),
            "copy_audio": self.copy_audio_check.isChecked(),
            "smooth": int(self.smooth_spin.value()),
            "human_refine": int(self.human_refine_spin.value()),
            "black_pct": float(self.black_pct_spin.value()),
            "white_pct": float(self.white_pct_spin.value()),
            "gamma": float(self.gamma_spin.value()),
            "detail_boost": int(self.detail_boost_spin.value()),
            "normal_strength": int(self.normal_strength_spin.value()),
            "normal_refine": int(self.normal_refine_spin.value()),
            "auto_mask_feather_px": int(self.auto_mask_feather_spin.value()),
            "auto_mask_expand_px": int(self.auto_mask_expand_spin.value()),
            "background_mode": self.background_mode_combo.currentText(),
            "background_gray": int(self.background_gray_spin.value()),
            "external_reference": {
                "source_mode": "cutout_video",
                "input_cutout_mask_enabled": True,
                "external_mask_enabled": False,
                "external_mask_path": "",
                "external_mask_invert": bool(self.external_mask_invert_check.isChecked()),
                "external_depth_path": self.external_depth_path_edit.text().strip(),
                "external_depth_weight": int(self.external_depth_weight_spin.value()),
                "external_depth_orientation_mode": self.external_depth_invert_check.currentText(),
            },
            "input_adjust": {
                "brightness": int(self.input_brightness_spin.value()),
                "contrast": int(self.input_contrast_spin.value()),
                "gamma": float(self.input_gamma_spin.value()),
                "shadow": int(self.input_shadow_spin.value()),
                "highlight": int(self.input_highlight_spin.value()),
                "sharpen": int(self.input_sharpen_spin.value()),
                "denoise": int(self.input_denoise_spin.value()),
            },
            "structure_model": self._structure_model_key() if hasattr(self, "_structure_model_key") else "4dhumans",
            "pointcloud": {
                "enabled": bool(self.pointcloud_enable_check.isChecked()),
                "density": self.pointcloud_density_combo.currentText(),
                "remove_outliers": bool(self.pointcloud_remove_outliers_check.isChecked()),
                "voxel_downsample": bool(self.pointcloud_voxel_check.isChecked()),
                "obj_sequence": False,
                "usd_sequence": bool(self.pointcloud_usd_check.isChecked()),
                "mesh_export": bool(self.mesh_export_check.isChecked()),
                "detail_mesh_export": bool(self.detail_mesh_export_check.isChecked()),
                "mesh_dense_segments": self.mesh_dense_segments_combo.currentText(),
                "garment_shell": bool(self.garment_shell_check.isChecked()),
                "hair_shell": bool(self.hair_shell_check.isChecked()),
                "segmentation_enabled": bool(self.segmentation_enable_check.isChecked()) if hasattr(self, "segmentation_enable_check") else True,
                "segmentation_provider": self.segmentation_provider_combo.currentText() if hasattr(self, "segmentation_provider_combo") else "Auto",
                "abc_sequence": False,
                "normal_relief_enabled": False,
                "temporal_stabilize": bool(self.pointcloud_temporal_check.isChecked()),
            },
            "anti_banding": int(self.anti_banding_spin.value()),
            "depth_smooth": int(self.depth_smooth_spin.value()),
            "edge_preserve": int(self.edge_preserve_spin.value()),
            "levels_in_black": int(self.levels_in_black_spin.value()),
            "levels_in_white": int(self.levels_in_white_spin.value()),
            "levels_out_black": int(self.levels_out_black_spin.value()),
            "levels_out_white": int(self.levels_out_white_spin.value()),
            "curve_points": self.levels_panel.getCurvePoints(),
            "tone": {
                "black": int(self.tone_black_spin.value()),
                "shadow": int(self.tone_shadow_spin.value()),
                "mid": int(self.tone_mid_spin.value()),
                "light": int(self.tone_light_spin.value()),
                "white": int(self.tone_white_spin.value()),
                "black_shift": int(self.tone_black_shift_spin.value()),
                "shadow_shift": int(self.tone_shadow_shift_spin.value()),
                "mid_shift": int(self.tone_mid_shift_spin.value()),
                "light_shift": int(self.tone_light_shift_spin.value()),
                "white_shift": int(self.tone_white_shift_spin.value()),
                "black_contrast": int(self.tone_black_contrast_spin.value()),
                "shadow_contrast": int(self.tone_shadow_contrast_spin.value()),
                "mid_contrast": int(self.tone_mid_contrast_spin.value()),
                "light_contrast": int(self.tone_light_contrast_spin.value()),
                "white_contrast": int(self.tone_white_contrast_spin.value()),
            },
        }

    def _apply_preset_payload(self, data: dict) -> None:
        def set_if(key: str, widget) -> None:  # noqa: ANN001
            if key in data:
                widget.setValue(data[key])
        set_if("output_long_side", self.long_side_spin)
        set_if("batch_size", self.batch_spin)
        set_if("process_res", self.process_res_spin)
        try:
            rng = data.get("processing_range", {}) if isinstance(data.get("processing_range", {}), dict) else {}
            if rng and hasattr(self, "_set_processing_values"):
                self._set_processing_values(int(rng.get("start", 0)), int(rng.get("end", -1)))
        except Exception:
            pass
        try:
            model_key = str(data.get("structure_model", "") or "").strip().lower()
            if model_key in {"4dhumans", "wham"} and hasattr(self, "structure_solver_combo"):
                self.structure_solver_combo.setCurrentText(self._structure_scheme_text(model_key))
        except Exception:
            pass
        if data.get("color_mode") in [self.color_combo.itemText(i) for i in range(self.color_combo.count())]:
            self.color_combo.setCurrentText(data["color_mode"])
        if data.get("encoder_mode"):
            self._set_encoder_combo_value(str(data.get("encoder_mode")))
        if data.get("normalize_mode") in NORMALIZE_MODES:
            self.normalize_mode_combo.setCurrentText(data["normalize_mode"])
        if "invert" in data:
            self.invert_check.setChecked(bool(data["invert"]))
        if "cache_enabled" in data:
            self.cache_enable_check.setChecked(bool(data["cache_enabled"]))
        if "copy_audio" in data:
            self.copy_audio_check.setChecked(bool(data["copy_audio"]))
        if data.get("background_mode") in [self.background_mode_combo.itemText(i) for i in range(self.background_mode_combo.count())]:
            self.background_mode_combo.setCurrentText(data["background_mode"])
        pointcloud = data.get("pointcloud", {}) if isinstance(data.get("pointcloud", {}), dict) else {}
        # Ignore legacy hidden point-cloud enable/mode values from old presets.
        # The current app flow is always structure-cache -> Mesh/Shell -> optional stable point cloud.
        self.pointcloud_enable_check.setChecked(True)
        if pointcloud.get("density") in [self.pointcloud_density_combo.itemText(i) for i in range(self.pointcloud_density_combo.count())]:
            self.pointcloud_density_combo.setCurrentText(pointcloud["density"])
        if "remove_outliers" in pointcloud:
            self.pointcloud_remove_outliers_check.setChecked(bool(pointcloud["remove_outliers"]))
        if "voxel_downsample" in pointcloud:
            self.pointcloud_voxel_check.setChecked(bool(pointcloud["voxel_downsample"]))
        if "temporal_stabilize" in pointcloud:
            self.pointcloud_temporal_check.setChecked(bool(pointcloud["temporal_stabilize"]))
        if "usd_sequence" in pointcloud:
            self.pointcloud_usd_check.setChecked(bool(pointcloud["usd_sequence"]))
        if "mesh_export" in pointcloud:
            self.mesh_export_check.setChecked(bool(pointcloud["mesh_export"]))
        if "detail_mesh_export" in pointcloud:
            self.detail_mesh_export_check.setChecked(bool(pointcloud["detail_mesh_export"]))
        if pointcloud.get("mesh_dense_segments") in [self.mesh_dense_segments_combo.itemText(i) for i in range(self.mesh_dense_segments_combo.count())]:
            self.mesh_dense_segments_combo.setCurrentText(pointcloud["mesh_dense_segments"])
        if "garment_shell" in pointcloud:
            self.garment_shell_check.setChecked(bool(pointcloud["garment_shell"]))
        if "hair_shell" in pointcloud:
            self.hair_shell_check.setChecked(bool(pointcloud["hair_shell"]))
        if "segmentation_enabled" in pointcloud and hasattr(self, "segmentation_enable_check"):
            self.segmentation_enable_check.setChecked(bool(pointcloud["segmentation_enabled"]))
        if pointcloud.get("segmentation_provider") in [self.segmentation_provider_combo.itemText(i) for i in range(self.segmentation_provider_combo.count())] if hasattr(self, "segmentation_provider_combo") else False:
            self.segmentation_provider_combo.setCurrentText(pointcloud["segmentation_provider"])

        for key, widget in [
            ("smooth", self.smooth_spin),
            ("human_refine", self.human_refine_spin),
            ("black_pct", self.black_pct_spin), ("white_pct", self.white_pct_spin), ("gamma", self.gamma_spin),
            ("detail_boost", self.detail_boost_spin), ("normal_strength", self.normal_strength_spin),
            ("normal_refine", self.normal_refine_spin),
            ("auto_mask_feather_px", self.auto_mask_feather_spin),
            ("auto_mask_expand_px", self.auto_mask_expand_spin),
            ("background_gray", self.background_gray_spin),
            ("anti_banding", self.anti_banding_spin),
            ("depth_smooth", self.depth_smooth_spin), ("edge_preserve", self.edge_preserve_spin),
            ("levels_in_black", self.levels_in_black_spin), ("levels_in_white", self.levels_in_white_spin),
            ("levels_out_black", self.levels_out_black_spin), ("levels_out_white", self.levels_out_white_spin),
        ]:
            set_if(key, widget)
        input_adjust = data.get("input_adjust", {}) if isinstance(data.get("input_adjust", {}), dict) else {}
        input_map = {
            "brightness": self.input_brightness_spin, "contrast": self.input_contrast_spin,
            "gamma": self.input_gamma_spin, "shadow": self.input_shadow_spin,
            "highlight": self.input_highlight_spin, "sharpen": self.input_sharpen_spin,
            "denoise": self.input_denoise_spin,
        }
        for key, widget in input_map.items():
            if key in input_adjust:
                widget.setValue(input_adjust[key])
        self.levels_panel.setCurvePoints(data.get("curve_points", [(0.0, 0.0), (1.0, 1.0)]), emit=False)
        tone = data.get("tone", {}) if isinstance(data.get("tone", {}), dict) else {}
        tone_map = {
            "black": self.tone_black_spin, "shadow": self.tone_shadow_spin, "mid": self.tone_mid_spin,
            "light": self.tone_light_spin, "white": self.tone_white_spin,
            "black_shift": self.tone_black_shift_spin, "shadow_shift": self.tone_shadow_shift_spin, "mid_shift": self.tone_mid_shift_spin,
            "light_shift": self.tone_light_shift_spin, "white_shift": self.tone_white_shift_spin,
            "black_contrast": self.tone_black_contrast_spin, "shadow_contrast": self.tone_shadow_contrast_spin,
            "mid_contrast": self.tone_mid_contrast_spin, "light_contrast": self.tone_light_contrast_spin,
            "white_contrast": self.tone_white_contrast_spin,
        }
        for key, widget in tone_map.items():
            if key in tone:
                widget.setValue(tone[key])
        external_reference = data.get("external_reference", {}) if isinstance(data.get("external_reference", {}), dict) else {}
        if "input_cutout_mask_enabled" in external_reference:
            self.input_cutout_mask_check.setChecked(bool(external_reference.get("input_cutout_mask_enabled")))
        self.external_mask_path_edit.setText("")
        if "external_mask_invert" in external_reference:
            self.external_mask_invert_check.setChecked(bool(external_reference.get("external_mask_invert")))
        if "external_depth_path" in external_reference:
            self.external_depth_path_edit.setText(str(external_reference.get("external_depth_path") or ""))
        if "external_depth_weight" in external_reference:
            self.external_depth_weight_spin.setValue(int(external_reference.get("external_depth_weight") or 0))
        if "external_depth_orientation_mode" in external_reference:
            mode = str(external_reference.get("external_depth_orientation_mode") or "自动方向")
            if mode in [self.external_depth_invert_check.itemText(i) for i in range(self.external_depth_invert_check.count())]:
                self.external_depth_invert_check.setCurrentText(mode)
        # Re-force the hidden compatibility controls after applying old presets.
        self.pointcloud_enable_check.setChecked(True)
        self.external_depth_path_edit.setText("")
        if hasattr(self, "source_cutout_radio"):
            source_mode = "cutout_video"
            self.source_cutout_radio.setChecked(True)
            self._apply_source_mode(source_mode)
        else:
            self.on_external_media_changed()
        try:
            self._on_density_mode_changed()
            self._update_conditional_visibility()
        except Exception:
            pass
        self._on_density_mode_changed()
        self.sync_levels_panel_from_controls()
        self.render_preview_from_cache()

    def apply_builtin_preset(self, preset_key: str) -> None:
        payload = BUILTIN_PRESETS.get(preset_key)
        if not isinstance(payload, dict):
            QMessageBox.warning(self, APP_NAME, f"未找到内置预设: {preset_key}")
            return
        self._apply_preset_payload(dict(payload))
        self.preview_status_label.setText(f"已应用内置预设：{payload.get('name', preset_key)}")
        self.log(f"已应用内置预设: {payload.get('name', preset_key)}")

    def export_preset_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出参数预设", str(PROJECT_DIR / "depth_preset.json"), "JSON (*.json)")
        if not path:
            return
        if not Path(path).suffix:
            path += ".json"
        Path(path).write_text(json.dumps(self._preset_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.log(f"已导出预设: {path}")

    def import_preset_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入参数预设", str(PROJECT_DIR), "JSON (*.json);;All Files (*.*)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise RuntimeError("预设文件格式错误。")
            self._apply_preset_payload(data)
            self.log(f"已导入预设: {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"导入预设失败: {exc}")

    def save_project_state(self, current_step: int = -1) -> None:
        """Automatically saves the current project state into the project.vhm file."""
        if not hasattr(self, "current_project_dir") or not self.current_project_dir:
            return
        proj_file = self.current_project_dir / "project.vhm"
        try:
            data = {}
            if proj_file.exists():
                data = json.loads(proj_file.read_text(encoding="utf-8"))
            data["preset"] = self._preset_payload()
            if current_step >= 0:
                data["workflow_step"] = current_step
            proj_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            try:
                self.log(f"保存项目进度失败: {exc}")
            except Exception:
                pass

    def load_project(self, project_dir: Path) -> None:
        self.current_project_dir = project_dir
        self.setWindowTitle(f"{APP_NAME} - {project_dir.name}")
        self.log(f"进入项目: {project_dir}")

        proj_file = project_dir / "project.vhm"
        if proj_file.exists():
            try:
                import json
                with open(proj_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                preset = data.get("preset", {})
                if preset.get("input_path") and os.path.isfile(preset["input_path"]):
                    # Automatically load the video if the project had one
                    self.load_video(preset["input_path"])

                self._apply_preset_payload(preset)
                if hasattr(self, "_restore_best_available_structure_scheme"):
                    self._restore_best_available_structure_scheme()
                if hasattr(self, "_update_structure_scheme_status_labels"):
                    self._update_structure_scheme_status_labels()
                if hasattr(self, "refresh_workflow_action_gates"):
                    self.refresh_workflow_action_gates()

                # Restore step. Step 0 is valid, so do not use truthiness.
                if "workflow_step" in data:
                    saved_step = int(data.get("workflow_step") or 0)
                    if hasattr(self, "set_workflow_step"):
                        self.set_workflow_step(saved_step)
                if hasattr(self, "_update_structure_scheme_status_labels"):
                    self._update_structure_scheme_status_labels()
                if hasattr(self, "refresh_workflow_action_gates"):
                    self.refresh_workflow_action_gates()
            except Exception as e:
                self.log(f"加载工程文件失败: {e}")
