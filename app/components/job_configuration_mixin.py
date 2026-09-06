# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import shutil

from depth_fusion_core import APP_NAME, DEFAULT_MATANYONE_MODEL_PATH, JobConfig, MAX_SAFE_LONG_SIDE_HINT, MODEL_IDS, Path, QMessageBox, cuda_total_memory_gb, estimate_vram_gb, is_direct_depth_video_workflow, scaled_size_from_long_side


class JobConfigurationMixin:
    def _structure_model_key(self) -> str:
        text = self.structure_solver_combo.currentText() if hasattr(self, "structure_solver_combo") else "4DHumans"
        return "wham" if "WHAM" in str(text).upper() else "4dhumans"

    def _pointcloud_mode(self) -> str:
        # Current GUI exposes only the Mesh/Shell main workflow. Older presets may
        # still contain "Legacy可见表面" in the hidden combo; never let that hidden
        # value switch export back to the old Depth/visible-surface branch.
        text = self.structure_solver_combo.currentText() if hasattr(self, "structure_solver_combo") else ""
        if "手" in text:
            return "fused_body_hand"
        return "fused_body"

    def _pointcloud_color_mode(self) -> str:
        return "xyz"

    def make_config(self) -> JobConfig:
        if not self.current_input or not self.video_info:
            raise RuntimeError("请先导入视频。")
        out_w, out_h = scaled_size_from_long_side(
            self.video_info.width,
            self.video_info.height,
            self.long_side_spin.value(),
        )
        pointcloud_mode_value = self._pointcloud_mode()
        structure_xyz_mode = pointcloud_mode_value != "structure_xyz"
        if not structure_xyz_mode:
            if self.black_pct_spin.value() >= self.white_pct_spin.value():
                raise RuntimeError("黑位裁切必须小于白位裁切。")
            if self.levels_in_black_spin.value() >= self.levels_in_white_spin.value():
                raise RuntimeError("曲线输入黑必须小于曲线输入白。")
        encoder_mode = self._current_encoder_mode()
        output_path = self._coerce_output_path_for_encoder(self.output_path_edit.text().strip(), out_w, out_h, encoder_mode)
        output_obj = Path(output_path)
        if os.path.abspath(output_path) == os.path.abspath(self.current_input):
            raise RuntimeError("输出路径不能覆盖原视频。")
        if structure_xyz_mode or self._is_png_sequence_mode(encoder_mode):
            output_obj.mkdir(parents=True, exist_ok=True)
        else:
            output_obj.parent.mkdir(parents=True, exist_ok=True)
        source_mode = "cutout_video"
        matting_enabled = False
        matting_model_path = os.path.normpath(self.matting_model_path_edit.text().strip() or str(DEFAULT_MATANYONE_MODEL_PATH))
        matting_mask_path = os.path.normpath(self.matting_mask_path_edit.text().strip())
        # Current main workflow only needs the main video. Alpha is optional helper; MatAnyone remains hidden compatibility.
        if matting_enabled:
            model_ok = Path(matting_model_path).is_file()
            mask_ok = bool(matting_mask_path) and Path(matting_mask_path).is_file()
            if not (model_ok and mask_ok):
                # Do not block the stable Mesh/Shell export. MatAnyone is optional compatibility.
                matting_enabled = False
                try:
                    self._update_matting_status_label()
                except Exception:
                    pass
                self.log("MatAnyone 未参与：缺少模型或第一帧 mask，已继续使用主视频结构流程。")
                self.preview_status_label.setText("MatAnyone 未参与：已继续主视频结构流程。")
        input_cutout_mask_enabled = True
        external_mask_path = ""
        external_mask_enabled = False
        # Main workflow no longer requires external Depth / 法线.
        # Shell geometry uses the mesh's own normals; Alpha is optional.
        external_depth_path = ""
        external_depth_enabled = False
        model_id = MODEL_IDS[self.model_combo.currentText()]
        normalize_mode_for_job = self.normalize_mode_combo.currentText()
        pointcloud_temporal_enabled = bool(self.pointcloud_temporal_check.isChecked())
        mesh_export_selected = bool(self.mesh_export_check.isChecked())
        detail_mesh_export_selected = bool(self.detail_mesh_export_check.isChecked())
        pointcloud_export_selected = bool(self.pointcloud_usd_check.isChecked())
        if structure_xyz_mode and not (mesh_export_selected or detail_mesh_export_selected or pointcloud_export_selected):
            raise RuntimeError("没有选择任何输出内容。请至少勾选低模 Mesh、细节 Mesh 或稳定点云之一。")
        pointcloud_enabled_value = bool(mesh_export_selected or detail_mesh_export_selected or pointcloud_export_selected) if structure_xyz_mode else bool(self.pointcloud_enable_check.isChecked())
        return JobConfig(
            input_path=self.current_input,
            output_path=output_path,
            output_width=out_w,
            output_height=out_h,
            model_id=model_id,
            device_mode=self.device_combo.currentText(),
            batch_size=int(self.batch_spin.value()),
            process_res=int(self.process_res_spin.value()),
            invert=self.invert_check.isChecked(),
            smooth=int(self.smooth_spin.value()),
            black_pct=float(self.black_pct_spin.value()),
            white_pct=float(self.white_pct_spin.value()),
            gamma=float(self.gamma_spin.value()),
            detail_boost=int(self.detail_boost_spin.value()),
            normal_strength=0,
            levels_in_black=int(self.levels_in_black_spin.value()),
            levels_in_white=int(self.levels_in_white_spin.value()),
            levels_out_black=int(self.levels_out_black_spin.value()),
            levels_out_white=int(self.levels_out_white_spin.value()),
            curve_points=tuple((float(x), float(y)) for x, y in self.levels_panel.getCurvePoints()),
            anti_banding=int(self.anti_banding_spin.value()),
            depth_smooth=int(self.depth_smooth_spin.value()),
            edge_preserve=int(self.edge_preserve_spin.value()),
            tone_black=int(self.tone_black_spin.value()),
            tone_shadow=int(self.tone_shadow_spin.value()),
            tone_mid=int(self.tone_mid_spin.value()),
            tone_light=int(self.tone_light_spin.value()),
            tone_white=int(self.tone_white_spin.value()),
            tone_black_shift=int(self.tone_black_shift_spin.value()),
            tone_shadow_shift=int(self.tone_shadow_shift_spin.value()),
            tone_mid_shift=int(self.tone_mid_shift_spin.value()),
            tone_light_shift=int(self.tone_light_shift_spin.value()),
            tone_white_shift=int(self.tone_white_shift_spin.value()),
            tone_black_contrast=int(self.tone_black_contrast_spin.value()),
            tone_shadow_contrast=int(self.tone_shadow_contrast_spin.value()),
            tone_mid_contrast=int(self.tone_mid_contrast_spin.value()),
            tone_light_contrast=int(self.tone_light_contrast_spin.value()),
            tone_white_contrast=int(self.tone_white_contrast_spin.value()),
            copy_audio=self.copy_audio_check.isChecked(),
            cache_enabled=self.cache_enable_check.isChecked(),
            normalize_mode=normalize_mode_for_job,
            human_refine=0,
            normal_refine=0,
            encoder_mode=encoder_mode,
            input_brightness=int(self.input_brightness_spin.value()),
            input_contrast=int(self.input_contrast_spin.value()),
            input_gamma=float(self.input_gamma_spin.value()),
            input_shadow=int(self.input_shadow_spin.value()),
            input_highlight=int(self.input_highlight_spin.value()),
            input_sharpen=int(self.input_sharpen_spin.value()),
            input_denoise=int(self.input_denoise_spin.value()),
            matting_enabled=matting_enabled,
            matting_mask_path=matting_mask_path,
            matting_model_path=matting_model_path,
            matting_max_size=0,
            auto_mask_feather_px=int(self.auto_mask_feather_spin.value()),
            auto_mask_expand_px=int(self.auto_mask_expand_spin.value()),
            background_mode=self.background_mode_combo.currentText(),
            background_gray=int(self.background_gray_spin.value()),
            external_mask_enabled=external_mask_enabled,
            external_mask_path=external_mask_path,
            external_mask_invert=self.external_mask_invert_check.isChecked(),
            input_cutout_mask_enabled=input_cutout_mask_enabled,
            external_depth_enabled=False,
            external_depth_path="",
            external_depth_weight=0,
            external_depth_invert=False,
            external_depth_orientation_mode="不使用",
            pointcloud_enabled=pointcloud_enabled_value,
            pointcloud_mode=pointcloud_mode_value,
            pointcloud_density=self.pointcloud_density_combo.currentText(),
            pointcloud_stride=int(self._effective_pointcloud_stride()),
            pointcloud_max_points=self._effective_pointcloud_max_points(),
            pointcloud_depth_near_percentile=1.0,
            pointcloud_depth_far_percentile=99.0,
            pointcloud_color_mode="xyz",
            pointcloud_coordinate_mode="blender",
            pointcloud_binary_ply=True,
            pointcloud_alpha_erode_px=1,
            pointcloud_alpha_dilate_px=0,
            pointcloud_alpha_feather_px=3,
            pointcloud_body_bbox_margin_px=12,
            pointcloud_remove_outliers=False if structure_xyz_mode else bool(self.pointcloud_remove_outliers_check.isChecked()),
            pointcloud_voxel_downsample=False if structure_xyz_mode else bool(self.pointcloud_voxel_check.isChecked()),
            pointcloud_temporal_depth_smooth=(0.52 if (structure_xyz_mode and pointcloud_temporal_enabled) else 0.0),
            pointcloud_temporal_center_smooth=(0.68 if (structure_xyz_mode and pointcloud_temporal_enabled) else 0.0),
            pointcloud_temporal_scale_smooth=(0.58 if (structure_xyz_mode and pointcloud_temporal_enabled) else 0.0),
            pointcloud_template_align_strength=1.0,
            pointcloud_obj_sequence=False,
            pointcloud_usd_sequence=pointcloud_export_selected,
            pointcloud_usd_max_points=max(1000, min(int(self._effective_pointcloud_max_points()), 800000)),
            pointcloud_usd_point_width=0.008,
            mesh_export_enabled=mesh_export_selected,
            detail_mesh_export_enabled=detail_mesh_export_selected,
            mesh_dense_segments=int((self.mesh_dense_segments_combo.currentText() or "中 2x").split()[1].replace("x", "")) if hasattr(self, "mesh_dense_segments_combo") else 2,
            garment_shell_enabled=True,
            garment_shell_offset=(float(self.garment_shell_offset_spin.value()) / 1000.0 if hasattr(self, "garment_shell_offset_spin") else 0.020),
            hair_shell_enabled=True,
            hair_shell_offset=(float(self.hair_shell_offset_spin.value()) / 1000.0 if hasattr(self, "hair_shell_offset_spin") else 0.035),
            segmentation_enabled=True,
            segmentation_provider="Auto",
            segmentation_use_cache=True,
            segmentation_fallback_geometry=False,
            pointcloud_abc_sequence=False,
            pointcloud_abc_max_points=120000,
            pointcloud_normal_relief_enabled=False,
            pointcloud_normal_relief_strength=0.0,
            pointcloud_normal_relief_gamma=1.6,
            pointcloud_hand_enabled=(pointcloud_mode_value == "fused_body_hand"),
            structure_model=self._structure_model_key() if structure_xyz_mode else "none",
            hand_model="none",
            occlusion_fill_enabled=structure_xyz_mode,
            processing_start_frame=int(self._processing_range_values()[0]) if hasattr(self, "_processing_range_values") else 0,
            processing_end_frame=int(self._processing_range_values()[1]) if hasattr(self, "_processing_range_values") else -1,
            mesh_preview_yaw=float(getattr(self, "mesh_preview_yaw", 0.0)),
            mesh_preview_pitch=float(getattr(self, "mesh_preview_pitch", 0.0)),
            project_dir=str(self.current_project_dir) if self.current_project_dir else "",
        )

    def _resource_risks(self, cfg: JobConfig, preview: bool = False) -> list[str]:
        risks: list[str] = []
        direct_depth_mode = False
        try:
            direct_depth_mode = is_direct_depth_video_workflow(cfg)
        except Exception:
            direct_depth_mode = False
        long_side = max(cfg.output_width, cfg.output_height)
        if long_side > MAX_SAFE_LONG_SIDE_HINT:
            risks.append(f"{'预览' if preview else '输出'}长边 {long_side} 超过建议上限 {MAX_SAFE_LONG_SIDE_HINT}")
        if not direct_depth_mode:
            est = estimate_vram_gb(cfg)
            total = cuda_total_memory_gb()
            if total > 0.0:
                risks.append(f"检测到显存约 {total:.1f}GB，当前参数估算峰值约 {est:.1f}GB")
                if est > total * 0.82:
                    risks.append("估算显存已接近/超过安全区，容易 CUDA OOM")
            else:
                if cfg.process_res > 1280:
                    risks.append(f"process_res={cfg.process_res} 较高")
                if cfg.batch_size > 1:
                    risks.append(f"批量帧数={cfg.batch_size}，显存占用会增加")
        if cfg.encoder_mode.startswith("FFmpeg") and shutil.which("ffmpeg") is None:
            risks.append("未检测到 ffmpeg，会回退 OpenCV mp4v；勾选原音频时无法合并声音")
        return risks

    def _confirm_resource_risk(self, cfg: JobConfig) -> bool:
        risks = self._resource_risks(cfg, preview=False)
        hard = [r for r in risks if "容易 CUDA OOM" in r or "超过建议上限" in r or "回退" in r]
        if not hard:
            return True
        reply = QMessageBox.question(
            self,
            APP_NAME,
            "当前参数有显存/性能风险：\n\n" + "\n".join(f"- {item}" for item in risks) + "\n\n仍然继续导出？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _confirm_preview_resource_risk(self, cfg: JobConfig) -> bool:
        risks = self._resource_risks(cfg, preview=True)
        hard = [r for r in risks if "容易 CUDA OOM" in r or "超过建议上限" in r]
        if not hard:
            return True
        reply = QMessageBox.question(
            self,
            APP_NAME,
            "当前预览参数有显存/性能风险：\n\n" + "\n".join(f"- {item}" for item in risks) + "\n\n仍然渲染当前帧？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

