# -*- coding: utf-8 -*-
from __future__ import annotations

from depth_fusion_core import APP_NAME, Optional, Path, QMessageBox, VideoInfo, cv2, describe_real_alpha_source, probe_video, short_error_message


class InputValidationMixin:
    def _is_image_path(self, path_text: str) -> bool:
        return Path(str(path_text or "")).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def _safe_probe_external_media(self, path_text: str) -> tuple[Optional[VideoInfo], str]:
        path = Path(str(path_text or "").strip())
        if not path.is_file():
            return None, "未选择"
        if self._is_image_path(str(path)):
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return None, "图片不可读"
            h, w = img.shape[:2]
            has_alpha = bool(img.ndim == 3 and img.shape[2] >= 4)
            return VideoInfo(path=str(path), width=w, height=h, fps=0.0, frame_count=1, has_alpha=has_alpha), f"图片 {w}x{h}"
        try:
            info = probe_video(str(path))
            return info, f"{info.width}x{info.height} / {info.fps:.3f}fps / {info.frame_count}帧"
        except Exception as exc:  # noqa: BLE001
            return None, f"不可读：{short_error_message(str(exc))}"

    def _main_video_alpha_state(self) -> tuple[bool, bool, str]:
        main_ok = bool(self.current_input and self.video_info)
        alpha_ok = False
        alpha_desc = "未检测"
        if main_ok:
            try:
                alpha_ok, alpha_desc = describe_real_alpha_source(self.current_input or "", int(self.preview_frame_spin.value()))
            except Exception as exc:  # noqa: BLE001
                alpha_ok = False
                alpha_desc = f"Alpha 检测失败：{short_error_message(str(exc))}"
        return main_ok, alpha_ok, alpha_desc

    def _update_external_media_status_label(self) -> None:
        if not hasattr(self, "external_status_label"):
            return
        main_loaded = bool(self.current_input and self.video_info)
        alpha_ok = False
        if main_loaded:
            try:
                alpha_ok, _desc = describe_real_alpha_source(self.current_input or "", int(self.preview_frame_spin.value()))
            except Exception:
                alpha_ok = False
        if hasattr(self, "top_main_chain_label"):
            self.top_main_chain_label.setText("主视频：已加载" if main_loaded else "主视频：未加载")
        if hasattr(self, "top_external_depth_label"):
            self.top_external_depth_label.setText("Alpha：黑底合成" if alpha_ok else "Alpha：未检测/可继续")
        parts = ["主视频" if main_loaded else "未导入主视频", "Alpha黑底" if alpha_ok else "无Alpha继续", "structure cache", "稳定Mesh", "Dense/Shell", "Mesh/点云导出"]
        self.external_chain_label.setText("操作链：" + " → ".join(parts))

    def validate_main_video_alpha_chain(self, silent: bool = False) -> bool:
        """Validate step 1: main video only. Alpha is optional helper, not a blocker."""
        main_ok, alpha_ok, alpha_desc = self._main_video_alpha_state()
        self.input_cutout_mask_check.setChecked(True)
        self.background_mode_combo.setCurrentText("背景白")
        if hasattr(self, "top_main_chain_label"):
            self.top_main_chain_label.setText("主视频：已加载" if main_ok else "主视频：未加载")
        if hasattr(self, "top_external_depth_label"):
            self.top_external_depth_label.setText("Alpha：黑底合成" if alpha_ok else "Alpha：未检测/可继续")
        detail = "主视频：未导入"
        if main_ok:
            detail = f"主视频：{self.video_info.width}x{self.video_info.height} / {self.video_info.fps:.3f}fps / {self.video_info.frame_count}帧"
        detail += f"\nAlpha：{alpha_desc}"
        if not main_ok:
            detail += "\n注意：先导入主视频。"
        elif alpha_ok:
            detail += "\n输入就绪：Alpha 会自动合成黑色背景。"
        else:
            detail += "\n输入就绪：无 Alpha，将按普通视频处理。"
        if hasattr(self, "external_status_label"):
            self.external_status_label.setText(detail)
        if hasattr(self, "external_chain_label"):
            step1 = "✓主视频" if main_ok else "×主视频"
            step2 = "Alpha黑底" if alpha_ok else "无Alpha继续"
            self.external_chain_label.setText(f"链路：{step1} → {step2} → 结构缓存 → 稳定 Mesh/Shell → 导出")
        if hasattr(self, "preview_status_label"):
            if main_ok:
                self.preview_status_label.setText("主视频已接入。下一步生成结构缓存。")
            elif not silent:
                self.preview_status_label.setText("请先导入主视频。")
        self._refresh_reference_preview_tiles()
        self._update_external_media_status_label()
        if not main_ok and not silent:
            QMessageBox.warning(self, APP_NAME, "请先导入主视频。")
        return bool(main_ok)

    def validate_external_reference_chain(self, silent: bool = False) -> bool:
        """Compatibility alias. Depth/法线 reference is no longer required."""
        return self.validate_main_video_alpha_chain(silent=silent)

    def on_external_media_changed(self) -> None:
        self._update_external_media_status_label()
        self.preview_subject_mask = None
        self.preview_depth_version += 1
        self.preview_depth = None
        self.preview_normal_map = None
        self.preview_depth_render_bgr = None
        self.preview_base_gray_cache = None
        self.preview_hist_gray_cache = None
        self.preview_base_key = None
        self.preview_depth_label.clearImage("等待 Mesh 预览")
        self.preview_big_btn.setEnabled(False)
        self._refresh_reference_preview_tiles()
        if hasattr(self, "preview_status_label"):
            self.preview_status_label.setText("输入设置已更新。请检查主视频和结构缓存状态。")
        self.validate_external_reference_chain(silent=True)
        if hasattr(self, "refresh_workflow_action_gates"):
            self.refresh_workflow_action_gates()

