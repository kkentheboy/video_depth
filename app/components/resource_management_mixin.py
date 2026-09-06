# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from depth_fusion_core import APP_NAME, PROJECT_CACHE_DIR, PROJECT_LOG_DIR, QMessageBox, clear_all_cache, clear_cache_entry, clear_cache_older_than, directory_size_bytes, event_log, format_bytes, frame_cache_root, list_cache_entries, os
from components.model_manager import LocalModelManagerDialog

class ResourceManagementMixin:
    def open_model_manager(self) -> None:
        if self._has_background_processing():
            QMessageBox.warning(self, APP_NAME, "当前有预览、导出或融合重建任务，结束后再管理模型。")
            return
        dlg = LocalModelManagerDialog(self)
        dlg.exec()

    def open_log_dir(self) -> None:
        PROJECT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        event_log(f"打开日志目录: {PROJECT_LOG_DIR}", channel="UI")
        try:
            os.startfile(str(PROJECT_LOG_DIR))  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"无法打开日志目录: {exc}")

    def open_cache_manager(self) -> None:
        if self._has_background_processing():
            QMessageBox.warning(self, APP_NAME, "当前有预览、导出或融合重建任务，结束后再管理缓存。")
            return

        cache_dir = (self.current_project_dir / "cache") if getattr(self, "current_project_dir", None) else PROJECT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        total_size = directory_size_bytes(cache_dir)
        entries = list_cache_entries(cache_dir, limit=10)
        detail_lines = []
        for entry in entries:
            is_current = ""
            try:
                if self.current_input and self.video_info and frame_cache_root(self.make_config()).resolve() == entry.path.resolve():
                    is_current = "  ← 当前项目"
            except Exception:
                is_current = ""
            detail_lines.append(f"- {entry.path.name}: {format_bytes(entry.size_bytes)}{is_current}")
        detail = "\n".join(detail_lines) if detail_lines else "暂无缓存目录。"

        box = QMessageBox(self)
        box.setWindowTitle(APP_NAME)
        box.setIcon(QMessageBox.Question)
        box.setText(
            f"帧缓存目录：{cache_dir}\n"
            f"总大小：{format_bytes(total_size)}\n\n"
            f"最近缓存：\n{detail}\n\n"
            "请选择清理方式。"
        )
        current_btn = box.addButton("清当前项目", QMessageBox.ActionRole)
        old_btn = box.addButton("清7天前", QMessageBox.ActionRole)
        all_btn = box.addButton("清全部", QMessageBox.DestructiveRole)
        cancel_btn = box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked == cancel_btn or clicked is None:
            return
        if clicked == current_btn:
            if not self.current_input or not self.video_info:
                QMessageBox.information(self, APP_NAME, "当前没有已导入视频，无法定位当前项目缓存。")
                return
            try:
                root = frame_cache_root(self.make_config())
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, APP_NAME, f"无法定位当前项目缓存: {exc}")
                return
            removed = clear_cache_entry(root)
            self.log(f"已清理当前项目缓存: {root.name} / {format_bytes(removed)}")
            QMessageBox.information(self, APP_NAME, f"已清理当前项目缓存：{format_bytes(removed)}")
            return
        if clicked == old_btn:
            count, removed = clear_cache_older_than(cache_dir, days=7)
            self.log(f"已清理 7 天前缓存: {count} 项 / {format_bytes(removed)}")
            QMessageBox.information(self, APP_NAME, f"已清理 7 天前缓存：{count} 项，{format_bytes(removed)}")
            return
        if clicked == all_btn:
            confirm = QMessageBox.question(
                self,
                APP_NAME,
                "确认清空全部帧缓存？这不会删除原视频或导出结果。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return
            removed = clear_all_cache(cache_dir)
            self.log(f"已清空全部帧缓存: {format_bytes(removed)}")
            QMessageBox.information(self, APP_NAME, f"已清空全部帧缓存：{format_bytes(removed)}")
