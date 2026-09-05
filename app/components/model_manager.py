# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import shutil
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QLabel, QPlainTextEdit, QHBoxLayout,
    QPushButton, QMessageBox,
)

from depth_fusion_core import (
    APP_NAME, PROJECT_DIR, PROJECT_HF_HUB, PROJECT_MODELS_DIR,
    clear_memory_model_cache, directory_size_bytes, format_bytes,
)


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)

def collect_unused_model_candidates() -> list[Path]:
    candidates: list[Path] = []
    for p in [
        PROJECT_HF_HUB / "models--old-depth--old-depth-cache",
        PROJECT_HF_HUB / "models--facebook--old-normal-cache-0.3b-torchscript",
        PROJECT_DIR / "models" / "video_depth_anything",
        PROJECT_DIR / "vendor" / "old-depth-vendor",
    ]:
        if p.exists():
            candidates.append(p)
    if PROJECT_HF_HUB.exists():
        candidates.extend(PROJECT_HF_HUB.rglob("*.incomplete"))
    return candidates

def delete_unused_models(candidates: list[Path]) -> tuple[list[str], list[str]]:
    removed: list[str] = []
    errors: list[str] = []
    for item in candidates:
        try:
            if item.is_dir():
                safe_rmtree(item)
            elif item.exists():
                item.unlink()
            if not item.exists():
                removed.append(str(item))
            else:
                errors.append(f"删除失败: {item}")
        except OSError as exc:
            errors.append(f"删除失败: {item} -> {exc}")
    return removed, errors

class LocalModelManagerDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("本地模型管理")
        self.resize(780, 540)
        layout = QVBoxLayout(self)
        note = QLabel("当前主线只需要 4DHumans/SMPL/FASHN。这里仅列出可清理的旧深度/旧法线缓存。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.text_box = QPlainTextEdit()
        self.text_box.setReadOnly(True)
        layout.addWidget(self.text_box, 1)
        row = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新")
        self.open_btn = QPushButton("打开 models")
        self.clear_mem_btn = QPushButton("清空内存缓存")
        self.delete_btn = QPushButton("删除旧缓存")
        self.close_btn = QPushButton("关闭")
        for btn in [self.refresh_btn, self.open_btn, self.clear_mem_btn, self.delete_btn]:
            row.addWidget(btn)
        row.addStretch(1)
        row.addWidget(self.close_btn)
        layout.addLayout(row)
        self.refresh_btn.clicked.connect(self.refresh)
        self.open_btn.clicked.connect(self.open_models_dir)
        self.clear_mem_btn.clicked.connect(self.clear_memory_cache)
        self.delete_btn.clicked.connect(self.delete_unused)
        self.close_btn.clicked.connect(self.accept)
        self.refresh()

    def refresh(self) -> None:
        lines = [f"项目目录: {PROJECT_DIR}", f"模型目录: {PROJECT_MODELS_DIR}", ""]
        candidates = collect_unused_model_candidates()
        if not candidates:
            lines.append("未发现旧深度/旧法线缓存。")
        total = 0
        for p in candidates:
            size = directory_size_bytes(p)
            total += size
            lines.append(f"[可删] {p.name}")
            lines.append(f"  大小: {format_bytes(size)}")
            lines.append(f"  路径: {p}")
            lines.append("")
        lines.append(f"合计: {format_bytes(total)}")
        self.text_box.setPlainText("\n".join(lines))

    def open_models_dir(self) -> None:
        PROJECT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(PROJECT_MODELS_DIR))  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"无法打开目录: {exc}")

    def clear_memory_cache(self) -> None:
        clear_memory_model_cache()
        QMessageBox.information(self, APP_NAME, "已清空内存模型缓存。")

    def delete_unused(self) -> None:
        candidates = collect_unused_model_candidates()
        if not candidates:
            QMessageBox.information(self, APP_NAME, "没有发现可删除旧缓存。")
            return
        preview = "\n".join(str(p) for p in candidates[:20])
        if QMessageBox.question(
            self,
            APP_NAME,
            "确认删除这些旧缓存？\n\n" + preview,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        removed, errors = delete_unused_models(candidates)
        clear_memory_model_cache()
        QMessageBox.information(self, APP_NAME, "删除完成。\n" + "\n".join(removed + errors))
        self.refresh()

