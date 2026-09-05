from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QLineEdit, QMessageBox
)
from PySide6.QtCore import Qt
import json

from depth_fusion_core import PROJECT_DIR, APP_NAME

class ProjectManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - 项目管理器")
        self.setMinimumSize(800, 500)
        
        self.projects_dir = PROJECT_DIR / "data" / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        
        self.selected_project_dir: Path | None = None
        
        layout = QVBoxLayout(self)
        
        # Make the dialog background dark
        self.setStyleSheet("QDialog { background: #18181b; } QLabel { color: #e4e4e7; }")
        
        title = QLabel("项目库 (Project Library)")
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        
        # Project List
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background: #27272a;
                color: #e4e4e7;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 4px;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #3f3f46;
            }
            QListWidget::item:selected {
                background: #2563eb;
                color: white;
                border-radius: 4px;
            }
        """)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget, 1)
        
        # Bottom Bar
        bottom_layout = QHBoxLayout()
        
        self.new_proj_edit = QLineEdit()
        self.new_proj_edit.setStyleSheet("background: #27272a; color: #e4e4e7; border: 1px solid #3f3f46; border-radius: 4px; padding: 0 8px;")
        self.new_proj_edit.setPlaceholderText("输入新项目名称...")
        self.new_proj_edit.setMinimumHeight(32)
        bottom_layout.addWidget(self.new_proj_edit, 1)
        
        new_btn = QPushButton("新建项目")
        new_btn.setMinimumHeight(32)
        new_btn.clicked.connect(self._create_new_project)
        bottom_layout.addWidget(new_btn)
        
        bottom_layout.addSpacing(20)
        
        open_btn = QPushButton("打开选中项目")
        open_btn.setMinimumHeight(32)
        open_btn.setMinimumWidth(120)
        open_btn.setStyleSheet("background: #2563eb; color: white; border: none; border-radius: 4px; font-weight: bold;")
        open_btn.clicked.connect(self._open_selected_project)
        bottom_layout.addWidget(open_btn)
        
        layout.addLayout(bottom_layout)
        
        self._refresh_list()
        
    def _refresh_list(self):
        self.list_widget.clear()
        projects = []
        for d in self.projects_dir.iterdir():
            if d.is_dir() and (d / "project.vhm").exists():
                # Get modified time of project.vhm if exists, else dir
                vhm = d / "project.vhm"
                mtime = vhm.stat().st_mtime if vhm.exists() else d.stat().st_mtime
                projects.append((mtime, d))
                
        projects.sort(key=lambda x: x[0], reverse=True)
        
        for mtime, d in projects:
            item = QListWidgetItem(d.name)
            item.setData(Qt.UserRole, d)
            self.list_widget.addItem(item)
            
    def _create_new_project(self):
        name = self.new_proj_edit.text().strip()
        if not name:
            idx = 1
            while True:
                candidate = f"未命名项目_{idx}"
                if not (self.projects_dir / candidate).exists():
                    name = candidate
                    break
                idx += 1
            
        # sanitize name
        name = "".join(c for c in name if c.isalnum() or c in " _-")
        if not name:
            QMessageBox.warning(self, "错误", "项目名称无效")
            return
            
        proj_dir = self.projects_dir / name
        if proj_dir.exists():
            QMessageBox.warning(self, "错误", "该项目已存在")
            return
            
        try:
            proj_dir.mkdir(parents=True)
            vhm = proj_dir / "project.vhm"
            vhm.write_text(json.dumps({"workflow_step": 1}, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建项目失败: {e}")
            return
            
        self.selected_project_dir = proj_dir
        self.accept()
        
    def _open_selected_project(self):
        items = self.list_widget.selectedItems()
        if not items:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        
        self.selected_project_dir = items[0].data(Qt.UserRole)
        self.accept()
        
    def _on_item_double_clicked(self, item):
        self.selected_project_dir = item.data(Qt.UserRole)
        self.accept()
