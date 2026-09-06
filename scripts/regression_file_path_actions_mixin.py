#!/usr/bin/env python3
"""Source-only ownership contract for MainWindow file/path actions."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "app" / "depth_fusion_ui.py"
MIXIN = ROOT / "app" / "components" / "file_path_actions_mixin.py"
TARGETS = {"pick_video", "pick_output_path", "open_output_dir"}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def class_methods(node: ast.ClassDef) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        child.name: child
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def main() -> None:
    if not MIXIN.is_file():
        raise SystemExit("components/file_path_actions_mixin.py is missing")

    ui_tree = parse(UI)
    mixin_tree = parse(MIXIN)

    main_classes = [n for n in ui_tree.body if isinstance(n, ast.ClassDef) and n.name == "MainWindow"]
    if len(main_classes) != 1:
        raise SystemExit(f"expected one MainWindow, got {len(main_classes)}")
    main_window = main_classes[0]

    mixin_classes = [
        n for n in mixin_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "FilePathActionsMixin"
    ]
    if len(mixin_classes) != 1:
        raise SystemExit(f"expected one FilePathActionsMixin, got {len(mixin_classes)}")
    mixin = mixin_classes[0]

    base_names = [base.id for base in main_window.bases if isinstance(base, ast.Name)]
    expected_prefix = ["ResourceManagementMixin", "FilePathActionsMixin", "QMainWindow"]
    if base_names[:3] != expected_prefix:
        raise SystemExit(f"unexpected MainWindow mixin order: {base_names}")

    main_methods = class_methods(main_window)
    leaked = TARGETS & main_methods.keys()
    if leaked:
        raise SystemExit(f"file/path methods returned to MainWindow: {sorted(leaked)}")

    mixin_methods = class_methods(mixin)
    missing = TARGETS - mixin_methods.keys()
    if missing:
        raise SystemExit(f"file/path mixin missing methods: {sorted(missing)}")

    for node in mixin_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "depth_fusion_ui":
            raise SystemExit("file/path mixin must not import depth_fusion_ui")
        if isinstance(node, ast.Import):
            if any(alias.name == "depth_fusion_ui" for alias in node.names):
                raise SystemExit("file/path mixin must not import depth_fusion_ui")

    ui_imports_mixin = False
    for node in ui_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "components.file_path_actions_mixin":
            if "FilePathActionsMixin" in {alias.name for alias in node.names}:
                ui_imports_mixin = True
    if not ui_imports_mixin:
        raise SystemExit("depth_fusion_ui must import FilePathActionsMixin")

    init_methods = [
        n for n in main_window.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    ]
    if len(init_methods) != 1:
        raise SystemExit("MainWindow.__init__ is missing or duplicated")
    init_attrs = {
        node.attr
        for node in ast.walk(init_methods[0])
        if isinstance(node, ast.Attribute)
    }
    missing_wiring = TARGETS - init_attrs
    if missing_wiring:
        raise SystemExit(f"MainWindow init no longer references file/path handlers: {sorted(missing_wiring)}")

    pick_video_attrs = {
        node.attr
        for node in ast.walk(mixin_methods["pick_video"])
        if isinstance(node, ast.Attribute)
    }
    if "load_video" not in pick_video_attrs:
        raise SystemExit("pick_video must continue delegating actual import to self.load_video")

    print("video_depth file path actions mixin authority: PASS")


if __name__ == "__main__":
    main()
