#!/usr/bin/env python3
"""Source-only ownership contract for MainWindow resource-management methods."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "app" / "depth_fusion_ui.py"
MIXIN = ROOT / "app" / "components" / "resource_management_mixin.py"
TARGETS = {"open_model_manager", "open_log_dir", "open_cache_manager"}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def methods(node: ast.ClassDef) -> set[str]:
    return {
        child.name
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def main() -> None:
    if not MIXIN.is_file():
        raise SystemExit("components/resource_management_mixin.py is missing")

    ui_tree = parse(UI)
    mixin_tree = parse(MIXIN)

    main_classes = [n for n in ui_tree.body if isinstance(n, ast.ClassDef) and n.name == "MainWindow"]
    if len(main_classes) != 1:
        raise SystemExit(f"expected one MainWindow, got {len(main_classes)}")
    main_window = main_classes[0]

    mixin_classes = [
        n for n in mixin_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ResourceManagementMixin"
    ]
    if len(mixin_classes) != 1:
        raise SystemExit(f"expected one ResourceManagementMixin, got {len(mixin_classes)}")
    mixin = mixin_classes[0]

    base_names = [base.id for base in main_window.bases if isinstance(base, ast.Name)]
    if "ResourceManagementMixin" not in base_names:
        raise SystemExit("MainWindow must inherit ResourceManagementMixin")
    if "QMainWindow" not in base_names:
        raise SystemExit("MainWindow must remain a QMainWindow")
    if base_names.index("ResourceManagementMixin") > base_names.index("QMainWindow"):
        raise SystemExit("ResourceManagementMixin must precede QMainWindow in MRO")

    leaked = TARGETS & methods(main_window)
    if leaked:
        raise SystemExit(f"resource-management methods returned to MainWindow: {sorted(leaked)}")

    missing = TARGETS - methods(mixin)
    if missing:
        raise SystemExit(f"resource-management mixin missing methods: {sorted(missing)}")

    for node in mixin_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "depth_fusion_ui":
            raise SystemExit("resource-management mixin must not import depth_fusion_ui")
        if isinstance(node, ast.Import):
            if any(alias.name == "depth_fusion_ui" for alias in node.names):
                raise SystemExit("resource-management mixin must not import depth_fusion_ui")

    ui_imports_mixin = False
    for node in ui_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "components.resource_management_mixin":
            if "ResourceManagementMixin" in {alias.name for alias in node.names}:
                ui_imports_mixin = True
    if not ui_imports_mixin:
        raise SystemExit("depth_fusion_ui must import ResourceManagementMixin")

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
        raise SystemExit(f"MainWindow init no longer references resource handlers: {sorted(missing_wiring)}")

    print("video_depth resource management mixin authority: PASS")


if __name__ == "__main__":
    main()
