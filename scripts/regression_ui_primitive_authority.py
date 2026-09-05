#!/usr/bin/env python3
"""Source-only contract for UI primitives moved out of depth_fusion_workers."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
WORKERS = APP / "depth_fusion_workers.py"
WIDGETS = APP / "components" / "widgets.py"
UI = APP / "depth_fusion_ui.py"

MOVED = {
    "DropLineEdit",
    "NoWheelSlider",
    "NoWheelSpinBox",
    "NoWheelDoubleSpinBox",
    "NoWheelComboBox",
    "PreviewImageLabel",
}

EXPECTED_BASES = {
    "DropLineEdit": "QLineEdit",
    "NoWheelSlider": "QSlider",
    "NoWheelSpinBox": "QSpinBox",
    "NoWheelDoubleSpinBox": "QDoubleSpinBox",
    "NoWheelComboBox": "QComboBox",
    "PreviewImageLabel": "QLabel",
}


def module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def top_level_classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


def base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def imported_names(tree: ast.Module, module: str) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


def main() -> None:
    worker_tree = module_tree(WORKERS)
    widget_tree = module_tree(WIDGETS)
    ui_tree = module_tree(UI)

    worker_classes = top_level_classes(worker_tree)
    widget_classes = top_level_classes(widget_tree)

    leaked = MOVED & set(worker_classes)
    if leaked:
        raise SystemExit(f"UI primitives returned to worker module: {sorted(leaked)}")

    missing = MOVED - set(widget_classes)
    if missing:
        raise SystemExit(f"UI primitives missing from components.widgets: {sorted(missing)}")

    for name, expected_base in EXPECTED_BASES.items():
        node = widget_classes[name]
        bases = {base_name(base) for base in node.bases}
        if expected_base not in bases:
            raise SystemExit(
                f"{name} base changed: expected {expected_base}, found {sorted(bases)}"
            )

    widget_worker_imports = imported_names(widget_tree, "depth_fusion_workers")
    if widget_worker_imports:
        raise SystemExit(
            "components.widgets must not depend on depth_fusion_workers: "
            f"{sorted(widget_worker_imports)}"
        )

    for path in sorted(APP.rglob("*.py")):
        tree = module_tree(path)
        bad = MOVED & imported_names(tree, "depth_fusion_workers")
        if bad:
            raise SystemExit(
                f"{path.relative_to(ROOT)} imports UI primitives from workers: {sorted(bad)}"
            )

    ui_widget_imports = imported_names(ui_tree, "components.widgets")
    missing_ui = MOVED - ui_widget_imports
    if missing_ui:
        raise SystemExit(
            f"depth_fusion_ui must import moved primitives from components.widgets: {sorted(missing_ui)}"
        )

    print("video_depth UI primitive authority: PASS")


if __name__ == "__main__":
    main()
