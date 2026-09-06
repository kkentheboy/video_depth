#!/usr/bin/env python3
"""Source-only ownership contract for project/preset state handling."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "app" / "depth_fusion_ui.py"
MIXIN = ROOT / "app" / "components" / "project_state_mixin.py"
TARGETS = {
    "_preset_payload",
    "_apply_preset_payload",
    "apply_builtin_preset",
    "export_preset_json",
    "import_preset_json",
    "save_project_state",
    "load_project",
}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def class_methods(node: ast.ClassDef) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        child.name: child
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def attrs(node: ast.AST) -> set[str]:
    return {
        child.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
    }


def require_mro_order(base_names: list[str], expected: list[str]) -> None:
    missing = [name for name in expected if name not in base_names]
    if missing:
        raise SystemExit(f"MainWindow MRO missing required bases: {missing}; actual={base_names}")
    positions = [base_names.index(name) for name in expected]
    if positions != sorted(positions):
        raise SystemExit(f"unexpected MainWindow mixin order: {base_names}")


def main() -> None:
    if not MIXIN.is_file():
        raise SystemExit("components/project_state_mixin.py is missing")

    ui_tree = parse(UI)
    mixin_tree = parse(MIXIN)

    mains = [n for n in ui_tree.body if isinstance(n, ast.ClassDef) and n.name == "MainWindow"]
    if len(mains) != 1:
        raise SystemExit(f"expected one MainWindow, got {len(mains)}")
    main_window = mains[0]

    mixins = [n for n in mixin_tree.body if isinstance(n, ast.ClassDef) and n.name == "ProjectStateMixin"]
    if len(mixins) != 1:
        raise SystemExit(f"expected one ProjectStateMixin, got {len(mixins)}")
    mixin = mixins[0]

    base_names = [base.id for base in main_window.bases if isinstance(base, ast.Name)]
    require_mro_order(
        base_names,
        ["ResourceManagementMixin", "FilePathActionsMixin", "ProjectStateMixin", "QMainWindow"],
    )

    main_methods = class_methods(main_window)
    leaked = TARGETS & main_methods.keys()
    if leaked:
        raise SystemExit(f"project-state methods returned to MainWindow: {sorted(leaked)}")

    mixin_methods = class_methods(mixin)
    missing = TARGETS - mixin_methods.keys()
    if missing:
        raise SystemExit(f"project-state mixin missing methods: {sorted(missing)}")

    for node in mixin_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "depth_fusion_ui":
            raise SystemExit("project-state mixin must not import depth_fusion_ui")
        if isinstance(node, ast.Import):
            if any(alias.name == "depth_fusion_ui" for alias in node.names):
                raise SystemExit("project-state mixin must not import depth_fusion_ui")

    ui_imports_mixin = False
    for node in ui_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "components.project_state_mixin":
            if "ProjectStateMixin" in {alias.name for alias in node.names}:
                ui_imports_mixin = True
    if not ui_imports_mixin:
        raise SystemExit("depth_fusion_ui must import ProjectStateMixin")

    required_calls = {
        "save_project_state": "_preset_payload",
        "export_preset_json": "_preset_payload",
        "load_project": "_apply_preset_payload",
        "import_preset_json": "_apply_preset_payload",
        "apply_builtin_preset": "_apply_preset_payload",
    }
    for method_name, required_attr in required_calls.items():
        if required_attr not in attrs(mixin_methods[method_name]):
            raise SystemExit(f"{method_name} must continue delegating through self.{required_attr}")

    print("video_depth project state mixin authority: PASS")


if __name__ == "__main__":
    main()
