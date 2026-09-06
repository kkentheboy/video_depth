#!/usr/bin/env python3
"""Source-only ownership contract for output routing and geometry actions."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "app" / "depth_fusion_ui.py"
MIXIN = ROOT / "app" / "components" / "output_routing_mixin.py"
TARGETS = {
    "_set_encoder_combo_value",
    "_current_encoder_mode",
    "_is_structure_output_mode",
    "_is_png_sequence_mode",
    "_default_output_path_for_encoder",
    "_coerce_output_path_for_encoder",
    "on_encoder_changed",
    "refresh_output_size",
    "on_output_geometry_changed",
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


def main() -> None:
    if not MIXIN.is_file():
        raise SystemExit("components/output_routing_mixin.py is missing")

    ui_tree = parse(UI)
    mixin_tree = parse(MIXIN)

    mains = [n for n in ui_tree.body if isinstance(n, ast.ClassDef) and n.name == "MainWindow"]
    if len(mains) != 1:
        raise SystemExit(f"expected one MainWindow, got {len(mains)}")
    main_window = mains[0]

    mixins = [n for n in mixin_tree.body if isinstance(n, ast.ClassDef) and n.name == "OutputRoutingMixin"]
    if len(mixins) != 1:
        raise SystemExit(f"expected one OutputRoutingMixin, got {len(mixins)}")
    mixin = mixins[0]

    base_names = [base.id for base in main_window.bases if isinstance(base, ast.Name)]
    required_order = [
        "ResourceManagementMixin",
        "FilePathActionsMixin",
        "ProjectStateMixin",
        "OutputRoutingMixin",
        "QMainWindow",
    ]
    positions = {}
    for name in required_order:
        if name not in base_names:
            raise SystemExit(f"MainWindow missing base {name}: {base_names}")
        positions[name] = base_names.index(name)
    if [positions[name] for name in required_order] != sorted(positions.values()):
        raise SystemExit(f"unexpected relative MainWindow mixin order: {base_names}")

    main_methods = class_methods(main_window)
    leaked = TARGETS & main_methods.keys()
    if leaked:
        raise SystemExit(f"output-routing methods returned to MainWindow: {sorted(leaked)}")

    mixin_methods = class_methods(mixin)
    missing = TARGETS - mixin_methods.keys()
    if missing:
        raise SystemExit(f"output-routing mixin missing methods: {sorted(missing)}")

    for node in mixin_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "depth_fusion_ui":
            raise SystemExit("output-routing mixin must not import depth_fusion_ui")
        if isinstance(node, ast.Import):
            if any(alias.name == "depth_fusion_ui" for alias in node.names):
                raise SystemExit("output-routing mixin must not import depth_fusion_ui")

    ui_imports_mixin = False
    for node in ui_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "components.output_routing_mixin":
            if "OutputRoutingMixin" in {alias.name for alias in node.names}:
                ui_imports_mixin = True
    if not ui_imports_mixin:
        raise SystemExit("depth_fusion_ui must import OutputRoutingMixin")

    required_calls = {
        "_is_structure_output_mode": {"_pointcloud_mode"},
        "_is_png_sequence_mode": {"_is_structure_output_mode"},
        "_default_output_path_for_encoder": {"_is_structure_output_mode", "_is_png_sequence_mode"},
        "_coerce_output_path_for_encoder": {
            "_default_output_path_for_encoder",
            "_is_structure_output_mode",
            "_is_png_sequence_mode",
        },
        "on_encoder_changed": {
            "_default_output_path_for_encoder",
            "_coerce_output_path_for_encoder",
            "_is_structure_output_mode",
        },
        "refresh_output_size": {"_default_output_path_for_encoder"},
        "on_output_geometry_changed": {"refresh_output_size"},
    }
    for method_name, required_attrs in required_calls.items():
        missing_attrs = required_attrs - attrs(mixin_methods[method_name])
        if missing_attrs:
            raise SystemExit(
                f"{method_name} lost routing delegation(s): {sorted(missing_attrs)}"
            )

    init_methods = [
        n for n in main_window.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    ]
    if len(init_methods) != 1:
        raise SystemExit("MainWindow.__init__ is missing or duplicated")
    init_attrs = attrs(init_methods[0])
    for handler in ("on_encoder_changed", "on_output_geometry_changed"):
        if handler not in init_attrs:
            raise SystemExit(f"MainWindow init no longer wires {handler}")

    print("video_depth output routing mixin authority: PASS")


if __name__ == "__main__":
    main()
