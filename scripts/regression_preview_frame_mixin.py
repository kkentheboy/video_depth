#!/usr/bin/env python3
"""Source-only ownership contract for preview-frame synchronization and playback."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
UI = APP / "depth_fusion_ui.py"
MIXIN = APP / "components" / "preview_frame_mixin.py"
PROCESSING = APP / "components" / "processing_range_mixin.py"
TARGETS = {
    "_preview_frame_control_pairs",
    "_all_preview_frame_controls",
    "set_preview_frame_range",
    "update_preview_frame_label",
    "_apply_preview_frame_value",
    "on_preview_frame_slider_changed",
    "on_preview_frame_spin_changed",
    "toggle_preview_playback",
    "_advance_preview_playback",
}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def named_class(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    if len(matches) != 1:
        raise SystemExit(f"expected one {name}, got {len(matches)}")
    return matches[0]


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
    for path in (MIXIN, PROCESSING):
        if not path.is_file():
            raise SystemExit(f"missing required source: {path.relative_to(ROOT)}")

    ui_tree = parse(UI)
    mixin_tree = parse(MIXIN)
    main_window = named_class(ui_tree, "MainWindow")
    mixin = named_class(mixin_tree, "PreviewFrameMixin")

    base_names = [base.id for base in main_window.bases if isinstance(base, ast.Name)]
    required_order = [
        "ResourceManagementMixin",
        "FilePathActionsMixin",
        "ProcessingRangeMixin",
        "PreviewFrameMixin",
        "ProjectStateMixin",
        "JobConfigurationMixin",
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
        raise SystemExit(f"preview-frame methods returned to MainWindow: {sorted(leaked)}")

    mixin_methods = class_methods(mixin)
    missing = TARGETS - mixin_methods.keys()
    if missing:
        raise SystemExit(f"preview-frame mixin missing methods: {sorted(missing)}")

    for node in mixin_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "depth_fusion_ui":
            raise SystemExit("preview-frame mixin must not import depth_fusion_ui")
        if isinstance(node, ast.Import):
            if any(alias.name == "depth_fusion_ui" for alias in node.names):
                raise SystemExit("preview-frame mixin must not import depth_fusion_ui")

    ui_imports_mixin = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "components.preview_frame_mixin"
        and "PreviewFrameMixin" in {alias.name for alias in node.names}
        for node in ui_tree.body
    )
    if not ui_imports_mixin:
        raise SystemExit("depth_fusion_ui must import PreviewFrameMixin")

    required_delegations = {
        "_all_preview_frame_controls": {"_preview_frame_control_pairs"},
        "set_preview_frame_range": {
            "_all_preview_frame_controls",
            "_set_processing_frame_range",
            "update_preview_frame_label",
        },
        "_apply_preview_frame_value": {
            "_all_preview_frame_controls",
            "update_preview_frame_label",
            "show_original_frame_immediately",
            "_refresh_reference_preview_tiles",
            "_schedule_active_mesh_preview_refresh",
        },
        "on_preview_frame_slider_changed": {"_apply_preview_frame_value"},
        "on_preview_frame_spin_changed": {"_apply_preview_frame_value"},
        "_advance_preview_playback": {"_processing_range_values", "_apply_preview_frame_value"},
    }
    for method_name, required_attrs in required_delegations.items():
        missing_attrs = required_attrs - attrs(mixin_methods[method_name])
        if missing_attrs:
            raise SystemExit(f"{method_name} lost preview-frame delegation(s): {sorted(missing_attrs)}")

    core_imports = set()
    for node in mixin_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "depth_fusion_core":
            core_imports.update(alias.name for alias in node.names)
    if core_imports != {"QTimer", "format_seconds"}:
        raise SystemExit(f"unexpected preview-frame core imports: {sorted(core_imports)}")

    init_methods = [n for n in main_window.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    if len(init_methods) != 1:
        raise SystemExit("MainWindow.__init__ is missing or duplicated")
    init_attrs = attrs(init_methods[0])
    required_init_refs = {
        "on_preview_frame_slider_changed",
        "on_preview_frame_spin_changed",
        "toggle_preview_playback",
        "_advance_preview_playback",
    }
    missing_init = required_init_refs - init_attrs
    if missing_init:
        raise SystemExit(f"MainWindow init no longer wires preview-frame handlers: {sorted(missing_init)}")

    processing_methods = class_methods(named_class(parse(PROCESSING), "ProcessingRangeMixin"))
    for handler in (
        "_on_processing_start_slider_changed",
        "_on_processing_end_slider_changed",
        "_on_processing_start_spin_changed",
        "_on_processing_end_spin_changed",
    ):
        if "_apply_preview_frame_value" not in attrs(processing_methods[handler]):
            raise SystemExit(f"{handler} must continue delegating into PreviewFrameMixin")

    print("video_depth preview frame mixin authority: PASS")


if __name__ == "__main__":
    main()
