#!/usr/bin/env python3
"""Source-only ownership contract for processing-range controls."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
UI = APP / "depth_fusion_ui.py"
MIXIN = APP / "components" / "processing_range_mixin.py"
PROJECT_STATE = APP / "components" / "project_state_mixin.py"
JOB_CONFIG = APP / "components" / "job_configuration_mixin.py"
TARGETS = {
    "_set_processing_frame_range",
    "_processing_range_values",
    "_refresh_processing_range_label",
    "_set_processing_values",
    "_on_processing_start_slider_changed",
    "_on_processing_end_slider_changed",
    "_on_processing_start_spin_changed",
    "_on_processing_end_spin_changed",
}
HANDLERS = {
    "_on_processing_start_slider_changed",
    "_on_processing_end_slider_changed",
    "_on_processing_start_spin_changed",
    "_on_processing_end_spin_changed",
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


def named_class(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    if len(matches) != 1:
        raise SystemExit(f"expected one {name}, got {len(matches)}")
    return matches[0]


def main() -> None:
    for path in (MIXIN, PROJECT_STATE, JOB_CONFIG):
        if not path.is_file():
            raise SystemExit(f"missing required source: {path.relative_to(ROOT)}")

    ui_tree = parse(UI)
    mixin_tree = parse(MIXIN)
    main_window = named_class(ui_tree, "MainWindow")
    mixin = named_class(mixin_tree, "ProcessingRangeMixin")

    base_names = [base.id for base in main_window.bases if isinstance(base, ast.Name)]
    required_order = [
        "ResourceManagementMixin",
        "FilePathActionsMixin",
        "ProcessingRangeMixin",
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
        raise SystemExit(f"processing-range methods returned to MainWindow: {sorted(leaked)}")

    mixin_methods = class_methods(mixin)
    missing = TARGETS - mixin_methods.keys()
    if missing:
        raise SystemExit(f"processing-range mixin missing methods: {sorted(missing)}")

    for node in mixin_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "depth_fusion_ui":
            raise SystemExit("processing-range mixin must not import depth_fusion_ui")
        if isinstance(node, ast.Import):
            if any(alias.name == "depth_fusion_ui" for alias in node.names):
                raise SystemExit("processing-range mixin must not import depth_fusion_ui")

    ui_imports_mixin = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "components.processing_range_mixin"
        and "ProcessingRangeMixin" in {alias.name for alias in node.names}
        for node in ui_tree.body
    )
    if not ui_imports_mixin:
        raise SystemExit("depth_fusion_ui must import ProcessingRangeMixin")

    required_delegations = {
        "_set_processing_frame_range": {"_refresh_processing_range_label"},
        "_refresh_processing_range_label": {"_processing_range_values"},
        "_set_processing_values": {"_processing_range_values", "_refresh_processing_range_label"},
    }
    for method_name, required_attrs in required_delegations.items():
        missing_attrs = required_attrs - attrs(mixin_methods[method_name])
        if missing_attrs:
            raise SystemExit(f"{method_name} lost range delegation(s): {sorted(missing_attrs)}")

    for handler in HANDLERS:
        handler_attrs = attrs(mixin_methods[handler])
        for required in ("_set_processing_values", "_apply_preview_frame_value"):
            if required not in handler_attrs:
                raise SystemExit(f"{handler} must continue delegating through self.{required}")

    init_methods = [
        n for n in main_window.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    ]
    if len(init_methods) != 1:
        raise SystemExit("MainWindow.__init__ is missing or duplicated")
    init_attrs = attrs(init_methods[0])
    missing_handlers = HANDLERS - init_attrs
    if missing_handlers:
        raise SystemExit(f"MainWindow init no longer wires processing handlers: {sorted(missing_handlers)}")

    project_methods = class_methods(named_class(parse(PROJECT_STATE), "ProjectStateMixin"))
    if "_processing_range_values" not in attrs(project_methods["_preset_payload"]):
        raise SystemExit("ProjectStateMixin._preset_payload must read the processing-range authority")

    config_methods = class_methods(named_class(parse(JOB_CONFIG), "JobConfigurationMixin"))
    if "_processing_range_values" not in attrs(config_methods["make_config"]):
        raise SystemExit("JobConfigurationMixin.make_config must read the processing-range authority")

    print("video_depth processing range mixin authority: PASS")


if __name__ == "__main__":
    main()
