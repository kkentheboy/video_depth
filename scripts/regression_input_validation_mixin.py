#!/usr/bin/env python3
"""Source-only ownership contract for input/media validation state."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
UI = APP / "depth_fusion_ui.py"
MIXIN = APP / "components" / "input_validation_mixin.py"
TARGETS = {
    "_is_image_path",
    "_safe_probe_external_media",
    "_main_video_alpha_state",
    "_update_external_media_status_label",
    "validate_main_video_alpha_chain",
    "validate_external_reference_chain",
    "on_external_media_changed",
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
    if not MIXIN.is_file():
        raise SystemExit("components/input_validation_mixin.py is missing")

    ui_tree = parse(UI)
    mixin_tree = parse(MIXIN)
    main_window = named_class(ui_tree, "MainWindow")
    mixin = named_class(mixin_tree, "InputValidationMixin")

    base_names = [base.id for base in main_window.bases if isinstance(base, ast.Name)]
    required_order = [
        "ResourceManagementMixin",
        "FilePathActionsMixin",
        "InputValidationMixin",
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
        raise SystemExit(f"input-validation methods returned to MainWindow: {sorted(leaked)}")

    mixin_methods = class_methods(mixin)
    missing = TARGETS - mixin_methods.keys()
    if missing:
        raise SystemExit(f"input-validation mixin missing methods: {sorted(missing)}")

    for node in mixin_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "depth_fusion_ui":
            raise SystemExit("input-validation mixin must not import depth_fusion_ui")
        if isinstance(node, ast.Import):
            if any(alias.name == "depth_fusion_ui" for alias in node.names):
                raise SystemExit("input-validation mixin must not import depth_fusion_ui")

    ui_imports_mixin = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "components.input_validation_mixin"
        and "InputValidationMixin" in {alias.name for alias in node.names}
        for node in ui_tree.body
    )
    if not ui_imports_mixin:
        raise SystemExit("depth_fusion_ui must import InputValidationMixin")

    required_delegations = {
        "_safe_probe_external_media": {"_is_image_path"},
        "validate_main_video_alpha_chain": {
            "_main_video_alpha_state",
            "_refresh_reference_preview_tiles",
            "_update_external_media_status_label",
        },
        "validate_external_reference_chain": {"validate_main_video_alpha_chain"},
        "on_external_media_changed": {
            "_update_external_media_status_label",
            "_refresh_reference_preview_tiles",
            "validate_external_reference_chain",
        },
    }
    for method_name, required_attrs in required_delegations.items():
        missing_attrs = required_attrs - attrs(mixin_methods[method_name])
        if missing_attrs:
            raise SystemExit(f"{method_name} lost input-validation delegation(s): {sorted(missing_attrs)}")

    core_imports = set()
    for node in mixin_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "depth_fusion_core":
            core_imports.update(alias.name for alias in node.names)
    required_core = {
        "APP_NAME",
        "Optional",
        "Path",
        "QMessageBox",
        "VideoInfo",
        "cv2",
        "describe_real_alpha_source",
        "probe_video",
        "short_error_message",
    }
    missing_core = required_core - core_imports
    if missing_core:
        raise SystemExit(f"input-validation mixin missing core dependencies: {sorted(missing_core)}")

    if "load_video" not in main_methods:
        raise SystemExit("MainWindow.load_video must remain the input lifecycle owner")
    if "validate_main_video_alpha_chain" not in attrs(main_methods["load_video"]):
        raise SystemExit("MainWindow.load_video must continue delegating validation to InputValidationMixin")

    init_methods = [n for n in main_window.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    if len(init_methods) != 1:
        raise SystemExit("MainWindow.__init__ is missing or duplicated")
    if "on_external_media_changed" not in attrs(init_methods[0]):
        raise SystemExit("MainWindow init no longer wires on_external_media_changed")

    print("video_depth input validation mixin authority: PASS")


if __name__ == "__main__":
    main()
