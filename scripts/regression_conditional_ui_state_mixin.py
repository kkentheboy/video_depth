from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_PATH = ROOT / "app" / "depth_fusion_ui.py"
MIXIN_PATH = ROOT / "app" / "components" / "conditional_ui_state_mixin.py"
INPUT_SOURCE_MIXIN_PATH = ROOT / "app" / "components" / "input_source_state_mixin.py"
EXPECTED_METHODS = {"_on_density_mode_changed", "_update_conditional_visibility"}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def direct_methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {node.name: node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def self_calls(node: ast.AST) -> set[str]:
    calls: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        target = sub.func
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
            calls.add(target.attr)
    return calls


def string_literals(node: ast.AST) -> set[str]:
    return {str(sub.value) for sub in ast.walk(node) if isinstance(sub, ast.Constant) and isinstance(sub.value, str)}


def main() -> None:
    ui_cls = class_node(parse(UI_PATH), "MainWindow")
    mixin_cls = class_node(parse(MIXIN_PATH), "ConditionalUiStateMixin")
    input_source_cls = class_node(parse(INPUT_SOURCE_MIXIN_PATH), "InputSourceStateMixin")
    ui_methods = direct_methods(ui_cls)
    mixin_methods = direct_methods(mixin_cls)
    input_source_methods = direct_methods(input_source_cls)

    assert not (EXPECTED_METHODS - mixin_methods.keys())
    assert not (EXPECTED_METHODS & ui_methods.keys())
    assert "_effective_pointcloud_stride" in ui_methods, "unused/trivial stride query must not be moved just to enlarge this mixin"
    assert "on_background_fill_changed" in ui_methods, "preview-render action must remain on MainWindow"
    assert "depth_fusion_ui" not in MIXIN_PATH.read_text(encoding="utf-8")

    bases = [ast.unparse(base) for base in ui_cls.bases]
    assert bases.index("InputSourceStateMixin") < bases.index("ConditionalUiStateMixin") < bases.index("StructureCacheStateMixin")

    visibility = mixin_methods["_update_conditional_visibility"]
    visibility_calls = self_calls(visibility)
    assert {"_current_source_mode", "_on_density_mode_changed"} <= visibility_calls
    literals = string_literals(visibility)
    assert {"matanyone", "external_mask", "背景灰"} <= literals

    density = mixin_methods["_on_density_mode_changed"]
    density_literals = string_literals(density)
    assert {"pointcloud_stride_row", "pointcloud_max_points_row", "pointcloud_stride_spin", "pointcloud_max_points_spin"} <= density_literals

    source_sync_calls = self_calls(input_source_methods["_sync_source_mode_radios"])
    source_apply_calls = self_calls(input_source_methods["_apply_source_mode"])
    assert "_update_conditional_visibility" in source_sync_calls
    assert "_update_conditional_visibility" in source_apply_calls

    background_calls = self_calls(ui_methods["on_background_fill_changed"])
    assert {"_update_conditional_visibility", "_switch_to_fusion_preview_for_curve_edit", "render_preview_from_cache"} <= background_calls

    print("conditional UI state mixin contract: PASS")


if __name__ == "__main__":
    main()
