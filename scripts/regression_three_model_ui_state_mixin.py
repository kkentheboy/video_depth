from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_PATH = ROOT / "app" / "depth_fusion_ui.py"
MIXIN_PATH = ROOT / "app" / "components" / "three_model_ui_state_mixin.py"
EXPECTED_METHODS = {"_effective_normal_strength", "_effective_normal_refine", "_update_three_model_status"}


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


def constant_return(node: ast.FunctionDef | ast.AsyncFunctionDef) -> object:
    returns = [sub for sub in ast.walk(node) if isinstance(sub, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Constant):
        return object()
    return returns[0].value.value


def main() -> None:
    ui_cls = class_node(parse(UI_PATH), "MainWindow")
    mixin_cls = class_node(parse(MIXIN_PATH), "ThreeModelUiStateMixin")
    ui_methods = direct_methods(ui_cls)
    mixin_methods = direct_methods(mixin_cls)

    assert not (EXPECTED_METHODS - mixin_methods.keys())
    assert not (EXPECTED_METHODS & ui_methods.keys())
    assert "on_three_model_controls_changed" in ui_methods, "preview/cache/render action must remain on MainWindow"
    assert "_effective_pointcloud_stride" in ui_methods, "unrelated pointcloud query must remain on MainWindow"
    assert "depth_fusion_ui" not in MIXIN_PATH.read_text(encoding="utf-8")

    bases = [ast.unparse(base) for base in ui_cls.bases]
    assert bases.index("ConditionalUiStateMixin") < bases.index("ThreeModelUiStateMixin") < bases.index("StructureCacheStateMixin")

    assert constant_return(mixin_methods["_effective_normal_strength"]) == 0
    assert constant_return(mixin_methods["_effective_normal_refine"]) == 0

    status_calls = self_calls(mixin_methods["_update_three_model_status"])
    assert {"_update_matting_status_label", "_update_external_media_status_label", "refresh_3d_model_status"} <= status_calls

    action_calls = self_calls(ui_methods["on_three_model_controls_changed"])
    assert {"_update_three_model_status", "_effective_normal_strength", "_effective_normal_refine", "render_preview_from_cache"} <= action_calls

    source = ast.get_source_segment(MIXIN_PATH.read_text(encoding="utf-8"), mixin_methods["_update_three_model_status"]) or ""
    assert "4DHumans" in source
    assert "Root稳定/时序去抖" in source
    assert "setEnabled(False)" in source

    print("three-model UI state mixin contract: PASS")


if __name__ == "__main__":
    main()
