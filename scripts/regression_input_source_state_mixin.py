from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_PATH = ROOT / "app" / "depth_fusion_ui.py"
MIXIN_PATH = ROOT / "app" / "components" / "input_source_state_mixin.py"
EXPECTED_METHODS = {
    "on_matting_controls_changed",
    "_update_matting_status_label",
    "_source_mode_from_current_controls",
    "_current_source_mode",
    "_sync_source_mode_radios",
    "_apply_source_mode",
}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def direct_methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {node.name: node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def self_calls(node: ast.AST) -> set[str]:
    calls: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            target = sub.func
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                calls.add(target.attr)
    return calls


def constant_return(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    returns = [sub for sub in ast.walk(node) if isinstance(sub, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Constant):
        return None
    value = returns[0].value.value
    return value if isinstance(value, str) else None


def is_pass_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return len(node.body) == 1 and isinstance(node.body[0], ast.Pass)


def main() -> None:
    ui_cls = class_node(parse(UI_PATH), "MainWindow")
    mixin_cls = class_node(parse(MIXIN_PATH), "InputSourceStateMixin")
    ui_methods = direct_methods(ui_cls)
    mixin_methods = direct_methods(mixin_cls)
    assert not (EXPECTED_METHODS - mixin_methods.keys())
    assert not (EXPECTED_METHODS & ui_methods.keys())
    assert "_update_conditional_visibility" in ui_methods
    assert "_update_conditional_visibility" not in mixin_methods
    assert "depth_fusion_ui" not in MIXIN_PATH.read_text(encoding="utf-8")

    bases = [ast.unparse(base) for base in ui_cls.bases]
    assert bases.index("InputValidationMixin") < bases.index("InputSourceStateMixin") < bases.index("StructureCacheStateMixin")
    assert is_pass_stub(mixin_methods["on_matting_controls_changed"])
    assert is_pass_stub(mixin_methods["_update_matting_status_label"])
    assert constant_return(mixin_methods["_source_mode_from_current_controls"]) == "cutout_video"
    assert constant_return(mixin_methods["_current_source_mode"]) == "cutout_video"
    assert "_update_conditional_visibility" in self_calls(mixin_methods["_sync_source_mode_radios"])
    assert {
        "_sync_source_mode_radios",
        "_update_matting_status_label",
        "_update_conditional_visibility",
        "on_external_media_changed",
    } <= self_calls(mixin_methods["_apply_source_mode"])
    assert "_update_matting_status_label" in self_calls(ui_methods["_update_three_model_status"])
    visibility_calls = self_calls(ui_methods["_update_conditional_visibility"])
    assert "_current_source_mode" in visibility_calls or "_source_mode_from_current_controls" in visibility_calls
    print("input source state mixin contract: PASS")


if __name__ == "__main__":
    main()
