from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_PATH = ROOT / "app" / "depth_fusion_ui.py"
MIXIN_PATH = ROOT / "app" / "components" / "ui_presentation_state_mixin.py"
EXPECTED_METHODS = {
    "_apply_style",
    "_install_button_cursor_policy",
    "_sync_button_cursor",
    "_refresh_widget_style",
    "_set_depth_preview_busy",
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
    mixin_cls = class_node(parse(MIXIN_PATH), "UiPresentationStateMixin")
    ui_methods = direct_methods(ui_cls)
    mixin_methods = direct_methods(mixin_cls)

    assert not (EXPECTED_METHODS - mixin_methods.keys())
    assert not (EXPECTED_METHODS & ui_methods.keys())
    assert "eventFilter" in ui_methods, "Qt event-dispatch integration must remain on MainWindow"
    assert "depth_fusion_ui" not in MIXIN_PATH.read_text(encoding="utf-8")

    bases = [ast.unparse(base) for base in ui_cls.bases]
    assert bases.index("ThreeModelUiStateMixin") < bases.index("UiPresentationStateMixin") < bases.index("StructureCacheStateMixin")

    apply_calls = self_calls(mixin_methods["_apply_style"])
    assert {"setStyleSheet", "_install_button_cursor_policy"} <= apply_calls
    apply_literals = string_literals(mixin_methods["_apply_style"])
    assert {"secondaryAction", "pathEdit", "previewStatusLabel", "infoLabel"} <= apply_literals

    cursor_calls = self_calls(mixin_methods["_install_button_cursor_policy"])
    assert {"findChildren", "_sync_button_cursor"} <= cursor_calls
    sync_literals = string_literals(mixin_methods["_set_depth_preview_busy"])
    assert {"计算中...", "busy", "1", "0"} <= sync_literals
    assert "_refresh_widget_style" in self_calls(mixin_methods["_set_depth_preview_busy"])

    event_calls = self_calls(ui_methods["eventFilter"])
    assert "_sync_button_cursor" in event_calls

    print("UI presentation state mixin contract: PASS")


if __name__ == "__main__":
    main()
