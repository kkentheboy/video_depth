from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_PATH = ROOT / "app" / "depth_fusion_ui.py"
MIXIN_PATH = ROOT / "app" / "components" / "event_console_state_mixin.py"
EXPECTED_METHODS = {"_append_event_console_line", "_on_worker_log_signal", "log", "clear_event_console"}


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


def constants(node: ast.AST) -> set[object]:
    return {sub.value for sub in ast.walk(node) if isinstance(sub, ast.Constant)}


def main() -> None:
    ui_cls = class_node(parse(UI_PATH), "MainWindow")
    mixin_cls = class_node(parse(MIXIN_PATH), "EventConsoleStateMixin")
    ui_methods = direct_methods(ui_cls)
    mixin_methods = direct_methods(mixin_cls)

    assert not (EXPECTED_METHODS - mixin_methods.keys())
    assert not (EXPECTED_METHODS & ui_methods.keys())
    assert "on_stage_changed" in ui_methods, "task progress orchestration must remain on MainWindow"
    assert "eventFilter" in ui_methods, "Qt event dispatch must remain on MainWindow"
    assert "depth_fusion_ui" not in MIXIN_PATH.read_text(encoding="utf-8")

    bases = [ast.unparse(base) for base in ui_cls.bases]
    assert bases.index("UiPresentationStateMixin") < bases.index("EventConsoleStateMixin") < bases.index("StructureCacheStateMixin")

    append = mixin_methods["_append_event_console_line"]
    append_consts = constants(append)
    assert {12, 64} <= append_consts
    assert "appendPlainText" in {sub.attr for sub in ast.walk(append) if isinstance(sub, ast.Attribute)}

    worker = mixin_methods["_on_worker_log_signal"]
    assert "_append_event_console_line" in self_calls(worker)
    assert "_event_console_listener_active" in constants(worker)

    log_source = ast.get_source_segment(MIXIN_PATH.read_text(encoding="utf-8"), mixin_methods["log"]) or ""
    clear_source = ast.get_source_segment(MIXIN_PATH.read_text(encoding="utf-8"), mixin_methods["clear_event_console"]) or ""
    assert 'channel="UI"' in log_source
    assert "事件控制台已清空" in clear_source and 'channel="UI"' in clear_source

    init_source = ast.get_source_segment(UI_PATH.read_text(encoding="utf-8"), ui_methods["__init__"]) or ""
    assert "self._event_console_recent: list[str] = []" in init_source
    assert "self.event_console_line.connect(self._append_event_console_line)" in init_source
    assert "add_event_listener(self._event_console_listener)" in init_source

    print("event console state mixin contract: PASS")


if __name__ == "__main__":
    main()
