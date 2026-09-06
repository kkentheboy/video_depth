from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_PATH = ROOT / "app" / "depth_fusion_ui.py"
MIXIN_PATH = ROOT / "app" / "components" / "structure_cache_state_mixin.py"
MODEL_MIXIN_PATH = ROOT / "app" / "components" / "model_configuration_mixin.py"
PROJECT_MIXIN_PATH = ROOT / "app" / "components" / "project_state_mixin.py"

EXPECTED_METHODS = {
    "_structure_cache_root_for_model",
    "_layer_cache_state_for_model",
    "_current_layer_cache_state",
    "_restore_best_available_structure_scheme",
    "_has_structure_cache",
    "_structure_scheme_text",
    "_config_for_structure_scheme",
    "_has_structure_cache_for_model",
    "_update_structure_scheme_status_labels",
}
ACTION_METHODS = {
    "select_structure_scheme",
    "start_structure_cache_generation",
    "start_4dhumans_structure_generation",
    "start_wham_structure_generation",
}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def direct_methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def self_calls(node: ast.AST) -> set[str]:
    calls: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        target = sub.func
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
            calls.add(target.attr)
    return calls


def main() -> None:
    ui_tree = parse(UI_PATH)
    mixin_tree = parse(MIXIN_PATH)
    model_tree = parse(MODEL_MIXIN_PATH)
    project_tree = parse(PROJECT_MIXIN_PATH)

    ui_cls = class_node(ui_tree, "MainWindow")
    mixin_cls = class_node(mixin_tree, "StructureCacheStateMixin")
    model_cls = class_node(model_tree, "ModelConfigurationMixin")
    project_cls = class_node(project_tree, "ProjectStateMixin")

    ui_methods = direct_methods(ui_cls)
    mixin_methods = direct_methods(mixin_cls)
    model_methods = direct_methods(model_cls)
    project_methods = direct_methods(project_cls)

    missing = EXPECTED_METHODS - mixin_methods.keys()
    assert not missing, f"StructureCacheStateMixin missing methods: {sorted(missing)}"
    leaked = EXPECTED_METHODS & ui_methods.keys()
    assert not leaked, f"structure cache state methods returned to MainWindow: {sorted(leaked)}"
    assert ACTION_METHODS <= ui_methods.keys(), "structure selection/generation actions must remain on MainWindow"
    assert not (ACTION_METHODS & mixin_methods.keys()), "action/thread methods must not move into StructureCacheStateMixin"

    source = MIXIN_PATH.read_text(encoding="utf-8")
    assert "depth_fusion_ui" not in source, "StructureCacheStateMixin must not reverse-import depth_fusion_ui"

    bases = [ast.unparse(base) for base in ui_cls.bases]
    assert "StructureCacheStateMixin" in bases
    assert bases.index("InputValidationMixin") < bases.index("StructureCacheStateMixin") < bases.index("ModelConfigurationMixin"), (
        "StructureCacheStateMixin must precede ModelConfigurationMixin because model scan delegates cache state to it"
    )

    layer_calls = self_calls(mixin_methods["_layer_cache_state_for_model"])
    assert {"_structure_cache_root_for_model", "_structure_model_key"} <= layer_calls
    current_calls = self_calls(mixin_methods["_current_layer_cache_state"])
    assert {"_layer_cache_state_for_model", "_structure_model_key"} <= current_calls
    restore_calls = self_calls(mixin_methods["_restore_best_available_structure_scheme"])
    assert {"_has_structure_cache_for_model", "_layer_cache_state_for_model", "_structure_scheme_text", "_structure_model_key"} <= restore_calls
    assert "make_config" in self_calls(mixin_methods["_has_structure_cache"])
    assert "make_config" in self_calls(mixin_methods["_config_for_structure_scheme"])
    assert "_config_for_structure_scheme" in self_calls(mixin_methods["_has_structure_cache_for_model"])
    assert "_has_structure_cache_for_model" in self_calls(mixin_methods["_update_structure_scheme_status_labels"])

    assert "_scan_3d_model_config" in model_methods
    assert "_has_structure_cache" in self_calls(model_methods["_scan_3d_model_config"]), (
        "ModelConfigurationMixin must continue delegating structure-cache truth to StructureCacheStateMixin"
    )

    assert "_apply_preset_payload" in project_methods
    assert "_structure_scheme_text" in self_calls(project_methods["_apply_preset_payload"]), (
        "project preset restore must use the shared structure-scheme naming authority"
    )
    assert "load_project_state" in project_methods
    load_calls = self_calls(project_methods["load_project_state"])
    assert {"_restore_best_available_structure_scheme", "_update_structure_scheme_status_labels"} <= load_calls, (
        "project load must restore and refresh structure-cache state through StructureCacheStateMixin"
    )

    select_calls = self_calls(ui_methods["select_structure_scheme"])
    assert {"_structure_scheme_text", "_update_structure_scheme_status_labels", "_has_structure_cache_for_model"} <= select_calls, (
        "MainWindow structure-scheme action must delegate state/query behavior to StructureCacheStateMixin"
    )
    assert "_on_structure_cache_finished" in ui_methods
    assert "_update_structure_scheme_status_labels" in self_calls(ui_methods["_on_structure_cache_finished"]), (
        "structure generation completion must refresh the shared cache-state authority"
    )

    print("structure cache state mixin contract: PASS")


if __name__ == "__main__":
    main()
