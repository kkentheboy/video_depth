from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_PATH = ROOT / "app" / "depth_fusion_ui.py"
MIXIN_PATH = ROOT / "app" / "components" / "model_configuration_mixin.py"
DEPLOYMENT_MIXIN_PATH = ROOT / "app" / "components" / "deployment_environment_mixin.py"

EXPECTED_METHODS = {
    "_looks_like_real_model_weight",
    "_candidate_weight_files",
    "_model_scan_roots",
    "_scan_3d_model_config",
    "_format_3d_scan_lines",
    "_is_3d_structure_model_configured",
    "_is_3d_hand_model_configured",
    "refresh_3d_model_status",
}
EXPECTED_CONSTANTS = {
    "_MODEL_WEIGHT_SUFFIXES",
    "_MODEL_SCAN_SKIP_PARTS",
    "_MODEL_SCAN_SKIP_FILENAMES",
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


def direct_assignments(cls: ast.ClassDef) -> dict[str, ast.AST | None]:
    values: dict[str, ast.AST | None] = {}
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            values[node.target.id] = node.value
    return values


def self_calls(fn: ast.AST) -> set[str]:
    calls: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
            calls.add(target.attr)
    return calls


def self_attrs(fn: ast.AST) -> set[str]:
    attrs: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            attrs.add(node.attr)
    return attrs


def main() -> None:
    ui_tree = parse(UI_PATH)
    mixin_tree = parse(MIXIN_PATH)
    deployment_tree = parse(DEPLOYMENT_MIXIN_PATH)
    ui_cls = class_node(ui_tree, "MainWindow")
    mixin_cls = class_node(mixin_tree, "ModelConfigurationMixin")
    deployment_cls = class_node(deployment_tree, "DeploymentEnvironmentMixin")
    ui_methods = direct_methods(ui_cls)
    mixin_methods = direct_methods(mixin_cls)
    deployment_methods = direct_methods(deployment_cls)
    ui_assignments = direct_assignments(ui_cls)
    mixin_assignments = direct_assignments(mixin_cls)

    missing_methods = EXPECTED_METHODS - mixin_methods.keys()
    assert not missing_methods, f"ModelConfigurationMixin missing methods: {sorted(missing_methods)}"
    leaked_methods = EXPECTED_METHODS & ui_methods.keys()
    assert not leaked_methods, f"model configuration methods returned to MainWindow: {sorted(leaked_methods)}"

    missing_constants = EXPECTED_CONSTANTS - mixin_assignments.keys()
    assert not missing_constants, f"ModelConfigurationMixin missing constants: {sorted(missing_constants)}"
    leaked_constants = EXPECTED_CONSTANTS & ui_assignments.keys()
    assert not leaked_constants, f"model configuration constants returned to MainWindow: {sorted(leaked_constants)}"

    mixin_source = MIXIN_PATH.read_text(encoding="utf-8")
    assert "depth_fusion_ui" not in mixin_source, "ModelConfigurationMixin must not reverse-import depth_fusion_ui"

    bases = [ast.unparse(base) for base in ui_cls.bases]
    assert "ModelConfigurationMixin" in bases, "MainWindow must inherit ModelConfigurationMixin"
    assert bases.index("InputValidationMixin") < bases.index("ModelConfigurationMixin") < bases.index("ProcessingRangeMixin"), (
        "ModelConfigurationMixin must stay between input validation and processing-range authorities"
    )

    assert "_looks_like_real_model_weight" in self_calls(mixin_methods["_candidate_weight_files"])
    scan_calls = self_calls(mixin_methods["_scan_3d_model_config"])
    assert {"_model_scan_roots", "_candidate_weight_files", "_has_structure_cache"} <= scan_calls, (
        "model scan must keep root discovery, weight filtering, and structure-cache delegation"
    )
    assert "_scan_3d_model_config" in self_calls(mixin_methods["_is_3d_structure_model_configured"])
    assert "_scan_3d_model_config" in self_calls(mixin_methods["_is_3d_hand_model_configured"])
    refresh_calls = self_calls(mixin_methods["refresh_3d_model_status"])
    assert {"_scan_3d_model_config", "_format_3d_scan_lines"} <= refresh_calls

    weight_attrs = self_attrs(mixin_methods["_looks_like_real_model_weight"])
    assert EXPECTED_CONSTANTS <= weight_attrs, "weight filtering must continue to read the single model-scan constant authority"

    for caller in ("_deployment_model_resource_note", "_format_deployment_environment_lines"):
        assert caller in deployment_methods, f"expected deployment caller missing from DeploymentEnvironmentMixin: {caller}"
        assert "_scan_3d_model_config" in self_calls(deployment_methods[caller]), (
            f"{caller} must delegate to ModelConfigurationMixin scan"
        )
    assert "_on_structure_cache_finished" in ui_methods
    assert "refresh_3d_model_status" in self_calls(ui_methods["_on_structure_cache_finished"]), (
        "structure-cache completion must continue to refresh the model configuration authority"
    )

    print("model configuration mixin contract: PASS")


if __name__ == "__main__":
    main()
