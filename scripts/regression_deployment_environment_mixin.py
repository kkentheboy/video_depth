from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_PATH = ROOT / "app" / "depth_fusion_ui.py"
MIXIN_PATH = ROOT / "app" / "components" / "deployment_environment_mixin.py"

EXPECTED_METHODS = {
    "_pip_package_for_module",
    "_deployment_missing_python_modules",
    "_deployment_model_resource_note",
    "_format_deployment_environment_lines",
    "refresh_deployment_environment_status",
}
SIDE_EFFECT_METHODS = {
    "install_deployment_python_dependencies",
    "_download_fashn_human_parser_model",
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
    ui_cls = class_node(ui_tree, "MainWindow")
    mixin_cls = class_node(mixin_tree, "DeploymentEnvironmentMixin")
    ui_methods = direct_methods(ui_cls)
    mixin_methods = direct_methods(mixin_cls)

    missing = EXPECTED_METHODS - mixin_methods.keys()
    assert not missing, f"DeploymentEnvironmentMixin missing methods: {sorted(missing)}"
    leaked = EXPECTED_METHODS & ui_methods.keys()
    assert not leaked, f"deployment environment methods returned to MainWindow: {sorted(leaked)}"

    missing_actions = SIDE_EFFECT_METHODS - ui_methods.keys()
    assert not missing_actions, f"side-effect deployment actions must remain in MainWindow: {sorted(missing_actions)}"
    assert not (SIDE_EFFECT_METHODS & mixin_methods.keys()), "side-effect deployment actions must not move into status mixin"

    source = MIXIN_PATH.read_text(encoding="utf-8")
    assert "depth_fusion_ui" not in source, "DeploymentEnvironmentMixin must not reverse-import depth_fusion_ui"

    bases = [ast.unparse(base) for base in ui_cls.bases]
    assert "DeploymentEnvironmentMixin" in bases
    assert bases.index("ModelConfigurationMixin") < bases.index("DeploymentEnvironmentMixin") < bases.index("ProcessingRangeMixin"), (
        "DeploymentEnvironmentMixin must stay between model configuration and processing-range authorities"
    )

    fmt_calls = self_calls(mixin_methods["_format_deployment_environment_lines"])
    assert {
        "_scan_3d_model_config",
        "_deployment_model_resource_note",
        "_deployment_missing_python_modules",
        "_pip_package_for_module",
    } <= fmt_calls, "deployment report must keep its model/dependency delegation chain"

    note_calls = self_calls(mixin_methods["_deployment_model_resource_note"])
    assert {"_scan_3d_model_config", "make_config"} <= note_calls, (
        "deployment model note must continue using model-configuration and JobConfig authorities"
    )

    refresh_calls = self_calls(mixin_methods["refresh_deployment_environment_status"])
    assert {"_format_deployment_environment_lines", "refresh_workflow_action_gates"} <= refresh_calls, (
        "deployment status refresh must keep report generation and workflow-gate refresh"
    )

    install_calls = self_calls(ui_methods["install_deployment_python_dependencies"])
    assert {
        "_deployment_missing_python_modules",
        "_download_fashn_human_parser_model",
        "refresh_deployment_environment_status",
    } <= install_calls, "deployment install action must delegate status/policy back to DeploymentEnvironmentMixin"
    download_calls = self_calls(ui_methods["_download_fashn_human_parser_model"])
    assert "refresh_deployment_environment_status" in download_calls, (
        "FASHN download action must refresh the shared deployment status authority"
    )

    direct_stdlib = set()
    core_imported = set()
    for node in mixin_tree.body:
        if isinstance(node, ast.Import):
            direct_stdlib.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module == "depth_fusion_core":
            core_imported.update(alias.asname or alias.name for alias in node.names)
    assert {"importlib", "shutil", "sys"} <= direct_stdlib, "stdlib dependencies must be imported directly"
    assert not ({"importlib", "shutil", "sys"} & core_imported), "stdlib dependencies must not be re-exported through depth_fusion_core"

    print("deployment environment mixin contract: PASS")


if __name__ == "__main__":
    main()
