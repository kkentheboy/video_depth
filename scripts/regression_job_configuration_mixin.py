#!/usr/bin/env python3
"""Source-only ownership contract for JobConfig construction and resource-risk checks."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "app" / "depth_fusion_ui.py"
MIXIN = ROOT / "app" / "components" / "job_configuration_mixin.py"
TARGETS = {
    "_structure_model_key",
    "_pointcloud_mode",
    "_pointcloud_color_mode",
    "make_config",
    "_resource_risks",
    "_confirm_resource_risk",
    "_confirm_preview_resource_risk",
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


def calls_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == name
        for child in ast.walk(node)
    )


def main() -> None:
    if not MIXIN.is_file():
        raise SystemExit("components/job_configuration_mixin.py is missing")

    ui_tree = parse(UI)
    mixin_tree = parse(MIXIN)

    mains = [n for n in ui_tree.body if isinstance(n, ast.ClassDef) and n.name == "MainWindow"]
    if len(mains) != 1:
        raise SystemExit(f"expected one MainWindow, got {len(mains)}")
    main_window = mains[0]

    mixins = [n for n in mixin_tree.body if isinstance(n, ast.ClassDef) and n.name == "JobConfigurationMixin"]
    if len(mixins) != 1:
        raise SystemExit(f"expected one JobConfigurationMixin, got {len(mixins)}")
    mixin = mixins[0]

    base_names = [base.id for base in main_window.bases if isinstance(base, ast.Name)]
    required_order = [
        "ResourceManagementMixin",
        "FilePathActionsMixin",
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
        raise SystemExit(f"job-configuration methods returned to MainWindow: {sorted(leaked)}")

    mixin_methods = class_methods(mixin)
    missing = TARGETS - mixin_methods.keys()
    if missing:
        raise SystemExit(f"job-configuration mixin missing methods: {sorted(missing)}")

    for node in mixin_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "depth_fusion_ui":
            raise SystemExit("job-configuration mixin must not import depth_fusion_ui")
        if isinstance(node, ast.Import):
            if any(alias.name == "depth_fusion_ui" for alias in node.names):
                raise SystemExit("job-configuration mixin must not import depth_fusion_ui")

    ui_imports_mixin = False
    for node in ui_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "components.job_configuration_mixin":
            if "JobConfigurationMixin" in {alias.name for alias in node.names}:
                ui_imports_mixin = True
    if not ui_imports_mixin:
        raise SystemExit("depth_fusion_ui must import JobConfigurationMixin")

    make_attrs = attrs(mixin_methods["make_config"])
    required_make_delegations = {
        "_pointcloud_mode",
        "_current_encoder_mode",
        "_coerce_output_path_for_encoder",
        "_is_png_sequence_mode",
        "_structure_model_key",
        "_processing_range_values",
    }
    missing_make = required_make_delegations - make_attrs
    if missing_make:
        raise SystemExit(f"make_config lost delegation(s): {sorted(missing_make)}")
    if not calls_name(mixin_methods["make_config"], "JobConfig"):
        raise SystemExit("make_config must continue constructing JobConfig directly")

    for method_name in ("_confirm_resource_risk", "_confirm_preview_resource_risk"):
        if "_resource_risks" not in attrs(mixin_methods[method_name]):
            raise SystemExit(f"{method_name} must continue delegating through self._resource_risks")

    core_imports = set()
    for node in mixin_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "depth_fusion_core":
            core_imports.update(alias.name for alias in node.names)
    required_core = {
        "JobConfig",
        "MAX_SAFE_LONG_SIDE_HINT",
        "cuda_total_memory_gb",
        "estimate_vram_gb",
        "is_direct_depth_video_workflow",
        "scaled_size_from_long_side",
    }
    missing_core = required_core - core_imports
    if missing_core:
        raise SystemExit(f"job-configuration mixin missing core dependencies: {sorted(missing_core)}")

    print("video_depth job configuration mixin authority: PASS")


if __name__ == "__main__":
    main()
