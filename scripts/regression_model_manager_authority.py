#!/usr/bin/env python3
"""Source-only contract for the LocalModelManager extraction."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
WORKERS = APP / "depth_fusion_workers.py"
MODEL_MANAGER = APP / "components" / "model_manager.py"
UI = APP / "depth_fusion_ui.py"

MOVED = {
    "safe_rmtree",
    "collect_unused_model_candidates",
    "delete_unused_models",
    "LocalModelManagerDialog",
}

WIDGET_BASES = {
    "QWidget",
    "QDialog",
    "QLabel",
    "QLineEdit",
    "QSlider",
    "QSpinBox",
    "QDoubleSpinBox",
    "QComboBox",
    "QMainWindow",
}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def top_level_defs(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def imports_from(tree: ast.Module, module: str) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


def base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def main() -> None:
    if not MODEL_MANAGER.is_file():
        raise SystemExit("components/model_manager.py is missing")

    worker_tree = parse(WORKERS)
    model_tree = parse(MODEL_MANAGER)
    ui_tree = parse(UI)

    worker_defs = top_level_defs(worker_tree)
    model_defs = top_level_defs(model_tree)

    leaked = MOVED & worker_defs
    if leaked:
        raise SystemExit(f"model-manager symbols returned to workers: {sorted(leaked)}")

    missing = MOVED - model_defs
    if missing:
        raise SystemExit(f"model-manager symbols missing from components.model_manager: {sorted(missing)}")

    if imports_from(model_tree, "depth_fusion_workers"):
        raise SystemExit("components.model_manager must not import depth_fusion_workers")

    for path in sorted(APP.rglob("*.py")):
        tree = parse(path)
        bad = MOVED & imports_from(tree, "depth_fusion_workers")
        if bad:
            raise SystemExit(
                f"{path.relative_to(ROOT)} imports model-manager symbols from workers: {sorted(bad)}"
            )

    ui_imports = imports_from(ui_tree, "components.model_manager")
    if "LocalModelManagerDialog" not in ui_imports:
        raise SystemExit(
            "depth_fusion_ui must import LocalModelManagerDialog from components.model_manager"
        )

    for node in worker_tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {base_name(base) for base in node.bases}
        bad_bases = bases & WIDGET_BASES
        if bad_bases:
            raise SystemExit(
                f"worker module owns QWidget-derived class {node.name}: {sorted(bad_bases)}"
            )

    dialog_nodes = [
        node
        for node in model_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LocalModelManagerDialog"
    ]
    if len(dialog_nodes) != 1:
        raise SystemExit("LocalModelManagerDialog definition count must be exactly one")
    if "QDialog" not in {base_name(base) for base in dialog_nodes[0].bases}:
        raise SystemExit("LocalModelManagerDialog must remain a QDialog")

    print("video_depth model manager authority: PASS")


if __name__ == "__main__":
    main()
