#!/usr/bin/env python3
"""Source-only ownership contract for structure/background workers."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
WORKERS = APP / "depth_fusion_workers.py"
STRUCTURE = APP / "structure_workers.py"
UI = APP / "depth_fusion_ui.py"

MOVED = {"StructureCacheWorker", "ModelPreloadWorker", "SegmentationCacheWorker"}
REQUIRED_CORE_IMPORTS = {
    "JobConfig",
    "PROJECT_DIR",
    "QObject",
    "Signal",
    "event_exception",
    "event_log",
    "structure_cache_root",
}
REQUIRED_SEGMENTATION_IMPORTS = {"generate_segmentation_sequence_cache"}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def top_level_classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def imports_from(tree: ast.Module, module: str) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


def base_names(node: ast.ClassDef) -> set[str]:
    out: set[str] = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            out.add(base.id)
        elif isinstance(base, ast.Attribute):
            out.add(base.attr)
    return out


def main() -> None:
    if not STRUCTURE.is_file():
        raise SystemExit("app/structure_workers.py is missing")

    worker_tree = parse(WORKERS)
    structure_tree = parse(STRUCTURE)
    ui_tree = parse(UI)

    worker_classes = top_level_classes(worker_tree)
    structure_classes = top_level_classes(structure_tree)

    leaked = MOVED & worker_classes.keys()
    if leaked:
        raise SystemExit(f"structure workers returned to monolith: {sorted(leaked)}")

    missing = MOVED - structure_classes.keys()
    if missing:
        raise SystemExit(f"structure worker authority missing classes: {sorted(missing)}")

    for name in sorted(MOVED):
        if "QObject" not in base_names(structure_classes[name]):
            raise SystemExit(f"{name} must remain a QObject worker")

    if imports_from(structure_tree, "depth_fusion_workers"):
        raise SystemExit("structure_workers must not import depth_fusion_workers")

    core_imports = imports_from(structure_tree, "depth_fusion_core")
    missing_core = REQUIRED_CORE_IMPORTS - core_imports
    if missing_core:
        raise SystemExit(f"structure worker authority missing core imports: {sorted(missing_core)}")

    segmentation_imports = imports_from(
        structure_tree, "segmentation_pipeline.segmentation_cache"
    )
    missing_segmentation = REQUIRED_SEGMENTATION_IMPORTS - segmentation_imports
    if missing_segmentation:
        raise SystemExit(
            "structure worker authority missing segmentation imports: "
            f"{sorted(missing_segmentation)}"
        )

    ui_structure_imports = imports_from(ui_tree, "structure_workers")
    if not MOVED.issubset(ui_structure_imports):
        raise SystemExit(f"UI missing structure worker imports: {sorted(MOVED - ui_structure_imports)}")

    bad_ui = MOVED & imports_from(ui_tree, "depth_fusion_workers")
    if bad_ui:
        raise SystemExit(f"UI still imports structure workers from monolith: {sorted(bad_ui)}")

    for path in sorted(APP.rglob("*.py")):
        if path == WORKERS:
            continue
        tree = parse(path)
        bad = MOVED & imports_from(tree, "depth_fusion_workers")
        if bad:
            raise SystemExit(
                f"{path.relative_to(ROOT)} imports structure workers from monolith: {sorted(bad)}"
            )

    print("video_depth structure worker authority: PASS")


if __name__ == "__main__":
    main()
