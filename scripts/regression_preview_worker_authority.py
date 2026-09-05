#!/usr/bin/env python3
"""Source-only ownership contract for preview/original-frame workers."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
WORKERS = APP / "depth_fusion_workers.py"
PREVIEW = APP / "preview_workers.py"
UI = APP / "depth_fusion_ui.py"

MOVED = {"PreviewWorker", "OriginalFrameWorker", "_BaseRebuildWorker"}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def class_names(tree: ast.Module) -> set[str]:
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def imports_from(tree: ast.Module, module: str) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


def base_names(node: ast.ClassDef) -> set[str]:
    out = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            out.add(base.id)
        elif isinstance(base, ast.Attribute):
            out.add(base.attr)
    return out


def main() -> None:
    if not PREVIEW.is_file():
        raise SystemExit("app/preview_workers.py is missing")

    worker_tree = parse(WORKERS)
    preview_tree = parse(PREVIEW)
    ui_tree = parse(UI)

    leaked = MOVED & class_names(worker_tree)
    if leaked:
        raise SystemExit(f"preview worker classes returned to monolith: {sorted(leaked)}")

    missing = MOVED - class_names(preview_tree)
    if missing:
        raise SystemExit(f"preview worker classes missing from authority: {sorted(missing)}")

    if imports_from(preview_tree, "depth_fusion_workers"):
        raise SystemExit("preview_workers must not import depth_fusion_workers")

    ui_imports = imports_from(ui_tree, "preview_workers")
    if not MOVED.issubset(ui_imports):
        raise SystemExit(f"UI missing preview worker imports: {sorted(MOVED - ui_imports)}")

    for path in sorted(APP.rglob("*.py")):
        tree = parse(path)
        bad = MOVED & imports_from(tree, "depth_fusion_workers")
        if bad:
            raise SystemExit(
                f"{path.relative_to(ROOT)} imports moved preview workers from monolith: {sorted(bad)}"
            )

    classes = {
        node.name: node
        for node in preview_tree.body
        if isinstance(node, ast.ClassDef) and node.name in MOVED
    }
    for name in MOVED:
        if "QObject" not in base_names(classes[name]):
            raise SystemExit(f"{name} must remain QObject-derived")

    required_core = {
        "JobConfig",
        "event_exception",
        "event_log",
        "make_base_gray_for_levels",
        "read_video_frame_bgr",
    }
    core_imports = imports_from(preview_tree, "depth_fusion_core")
    if not required_core.issubset(core_imports):
        raise SystemExit(
            f"preview worker core imports changed/missing: {sorted(required_core - core_imports)}"
        )

    print("video_depth preview worker authority: PASS")


if __name__ == "__main__":
    main()
