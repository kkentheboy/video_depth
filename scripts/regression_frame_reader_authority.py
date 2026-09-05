#!/usr/bin/env python3
"""Source-only ownership contract for sequential frame readers."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
WORKERS = APP / "depth_fusion_workers.py"
READERS = APP / "depth_pipeline" / "frame_readers.py"

MOVED = {
    "_resize_gray01_for_worker",
    "_SequentialBgrFrameReader",
    "_SequentialRgbaFrameReader",
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


def main() -> None:
    if not READERS.is_file():
        raise SystemExit("depth_pipeline/frame_readers.py is missing")

    worker_tree = parse(WORKERS)
    reader_tree = parse(READERS)

    leaked = MOVED & top_level_defs(worker_tree)
    if leaked:
        raise SystemExit(f"frame-reader definitions returned to workers: {sorted(leaked)}")

    missing = MOVED - top_level_defs(reader_tree)
    if missing:
        raise SystemExit(f"frame-reader definitions missing from authority: {sorted(missing)}")

    worker_imports = imports_from(worker_tree, "depth_pipeline.frame_readers")
    if not MOVED.issubset(worker_imports):
        raise SystemExit(f"workers missing frame-reader imports: {sorted(MOVED - worker_imports)}")

    if imports_from(reader_tree, "depth_fusion_workers"):
        raise SystemExit("depth_pipeline.frame_readers must not import depth_fusion_workers")

    for path in sorted(APP.rglob("*.py")):
        if path == WORKERS:
            continue
        tree = parse(path)
        bad = MOVED & imports_from(tree, "depth_fusion_workers")
        if bad:
            raise SystemExit(
                f"{path.relative_to(ROOT)} imports frame-reader helpers from workers: {sorted(bad)}"
            )

    print("video_depth frame reader authority: PASS")


if __name__ == "__main__":
    main()
