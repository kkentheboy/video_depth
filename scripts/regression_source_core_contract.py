#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
CORE_PATH = APP / "depth_fusion_core.py"
WORKERS_PATH = APP / "depth_fusion_workers.py"
CACHE_STATE_PATH = APP / "depth_pipeline" / "cache_state.py"

CACHE_API = {
    "alpha_cache_signature",
    "cache_entry_matches",
    "depth_cache_signature",
    "normal_cache_signature",
    "record_cache_error",
    "record_cache_frame",
    "summarize_cache_validation",
    "validate_geometry_cache",
    "write_pipeline_state",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def parse(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except SyntaxError as exc:
        fail(f"syntax error in {path.relative_to(ROOT)}: {exc}")


def bound_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()

    def bind_target(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                bind_target(item)

    def visit_stmt(stmt: ast.stmt) -> None:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
            return
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                names.add(alias.asname or alias.name.split(".")[0])
            return
        if isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
            return
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    bind_target(target)
            else:
                bind_target(stmt.target)
            return
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            bind_target(stmt.target)
            for child in [*stmt.body, *stmt.orelse]:
                visit_stmt(child)
            return
        if isinstance(stmt, (ast.If, ast.While)):
            for child in [*stmt.body, *stmt.orelse]:
                visit_stmt(child)
            return
        if isinstance(stmt, ast.Try):
            for child in [*stmt.body, *stmt.orelse, *stmt.finalbody]:
                visit_stmt(child)
            for handler in stmt.handlers:
                if handler.name:
                    names.add(handler.name)
                for child in handler.body:
                    visit_stmt(child)
            return
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                if item.optional_vars is not None:
                    bind_target(item.optional_vars)
            for child in stmt.body:
                visit_stmt(child)

    for statement in tree.body:
        visit_stmt(statement)
    return names


def imports_from(tree: ast.Module, module: str) -> set[str]:
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            out.update(alias.name for alias in node.names if alias.name != "*")
    return out


def local_function_names(tree: ast.Module) -> set[str]:
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def main() -> None:
    # Compile-level syntax coverage for every application Python file.
    py_files = sorted(APP.rglob("*.py"))
    if not py_files:
        fail("no app Python sources found")
    for path in py_files:
        parse(path)

    core = parse(CORE_PATH)
    workers = parse(WORKERS_PATH)
    cache_state = parse(CACHE_STATE_PATH)

    core_exports = bound_names(core)
    worker_core_imports = imports_from(workers, "depth_fusion_core")
    missing_core_exports = sorted(worker_core_imports - core_exports)
    if missing_core_exports:
        fail(
            "depth_fusion_workers imports names that depth_fusion_core does not export: "
            + ", ".join(missing_core_exports)
        )

    cache_exports = bound_names(cache_state)
    missing_cache_api = sorted(CACHE_API - cache_exports)
    if missing_cache_api:
        fail("cache_state is missing required cache API: " + ", ".join(missing_cache_api))

    worker_local_functions = local_function_names(workers)
    shadowed_cache_api = sorted(CACHE_API & worker_local_functions)
    if shadowed_cache_api:
        fail(
            "depth_fusion_workers shadows authoritative cache_state APIs with local functions/stubs: "
            + ", ".join(shadowed_cache_api)
        )

    print(
        f"video_depth source/core contract: PASS files={len(py_files)} "
        f"core_imports={len(worker_core_imports)} cache_api={len(CACHE_API)}"
    )


if __name__ == "__main__":
    main()
