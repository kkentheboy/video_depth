# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path


def _normalize_windows_home(project_root: Path) -> None:
    """Keep Linux-first model repos from crashing on Windows when HOME is unset."""
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE") or str(Path.home())
    os.environ["HOME"] = home
    project_root = Path(__file__).resolve().parent.parent.parent
    os.environ.setdefault("HF_HOME", str(project_root / "data" / "models" / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(project_root / "data" / "models" / "torch"))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _add_app_to_path(project_root: Path) -> None:
    app_dir = project_root / "app"
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))


def _runner(model: str):
    key = str(model or "4dhumans").strip().lower()
    if key == "wham":
        from structure_pipeline.wham_runner import WhamRunner
        return WhamRunner()
    if key in {"4dhumans", "4d-humans", "hmr2"}:
        from structure_pipeline.fourdhumans_runner import FourDHumansRunner
        return FourDHumansRunner()
    raise RuntimeError(f"Unsupported structure model: {model}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate unified structure cache for XYZ-only human point cloud export.")
    parser.add_argument("--input", required=True, help="Input RGB video/image path")
    parser.add_argument("--cache-root", required=True, help="Project frame cache root")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]), help="Project root")
    parser.add_argument("--model", default="4dhumans", choices=["4dhumans", "wham"], help="Body structure solver")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--max-side", type=int, default=1024, help="Inference frame max side; clamped to <=1024 for 12GB VRAM")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    _normalize_windows_home(project_root)
    _add_app_to_path(project_root)
    max_side = max(256, min(1024, int(args.max_side or 1024)))
    cache_root = Path(args.cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "event": "structure_cache_start",
        "model": args.model,
        "input": str(Path(args.input).resolve()),
        "cache_root": str(cache_root),
        "max_side": max_side,
        "vram_policy": "serial_model_batch1_maxside1024",
    }, ensure_ascii=False))

    runner = _runner(args.model)
    result = runner.run_video_to_cache(
        args.input,
        cache_root,
        project_root,
        start_frame=int(args.start_frame),
        end_frame=int(args.end_frame),
        max_side=max_side,
        log=lambda text: print(text, flush=True),
    )

    # Best-effort GPU cleanup. The external model usually runs in a subprocess,
    # but this keeps the wrapper safe if a repo imports torch in-process later.
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    print(json.dumps({"event": "structure_cache_done", "result": result}, ensure_ascii=False))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
