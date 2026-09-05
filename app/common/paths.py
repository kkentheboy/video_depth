# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import hashlib
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = _PACKAGE_DIR.parent if _PACKAGE_DIR.name.lower() == "app" else _PACKAGE_DIR
PROJECT_DATA_DIR = PROJECT_DIR / "data"
EXPORTS_DIR = PROJECT_DATA_DIR / "exports"


def safe_stem(text: str, fallback: str = "video") -> str:
    stem = Path(str(text or fallback)).stem or fallback
    stem = re.sub(r"[\\/:*?\"<>|\s]+", "_", stem, flags=re.UNICODE).strip("._ ")
    return stem or fallback


def short_file_digest(path_text: str) -> str:
    p = Path(str(path_text or ""))
    try:
        stat = p.stat()
        raw = f"{p.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    except Exception:
        raw = str(path_text or "")
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:8]


def pointcloud_export_root(input_path: str, output_path: str | None = None) -> Path:
    """Return the final export folder for point cloud / Mesh output.

    Current Mesh/Shell workflow treats a suffix-less UI path as the exact final
    export directory. Legacy file paths like ``D:/out/name.mp4`` still map to
    ``D:/out/name_pointcloud`` for compatibility.
    """
    stem = safe_stem(input_path)
    digest = short_file_digest(input_path)
    out_text = str(output_path or "").strip()
    if out_text:
        out = Path(out_text)
        try:
            if out.suffix:
                return out.with_name(f"{safe_stem(out.stem, stem)}_pointcloud")
            return out
        except Exception:
            pass
    return EXPORTS_DIR / f"{stem}_{digest}_pointcloud"


def ensure_clean_dir(path: Path, keep_existing: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if not keep_existing:
        for child in path.iterdir():
            if child.is_file() or child.is_symlink():
                try:
                    child.unlink()
                except Exception:
                    pass
    return path
