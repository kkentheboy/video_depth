from __future__ import annotations

import importlib
import sys


MIN_PYTHON = (3, 10)
REQUIRED_MODULES = (
    "PySide6",
    "numpy",
    "cv2",
    "PIL",
    "scipy",
    "transformers",
    "safetensors",
    "huggingface_hub",
    "timm",
    "einops",
    "trimesh",
    "imageio",
    "requests",
)


def main() -> int:
    if sys.version_info < MIN_PYTHON:
        print(
            f"[ERROR] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required; "
            f"current={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        return 1

    failed: list[str] = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failed.append(f"{module_name}: {type(exc).__name__}: {exc}")

    if failed:
        print("[ERROR] Base dependency import smoke failed:")
        for item in failed:
            print(f"  - {item}")
        return 1

    print(
        "[OK] Base dependency smoke passed "
        f"on Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
