# -*- coding: utf-8 -*-
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TARGET = ROOT / "data" / "models" / "segmentation" / "fashn_human_parser"


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "缺少 huggingface_hub。先运行：python -m pip install huggingface_hub transformers pillow"
        ) from exc
    TARGET.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="fashn-ai/fashn-human-parser",
        local_dir=str(TARGET),
        local_dir_use_symlinks=False,
    )
    print("FASHN Human Parser 已下载到:", TARGET)


if __name__ == "__main__":
    main()
