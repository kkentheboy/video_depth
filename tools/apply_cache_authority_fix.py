#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKERS = ROOT / "app" / "depth_fusion_workers.py"

OLD_STUB_BLOCK = '''def cleanup_stale_output_frames(*a, **k): return None\ndef clean_output_dir_frames(*a, **k): return None\ndef existing_frame_is_complete(*a, **k): return False\ndef validate_pointcloud_outputs(*a, **k): return {}\ndef write_pipeline_state(root: Path, cfg: JobConfig | None = None, status: str = "", extra: dict | None = None) -> None:\n    try:\n        root = Path(root); root.mkdir(parents=True, exist_ok=True)\n        with (root / "pipeline_state.json").open("w", encoding="utf-8") as f:\n            json.dump({"status": status, "extra": extra or {}}, f, ensure_ascii=False, indent=2, default=str)\n    except Exception:\n        pass\ndef alpha_cache_signature(*a, **k): return ""\ndef cache_entry_matches(*a, **k): return False\ndef depth_cache_signature(*a, **k): return ""\ndef normal_cache_signature(*a, **k): return ""\ndef record_cache_error(*a, **k): return None\ndef record_cache_frame(*a, **k): return None\ndef summarize_cache_validation(*a, **k): return ""\ndef validate_geometry_cache(*a, **k): return {}\n'''

NEW_STUB_BLOCK = '''def cleanup_stale_output_frames(*a, **k): return None\ndef clean_output_dir_frames(*a, **k): return None\ndef existing_frame_is_complete(*a, **k): return False\ndef validate_pointcloud_outputs(*a, **k): return {}\n'''

CACHE_IMPORT = '''from depth_pipeline.cache_state import (\n    alpha_cache_signature,\n    cache_entry_matches,\n    depth_cache_signature,\n    normal_cache_signature,\n    record_cache_error,\n    record_cache_frame,\n    summarize_cache_validation,\n    validate_geometry_cache,\n    write_pipeline_state,\n)\n'''

IMPORT_ANCHOR = '''from common.cache import frame_stem\n'''


def main() -> None:
    text = WORKERS.read_text(encoding="utf-8-sig")
    if text.count(OLD_STUB_BLOCK) != 1:
        raise SystemExit(f"expected one cache stub block, found {text.count(OLD_STUB_BLOCK)}")
    if text.count(IMPORT_ANCHOR) != 1:
        raise SystemExit("common.cache import anchor changed")
    if "from depth_pipeline.cache_state import (" in text:
        raise SystemExit("cache_state authority import already exists")

    text = text.replace(IMPORT_ANCHOR, CACHE_IMPORT + IMPORT_ANCHOR, 1)
    text = text.replace(OLD_STUB_BLOCK, NEW_STUB_BLOCK, 1)
    WORKERS.write_text(text, encoding="utf-8")

    regression = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from depth_pipeline.cache_state import (
    CACHE_STATE_VERSION,
    alpha_cache_signature,
    cache_entry_matches,
    cache_manifest_path,
    depth_cache_signature,
    normal_cache_signature,
    record_cache_error,
    record_cache_frame,
    summarize_cache_validation,
    validate_cache_file,
    validate_geometry_cache,
    write_pipeline_state,
    pipeline_state_path,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def cfg(**updates):
    base = dict(
        input_path="input.mp4",
        model_id="model-a",
        device_mode="cpu",
        process_res=512,
        input_brightness=0,
        input_contrast=0,
        input_gamma=1.0,
        input_shadow=0,
        input_highlight=0,
        input_sharpen=0,
        input_denoise=0,
        matting_enabled=False,
        matting_model_path="",
        matting_mask_path="",
        matting_max_size=0,
        external_mask_enabled=False,
        external_mask_path="",
        external_mask_invert=False,
        external_mask_frame_offset=0,
        input_cutout_mask_enabled=False,
        auto_mask_feather_px=0,
        auto_mask_expand_px=0,
    )
    base.update(updates)
    return SimpleNamespace(**base)


def main() -> None:
    c = cfg()
    depth_sig = depth_cache_signature(c)
    alpha_sig = alpha_cache_signature(c)
    normal_sig = normal_cache_signature(c)
    check(len(depth_sig) == 20 and len(alpha_sig) == 20 and len(normal_sig) == 20, "cache signatures must be non-empty SHA1 prefixes")
    check(depth_sig == depth_cache_signature(c), "depth signature must be deterministic")
    check(depth_sig != depth_cache_signature(cfg(process_res=768)), "depth signature must change with process resolution")
    check(alpha_sig != alpha_cache_signature(cfg(external_mask_enabled=True, external_mask_path="mask.png")), "alpha signature must change with mask configuration")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "cache"
        depth0 = root / "depth" / "00000000.npy"
        depth1 = root / "depth" / "00000001.npy"
        depth0.parent.mkdir(parents=True, exist_ok=True)
        arr0 = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(4, 6)
        arr1 = np.ones((4, 6), dtype=np.float32)
        np.save(depth0, arr0)
        np.save(depth1, arr1)

        record_cache_frame(root, "depth", 0, depth0, depth_sig, arr0, status="ok", extra={"source": "regression"})
        check(cache_entry_matches(root, "depth", 0, depth0, depth_sig, allow_legacy=False), "recorded cache frame must be reusable")
        check(not cache_entry_matches(root, "depth", 0, depth0, "wrong-signature", allow_legacy=False), "wrong signature must invalidate cache")

        manifest = json.loads(cache_manifest_path(root, "depth").read_text(encoding="utf-8"))
        check(manifest.get("version") == CACHE_STATE_VERSION, "cache manifest version mismatch")
        check(manifest["frames"]["00000000"]["status"] == "ok", "cache frame status not persisted")
        check(manifest["frames"]["00000000"]["source"] == "regression", "cache frame metadata not persisted")

        record_cache_error(root, "depth", 1, depth_sig, "synthetic-error")
        manifest = json.loads(cache_manifest_path(root, "depth").read_text(encoding="utf-8"))
        check(manifest["frames"]["00000001"]["status"] == "error", "cache error status not persisted")
        check(not cache_entry_matches(root, "depth", 1, depth1, depth_sig, allow_legacy=False), "error frame must never be reusable")

        ok, reason, info = validate_cache_file(depth0, "depth")
        check(ok and reason == "ok" and info.get("shape") == [4, 6], f"valid depth cache rejected: {reason} {info}")

        # Re-record frame 1 as valid so geometry validation can go green.
        record_cache_frame(root, "depth", 1, depth1, depth_sig, arr1, status="ok")
        validation = validate_geometry_cache(
            root,
            c,
            expected_frames=2,
            frame_depth_path_builder=lambda cache_root, idx: Path(cache_root) / "depth" / f"{idx:08d}.npy",
            frame_alpha_path_builder=lambda cache_root, idx: Path(cache_root) / "alpha" / f"{idx:08d}.npy",
            frame_normal_path_builder=lambda cache_root, idx: Path(cache_root) / "normal" / f"{idx:08d}.npy",
            include_alpha=False,
            include_normal=False,
        )
        check(validation.get("status") == "green", f"complete depth cache should validate green: {validation}")
        summary = summarize_cache_validation(validation)
        check("depth: ok=2/2" in summary, f"cache validation summary changed: {summary}")

        write_pipeline_state(root, c, total_frames=2, status="complete", extra={"regression": True})
        state_path = pipeline_state_path(root)
        check(state_path.is_file(), "pipeline state must be written under cache meta authority")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        check(state.get("status") == "complete" and state.get("total_frames") == 2, f"pipeline state payload invalid: {state}")
        check(state.get("regression") is True, "pipeline state extra metadata missing")

    print("video_depth cache-state regression: PASS")


if __name__ == "__main__":
    main()
'''
    (ROOT / "scripts" / "regression_cache_state.py").write_text(regression, encoding="utf-8")
    print("video_depth cache authority repair staged")


if __name__ == "__main__":
    main()
