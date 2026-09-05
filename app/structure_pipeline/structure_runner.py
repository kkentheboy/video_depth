# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .smpl_config import StructureConfig
from .structure_cache import StructureFrame
from .external_output_importer import import_structure_outputs

LogFn = Callable[[str], None]


def _video_pix_fmt_may_have_alpha(path: Path) -> bool:
    """Cheap metadata check for video alpha, used only in worker stages."""
    suffix = Path(path).suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}:
        return False
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return False
    try:
        proc = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=pix_fmt", "-of", "default=nw=1:nk=1", str(path),
            ],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=3, check=False,
        )
        pix_fmt = (proc.stdout or "").strip().lower()
    except Exception:
        return False
    return any(token in pix_fmt for token in ("rgba", "bgra", "argb", "abgr", "yuva"))


class _SequentialInputAlphaReader:
    """Read RGBA alpha from an input video with one ffmpeg process.

    Per-frame ffmpeg launches made structure-cache frame export much slower after
    Alpha was connected into the 4D/WHAM input path. This reader keeps decoding
    sequentially for the normal export loop.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._proc: subprocess.Popen | None = None
        self._last_index = -1
        self._last_alpha: np.ndarray | None = None
        self.width = 0
        self.height = 0
        self.stride = 0
        self.available = False
        suffix = self.path.suffix.lower()
        self._image_alpha: np.ndarray | None = None
        if suffix in {".png", ".tif", ".tiff", ".webp"}:
            img = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] >= 4:
                self._image_alpha = img[..., 3].astype(np.float32) / 255.0
                self.available = True
            return
        try:
            if not _video_pix_fmt_may_have_alpha(self.path) or not shutil.which("ffmpeg"):
                return
            from depth_fusion_core import probe_video
            info = probe_video(str(self.path))
            self.width = int(info.width)
            self.height = int(info.height)
            self.stride = self.width * self.height * 4
            self.available = self.width > 0 and self.height > 0 and self.stride > 0
        except Exception:
            self.available = False

    def _open(self) -> bool:
        if not self.available or self._image_alpha is not None:
            return False
        if self._proc is not None and self._proc.stdout is not None:
            return True
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return False
        self._proc = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(self.path), "-an", "-sn", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._last_index = -1
        self._last_alpha = None
        return self._proc.stdout is not None

    def _restart(self) -> bool:
        self.close()
        return self._open()

    def _read_next_alpha(self) -> np.ndarray | None:
        if self._proc is None or self._proc.stdout is None or self.stride <= 0:
            return None
        buf = self._proc.stdout.read(self.stride)
        if len(buf) < self.stride:
            return None
        self._last_index += 1
        rgba = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width, 4))
        self._last_alpha = rgba[..., 3].astype(np.float32) / 255.0
        return self._last_alpha

    def read(self, frame_index: int, shape_hw: tuple[int, int]) -> np.ndarray | None:
        if not self.available:
            return None
        if self._image_alpha is not None:
            alpha = self._image_alpha
        else:
            target = max(0, int(frame_index))
            if self._last_alpha is not None and target == self._last_index:
                alpha = self._last_alpha
            else:
                if target < self._last_index:
                    if not self._restart():
                        return None
                elif not self._open():
                    return None
                alpha = None
                while self._last_index < target:
                    alpha = self._read_next_alpha()
                    if alpha is None:
                        return None
                alpha = self._last_alpha
        if alpha is None:
            return None
        coverage = float(np.mean(alpha > 0.01)) if alpha.size else 0.0
        if not (0.0005 < coverage < 0.9995):
            return None
        th, tw = shape_hw
        out = np.asarray(alpha, dtype=np.float32)
        if out.shape[:2] != (th, tw):
            out = cv2.resize(out, (tw, th), interpolation=cv2.INTER_LINEAR)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdout is not None:
                    self._proc.stdout.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=1)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        self._last_index = -1
        self._last_alpha = None


class StructureRunnerBase:
    """External structure model adapter interface.

    infer_frame() is kept for compatibility. Real 4DHumans / WHAM usage is video
    or frame-sequence based, so concrete adapters implement run_video_to_cache().
    """

    model_name = "structure"

    def __init__(self, config: StructureConfig | None = None) -> None:
        self.config = config or StructureConfig()

    def infer_frame(self, frame_bgr: np.ndarray, frame_index: int) -> StructureFrame:  # pragma: no cover - interface
        raise NotImplementedError

    def run_video_to_cache(
        self,
        input_path: str | Path,
        cache_root: str | Path,
        project_root: str | Path,
        *,
        start_frame: int = 0,
        end_frame: int = -1,
        max_side: int = 720,
        log: LogFn | None = None,
    ) -> dict:
        raise NotImplementedError

    def _log(self, log: LogFn | None, text: str) -> None:
        if log is not None:
            log(text)


class NoopStructureRunner(StructureRunnerBase):
    model_name = "none"

    def infer_frame(self, frame_bgr: np.ndarray, frame_index: int) -> StructureFrame:  # noqa: ARG002
        return StructureFrame(frame_index=int(frame_index), confidence=0.0, model_name="none")

    def run_video_to_cache(self, *args, **kwargs) -> dict:  # noqa: ANN002, ANN003
        return {"ok": False, "reason": "noop_runner"}


class ExternalCommandStructureRunner(StructureRunnerBase):
    """Base class for repo-backed command runners.

    The runner does not pretend the third-party repo is a Python library. It
    launches the official scripts as subprocesses, then imports OBJ/NPZ/PKL
    outputs into the stable structure cache used by this app.
    """

    repo_dir_name = ""
    env_command_var = ""
    output_subdir = "structure_out"
    confidence = 0.85

    def _project_python(self, project_root: Path) -> str:
        venv_py = project_root / ".venv" / "Scripts" / "python.exe"
        if venv_py.exists():
            return str(venv_py)
        return sys.executable

    def _repo_dir(self, project_root: Path) -> Path:
        return project_root / "data" / "models" / "external_repos" / self.repo_dir_name

    def _run_dir(self, project_root: Path, input_path: Path, cache_root: Path) -> Path:
        key = hashlib.sha1(f"{self.model_name}|{input_path}|{cache_root}".encode("utf-8", errors="ignore")).hexdigest()[:12]
        path = project_root / "data" / "cache" / "structure_runs" / f"{self.model_name}_{key}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _prepare_run_dirs(self, frames_dir: Path, output_dir: Path, log: LogFn | None = None) -> None:
        """Start every external-model run from clean frame/output folders."""
        for folder in (frames_dir, output_dir):
            try:
                if folder.exists():
                    shutil.rmtree(folder, ignore_errors=True)
                folder.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Cannot prepare structure run folder {folder}: {exc}") from exc
        self._log(log, f"Cleaned structure run folders: frames={frames_dir}, output={output_dir}")

    def _export_frames(self, input_path: Path, frames_dir: Path, *, start_frame: int, end_frame: int, max_side: int, log: LogFn | None = None) -> int:
        frames_dir.mkdir(parents=True, exist_ok=True)
        if input_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            img_raw = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
            if img_raw is None:
                raise RuntimeError(f"Cannot read image: {input_path}")
            img = self._rgba_or_bgr_to_black_bgr(img_raw)
            img = self._resize_keep_aspect(img, max_side)
            cv2.imwrite(str(frames_dir / "frame_000001.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            return 1
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_path}")
        try:
            from depth_fusion_core import probe_video
            total = probe_video(str(input_path)).frame_count
        except Exception:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start = max(0, int(start_frame))
        end = int(end_frame)
        if end < 0 and total > 0:
            end = total - 1
        alpha_reader = _SequentialInputAlphaReader(input_path)
        if alpha_reader.available:
            self._log(log, "Input Alpha: using one-pass ffmpeg stream for black-background structure frames.")
        expected = 0
        if total > 0 and end >= start:
            expected = max(1, int(end - start + 1))
        self._log(log, f"[Progress] 抽帧准备 {0}/{expected if expected > 0 else 0}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        count = 0
        idx = start
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if end >= 0 and idx > end:
                    break
                frame = self._composite_input_alpha_on_black(input_path, frame, idx, alpha_reader)
                frame = self._resize_keep_aspect(frame, max_side)
                cv2.imwrite(str(frames_dir / f"frame_{idx + 1:06d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                count += 1
                if expected > 0 and (count == 1 or count == expected or count % 10 == 0):
                    self._log(log, f"[Progress] 抽帧黑底合成 {count}/{expected}")
                idx += 1
        finally:
            alpha_reader.close()
            cap.release()
        self._log(log, f"[Progress] 抽帧黑底合成 {count}/{expected if expected > 0 else max(1, count)}")
        self._log(log, f"Exported {count} frames for {self.model_name}: {frames_dir}")
        return count

    @staticmethod
    def _rgba_or_bgr_to_black_bgr(frame: np.ndarray) -> np.ndarray:
        """Return BGR frame composited over black when a real alpha channel exists."""
        img = np.asarray(frame)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] >= 4:
            bgr = img[..., :3].astype(np.float32)
            alpha = np.clip(img[..., 3].astype(np.float32) / 255.0, 0.0, 1.0)
            if 0.0005 < float(np.mean(alpha > 0.01)) < 0.9995:
                out = bgr * alpha[..., None]
                return np.clip(out + 0.5, 0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] >= 3:
            return img[..., :3].copy()
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def _composite_input_alpha_on_black(self, input_path: Path, frame_bgr: np.ndarray, frame_index: int, alpha_reader: _SequentialInputAlphaReader | None = None) -> np.ndarray:
        """Use source RGBA alpha for detector input frames when available.

        4D/WHAM should see the same clean black-background frame as parsing/preview.
        This makes alpha part of the structure-solving input
        without turning alpha into the final point-cloud geometry.
        """
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return frame_bgr
        alpha = None
        if alpha_reader is not None:
            try:
                alpha = alpha_reader.read(int(frame_index), frame.shape[:2])
            except Exception:
                alpha = None
        if alpha is None:
            try:
                from depth_fusion_core import read_video_frame_alpha01
                alpha = read_video_frame_alpha01(str(input_path), int(frame_index), frame.shape[:2], allow_slow_video_alpha=False)
            except Exception:
                alpha = None
        if alpha is None:
            return frame_bgr
        a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
        if a.ndim == 3:
            a = a[:, :, 0]
        if a.shape[:2] != frame.shape[:2]:
            a = cv2.resize(a, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        if not (0.0005 < float(np.mean(a > 0.01)) < 0.9995):
            return frame_bgr
        out = frame[..., :3].astype(np.float32) * a[..., None]
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    @staticmethod
    def _resize_keep_aspect(frame: np.ndarray, max_side: int) -> np.ndarray:
        if not max_side or max_side <= 0:
            return frame
        h, w = frame.shape[:2]
        scale = float(max_side) / float(max(h, w))
        if scale >= 1.0:
            return frame
        return cv2.resize(frame, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


    def _frames_to_temp_video(self, frames_dir: Path, output_path: Path, *, fps: float = 25.0, log: LogFn | None = None) -> Path:
        """Build a temporary black-background MP4 from exported frames.

        WHAM's default demo entry is video-oriented. Feeding this generated video
        keeps WHAM on the same preprocessed input as 4D-Humans/parsing instead
        of silently reading the raw source video.
        """
        frames = sorted(Path(frames_dir).glob("frame_*.jpg"))
        if not frames:
            raise RuntimeError(f"No exported frames found for video conversion: {frames_dir}")
        first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Cannot read exported frame: {frames[0]}")
        h, w = first.shape[:2]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps if fps and fps > 0 else 25.0), (int(w), int(h)))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create temp structure video: {output_path}")
        count = 0
        try:
            for fp in frames:
                img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                writer.write(img)
                count += 1
        finally:
            writer.release()
        if count <= 0:
            raise RuntimeError("No frames written to temp structure video.")
        self._log(log, f"Generated preprocessed temp video for video-based solver: {output_path} ({count} frames)")
        return output_path

    def _run_command(self, cmd: list[str], cwd: Path, log: LogFn | None = None, env: dict | None = None) -> int:
        self._log(log, "RUN: " + " ".join(str(x) for x in cmd))
        self._log(log, "[Progress] 外部结构模型运行 0/0")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        lines: list[str] = []
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                lines.append(line)
                self._log(log, f"[{self.model_name}] {line}")
        self._last_command_output = lines
        code = int(proc.wait())
        self._log(log, f"[Progress] 外部结构模型完成 {1 if code == 0 else 0}/1")
        return code

    def _missing_dependency_result(
        self,
        *,
        command_exit_code: int,
        run_dir: Path,
        output_dir: Path,
    ) -> dict | None:
        lines = list(getattr(self, "_last_command_output", []) or [])
        text = "\n".join(lines)
        missing = sorted(set(re.findall(r"No module named ['\"]([^'\"]+)['\"]", text)))
        if missing:
            return {
                "ok": False,
                "reason": "missing_python_modules",
                "missing_modules": missing,
                "command_exit_code": int(command_exit_code),
                "run_dir": str(run_dir),
                "output_dir": str(output_dir),
            }
        if "expected str, bytes or os.PathLike object, not NoneType" in text and "HOME" in text:
            return {
                "ok": False,
                "reason": "missing_home_env",
                "command_exit_code": int(command_exit_code),
                "run_dir": str(run_dir),
                "output_dir": str(output_dir),
            }
        return None

    def _env(self, project_root: Path, repo_dir: Path) -> dict:
        env = dict(os.environ)

        # Some Linux-first research repos, including 4D-Humans/HMR2, build
        # cache paths from os.environ["HOME"]. On Windows HOME is often absent,
        # which makes os.path.join(None, ".cache") crash before inference starts.
        # Normalize it here for every external subprocess.
        home = (
            env.get("HOME")
            or env.get("USERPROFILE")
            or str(Path.home())
        )
        env["HOME"] = home
        env.setdefault("USERPROFILE", home)

        old = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(repo_dir) + (os.pathsep + old if old else "")
        env.setdefault("HF_HOME", str(project_root / "models" / "huggingface"))
        env.setdefault("TORCH_HOME", str(project_root / "models" / "torch"))
        return env

    def _import_outputs(self, output_dir: Path, cache_root: Path, project_root: Path, log: LogFn | None = None) -> dict:
        self._log(log, "[Progress] 导入结构输出 0/0")
        result = import_structure_outputs(
            output_dir,
            cache_root,
            model_name=self.model_name,
            confidence=float(self.confidence),
            project_root=project_root,
        )
        self._log(log, f"[Progress] 导入结构输出 {1 if bool(result.get('ok')) else 0}/1")
        self._log(log, f"Import result: {result}")
        return result

    def _manual_command(self) -> str:
        return str(os.environ.get(self.env_command_var, "")).strip()

    def _split_manual_command(self, command: str, frames_dir: Path, output_dir: Path, input_path: Path, cache_root: Path) -> list[str]:
        text = command.format(
            frames_dir=str(frames_dir),
            output_dir=str(output_dir),
            input=str(input_path),
            cache_root=str(cache_root),
        )
        try:
            import shlex
            return shlex.split(text, posix=False)
        except Exception:
            return text.split()

    def _copy_smpl_neutral_to_repo_data(self, project_root: Path, repo_dir: Path, log: LogFn | None = None) -> None:
        smpl_root = project_root / "data" / "models" / "checkpoints" / "smpl"
        candidates = list(smpl_root.glob("*neutral*.pkl")) + list(smpl_root.glob("SMPL_NEUTRAL.pkl"))
        if not candidates:
            self._log(log, "Neutral SMPL not found; repo may fail if it requires SMPL data.")
            return
        
        # 1. Standard format (for 4DHumans / HMR2 etc)
        dst_dir1 = repo_dir / "data"
        dst_dir1.mkdir(parents=True, exist_ok=True)
        dst1 = dst_dir1 / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
        
        # 2. WHAM format
        dst_dir2 = repo_dir / "dataset" / "body_models" / "smpl"
        dst_dir2.mkdir(parents=True, exist_ok=True)
        dst2 = dst_dir2 / "SMPL_NEUTRAL.pkl"
        
        try:
            shutil.copyfile(candidates[0], dst1)
            shutil.copyfile(candidates[0], dst2)
            self._log(log, f"Copied neutral SMPL to {dst1} and {dst2}")
        except Exception as exc:
            self._log(log, f"SMPL copy failed: {exc}")
