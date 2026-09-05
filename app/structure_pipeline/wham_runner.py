# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path

from .structure_runner import ExternalCommandStructureRunner, LogFn


class WhamRunner(ExternalCommandStructureRunner):
    model_name = "wham"
    repo_dir_name = "WHAM"
    env_command_var = "DEPTH_FUSION_WHAM_CMD"
    output_subdir = "wham_out"
    confidence = 0.88

    def _env(self, project_root: Path, repo_dir: Path) -> dict:
        """Extend base env to include ViTPose third-party paths.

        WHAM's DetectionModel uses ViTPose (a Vision Transformer pose estimator)
        whose custom ``ViT`` backbone class must be registered with the mmpose
        ``BACKBONES`` registry.  The ViT class lives in
        ``third-party/ViTPose/mmpose/models/backbones/vit.py`` and is only
        importable when the ViTPose root is on ``PYTHONPATH``.  Without it,
        mmpose raises ``KeyError: 'ViT is not in the models registry'``.

        We also add ``mmcv_custom`` (custom optimizer constructors used by
        ViTPose configs) to avoid potential import failures.
        """
        env = super()._env(project_root, repo_dir)
        vitpose_dir = repo_dir / "third-party" / "ViTPose"
        extra_paths: list[str] = []
        if vitpose_dir.is_dir():
            extra_paths.append(str(vitpose_dir))
        mmcv_custom_dir = vitpose_dir / "mmcv_custom"
        if mmcv_custom_dir.is_dir():
            extra_paths.append(str(mmcv_custom_dir))
        if extra_paths:
            old = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(extra_paths) + (os.pathsep + old if old else "")
        return env

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
        project_root = Path(project_root)
        input_path = Path(input_path)
        cache_root = Path(cache_root)
        repo_dir = self._repo_dir(project_root)
        if not repo_dir.exists():
            return {"ok": False, "reason": "repo_missing", "repo": str(repo_dir)}
        run_dir = self._run_dir(project_root, input_path, cache_root)
        frames_dir = run_dir / "frames"
        output_dir = run_dir / self.output_subdir
        self._prepare_run_dirs(frames_dir, output_dir, log)
        self._copy_smpl_neutral_to_repo_data(project_root, repo_dir, log)
        # WHAM is video-oriented, but frames are also exported for fallback / manual use.
        self._export_frames(input_path, frames_dir, start_frame=start_frame, end_frame=end_frame, max_side=max_side, log=log)
        try:
            import cv2
            cap = cv2.VideoCapture(str(input_path))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0) if cap.isOpened() else 25.0
            cap.release()
        except Exception:
            fps = 25.0
        wham_input_video = self._frames_to_temp_video(frames_dir, run_dir / "wham_preprocessed_black.mp4", fps=fps, log=log)

        py = self._project_python(project_root)
        env = self._env(project_root, repo_dir)
        manual = self._manual_command()
        commands: list[list[str]] = []
        if manual:
            commands.append(self._split_manual_command(manual, frames_dir, output_dir, wham_input_video, cache_root))
        commands.extend([
            [py, "demo.py", "--video", str(wham_input_video), "--output_pth", str(output_dir), "--save_pkl"],
        ])

        last_code = 1
        for cmd in commands:
            last_code = self._run_command(cmd, repo_dir, log, env)
            if last_code != 0:
                dep_result = self._missing_dependency_result(command_exit_code=last_code, run_dir=run_dir, output_dir=output_dir)
                if dep_result is not None:
                    return dep_result
            result = self._import_outputs(output_dir, cache_root, project_root, log)
            if result.get("ok"):
                result.update({"command_exit_code": last_code, "run_dir": str(run_dir), "world_trajectory_expected": True})
                return result

        # Some WHAM workflows write checkpoints/results inside the repo tree.
        result = self._import_outputs(repo_dir, cache_root, project_root, log)
        if result.get("ok"):
            result.update({"command_exit_code": last_code, "run_dir": str(run_dir), "world_trajectory_expected": True})
            return result
        return {"ok": False, "reason": "no_importable_outputs", "command_exit_code": last_code, "run_dir": str(run_dir), "output_dir": str(output_dir)}
