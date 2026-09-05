# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from .structure_runner import ExternalCommandStructureRunner, LogFn


class FourDHumansRunner(ExternalCommandStructureRunner):
    model_name = "4dhumans"
    repo_dir_name = "4D-Humans"
    env_command_var = "DEPTH_FUSION_4DHUMANS_CMD"
    output_subdir = "4dhumans_out"
    confidence = 0.82

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
        self._export_frames(input_path, frames_dir, start_frame=start_frame, end_frame=end_frame, max_side=max_side, log=log)

        py = self._project_python(project_root)
        env = self._env(project_root, repo_dir)

        # Trigger 4D-Humans official checkpoint download when the repo version supports it.
        code = self._run_command([
            py,
            "-c",
            "from hmr2.configs import CACHE_DIR_4DHUMANS; from hmr2.models import download_models; print(CACHE_DIR_4DHUMANS); download_models(CACHE_DIR_4DHUMANS)",
        ], repo_dir, log, env)
        if code != 0:
            dep_result = self._missing_dependency_result(command_exit_code=code, run_dir=run_dir, output_dir=output_dir)
            if dep_result is not None:
                return dep_result

        manual = self._manual_command()
        commands: list[list[str]] = []
        if manual:
            commands.append(self._split_manual_command(manual, frames_dir, output_dir, input_path, cache_root))
        commands.extend([
            [py, "demo.py", "--img_folder", str(frames_dir), "--out_folder", str(output_dir), "--batch_size", "1", "--save_mesh", "--full_frame", "--no_render"],
            [py, "demo.py", "--img_folder", str(frames_dir), "--out_folder", str(output_dir), "--save_mesh", "--no_render"],
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
                result.update({"command_exit_code": last_code, "run_dir": str(run_dir)})
                return result

        return {"ok": False, "reason": "no_importable_outputs", "command_exit_code": last_code, "run_dir": str(run_dir), "output_dir": str(output_dir)}
