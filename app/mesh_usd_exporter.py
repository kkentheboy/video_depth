# -*- coding: utf-8 -*-
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np



def deterministic_limit_indices(count: int, max_count: int, seed: int = 0):
    count = int(count)
    max_count = int(max_count)
    if count <= max_count or max_count <= 0:
        return None
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    idx = rng.choice(count, size=max_count, replace=False)
    idx.sort()
    return idx.astype(np.int64)



def _fmt_float(v: float) -> str:
    if not np.isfinite(v):
        v = 0.0
    return f"{float(v):.6g}"


import io

def _fmt_point_list(points: np.ndarray) -> str:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
    if len(pts) == 0:
        return "[]"
    bio = io.BytesIO()
    np.savetxt(bio, pts, fmt="(%.6g, %.6g, %.6g)", newline=", ")
    s = bio.getvalue().decode('utf-8')
    if s.endswith(", "):
        s = s[:-2]
    return "[" + s + "]"


def _fmt_extent(points: np.ndarray) -> str:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return "[(0, 0, 0), (0, 0, 0)]"
    pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return (
        "["
        f"({_fmt_float(mn[0])}, {_fmt_float(mn[1])}, {_fmt_float(mn[2])}), "
        f"({_fmt_float(mx[0])}, {_fmt_float(mx[1])}, {_fmt_float(mx[2])})"
        "]"
    )


def _fmt_color_list(colors: np.ndarray) -> str:
    cols = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
    if cols.size and cols.max() > 1.0:
        cols = cols / 255.0
    cols = np.clip(cols, 0.0, 1.0)
    if len(cols) == 0:
        return "[]"
    bio = io.BytesIO()
    np.savetxt(bio, cols, fmt="(%.6g, %.6g, %.6g)", newline=", ")
    s = bio.getvalue().decode('utf-8')
    if s.endswith(", "):
        s = s[:-2]
    return "[" + s + "]"


def _fmt_float_array(values: np.ndarray) -> str:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    if len(vals) == 0:
        return "[]"
    bio = io.BytesIO()
    np.savetxt(bio, vals, fmt="%.6g", newline=", ")
    s = bio.getvalue().decode('utf-8')
    if s.endswith(", "):
        s = s[:-2]
    return "[" + s + "]"


def _fmt_int_array(values: np.ndarray) -> str:
    vals = np.asarray(values, dtype=np.int32).reshape(-1)
    if len(vals) == 0:
        return "[]"
    bio = io.BytesIO()
    np.savetxt(bio, vals, fmt="%d", newline=", ")
    s = bio.getvalue().decode('utf-8')
    if s.endswith(", "):
        s = s[:-2]
    return "[" + s + "]"


def _take_or_fill_array(values: np.ndarray | None, count: int, *, shape: tuple[int, ...], dtype: np.dtype, fill_value: float | int) -> np.ndarray:
    if values is None:
        return np.full((count, *shape), fill_value, dtype=dtype) if shape else np.full((count,), fill_value, dtype=dtype)
    arr = np.asarray(values, dtype=dtype)
    try:
        arr = arr.reshape(-1, *shape) if shape else arr.reshape(-1)
    except ValueError:
        return np.full((count, *shape), fill_value, dtype=dtype) if shape else np.full((count,), fill_value, dtype=dtype)
    arr = arr[:count]
    if len(arr) < count:
        pad_shape = (count - len(arr), *shape) if shape else (count - len(arr),)
        pad = np.full(pad_shape, fill_value, dtype=dtype)
        arr = np.concatenate([arr, pad], axis=0)
    return arr.astype(dtype, copy=False)


def _normalize_frame_arrays(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    source_id: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    *,
    point_width: float = 0.008,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    count = int(len(pts))
    cols = _take_or_fill_array(colors, count, shape=(3,), dtype=np.uint8, fill_value=220)
    src = _take_or_fill_array(source_id, count, shape=(), dtype=np.int32, fill_value=0)
    conf = _take_or_fill_array(confidence, count, shape=(), dtype=np.float32, fill_value=1.0)
    conf = np.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    widths = np.full((count,), max(0.0001, float(point_width)), dtype=np.float32)
    return pts, cols, src, conf, widths


class UsdPointSequenceWriter:
    """Streaming USDA writer for Blender-readable animated point clouds.

    The result is one `UsdGeomPoints` prim with time-sampled `points`, `extent`
    and `widths`. Optional debug primvars can be enabled by constructor flags,
    but the project default is XYZ-only: no RGB color, source_id or confidence.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float = 24.0,
        start_frame: int = 1,
        end_frame: int = 1,
        point_width: float = 0.008,
        max_points_per_frame: int = 60_000,
        label: str = "pointcloud",
        include_colors: bool = False,
        include_source_id: bool = False,
        include_confidence: bool = False,
        xform_translate: np.ndarray | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps) if fps and fps > 0 else 24.0
        self.start_frame = max(1, int(start_frame))
        self.end_frame = max(self.start_frame, int(end_frame))
        self.point_width = max(0.0001, float(point_width))
        self.max_points_per_frame = max(100, int(max_points_per_frame))
        self.label = _safe_usd_identifier(label or "pointcloud")
        self.include_colors = bool(include_colors)
        self.include_source_id = bool(include_source_id)
        self.include_confidence = bool(include_confidence)
        if xform_translate is None:
            self.xform_translate = np.zeros(3, dtype=np.float32)
        else:
            self.xform_translate = np.asarray(xform_translate, dtype=np.float32).reshape(3)
            self.xform_translate = np.nan_to_num(self.xform_translate, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self._closed = False
        self._count = 0
        self._first_sample = True
        self._tmp_dir = self.path.with_suffix(self.path.suffix + ".tmp")
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_files: dict[str, object] = {
            "points": (self._tmp_dir / "points.samples").open("w", encoding="utf-8", newline="\n"),
            "extent": (self._tmp_dir / "extent.samples").open("w", encoding="utf-8", newline="\n"),
            "widths": (self._tmp_dir / "widths.samples").open("w", encoding="utf-8", newline="\n"),
        }
        if self.include_colors:
            self._tmp_files["colors"] = (self._tmp_dir / "colors.samples").open("w", encoding="utf-8", newline="\n")
        if self.include_source_id:
            self._tmp_files["source_id"] = (self._tmp_dir / "source_id.samples").open("w", encoding="utf-8", newline="\n")
        if self.include_confidence:
            self._tmp_files["confidence"] = (self._tmp_dir / "confidence.samples").open("w", encoding="utf-8", newline="\n")

    def _append_sample(self, key: str, frame_no: int, value: str) -> None:
        f = self._tmp_files[key]
        if not self._first_sample:
            f.write(",\n")
        f.write(f"            {int(frame_no)}: {value}")

    def add_frame(
        self,
        frame_index: int,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        source_id: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
    ) -> int:
        if self._closed:
            return 0
        frame_no = int(frame_index) + 1
        pts, cols, src, conf, widths = _normalize_frame_arrays(
            points,
            colors,
            source_id,
            confidence,
            point_width=self.point_width,
        )
        limiter = deterministic_limit_indices(len(pts), self.max_points_per_frame, 4100003 + int(frame_index))
        if limiter is not None:
            pts = pts[limiter]
            cols = cols[limiter]
            src = src[limiter]
            conf = conf[limiter]
            widths = widths[limiter]

        first_before = self._first_sample
        self._append_sample("points", frame_no, _fmt_point_list(pts))
        self._first_sample = first_before
        self._append_sample("extent", frame_no, _fmt_extent(pts))
        self._first_sample = first_before
        if self.include_colors:
            self._append_sample("colors", frame_no, _fmt_color_list(cols))
            self._first_sample = first_before
        self._append_sample("widths", frame_no, _fmt_float_array(widths))
        self._first_sample = first_before
        if self.include_source_id:
            self._append_sample("source_id", frame_no, _fmt_int_array(src))
            self._first_sample = first_before
        if self.include_confidence:
            self._append_sample("confidence", frame_no, _fmt_float_array(conf))
            self._first_sample = first_before
        self._first_sample = False
        self._count += 1
        return int(len(pts))

    def _close_tmp_files(self) -> None:
        for f in self._tmp_files.values():
            try:
                f.close()
            except Exception:
                pass

    def _copy_tmp(self, out, key: str) -> None:  # noqa: ANN001
        src = self._tmp_dir / f"{key}.samples"
        if src.exists():
            with src.open("r", encoding="utf-8") as f:
                shutil.copyfileobj(f, out)

    def close(self) -> None:
        if self._closed:
            return
        self._close_tmp_files()
        with self.path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("#usda 1.0\n")
            f.write("(\n")
            f.write(f"    defaultPrim = \"{self.label}\"\n")
            f.write(f"    startTimeCode = {self.start_frame}\n")
            f.write(f"    endTimeCode = {self.end_frame}\n")
            f.write(f"    framesPerSecond = {_fmt_float(self.fps)}\n")
            f.write(f"    timeCodesPerSecond = {_fmt_float(self.fps)}\n")
            f.write(")\n\n")
            f.write(f"def Xform \"{self.label}\"\n{{\n")
            f.write("    custom string qflux_note = \"Animated XYZ-only USDA point cloud cache.\"\n")
            tx, ty, tz = self.xform_translate
            if float(np.linalg.norm(self.xform_translate)) > 1e-9:
                f.write(f"    double3 xformOp:translate = ({_fmt_float(tx)}, {_fmt_float(ty)}, {_fmt_float(tz)})\n")
                f.write("    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n")
            f.write("\n    def Points \"dynamic_points\"\n    {\n")
            f.write("        point3f[] points.timeSamples = {\n")
            self._copy_tmp(f, "points")
            f.write("\n        }\n")
            f.write("        float3[] extent.timeSamples = {\n")
            self._copy_tmp(f, "extent")
            f.write("\n        }\n")
            if self.include_colors:
                f.write("        color3f[] primvars:displayColor.timeSamples = {\n")
                self._copy_tmp(f, "colors")
                f.write("\n        } (\n")
                f.write("            interpolation = \"vertex\"\n")
                f.write("        )\n")
            f.write("        float[] widths.timeSamples = {\n")
            self._copy_tmp(f, "widths")
            f.write("\n        }\n")
            if self.include_source_id:
                f.write("        int[] primvars:source_id.timeSamples = {\n")
                self._copy_tmp(f, "source_id")
                f.write("\n        } (\n")
                f.write("            interpolation = \"vertex\"\n")
                f.write("        )\n")
            if self.include_confidence:
                f.write("        float[] primvars:confidence.timeSamples = {\n")
                self._copy_tmp(f, "confidence")
                f.write("\n        } (\n")
                f.write("            interpolation = \"vertex\"\n")
                f.write("        )\n")
            f.write("    }\n")
            f.write("}\n")
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._closed = True

    @property
    def frame_count(self) -> int:
        return int(self._count)

    def __enter__(self) -> "UsdPointSequenceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()


def _safe_usd_identifier(value: str) -> str:
    text = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(value))
    if not text or text[0].isdigit():
        text = "pc_" + text
    return text


class UsdMeshSequenceWriter:
    """Streaming USDA writer for fixed-topology animated meshes.

    `faceVertexIndices` and `faceVertexCounts` are written once. Only `points`
    and `extent` are time-sampled, so Blender sees stable vertex IDs across time.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        faces: np.ndarray,
        fps: float = 24.0,
        start_frame: int = 1,
        end_frame: int = 1,
        label: str = "mesh_sequence",
        mesh_name: str = "dynamic_mesh",
        xform_translate: np.ndarray | None = None,
        display_color: tuple[float, float, float] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        self.fps = float(fps) if fps and fps > 0 else 24.0
        self.start_frame = max(1, int(start_frame))
        self.end_frame = max(self.start_frame, int(end_frame))
        self.label = _safe_usd_identifier(label or "mesh_sequence")
        self.mesh_name = _safe_usd_identifier(mesh_name or "dynamic_mesh")
        self.display_color = display_color
        if xform_translate is None:
            self.xform_translate = np.zeros(3, dtype=np.float32)
        else:
            self.xform_translate = np.asarray(xform_translate, dtype=np.float32).reshape(3)
            self.xform_translate = np.nan_to_num(self.xform_translate, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self._closed = False
        self._count = 0
        self._first_sample = True
        self._tmp_dir = self.path.with_suffix(self.path.suffix + ".tmp")
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_files: dict[str, object] = {
            "points": (self._tmp_dir / "points.samples").open("w", encoding="utf-8", newline="\n"),
            "extent": (self._tmp_dir / "extent.samples").open("w", encoding="utf-8", newline="\n"),
        }

    def _append_sample(self, key: str, frame_no: int, value: str) -> None:
        f = self._tmp_files[key]
        if not self._first_sample:
            f.write(",\n")
        f.write(f"            {int(frame_no)}: {value}")

    def add_frame(self, frame_index: int, vertices: np.ndarray) -> int:
        if self._closed:
            return 0
        frame_no = int(frame_index) + 1
        pts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        first_before = self._first_sample
        self._append_sample("points", frame_no, _fmt_point_list(pts))
        self._first_sample = first_before
        self._append_sample("extent", frame_no, _fmt_extent(pts))
        self._first_sample = False
        self._count += 1
        return int(len(pts))

    def _close_tmp_files(self) -> None:
        for f in self._tmp_files.values():
            try:
                f.close()
            except Exception:
                pass

    def _copy_tmp(self, out, key: str) -> None:  # noqa: ANN001
        src = self._tmp_dir / f"{key}.samples"
        if src.exists():
            with src.open("r", encoding="utf-8") as f:
                shutil.copyfileobj(f, out)

    def close(self) -> None:
        if self._closed:
            return
        self._close_tmp_files()
        flat_indices = self.faces.reshape(-1)
        counts = np.full((len(self.faces),), 3, dtype=np.int32)
        with self.path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("#usda 1.0\n")
            f.write("(\n")
            f.write(f"    defaultPrim = \"{self.label}\"\n")
            f.write(f"    startTimeCode = {self.start_frame}\n")
            f.write(f"    endTimeCode = {self.end_frame}\n")
            f.write(f"    framesPerSecond = {_fmt_float(self.fps)}\n")
            f.write(f"    timeCodesPerSecond = {_fmt_float(self.fps)}\n")
            f.write(")\n\n")
            f.write(f"def Xform \"{self.label}\"\n{{\n")
            f.write("    custom string qflux_note = \"Fixed-topology animated mesh. Vertex IDs are stable across frames.\"\n")
            tx, ty, tz = self.xform_translate
            if float(np.linalg.norm(self.xform_translate)) > 1e-9:
                f.write(f"    double3 xformOp:translate = ({_fmt_float(tx)}, {_fmt_float(ty)}, {_fmt_float(tz)})\n")
                f.write("    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n")
            f.write(f"\n    def Mesh \"{self.mesh_name}\"\n    {{\n")
            f.write("        uniform int[] faceVertexCounts = ")
            f.write(_fmt_int_array(counts))
            f.write("\n")
            f.write("        uniform int[] faceVertexIndices = ")
            f.write(_fmt_int_array(flat_indices))
            f.write("\n")
            # No explicit normals/UV are authored. Blender/USD will compute face
            # normals from the right-handed face winding below.
            f.write("        uniform token orientation = \"rightHanded\"\n")
            f.write("        point3f[] points.timeSamples = {\n")
            self._copy_tmp(f, "points")
            f.write("\n        }\n")
            f.write("        float3[] extent.timeSamples = {\n")
            self._copy_tmp(f, "extent")
            f.write("\n        }\n")
            if self.display_color is not None:
                c = np.clip(np.asarray(self.display_color, dtype=np.float32).reshape(3), 0.0, 1.0)
                f.write(f"        color3f[] primvars:displayColor = [({_fmt_float(c[0])}, {_fmt_float(c[1])}, {_fmt_float(c[2])})] (\n")
                f.write("            interpolation = \"constant\"\n")
                f.write("        )\n")
            f.write("        uniform token subdivisionScheme = \"none\"\n")
            f.write("    }\n")
            f.write("}\n")
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._closed = True

    @property
    def frame_count(self) -> int:
        return int(self._count)

    def __enter__(self) -> "UsdMeshSequenceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()


class UsdLayeredMeshSequenceWriter:
    """Streaming USDA writer with multiple fixed-topology Mesh prims in one file."""

    def __init__(
        self,
        path: str | Path,
        *,
        layers: dict[str, dict],
        fps: float = 24.0,
        start_frame: int = 1,
        end_frame: int = 1,
        label: str = "combined_mesh_sequence",
        xform_translate: np.ndarray | None = None,
        note: str = "Layered Body/Garment/Hair animated mesh.",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.layers: dict[str, dict] = {}
        for name, spec in (layers or {}).items():
            safe = _safe_usd_identifier(name)
            faces = np.asarray(spec.get("faces", np.zeros((0, 3), dtype=np.int32)), dtype=np.int32).reshape(-1, 3)
            color = spec.get("color", (0.65, 0.68, 0.74))
            self.layers[safe] = {"faces": faces, "color": color}
        self.fps = float(fps) if fps and fps > 0 else 24.0
        self.start_frame = max(1, int(start_frame))
        self.end_frame = max(self.start_frame, int(end_frame))
        self.label = _safe_usd_identifier(label or "combined_mesh_sequence")
        self.note = str(note or "Layered animated mesh.")
        if xform_translate is None:
            self.xform_translate = np.zeros(3, dtype=np.float32)
        else:
            self.xform_translate = np.asarray(xform_translate, dtype=np.float32).reshape(3)
            self.xform_translate = np.nan_to_num(self.xform_translate, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self._closed = False
        self._count = 0
        self._tmp_dir = self.path.with_suffix(self.path.suffix + ".tmp")
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_files: dict[tuple[str, str], object] = {}
        self._first_sample: dict[tuple[str, str], bool] = {}
        for name in self.layers:
            for key in ("points", "extent"):
                fp = self._tmp_dir / f"{name}.{key}.samples"
                self._tmp_files[(name, key)] = fp.open("w", encoding="utf-8", newline="\n")
                self._first_sample[(name, key)] = True

    def _append_sample(self, name: str, key: str, frame_no: int, value: str) -> None:
        f = self._tmp_files[(name, key)]
        if not self._first_sample[(name, key)]:
            f.write(",\n")
        f.write(f"            {int(frame_no)}: {value}")
        self._first_sample[(name, key)] = False

    def add_frame(self, frame_index: int, layer_vertices: dict[str, np.ndarray]) -> int:
        if self._closed:
            return 0
        frame_no = int(frame_index) + 1
        written = 0
        for raw_name in self.layers:
            pts = np.asarray(layer_vertices.get(raw_name, np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
            pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            self._append_sample(raw_name, "points", frame_no, _fmt_point_list(pts))
            self._append_sample(raw_name, "extent", frame_no, _fmt_extent(pts))
            written += int(len(pts))
        self._count += 1
        return written

    def _close_tmp_files(self) -> None:
        for f in self._tmp_files.values():
            try:
                f.close()
            except Exception:
                pass

    def _copy_tmp(self, out, name: str, key: str) -> None:  # noqa: ANN001
        src = self._tmp_dir / f"{name}.{key}.samples"
        if src.exists():
            with src.open("r", encoding="utf-8") as f:
                shutil.copyfileobj(f, out)

    def close(self) -> None:
        if self._closed:
            return
        self._close_tmp_files()
        with self.path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("#usda 1.0\n")
            f.write("(\n")
            f.write(f"    defaultPrim = \"{self.label}\"\n")
            f.write(f"    startTimeCode = {self.start_frame}\n")
            f.write(f"    endTimeCode = {self.end_frame}\n")
            f.write(f"    framesPerSecond = {_fmt_float(self.fps)}\n")
            f.write(f"    timeCodesPerSecond = {_fmt_float(self.fps)}\n")
            f.write(")\n\n")
            f.write(f"def Xform \"{self.label}\"\n{{\n")
            f.write(f"    custom string qflux_note = \"{self.note}\"\n")
            tx, ty, tz = self.xform_translate
            if float(np.linalg.norm(self.xform_translate)) > 1e-9:
                f.write(f"    double3 xformOp:translate = ({_fmt_float(tx)}, {_fmt_float(ty)}, {_fmt_float(tz)})\n")
                f.write("    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n")
            for name, spec in self.layers.items():
                faces = np.asarray(spec["faces"], dtype=np.int32).reshape(-1, 3)
                flat_indices = faces.reshape(-1)
                counts = np.full((len(faces),), 3, dtype=np.int32)
                color = np.clip(np.asarray(spec.get("color", (0.65, 0.68, 0.74)), dtype=np.float32).reshape(3), 0.0, 1.0)
                f.write(f"\n    def Mesh \"{name}\"\n    {{\n")
                f.write("        uniform int[] faceVertexCounts = ")
                f.write(_fmt_int_array(counts))
                f.write("\n")
                f.write("        uniform int[] faceVertexIndices = ")
                f.write(_fmt_int_array(flat_indices))
                f.write("\n")
                f.write("        uniform token orientation = \"rightHanded\"\n")
                f.write("        point3f[] points.timeSamples = {\n")
                self._copy_tmp(f, name, "points")
                f.write("\n        }\n")
                f.write("        float3[] extent.timeSamples = {\n")
                self._copy_tmp(f, name, "extent")
                f.write("\n        }\n")
                f.write(f"        color3f[] primvars:displayColor = [({_fmt_float(color[0])}, {_fmt_float(color[1])}, {_fmt_float(color[2])})] (\n")
                f.write("            interpolation = \"constant\"\n")
                f.write("        )\n")
                f.write("        uniform token subdivisionScheme = \"none\"\n")
                f.write("    }\n")
            f.write("}\n")
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._closed = True

    @property
    def frame_count(self) -> int:
        return int(self._count)

    def __enter__(self) -> "UsdLayeredMeshSequenceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()
