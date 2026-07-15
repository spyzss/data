"""Read-only projections from Canonical QC episodes into legacy QC contracts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np

from qc_common.keypoints import EGODATA_HAND21_INDEX_TO_ACCEPTANCE_BASE
from qc_common.types import ClipInputs
from qc_pipeline.context import AssetContext

from .contracts import CanonicalQcEpisode
from .errors import CanonicalInputError
from .validation import validate_episode
from .video_probe import probe_video


def _fail(field: str, detail: str) -> None:
    raise CanonicalInputError("source_integrity_error", field, detail)


def _readonly(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    result = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    result.flags.writeable = False
    return result


def _range(source_range: tuple[int, int] | None, frame_count: int) -> tuple[int, int]:
    if source_range is None:
        return 0, frame_count
    if (
        type(source_range) is not tuple
        or len(source_range) != 2
        or type(source_range[0]) is not int
        or type(source_range[1]) is not int
    ):
        raise ValueError("source_range must be an integer half-open range")
    start, end = source_range
    if start < 0 or end <= start or end > frame_count:
        raise ValueError(
            f"source_range must be a non-empty half-open range inside [0, {frame_count})"
        )
    return start, end


class CanonicalQcBridge:
    """Project one already-validated immutable episode without repairing it."""

    __slots__ = ("_episode", "_source_root")

    def __init__(self, episode: CanonicalQcEpisode, *, source_root: Path) -> None:
        validate_episode(episode)
        root = Path(source_root).expanduser()
        if not root.is_absolute():
            root = root.absolute()
        try:
            resolved = root.resolve(strict=True)
        except OSError as exc:
            _fail("source_root", f"cannot resolve supplied source root: {exc}")
        if not resolved.is_dir():
            _fail("source_root", "must be an existing directory")
        if resolved != root.absolute():
            _fail("source_root", "must not traverse a filesystem symlink")
        self._episode = episode
        self._source_root = resolved

    @property
    def episode(self) -> CanonicalQcEpisode:
        return self._episode

    @property
    def source_root(self) -> Path:
        return self._source_root

    def _declared_path(self, relative_path: str, *, field: str) -> Path:
        pure = PurePosixPath(relative_path)
        if (
            pure.is_absolute()
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
        ):
            _fail(field, "must be a normalized relative POSIX path")
        lexical = self._source_root.joinpath(*pure.parts)
        try:
            resolved = lexical.resolve(strict=True)
            resolved.relative_to(self._source_root)
        except (OSError, ValueError):
            _fail(field, "declared path does not resolve inside source_root")
        if resolved != lexical.absolute():
            _fail(field, "declared path must not traverse a filesystem symlink")
        if not resolved.is_file():
            _fail(field, "declared path must be a regular file")
        return resolved

    def verify_sources(self) -> Mapping[str, Path]:
        """Stream-verify every declared source without copying it."""
        verified: dict[str, Path] = {}
        for index, item in enumerate(self._episode.provenance.source_files):
            field = f"provenance.source_files[{index}]"
            path = self._declared_path(item.relative_path, field=f"{field}.relative_path")
            try:
                before = path.stat()
                digest_builder = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest_builder.update(chunk)
                after = path.stat()
            except OSError as exc:
                _fail(field, f"cannot verify source provenance: {exc}")
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                _fail(field, "source changed while verifying provenance")
            if after.st_size != item.size_bytes:
                _fail(f"{field}.size_bytes", "source size does not match provenance")
            if digest_builder.hexdigest() != item.sha256:
                _fail(f"{field}.sha256", "source hash does not match provenance")
            verified[item.relative_path] = path
        return MappingProxyType(verified)

    def video_path(self) -> Path:
        video = self._episode.main_video
        declared = [
            item
            for item in self._episode.provenance.source_files
            if item.role == "main_video"
        ]
        if len(declared) != 1 or declared[0].relative_path != video.path:
            _fail("main_video.path", "must match the sole provenance main_video")
        verified = self.verify_sources()
        path = verified[video.path]
        if video.sha256 != declared[0].sha256:
            _fail("main_video.sha256", "source file does not match canonical provenance")
        physical_frame_count = probe_video(path).frame_count
        _physical_start, physical_end = video.source_frame_range
        if physical_end > physical_frame_count:
            _fail(
                "main_video.source_frame_range",
                f"ends at {physical_end}, beyond physical MP4 frame count "
                f"{physical_frame_count}",
            )
        return path

    def semantic_payload(self) -> Mapping[str, Any]:
        return self._semantic_payload()

    def _semantic_payload(self) -> Mapping[str, Any]:
        semantics = self._episode.semantics
        subtasks = tuple(
            MappingProxyType(
                {
                    "subtask_id": item.subtask_id,
                    "start_frame": item.start_frame,
                    "end_frame_exclusive": item.end_frame_exclusive,
                    "description_cn": item.description_cn,
                    "description_en": item.description_en,
                }
            )
            for item in semantics.subtask_sequence
        )
        return MappingProxyType(
            {
                "scene_id": semantics.scene_id,
                "task_id": semantics.task_id,
                "task_category": semantics.task_category,
                "task_cn": semantics.task_cn,
                "task_en": semantics.task_en,
                "description_cn": semantics.description_cn,
                "description_en": semantics.description_en,
                "subtask_sequence": subtasks,
            }
        )

    def clip_inputs(
        self, source_range: tuple[int, int] | None = None
    ) -> ClipInputs:
        episode = self._episode
        start, end = _range(source_range, episode.time_axis.frame_count)
        points = episode.observation.hand_keypoints_3d[start:end]
        keypoints = {
            f"{side}{base_name}": _readonly(
                points[:, hand_index, joint_index, :]
            )
            for hand_index, side in enumerate(("left", "right"))
            for joint_index, base_name in EGODATA_HAND21_INDEX_TO_ACCEPTANCE_BASE.items()
        }
        payload = self._semantic_payload()
        legacy_text_label = dict(payload)
        legacy_text_label.update(
            {
                "scene": episode.semantics.scene_id,
                "task": episode.semantics.task_id,
                "text_en": episode.semantics.description_en,
            }
        )
        quality = episode.supplier_evidence.hand_quality
        status = None
        if (
            quality is not None
            and quality.provided
            and quality.status is not None
        ):
            status = _readonly(quality.status[start:end])
        clip = ClipInputs(
            episode_idx=0,
            frame_indices=list(range(start, end)),
            keypoints=keypoints,
            rotations=None,
            confidences=None,
            quality_hand=None,
            instruction=episode.semantics.description_en,
            text_label=legacy_text_label,
            intrinsics=_readonly(episode.calibration.intrinsic_matrix),
            fps=episode.time_axis.fps_num / episode.time_axis.fps_den,
            supplier_hand_quality_status=status,
            hand_keypoints_3d=_readonly(points),
            hand_joint_valid_3d=_readonly(
                episode.observation.hand_joint_valid_3d[start:end]
            ),
            timestamps_ns=_readonly(episode.time_axis.timestamps_ns[start:end]),
            hand_keypoints_2d=_readonly(
                episode.observation.hand_keypoints_2d[start:end]
            ),
            hand_joint_valid_2d=_readonly(
                episode.observation.hand_joint_valid_2d[start:end]
            ),
        )
        setattr(clip, "asset_id", episode.identity.asset_id)
        setattr(clip, "supplier_id", episode.identity.supplier_id)
        setattr(clip, "source_path", str(self._source_root))
        setattr(clip, "clip_start_frame", start)
        setattr(clip, "clip_end_frame", end - 1)
        return clip

    def physical_video_range(
        self, source_range: tuple[int, int] | None = None
    ) -> tuple[int, int]:
        """Compose a logical half-open slice with the physical MP4 placement."""
        start, end = _range(source_range, self._episode.time_axis.frame_count)
        physical_start, _physical_end = self._episode.main_video.source_frame_range
        return physical_start + start, physical_start + end

    def asset_context(
        self,
        *,
        batch_root: Path,
        report_path: Path,
        source_range: tuple[int, int] | None = None,
        metadata: Mapping[str, Any] | None = None,
        supplemental_source_files: Mapping[str, Any] | None = None,
    ) -> AssetContext:
        self.verify_sources()
        start, end = _range(source_range, self._episode.time_axis.frame_count)
        selected_range = None if source_range is None else (start, end)
        batch = Path(batch_root).resolve()
        try:
            root_prefix = self._source_root.relative_to(batch)
        except ValueError:
            _fail("source_root", "must be inside AssetContext.batch_root")
        source_files: dict[str, Any] = dict(supplemental_source_files or {})
        provenance_rows: list[dict[str, Any]] = []
        for index, item in enumerate(self._episode.provenance.source_files):
            relative = (root_prefix / item.relative_path).as_posix()
            provenance_rows.append(
                {
                    "relative_path": item.relative_path,
                    "role": item.role,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
            )
            source_files[f"canonical_source_{index:03d}"] = {
                "path": relative,
                "role": item.role,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            if item.role == "main_video":
                source_files["video"] = {"path": relative}
            elif item.role == "episode_data":
                name = "hdf5" if self._episode.identity.source_format == "hdf5" else "parquet"
                source_files[name] = {"path": relative}
        source_files["canonical_provenance"] = {
            "source_format": self._episode.identity.source_format,
            "source_schema_version": self._episode.identity.source_schema_version,
            "source_fingerprint": self._episode.provenance.source_fingerprint,
            "adapter_id": self._episode.provenance.adapter_id,
            "adapter_version": self._episode.provenance.adapter_version,
            "source_files": provenance_rows,
            "main_video_source_frame_range": list(
                self._episode.main_video.source_frame_range
            ),
        }
        projected_metadata = dict(metadata or {})
        projected_metadata.update(
            {
                "canonical_episode": self._episode,
                "canonical_source_root": str(self._source_root),
                "supplier": self._episode.identity.supplier_id,
                "supplier_id": self._episode.identity.supplier_id,
            }
        )
        return AssetContext(
            asset_id=self._episode.identity.asset_id,
            batch_root=batch,
            report_path=report_path,
            source_files=source_files,
            source_range=selected_range,
            metadata=projected_metadata,
        )


__all__ = ["CanonicalQcBridge"]
