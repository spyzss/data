"""Lazy, bounded evidence generation for human issue review."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
import uuid
from urllib.parse import quote
from typing import Any

from qc_pipeline.context import AssetContext


class EvidenceError(RuntimeError):
    """Raised when issue evidence cannot be resolved safely."""


@dataclass(frozen=True)
class EvidenceView:
    issue_id: str
    start_frame: int
    end_frame_exclusive: int
    clip_url: str
    overlay_url: str | None
    generation_error: str | None

    @property
    def end_frame(self) -> int:
        return self.end_frame_exclusive


@dataclass(frozen=True)
class _Entry:
    issue: Mapping[str, Any]
    context: AssetContext
    source_path: Path
    source_hash: str
    start_frame: int
    end_frame_exclusive: int
    existing_clip: Path | None
    existing_overlay: Path | None


def _strict_frame(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EvidenceError(f"{field} must be a non-negative integer")
    return value


class EvidenceService:
    """Resolve existing evidence or lazily generate one issue-sized window."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        ffmpeg_runner: Callable[[Sequence[str], Path], None] | None = None,
        overlay_renderer: Callable[[Mapping[str, Any], Path, int, int], None] | None = None,
        url_prefix: str = "/evidence",
    ) -> None:
        self.cache_root = Path(cache_root).resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._ffmpeg_runner = ffmpeg_runner or self._run_ffmpeg
        self._overlay_renderer = overlay_renderer
        self._url_prefix = "/" + url_prefix.strip("/") if url_prefix.strip("/") else ""
        self._entries: dict[str, _Entry] = {}

    @staticmethod
    def _run_ffmpeg(command: Sequence[str], output: Path) -> None:
        subprocess.run(list(command), check=True, capture_output=True)

    @staticmethod
    def _inside(root: Path, value: str | Path, *, label: str) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise EvidenceError(f"{label} must stay inside batch root") from exc
        return resolved

    @staticmethod
    def _source_path(context: AssetContext) -> Path:
        video = context.source_files.get("video")
        if not isinstance(video, Mapping) or not isinstance(video.get("path"), str):
            raise EvidenceError("asset context has no source_files.video.path")
        path = EvidenceService._inside(context.batch_root, video["path"], label="video path")
        if not path.is_file():
            raise EvidenceError(f"video source does not exist: {path}")
        return path

    @staticmethod
    def _source_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _window(issue: Mapping[str, Any], context: AssetContext) -> tuple[int, int]:
        nested = issue.get("window")
        nested = nested if isinstance(nested, Mapping) else {}
        start_value = issue.get("start_frame", nested.get("start_frame"))
        end_value = issue.get(
            "end_frame_exclusive",
            issue.get("end_frame", nested.get("end_frame_exclusive", nested.get("end_frame"))),
        )
        start = _strict_frame(start_value, "issue start_frame")
        end = _strict_frame(end_value, "issue end_frame")
        if end <= start:
            raise EvidenceError("issue frame window must be non-empty half-open range")
        if context.source_range is not None:
            range_start, range_end = context.source_range
            if start < range_start or end > range_end:
                raise EvidenceError("issue frame window exceeds asset source_range")
        return start, end

    @staticmethod
    def _evidence_rows(issue: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        raw = issue.get("evidence", [])
        if isinstance(raw, Mapping):
            raw = [raw]
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            return []
        return [row for row in raw if isinstance(row, Mapping)]

    def _existing_paths(
        self, issue: Mapping[str, Any], context: AssetContext
    ) -> tuple[Path | None, Path | None]:
        clip: Path | None = None
        overlay: Path | None = None
        for row in self._evidence_rows(issue):
            raw_path = row.get("path", row.get("source_path"))
            if not isinstance(raw_path, str) or not raw_path:
                continue
            kind = str(row.get("kind", row.get("evidence_type", ""))).lower()
            path = self._inside(context.batch_root, raw_path, label="evidence path")
            if kind in {"clip", "video_clip", "issue_clip"}:
                if path.is_file():
                    clip = path
            elif kind in {"overlay", "skeleton_overlay", "combined_overlay"}:
                if path.is_file():
                    overlay = path
        return clip, overlay

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "issue"

    def _cache_paths(self, entry: _Entry) -> tuple[Path, Path]:
        key = hashlib.sha256(
            f"{entry.context.asset_id}:{entry.issue.get('issue_id')}:{entry.source_hash}:"
            f"{entry.start_frame}:{entry.end_frame_exclusive}".encode()
        ).hexdigest()[:24]
        name = self._safe_name(str(entry.issue.get("issue_id", "issue")))
        directory = self.cache_root / entry.context.asset_id
        return (
            directory / f"{name}-{entry.start_frame}-{entry.end_frame_exclusive}-{key}.mp4",
            directory / f"{name}-{entry.start_frame}-{entry.end_frame_exclusive}-{key}.png",
        )

    def _url(self, path: Path, context: AssetContext) -> str:
        try:
            relative = path.resolve().relative_to(context.batch_root.resolve())
        except ValueError:
            try:
                relative = path.resolve().relative_to(self.cache_root.resolve())
            except ValueError:
                relative = Path(path.name)
        encoded = quote(relative.as_posix(), safe="/._-")
        return f"{self._url_prefix}/{encoded}" if self._url_prefix else encoded

    def _entry(self, issue_id: str) -> _Entry:
        try:
            return self._entries[issue_id]
        except KeyError as exc:
            raise EvidenceError(f"issue {issue_id!r} has not been resolved") from exc

    def _atomic_generate(
        self,
        output: Path,
        generator: Callable[[Path], None],
    ) -> Path:
        if output.is_file():
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
        try:
            generator(temporary)
            if not temporary.is_file():
                raise EvidenceError(f"evidence generator did not create {temporary}")
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        return output

    def ensure_issue_clip(self, issue_id: str) -> Path:
        entry = self._entry(issue_id)
        if entry.existing_clip is not None:
            return entry.existing_clip
        output, _ = self._cache_paths(entry)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(entry.source_path),
            "-vf",
            f"trim=start_frame={entry.start_frame}:end_frame={entry.end_frame_exclusive},setpts=PTS-STARTPTS",
            "-an",
            str(output),
        ]
        return self._atomic_generate(
            output,
            lambda temporary: self._ffmpeg_runner(
                [*command[:-1], str(temporary)], temporary
            ),
        )

    @staticmethod
    def _slice_skeleton(issue: Mapping[str, Any], start: int, end: int) -> dict[str, Any]:
        skeleton = issue.get("skeleton")
        if not isinstance(skeleton, Mapping):
            raise EvidenceError("issue has no skeleton payload")
        result = deepcopy(dict(skeleton))
        frames = result.get("frames")
        if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes, bytearray)):
            selected = []
            for frame in list(frames)[start:end]:
                if isinstance(frame, Mapping):
                    frame_copy = deepcopy(dict(frame))
                    points = frame_copy.get("points")
                    if isinstance(points, Sequence) and not isinstance(points, (str, bytes, bytearray)):
                        frame_copy["points"] = deepcopy(list(points)[:21])
                    selected.append(frame_copy)
                else:
                    selected.append(deepcopy(frame))
            result["frames"] = selected
        return result

    def ensure_skeleton_overlay(self, issue_id: str) -> Path:
        entry = self._entry(issue_id)
        if entry.existing_overlay is not None:
            return entry.existing_overlay
        if self._overlay_renderer is None:
            raise EvidenceError("skeleton overlay renderer is not configured")
        output = self._cache_paths(entry)[1]
        issue_copy = deepcopy(dict(entry.issue))
        issue_copy["skeleton"] = self._slice_skeleton(
            entry.issue, entry.start_frame, entry.end_frame_exclusive
        )
        return self._atomic_generate(
            output,
            lambda temporary: self._overlay_renderer(
                issue_copy,
                temporary,
                entry.start_frame,
                entry.end_frame_exclusive,
            ),
        )

    def resolve(self, issue: Mapping[str, Any], asset_context: AssetContext) -> EvidenceView:
        if not isinstance(issue, Mapping):
            raise TypeError("issue must be a mapping")
        if not isinstance(asset_context, AssetContext):
            raise TypeError("asset_context must be an AssetContext")
        issue_id = issue.get("issue_id")
        if not isinstance(issue_id, str) or not issue_id:
            raise EvidenceError("issue is missing issue_id")
        source_path = self._source_path(asset_context)
        start, end = self._window(issue, asset_context)
        clip, overlay = self._existing_paths(issue, asset_context)
        entry = _Entry(
            issue=deepcopy(dict(issue)),
            context=asset_context,
            source_path=source_path,
            source_hash=self._source_hash(source_path),
            start_frame=start,
            end_frame_exclusive=end,
            existing_clip=clip,
            existing_overlay=overlay,
        )
        self._entries[issue_id] = entry
        clip_path = self.ensure_issue_clip(issue_id)
        overlay_url: str | None = None
        generation_error: str | None = None
        if overlay is not None:
            overlay_url = self._url(overlay, asset_context)
        elif isinstance(issue.get("skeleton"), Mapping) and self._overlay_renderer is not None:
            try:
                overlay_url = self._url(self.ensure_skeleton_overlay(issue_id), asset_context)
            except Exception as exc:  # overlay is optional for human review
                generation_error = str(exc)
        return EvidenceView(
            issue_id=issue_id,
            start_frame=start,
            end_frame_exclusive=end,
            clip_url=self._url(clip_path, asset_context),
            overlay_url=overlay_url,
            generation_error=generation_error,
        )


__all__ = ["EvidenceError", "EvidenceService", "EvidenceView"]
