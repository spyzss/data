"""QY episode discovery and canonical supplier manifests."""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from acceptance_pull.supplier_adapters.qingyu_hand_pose import (
    QingyuHandPoseSession,
)


QY_CAMERAS = (
    "left_cam_left",
    "left_cam_right",
    "mid_cam_left",
    "mid_cam_right",
    "right_cam_left",
    "right_cam_right",
)

_DEFAULT_CAMERA_SELECTION_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "qy_camera_selection.yaml"
)
_default_camera_selection_payload = yaml.safe_load(
    _DEFAULT_CAMERA_SELECTION_PATH.read_text(encoding="utf-8")
)
if not isinstance(_default_camera_selection_payload, dict):
    raise ValueError("QY default camera-selection config must be an object")
DEFAULT_CAMERA_SELECTION_CONFIG: dict[str, Any] = dict(
    _default_camera_selection_payload
)

_REQUIRED_PATHS = {
    "episode_manifest": "episode_manifest.json",
    "observations_2d": "hand_pose/observations_2d.parquet",
    "trajectory_3d": "hand_pose/trajectory_3d.parquet",
    "coordinate_system": "hand_pose/coordinate_system.json",
    "quality": "hand_pose/quality.json",
    "timebase": "timestamps/episode_timebase.json",
    "semantic": "semantic/annotation.json",
    "review_video": "review/review.mp4",
}

MANIFEST_COLUMNS = [
    "schema_version",
    "supplier",
    "supplier_id",
    "supplier_name",
    "supplier_alias",
    "asset_id",
    "source_granularity",
    "category",
    "task_name",
    "episode_id",
    "episode_root",
    "episode_manifest_path",
    "observations_2d_path",
    "trajectory_3d_path",
    "coordinate_system_path",
    "quality_path",
    "timebase_path",
    "semantic_path",
    "review_video_path",
    "sam3_model_path",
    "primary_camera",
    "requested_primary_camera",
    "primary_camera_source",
    "primary_video_path",
    *[f"{camera}_video_path" for camera in QY_CAMERAS],
    "video_frame_count",
    "video_width",
    "video_height",
    "source_frame_count",
    "start_frame",
    "end_frame",
    "fps",
    "frame_coordinate_system",
    "timebase_status",
    "timebase_source",
    "frame_mapping_status",
    "frame_mapping_source",
    "source_video_identity_assumed",
    "skeleton_2d_status",
    "skeleton_3d_status",
    "skeleton_3d_valid_row_count",
    "skeleton_3d_coverage_status",
    "joint_topology_status",
    "coordinate_system_status",
    "reference_camera_status",
    "camera_coverage",
    "camera_recommendation_score",
    "camera_second_best_score",
    "camera_recommendation_reason",
    "camera_selection_config",
    "skeleton_3d_audit",
    "file_inventory",
    "adapter_status",
    "reason",
]


def _logical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a finite number") from None
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be a finite number")
    return numeric


def _integer(value: Any, field: str) -> int:
    numeric = _finite_number(value, field)
    if not numeric.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(numeric)


def camera_selection_config(
    override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CAMERA_SELECTION_CONFIG))
    if override is not None:
        allowed = {
            "schema_version",
            "minimum_hand_coverage",
            "minimum_both_hand_coverage",
            "minimum_valid_joint_ratio",
            "minimum_timeline_match_ratio",
            "tie_tolerance",
            "weights",
        }
        unknown = sorted(set(override) - allowed)
        if unknown:
            raise ValueError(
                f"unknown QY camera-selection field: {unknown[0]}"
            )
        for name in (
            "schema_version",
            "minimum_hand_coverage",
            "minimum_both_hand_coverage",
            "minimum_valid_joint_ratio",
            "minimum_timeline_match_ratio",
            "tie_tolerance",
        ):
            if name in override:
                config[name] = override[name]
        if "weights" in override:
            if not isinstance(override["weights"], Mapping):
                raise ValueError("QY camera-selection weights must be an object")
            config["weights"].update(dict(override["weights"]))
    if config["schema_version"] != "qy.camera_selection.v1":
        raise ValueError("unsupported QY camera-selection schema_version")
    for name in (
        "minimum_hand_coverage",
        "minimum_both_hand_coverage",
        "minimum_valid_joint_ratio",
        "minimum_timeline_match_ratio",
        "tie_tolerance",
    ):
        value = _finite_number(config[name], name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
        config[name] = value
    expected_weights = {
        "minimum_hand_coverage",
        "both_hand_coverage",
        "valid_joint_ratio",
        "timeline_match_ratio",
    }
    if set(config["weights"]) != expected_weights:
        raise ValueError("QY camera-selection weights have unknown or missing fields")
    for name in expected_weights:
        config["weights"][name] = _finite_number(
            config["weights"][name], f"weights.{name}"
        )
        if config["weights"][name] < 0:
            raise ValueError(f"weights.{name} must be >= 0")
    weight_sum = sum(config["weights"].values())
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("QY camera-selection weights must sum to 1")
    return config


def discover_qingyu_episodes(root: Path) -> list[Path]:
    logical_root = _logical_absolute(root)
    return sorted(
        {path.parent for path in logical_root.rglob("episode_manifest.json")},
        key=lambda path: path.relative_to(logical_root).as_posix(),
    )


def _episode_identity(root: Path, episode: Path) -> tuple[str, str, str, str]:
    relative = episode.relative_to(root)
    if len(relative.parts) != 3:
        raise ValueError(
            f"QY episode must be category/task/episode: {relative.as_posix()}"
        )
    category_dir, task_name, episode_id = relative.parts
    if not category_dir.startswith("category__") or not category_dir[10:]:
        raise ValueError(f"invalid QY category directory: {category_dir}")
    category = category_dir[10:]
    asset_id = f"qy__{category}__{task_name}__{episode_id}"
    return category, task_name, episode_id, asset_id


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid QY {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"QY {label} must be an object: {path}")
    return value


def _safe_episode_relative_path(episode: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing {field}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must stay inside the episode")
    return episode.joinpath(*relative.parts)


def _parse_timebase(path: Path, episode: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json_object(path, "timebase")
    episode_id = str(payload.get("episode_id") or "").strip()
    if episode_id != episode.name:
        raise ValueError(
            "QY timebase episode_id does not match episode directory"
        )
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise ValueError("QY timebase videos must be an array")
    parsed: dict[str, dict[str, Any]] = {}
    seen_cameras: set[str] = set()
    for row_index, raw in enumerate(videos):
        if not isinstance(raw, Mapping):
            raise ValueError(f"QY timebase videos[{row_index}] must be an object")
        camera = str(raw.get("camera") or "").strip()
        if camera not in QY_CAMERAS:
            raise ValueError(f"unknown QY timebase camera: {camera or '<missing>'}")
        if camera in seen_cameras:
            raise ValueError(f"duplicate QY timebase camera: {camera}")
        seen_cameras.add(camera)
        if str(raw.get("status") or "").strip().lower() != "ok":
            continue
        video_path = _safe_episode_relative_path(
            episode, raw.get("video"), f"videos[{row_index}].video"
        )
        expected = episode / "videos" / f"{camera}.mp4"
        if _logical_absolute(video_path) != _logical_absolute(expected):
            raise ValueError(f"QY timebase video path does not match camera {camera}")
        frames = _integer(raw.get("frames"), "frames")
        width = _integer(raw.get("width"), "width")
        height = _integer(raw.get("height"), "height")
        fps = _finite_number(raw.get("fps"), "fps")
        start = _integer(raw.get("source_start_frame"), "source_start_frame")
        end = _integer(raw.get("source_end_frame"), "source_end_frame")
        timestamp_start = _finite_number(
            raw.get("source_start_timestamp"), "source_start_timestamp"
        )
        timestamp_end = _finite_number(
            raw.get("source_end_timestamp"), "source_end_timestamp"
        )
        if frames < 1 or width < 1 or height < 1 or fps <= 0:
            raise ValueError(f"invalid QY physical video metadata for {camera}")
        if start < 0 or end < start:
            raise ValueError(f"invalid QY source-frame range for {camera}")
        if timestamp_end < timestamp_start:
            raise ValueError(f"invalid QY timestamp range for {camera}")
        parsed[camera] = {
            "camera": camera,
            "video_path": video_path,
            "frames": frames,
            "width": width,
            "height": height,
            "fps": fps,
            "source_start_frame": start,
            "source_end_frame": end,
            "source_frame_count": end - start + 1,
            "source_start_timestamp": timestamp_start,
            "source_end_timestamp": timestamp_end,
        }
    return parsed


def load_qingyu_timebase(
    path: Path,
    *,
    episode_root: Path,
) -> dict[str, dict[str, Any]]:
    """Load the confirmed QY episode timebase without inferring frame mappings."""
    return _parse_timebase(path, episode_root)


def _common_source_range(
    timebase: Mapping[str, Mapping[str, Any]],
) -> tuple[int, int] | None:
    ranges = {
        (int(row["source_start_frame"]), int(row["source_end_frame"]))
        for row in timebase.values()
    }
    return next(iter(ranges)) if len(ranges) == 1 else None


def _select_camera(
    audits: Mapping[str, Mapping[str, Any]],
    *,
    requested: str | None,
    config: Mapping[str, Any],
) -> tuple[str | None, str | None, float | None, float | None, str]:
    scored = [
        (camera, float(audit["score"]))
        for camera, audit in audits.items()
        if audit.get("score") is not None
    ]
    eligible = [
        (camera, float(audit["score"]))
        for camera, audit in audits.items()
        if audit.get("eligible") is True and audit.get("score") is not None
    ]
    if requested is not None:
        if requested not in QY_CAMERAS:
            raise ValueError(f"unsupported QY primary camera: {requested}")
        if audits.get(requested, {}).get("explicit_camera_eligible") is True:
            selected_score = float(audits[requested]["score"])
            other_scores = sorted(
                (score for camera, score in scored if camera != requested),
                reverse=True,
            )
            return (
                requested,
                "explicit_config",
                selected_score,
                other_scores[0] if other_scores else None,
                "explicit_camera_valid",
            )
        return (
            None,
            None,
            None,
            None,
            str(
                audits.get(requested, {}).get("reason")
                or "explicit_camera_ineligible"
            ),
        )
    if not eligible:
        return None, None, None, None, "no_eligible_camera"
    scores = sorted((score for _, score in eligible), reverse=True)
    best_score = scores[0]
    second_score = scores[1] if len(scores) > 1 else None
    if (
        second_score is not None
        and best_score - second_score <= float(config["tie_tolerance"])
    ):
        return None, None, best_score, second_score, "best_camera_tied"
    winners = [camera for camera, score in eligible if score == best_score]
    if len(winners) != 1:
        return None, None, best_score, second_score, "best_camera_tied"
    other_scores = sorted(
        (score for camera, score in scored if camera != winners[0]),
        reverse=True,
    )
    return (
        winners[0],
        "coverage_recommendation",
        best_score,
        second_score if second_score is not None else (
            other_scores[0] if other_scores else None
        ),
        "unique_best_qualified_camera",
    )


def build_qingyu_manifest(
    root: Path,
    *,
    primary_camera: str | None = None,
    camera_selection: Mapping[str, Any] | None = None,
    max_assets: int | None = None,
    sam3_model: Path | None = None,
) -> list[dict[str, str]]:
    if max_assets is not None and max_assets < 1:
        raise ValueError("max_assets must be >= 1")
    logical_root = _logical_absolute(root)
    selection = camera_selection_config(camera_selection)
    episodes = discover_qingyu_episodes(logical_root)
    if max_assets is not None:
        episodes = episodes[:max_assets]
    rows: list[dict[str, str]] = []
    for episode in episodes:
        category, task_name, episode_id, asset_id = _episode_identity(
            logical_root, episode
        )
        paths = {
            name: episode / relative for name, relative in _REQUIRED_PATHS.items()
        }
        inventory = {
            name: {
                "logical_path": relative,
                "status": "present" if paths[name].is_file() else "missing",
            }
            for name, relative in _REQUIRED_PATHS.items()
        }
        required_present = all(
            item["status"] == "present"
            for name, item in inventory.items()
            if name != "timebase"
        )
        timebase: dict[str, dict[str, Any]] = {}
        timebase_error: str | None = None
        try:
            if paths["timebase"].is_file():
                timebase = _parse_timebase(paths["timebase"], episode)
                for record in timebase.values():
                    record["timebase_source"] = "supplier_episode_timebase"
                    record["frame_mapping_source"] = (
                        "explicit_source_frame_index_to_video_frame"
                    )
                    record["source_video_identity_assumed"] = False
        except ValueError as exc:
            timebase_error = str(exc)
        video_paths = {
            camera: episode / "videos" / f"{camera}.mp4" for camera in QY_CAMERAS
        }
        session = QingyuHandPoseSession(
            observations_path=paths["observations_2d"],
            trajectory_path=paths["trajectory_3d"],
        )
        camera_audits: dict[str, dict[str, Any]] = {}
        observations_error: str | None = None
        if (
            not paths["timebase"].is_file()
            and paths["observations_2d"].is_file()
        ):
            try:
                for camera in QY_CAMERAS:
                    derived = session.derive_camera_timebase(
                        camera=camera,
                        video_path=video_paths[camera],
                    )
                    if derived is not None:
                        timebase[camera] = derived
            except ValueError as exc:
                observations_error = str(exc)
        if paths["observations_2d"].is_file() and observations_error is None:
            try:
                for camera in QY_CAMERAS:
                    camera_audits[camera] = session.camera_coverage(
                        camera=camera,
                        timebase=timebase.get(camera),
                        video_available=video_paths[camera].is_file(),
                        config=selection,
                    )
            except ValueError as exc:
                observations_error = str(exc)
                camera_audits = {
                    camera: {
                        "camera": camera,
                        "eligible": False,
                        "video_available": video_paths[camera].is_file(),
                        "timebase_status": (
                            "valid" if camera in timebase else "input_missing"
                        ),
                        "frame_mapping_status": "mapping_unverified",
                        "source_video_identity_assumed": False,
                        "score": None,
                        "reason": "observations_2d_invalid",
                    }
                    for camera in QY_CAMERAS
                }
        elif observations_error is None:
            camera_audits = {
                camera: {
                    "camera": camera,
                    "eligible": False,
                    "video_available": video_paths[camera].is_file(),
                    "timebase_status": "valid" if camera in timebase else "input_missing",
                    "frame_mapping_status": "mapping_unverified",
                    "source_video_identity_assumed": False,
                    "score": None,
                    "reason": "observations_2d_missing",
                }
                for camera in QY_CAMERAS
            }
        else:
            camera_audits = {
                camera: {
                    "camera": camera,
                    "eligible": False,
                    "explicit_camera_eligible": False,
                    "video_available": video_paths[camera].is_file(),
                    "timebase_status": (
                        "derived"
                        if timebase.get(camera, {}).get("timebase_source")
                        == "derived_from_video_and_observations_2d"
                        else "valid" if camera in timebase else "input_missing"
                    ),
                    "frame_mapping_status": "mapping_unverified",
                    "source_video_identity_assumed": False,
                    "score": None,
                    "reason": "observations_2d_invalid",
                }
                for camera in QY_CAMERAS
            }
        selected, selected_source, best_score, second_score, recommendation_reason = _select_camera(
            camera_audits,
            requested=primary_camera,
            config=selection,
        )
        selected_timebase = timebase.get(selected) if selected is not None else None
        common_range = _common_source_range(timebase)
        source_range = (
            (
                int(selected_timebase["source_start_frame"]),
                int(selected_timebase["source_end_frame"]),
            )
            if selected_timebase is not None
            else common_range
        )
        trajectory_error: str | None = None
        if paths["trajectory_3d"].is_file() and source_range is not None:
            try:
                trajectory_audit = session.audit_3d(
                    start_frame=source_range[0], end_frame=source_range[1]
                )
            except ValueError as exc:
                trajectory_error = str(exc)
                trajectory_audit = {
                    "row_count": 0,
                    "valid_row_count": 0,
                    "status": "input_invalid",
                    "coverage_status": "input_invalid",
                    "read_error": trajectory_error,
                }
        else:
            trajectory_audit = {
                "row_count": 0,
                "valid_row_count": 0,
                "status": "no_valid_output",
                "coverage_status": "no_valid_output",
            }
        coverage_status = str(
            trajectory_audit.get("coverage_status")
            or ("no_valid_output" if not trajectory_audit.get("valid_row_count") else "unverified")
        )
        skeleton_3d_status = str(trajectory_audit["status"])
        if skeleton_3d_status == "valid":
            if coverage_status == "no_valid_output":
                skeleton_3d_status = "no_valid_output"
            elif coverage_status == "sparse":
                skeleton_3d_status = "input_missing"
        coordinate_system_status = "input_missing"
        if paths["coordinate_system"].is_file():
            try:
                _read_json_object(paths["coordinate_system"], "coordinate system")
                coordinate_system_status = (
                    "contract_declared_camera_frame_meters_schema_unverified"
                )
            except ValueError:
                coordinate_system_status = "input_invalid"
        if observations_error is not None:
            adapter_status = "input_invalid"
            reason = "observations_2d_invalid"
        elif timebase_error is not None:
            adapter_status = "input_invalid"
            reason = "timebase_invalid"
        elif trajectory_error is not None:
            adapter_status = "input_invalid"
            reason = "trajectory_3d_invalid"
        elif coordinate_system_status == "input_invalid":
            adapter_status = "input_invalid"
            reason = "coordinate_system_invalid"
        elif selected is None:
            adapter_status = "input_missing"
            reason = (
                recommendation_reason
                if primary_camera is not None
                else "primary_camera_missing"
            )
        elif not required_present:
            adapter_status = "input_missing"
            reason = "required_file_missing"
        elif skeleton_3d_status == "no_valid_output":
            adapter_status = "input_missing"
            reason = "no_valid_3d_skeleton"
        elif skeleton_3d_status == "input_invalid":
            adapter_status = "input_invalid"
            reason = "invalid_3d_skeleton"
        elif skeleton_3d_status == "input_missing":
            adapter_status = "input_missing"
            reason = "incomplete_3d_skeleton"
        else:
            adapter_status = "ready"
            reason = "ready"
        start_frame = source_range[0] if source_range is not None else None
        end_frame = source_range[1] if source_range is not None else None
        source_frame_count = (
            end_frame - start_frame + 1
            if start_frame is not None and end_frame is not None
            else None
        )
        selected_audit = camera_audits.get(selected, {}) if selected else {}
        row = {
            "schema_version": "supplier_manifest.qy.v1",
            "supplier": "qy",
            "supplier_id": "qy",
            "supplier_name": "QY",
            "supplier_alias": "qingyu",
            "asset_id": asset_id,
            "source_granularity": "episode",
            "category": category,
            "task_name": task_name,
            "episode_id": episode_id,
            "episode_root": str(episode),
            "episode_manifest_path": str(paths["episode_manifest"]),
            "observations_2d_path": str(paths["observations_2d"]),
            "trajectory_3d_path": str(paths["trajectory_3d"]),
            "coordinate_system_path": str(paths["coordinate_system"]),
            "quality_path": str(paths["quality"]),
            "timebase_path": (
                str(paths["timebase"]) if paths["timebase"].is_file() else ""
            ),
            "semantic_path": str(paths["semantic"]),
            "review_video_path": str(paths["review_video"]),
            "sam3_model_path": (
                str(_logical_absolute(sam3_model)) if sam3_model is not None else ""
            ),
            "primary_camera": selected or "",
            "requested_primary_camera": primary_camera or "",
            "primary_camera_source": selected_source or "",
            "primary_video_path": str(video_paths[selected]) if selected else "",
            **{
                f"{camera}_video_path": str(video_paths[camera])
                for camera in QY_CAMERAS
            },
            "video_frame_count": (
                str(selected_timebase["frames"]) if selected_timebase is not None else ""
            ),
            "video_width": (
                str(selected_timebase["width"]) if selected_timebase is not None else ""
            ),
            "video_height": (
                str(selected_timebase["height"]) if selected_timebase is not None else ""
            ),
            "source_frame_count": str(source_frame_count) if source_frame_count is not None else "",
            "start_frame": str(start_frame) if start_frame is not None else "",
            "end_frame": str(end_frame) if end_frame is not None else "",
            "fps": str(selected_timebase["fps"]) if selected_timebase is not None else "",
            "frame_coordinate_system": "source_inclusive",
            "timebase_status": (
                "input_invalid" if timebase_error is not None else (
                    str(selected_audit.get("timebase_status") or "input_missing")
                )
            ),
            "timebase_source": (
                str(selected_timebase.get("timebase_source") or "")
                if selected_timebase is not None
                else ""
            ),
            "frame_mapping_status": str(
                selected_audit.get("frame_mapping_status") or "mapping_unverified"
            ),
            "frame_mapping_source": (
                str(selected_timebase.get("frame_mapping_source") or "")
                if selected_timebase is not None
                else ""
            ),
            "source_video_identity_assumed": _json(False),
            "skeleton_2d_status": (
                "valid"
                if selected_audit.get("frame_mapping_status") == "verified"
                else "input_missing"
            ),
            "skeleton_3d_status": skeleton_3d_status,
            "skeleton_3d_valid_row_count": str(trajectory_audit.get("valid_row_count", 0)),
            "skeleton_3d_coverage_status": coverage_status,
            "joint_topology_status": "unverified",
            "coordinate_system_status": coordinate_system_status,
            "reference_camera_status": str(
                trajectory_audit.get("reference_camera_status") or "mapping_unverified"
            ),
            "camera_coverage": _json(camera_audits),
            "camera_recommendation_score": "" if best_score is None else str(best_score),
            "camera_second_best_score": "" if second_score is None else str(second_score),
            "camera_recommendation_reason": recommendation_reason,
            "camera_selection_config": _json(selection),
            "skeleton_3d_audit": _json(trajectory_audit),
            "file_inventory": _json(inventory),
            "adapter_status": adapter_status,
            "reason": reason,
        }
        rows.append({column: str(row.get(column, "")) for column in MANIFEST_COLUMNS})
    return rows


def write_qingyu_manifest(rows: list[dict[str, str]], output_dir: Path) -> Path:
    path = output_dir / "manifests" / "supplier_manifest_qy.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


__all__ = [
    "DEFAULT_CAMERA_SELECTION_CONFIG",
    "MANIFEST_COLUMNS",
    "QY_CAMERAS",
    "build_qingyu_manifest",
    "camera_selection_config",
    "discover_qingyu_episodes",
    "load_qingyu_timebase",
    "write_qingyu_manifest",
]
