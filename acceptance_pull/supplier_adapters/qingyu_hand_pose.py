"""Strict QY hand-pose loading with explicit source-frame lineage."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from acceptance_pull.supplier_adapters.structured_audit import (
    audit_video_metadata,
)
from qc_common.types import ClipInputs


QY_HANDS = ("left", "right")
_OBSERVATION_COLUMNS = frozenset(
    {
        "camera",
        "video_frame",
        "source_frame_index",
        "timestamp",
        "pred_keypoints_2d",
    }
)
_TRAJECTORY_COLUMNS = frozenset(
    {
        "source_step",
        "timestamp_seconds",
        "hand",
        "keypoints_3d_ref",
        "reference_camera",
    }
)


class QingyuSourceReadError(ValueError):
    """A QY source exists but cannot be decoded as the declared format."""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _text(value: Any) -> str:
    return "" if _is_missing(value) else str(value).strip()


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or _is_missing(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _finite_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _hand(value: Any) -> str | None:
    text = _text(value).lower()
    aliases = {
        "left": "left",
        "right": "right",
        "l": "left",
        "r": "right",
        "left_hand": "left",
        "right_hand": "right",
    }
    return aliases.get(text)


def _strict_points(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if _is_missing(value):
        return None
    try:
        array = np.asarray(value)
        if array.ndim == 1 and len(array) > 0:
            array = np.stack(list(array), axis=0)
        array = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if array.shape != shape or not np.isfinite(array).all():
        return None
    return array


def _topology_agnostic_keypoints(points: np.ndarray) -> dict[str, np.ndarray]:
    """Expose stable raw indices without assigning anatomical joint names."""
    return {
        f"{side}QYJoint{joint_index:02d}": np.asarray(
            points[:, hand_index, joint_index, :], dtype=np.float32
        )
        for hand_index, side in enumerate(QY_HANDS)
        for joint_index in range(21)
    }


def _observation_priority(row: Mapping[str, Any]) -> tuple[int, float, int]:
    status_priority = {
        "ok": 3,
        "valid": 3,
        "pass": 3,
        "review": 2,
        "warn": 1,
    }
    score = row.get("score")
    return (
        status_priority.get(str(row.get("status") or "").lower(), 0),
        float(score) if score is not None else float("-inf"),
        -int(row["row_index"]),
    )


def _equivalent_observation(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    left_points = left.get("points")
    right_points = right.get("points")
    if not isinstance(left_points, np.ndarray) or not isinstance(
        right_points, np.ndarray
    ):
        return False
    left_timestamp = left.get("timestamp")
    right_timestamp = right.get("timestamp")
    if left_timestamp is None or right_timestamp is None:
        return False
    return bool(
        np.array_equal(left_points, right_points)
        and math.isclose(
            float(left_timestamp),
            float(right_timestamp),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )


def _deduplicate_observations(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Deduplicate only equivalent rows for the same camera-frame hand.

    Mapping repeats across left/right hands are deliberately unrelated to this
    grouping.  A group containing different strict points or timestamps is
    rejected as a whole so no conflicting observation is selected silently.
    """
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in records:
        source_frame = row.get("source_frame_index")
        video_frame = row.get("video_frame")
        hand = row.get("hand")
        if (
            source_frame is None
            or video_frame is None
            or hand not in QY_HANDS
            or row.get("points") is None
            or row.get("timestamp") is None
        ):
            continue
        grouped.setdefault(
            (int(source_frame), int(video_frame), str(hand)), []
        ).append(row)

    selected: list[dict[str, Any]] = []
    equivalent_duplicate_count = 0
    conflicting_group_count = 0
    for key in sorted(grouped):
        rows = grouped[key]
        reference = rows[0]
        if not all(
            _equivalent_observation(reference, candidate)
            for candidate in rows[1:]
        ):
            conflicting_group_count += 1
            continue
        selected.append(max(rows, key=_observation_priority))
        equivalent_duplicate_count += len(rows) - 1
    return selected, equivalent_duplicate_count, conflicting_group_count


class QingyuHandPoseSession:
    """Read each QY Parquet once and expose strict indexed views."""

    def __init__(
        self,
        *,
        observations_path: Path,
        trajectory_path: Path | None = None,
    ) -> None:
        self.observations_path = observations_path
        self.trajectory_path = trajectory_path
        self._observations: pd.DataFrame | None = None
        self._trajectory: pd.DataFrame | None = None
        self._observation_records: tuple[dict[str, Any], ...] | None = None
        self._trajectory_audit: dict[str, Any] | None = None
        self._trajectory_index: dict[tuple[int, str], dict[str, Any]] | None = None
        self._timebase_derivation_failures: dict[str, str] = {}

    @property
    def observations(self) -> pd.DataFrame:
        if self._observations is None:
            try:
                self._observations = pd.read_parquet(self.observations_path)
            except Exception as exc:
                raise QingyuSourceReadError(
                    f"invalid QY observations_2d parquet: {exc}"
                ) from exc
        return self._observations

    @property
    def trajectory(self) -> pd.DataFrame:
        if self.trajectory_path is None:
            raise ValueError("QY trajectory_path is required for 3D loading")
        if self._trajectory is None:
            try:
                self._trajectory = pd.read_parquet(self.trajectory_path)
            except Exception as exc:
                raise QingyuSourceReadError(
                    f"invalid QY trajectory_3d parquet: {exc}"
                ) from exc
        return self._trajectory

    def _parsed_observations(self) -> tuple[dict[str, Any], ...]:
        if self._observation_records is not None:
            return self._observation_records
        frame = self.observations
        hand_column = (
            "hand/current_hand"
            if "hand/current_hand" in frame.columns
            else "hand" if "hand" in frame.columns else None
        )
        missing_columns = sorted(_OBSERVATION_COLUMNS - set(frame.columns))
        records: list[dict[str, Any]] = []
        for row_index, row in frame.iterrows():
            points = _strict_points(row.get("pred_keypoints_2d"), (21, 2))
            finite_joint_count = 21 if points is not None else 0
            records.append(
                {
                    "row_index": int(row_index),
                    "camera": _text(row.get("camera")),
                    "hand": _hand(row.get(hand_column)) if hand_column else None,
                    "video_frame": _integer(row.get("video_frame")),
                    "source_frame_index": _integer(row.get("source_frame_index")),
                    "timestamp": _finite_float(row.get("timestamp")),
                    "status": _text(row.get("status")).lower(),
                    "score": _finite_float(row.get("score")),
                    "points": points,
                    "finite_joint_count": finite_joint_count,
                    "shape_valid": points is not None,
                    "missing_columns": missing_columns
                    + ([] if hand_column else ["hand/current_hand"]),
                }
            )
        self._observation_records = tuple(records)
        return self._observation_records

    def audit_2d(self) -> dict[str, Any]:
        records = self._parsed_observations()
        return {
            "row_count": len(records),
            "valid_shape_row_count": sum(row["shape_valid"] for row in records),
            "invalid_shape_row_count": sum(not row["shape_valid"] for row in records),
            "cameras": sorted({str(row["camera"]) for row in records if row["camera"]}),
        }

    def derive_camera_timebase(
        self,
        *,
        camera: str,
        video_path: Path,
    ) -> dict[str, Any] | None:
        """Derive physical video metadata plus the observed source range.

        This is deliberately not a supplier timebase.  Source/video pairing
        remains the explicit mapping carried by each observations_2d row.
        Conflicts are audited by :meth:`camera_coverage` rather than repaired.
        """
        records = [
            row for row in self._parsed_observations() if row["camera"] == camera
        ]
        source_frames = [
            int(row["source_frame_index"])
            for row in records
            if row["source_frame_index"] is not None
        ]
        timestamps = [
            float(row["timestamp"])
            for row in records
            if row["timestamp"] is not None
        ]
        if not records:
            self._timebase_derivation_failures[camera] = (
                "primary_camera_2d_missing"
            )
            return None
        if not source_frames or not timestamps:
            self._timebase_derivation_failures[camera] = "frame_mapping_missing"
            return None
        video_metadata = audit_video_metadata(video_path)
        if video_metadata.get("status") != "pass":
            self._timebase_derivation_failures[camera] = str(
                video_metadata.get("reason") or "video_metadata_invalid"
            )
            return None
        frames = int(video_metadata["frame_count"])
        width = int(video_metadata["width"])
        height = int(video_metadata["height"])
        fps = float(video_metadata["fps"])
        self._timebase_derivation_failures.pop(camera, None)
        start = min(source_frames)
        end = max(source_frames)
        return {
            "camera": camera,
            "video_path": video_path,
            "frames": frames,
            "width": width,
            "height": height,
            "fps": fps,
            "source_start_frame": start,
            "source_end_frame": end,
            "source_frame_count": end - start + 1,
            "source_start_timestamp": min(timestamps),
            "source_end_timestamp": max(timestamps),
            "timebase_source": "derived_from_video_and_observations_2d",
            "frame_mapping_source": (
                "explicit_source_frame_index_to_video_frame"
            ),
            "source_video_identity_assumed": False,
        }

    def camera_coverage(
        self,
        *,
        camera: str,
        timebase: Mapping[str, Any] | None,
        video_available: bool,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        records = [
            row for row in self._parsed_observations() if row["camera"] == camera
        ]
        row_count = len(records)
        result: dict[str, Any] = {
            "camera": camera,
            "row_count": row_count,
            "video_available": bool(video_available),
            "timebase_status": (
                "derived"
                if timebase is not None
                and timebase.get("timebase_source")
                == "derived_from_video_and_observations_2d"
                else "valid" if timebase is not None else "input_missing"
            ),
            "timebase_source": (
                timebase.get("timebase_source") if timebase is not None else None
            ),
            "frame_mapping_source": (
                timebase.get("frame_mapping_source")
                if timebase is not None
                else None
            ),
            "frame_mapping_status": "mapping_unverified",
            "eligible": False,
            "explicit_camera_eligible": False,
            "source_video_identity_assumed": False,
        }
        if timebase is None:
            if not records:
                reason = "primary_camera_2d_missing"
            elif not video_available:
                reason = "primary_video_missing"
            else:
                reason = self._timebase_derivation_failures.get(
                    camera,
                    "video_metadata_invalid_or_timebase_missing",
                )
            result.update(
                {
                    "video_frame_count": None,
                    "source_frame_count": None,
                    "invalid_shape_row_count": sum(
                        not row["shape_valid"] for row in records
                    ),
                    "score": None,
                    "reason": reason,
                }
            )
            return result

        start = int(timebase["source_start_frame"])
        end = int(timebase["source_end_frame"])
        video_frames = int(timebase["frames"])
        fps = float(timebase["fps"])
        timestamp_start = float(timebase["source_start_timestamp"])
        timestamp_end = float(timebase["source_end_timestamp"])
        source_count = end - start + 1
        tolerance = 1.0 / fps
        mapping_rows: list[dict[str, Any]] = []
        invalid_shape_count = 0
        source_frame_out_of_range_count = 0
        video_frame_out_of_range_count = 0
        timestamp_out_of_range_count = 0
        missing_mapping_count = 0
        invalid_hand_count = 0
        for row in records:
            if not row["shape_valid"]:
                invalid_shape_count += 1
            hand = row["hand"]
            source_frame = row["source_frame_index"]
            video_frame = row["video_frame"]
            timestamp = row["timestamp"]
            if hand not in QY_HANDS:
                invalid_hand_count += 1
            if source_frame is None or video_frame is None or timestamp is None:
                missing_mapping_count += 1
                continue
            if not start <= source_frame <= end:
                source_frame_out_of_range_count += 1
                continue
            if not 0 <= video_frame < video_frames:
                video_frame_out_of_range_count += 1
                continue
            if not timestamp_start - tolerance <= timestamp <= timestamp_end + tolerance:
                timestamp_out_of_range_count += 1
                continue
            mapping_rows.append(row)

        pair_counts: dict[tuple[int, int], int] = {}
        source_targets: dict[int, set[int]] = {}
        video_targets: dict[int, set[int]] = {}
        for row in mapping_rows:
            source_frame = int(row["source_frame_index"])
            video_frame = int(row["video_frame"])
            pair = (source_frame, video_frame)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            source_targets.setdefault(source_frame, set()).add(video_frame)
            video_targets.setdefault(video_frame, set()).add(source_frame)

        source_conflict_keys = {
            source for source, targets in source_targets.items() if len(targets) > 1
        }
        video_conflict_keys = {
            video for video, targets in video_targets.items() if len(targets) > 1
        }
        source_conflict_count = len(source_conflict_keys)
        video_conflict_count = len(video_conflict_keys)
        conflict_count = source_conflict_count + video_conflict_count
        duplicate_count = sum(count - 1 for count in pair_counts.values())
        repeated_pair_count = sum(count > 1 for count in pair_counts.values())
        unique_pair_count = len(pair_counts)
        global_source_to_video = {
            source: next(iter(targets))
            for source, targets in source_targets.items()
            if len(targets) == 1
        }

        observation_rows = [
            row
            for row in mapping_rows
            if row["hand"] in QY_HANDS and row["points"] is not None
        ]
        (
            deduplicated_rows,
            equivalent_duplicate_count,
            conflicting_observation_count,
        ) = _deduplicate_observations(observation_rows)
        by_hand = {
            hand: {
                int(row["source_frame_index"])
                for row in deduplicated_rows
                if row["hand"] == hand
            }
            for hand in QY_HANDS
        }
        left_coverage = len(by_hand["left"]) / source_count
        right_coverage = len(by_hand["right"]) / source_count
        minimum_hand_coverage = min(left_coverage, right_coverage)
        both_hand_coverage = len(by_hand["left"] & by_hand["right"]) / source_count
        normalized_observation_count = max(
            row_count - equivalent_duplicate_count,
            0,
        )
        finite_joint_count = sum(
            int(row["finite_joint_count"]) for row in deduplicated_rows
        )
        valid_joint_ratio = (
            finite_joint_count / (normalized_observation_count * 21)
            if normalized_observation_count
            else 0.0
        )
        normalized_mapping_count = max(
            len(mapping_rows) - equivalent_duplicate_count,
            0,
        )
        timeline_match_ratio = (
            normalized_mapping_count / normalized_observation_count
            if normalized_observation_count
            else 0.0
        )
        invalid_mapping_count = (
            missing_mapping_count
            + source_frame_out_of_range_count
            + video_frame_out_of_range_count
            + timestamp_out_of_range_count
        )
        mapping_verified = bool(
            row_count
            and invalid_mapping_count == 0
            and conflict_count == 0
            and unique_pair_count > 0
        )
        observation_status = (
            "input_invalid"
            if conflicting_observation_count
            else "valid_deduplicated"
            if equivalent_duplicate_count
            else "valid"
        )
        weights = config["weights"]
        score = (
            float(weights["minimum_hand_coverage"]) * minimum_hand_coverage
            + float(weights["both_hand_coverage"]) * both_hand_coverage
            + float(weights["valid_joint_ratio"]) * valid_joint_ratio
            + float(weights["timeline_match_ratio"]) * timeline_match_ratio
        )
        eligible = bool(
            video_available
            and mapping_verified
            and invalid_shape_count == 0
            and invalid_hand_count == 0
            and conflicting_observation_count == 0
            and minimum_hand_coverage
            >= float(config["minimum_hand_coverage"])
            and both_hand_coverage
            >= float(config["minimum_both_hand_coverage"])
            and valid_joint_ratio >= float(config["minimum_valid_joint_ratio"])
            and timeline_match_ratio
            >= float(config["minimum_timeline_match_ratio"])
        )
        explicit_camera_eligible = bool(
            video_available
            and mapping_verified
            and invalid_shape_count == 0
            and invalid_hand_count == 0
            and conflicting_observation_count == 0
            and bool(deduplicated_rows)
        )
        if not records:
            reason = "primary_camera_2d_missing"
        elif not video_available:
            reason = "primary_video_missing"
        elif invalid_shape_count == row_count:
            reason = "no_valid_2d_points"
        elif source_conflict_count:
            reason = "source_to_video_conflict"
        elif video_conflict_count:
            reason = "video_to_source_conflict"
        elif video_frame_out_of_range_count:
            reason = "video_frame_out_of_range"
        elif source_frame_out_of_range_count:
            reason = "source_frame_out_of_range"
        elif timestamp_out_of_range_count:
            reason = "timestamp_mapping_invalid"
        elif missing_mapping_count:
            reason = "frame_mapping_missing"
        elif invalid_hand_count:
            reason = "hand_identity_invalid"
        elif conflicting_observation_count:
            reason = "conflicting_same_hand_observations"
        else:
            reason = "eligible" if eligible else "minimum_qualification_not_met"
        result.update(
            {
                "video_frame_count": video_frames,
                "video_width": (
                    int(timebase["width"]) if timebase.get("width") is not None else None
                ),
                "video_height": (
                    int(timebase["height"])
                    if timebase.get("height") is not None
                    else None
                ),
                "fps": fps,
                "source_start_frame": start,
                "source_end_frame": end,
                "source_frame_count": source_count,
                "left_hand_coverage": left_coverage,
                "right_hand_coverage": right_coverage,
                "minimum_hand_coverage": minimum_hand_coverage,
                "both_hand_coverage": both_hand_coverage,
                "valid_joint_ratio": valid_joint_ratio,
                "timeline_match_ratio": timeline_match_ratio,
                "explicit_mapping_valid_count": len(mapping_rows),
                "invalid_mapping_row_count": invalid_mapping_count,
                "missing_mapping_row_count": missing_mapping_count,
                "source_frame_out_of_range_count": source_frame_out_of_range_count,
                "video_frame_out_of_range_count": video_frame_out_of_range_count,
                "timestamp_out_of_range_count": timestamp_out_of_range_count,
                "invalid_hand_row_count": invalid_hand_count,
                "invalid_shape_row_count": invalid_shape_count,
                "mapping_conflict_count": conflict_count,
                "source_to_video_conflict_count": source_conflict_count,
                "video_to_source_conflict_count": video_conflict_count,
                "duplicate_mapping_count": duplicate_count,
                "repeated_mapping_pair_count": repeated_pair_count,
                "unique_mapping_pair_count": unique_pair_count,
                "mapping_warning": (
                    "repeated_identical_mapping_pairs"
                    if duplicate_count
                    else None
                ),
                "deduplicated_observation_row_count": len(deduplicated_rows),
                "equivalent_duplicate_observation_count": (
                    equivalent_duplicate_count
                ),
                "conflicting_same_hand_observation_count": (
                    conflicting_observation_count
                ),
                "observation_status": observation_status,
                "observation_deduplication_policy": (
                    "equivalent_strict_points_and_timestamp_then_status_score_row_index"
                ),
                "source_to_video_mapping": {
                    str(source): video
                    for source, video in sorted(global_source_to_video.items())
                },
                "frame_mapping_status": (
                    "verified" if mapping_verified else "mapping_unverified"
                ),
                "score": score,
                "eligible": eligible,
                "explicit_camera_eligible": explicit_camera_eligible,
                "reason": reason,
            }
        )
        return result

    def _parse_trajectory(self) -> None:
        if self._trajectory_audit is not None:
            return
        frame = self.trajectory
        missing_columns = sorted(_TRAJECTORY_COLUMNS - set(frame.columns))
        candidates: dict[tuple[int, str], dict[str, Any]] = {}
        duplicate_keys: set[tuple[int, str]] = set()
        invalid_row_count = 0
        null_row_count = 0
        invalid_reference_camera_count = 0
        reference_cameras: set[str] = set()
        for row_index, row in frame.iterrows():
            raw_values = (
                row.get("source_step"),
                row.get("hand"),
                row.get("keypoints_3d_ref"),
            )
            if all(_is_missing(value) for value in raw_values):
                null_row_count += 1
                continue
            step = _integer(row.get("source_step"))
            hand = _hand(row.get("hand"))
            points = _strict_points(row.get("keypoints_3d_ref"), (21, 3))
            timestamp = _finite_float(row.get("timestamp_seconds"))
            reference_camera = _text(row.get("reference_camera"))
            reference_valid = reference_camera in {
                "left_cam_left",
                "left_cam_right",
                "mid_cam_left",
                "mid_cam_right",
                "right_cam_left",
                "right_cam_right",
            }
            if not reference_valid:
                invalid_reference_camera_count += 1
            if (
                missing_columns
                or step is None
                or hand is None
                or points is None
                or not reference_valid
            ):
                invalid_row_count += 1
                continue
            reference_cameras.add(reference_camera)
            key = (step, hand)
            if key in candidates:
                duplicate_keys.add(key)
                continue
            candidates[key] = {
                "row_index": int(row_index),
                "source_step": step,
                "hand": hand,
                "points": points,
                "timestamp_seconds": timestamp,
                "reference_camera": reference_camera,
                "quality_tier": _text(row.get("quality_tier")) or None,
                "trajectory_quality": _text(row.get("trajectory_quality")) or None,
                "joint_mean_reprojection_error_px": row.get(
                    "joint_mean_reprojection_error_px"
                ),
                "joint_max_reprojection_error_px": row.get(
                    "joint_max_reprojection_error_px"
                ),
            }
        for key in duplicate_keys:
            candidates.pop(key, None)
        duplicate_count = len(duplicate_keys)
        no_substantive_rows = len(frame) == null_row_count
        reference_camera_status = (
            "no_valid_output"
            if no_substantive_rows
            else "verified"
            if len(reference_cameras) == 1
            and invalid_reference_camera_count == 0
            and not missing_columns
            else "mapping_invalid"
        )
        if reference_camera_status != "verified":
            candidates.clear()
        valid_count = len(candidates)
        if reference_camera_status == "mapping_invalid":
            status = "input_invalid"
        elif valid_count == 0 and invalid_row_count == 0 and duplicate_count == 0:
            status = "no_valid_output"
        elif missing_columns or invalid_row_count or duplicate_count:
            status = "input_invalid"
        else:
            status = "valid"
        self._trajectory_index = candidates
        self._trajectory_audit = {
            "row_count": len(frame),
            "valid_row_count": valid_count,
            "invalid_row_count": invalid_row_count,
            "null_row_count": null_row_count,
            "duplicate_hand_step_count": duplicate_count,
            "missing_columns": missing_columns,
            "reference_camera_status": reference_camera_status,
            "reference_camera": (
                next(iter(reference_cameras))
                if reference_camera_status == "verified"
                else None
            ),
            "invalid_reference_camera_count": invalid_reference_camera_count,
            "status": status,
        }

    def audit_3d(
        self,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> dict[str, Any]:
        self._parse_trajectory()
        assert self._trajectory_audit is not None
        assert self._trajectory_index is not None
        audit = dict(self._trajectory_audit)
        if start_frame is None or end_frame is None:
            return audit
        required_count = (end_frame - start_frame + 1) * 2
        in_range = {
            key: row
            for key, row in self._trajectory_index.items()
            if start_frame <= key[0] <= end_frame
        }
        audit["valid_row_count"] = len(in_range)
        audit["expected_hand_step_count"] = required_count
        audit["out_of_range_valid_row_count"] = len(self._trajectory_index) - len(in_range)
        if audit["out_of_range_valid_row_count"]:
            audit["status"] = "input_invalid"
            audit["coverage_status"] = "mapping_invalid"
        elif not in_range:
            audit["coverage_status"] = "no_valid_output"
        elif len(in_range) < required_count:
            audit["coverage_status"] = "sparse"
        else:
            audit["coverage_status"] = "complete"
        return audit

    def direct_2d_index(
        self,
        *,
        camera: str,
        start_frame: int,
        end_frame: int,
        video_frame_count: int,
        timestamp_start: float,
        timestamp_end: float,
        fps: float,
    ) -> dict[tuple[int, str], dict[str, Any]]:
        timebase = {
            "source_start_frame": start_frame,
            "source_end_frame": end_frame,
            "frames": video_frame_count,
            "fps": fps,
            "source_start_timestamp": timestamp_start,
            "source_end_timestamp": timestamp_end,
        }
        # Thresholds are irrelevant to mapping validation here.
        audit = self.camera_coverage(
            camera=camera,
            timebase=timebase,
            video_available=True,
            config={
                "minimum_hand_coverage": 0.0,
                "minimum_both_hand_coverage": 0.0,
                "minimum_valid_joint_ratio": 0.0,
                "minimum_timeline_match_ratio": 0.0,
                "weights": {
                    "minimum_hand_coverage": 0.25,
                    "both_hand_coverage": 0.25,
                    "valid_joint_ratio": 0.25,
                    "timeline_match_ratio": 0.25,
                },
            },
        )
        if (
            audit["frame_mapping_status"] != "verified"
            or audit.get("observation_status") == "input_invalid"
        ):
            return {}
        records = [
            row
            for row in self._parsed_observations()
            if row["camera"] == camera
            and row["hand"] in QY_HANDS
            and row["points"] is not None
            and row["source_frame_index"] is not None
            and row["video_frame"] is not None
            and row["timestamp"] is not None
        ]
        deduplicated, _, conflicts = _deduplicate_observations(records)
        if conflicts:
            return {}
        return {
            (int(row["source_frame_index"]), str(row["hand"])): {
                "video_frame": int(row["video_frame"]),
                "timestamp": float(row["timestamp"]),
                "keypoints_2d": np.asarray(row["points"], dtype=np.float32),
            }
            for row in deduplicated
        }

    def build_clip(
        self,
        *,
        asset_id: str,
        start_frame: int,
        end_frame: int,
        primary_camera: str,
        fps: float,
        episode_idx: int,
        text_label: Mapping[str, Any] | None = None,
        coordinate_system_status: str = "mapping_unverified",
    ) -> ClipInputs:
        self._parse_trajectory()
        assert self._trajectory_index is not None
        frame_count = end_frame - start_frame + 1
        points = np.full((frame_count, 2, 21, 3), np.nan, dtype=np.float32)
        valid = np.zeros((frame_count, 2, 21), dtype=np.bool_)
        timestamps_ns = np.full(frame_count, -1, dtype=np.int64)
        supplier_signals: list[dict[str, Any]] = []
        for (source_step, hand), row in self._trajectory_index.items():
            if not start_frame <= source_step <= end_frame:
                continue
            offset = source_step - start_frame
            hand_index = QY_HANDS.index(hand)
            points[offset, hand_index] = row["points"]
            valid[offset, hand_index] = True
            timestamp = row["timestamp_seconds"]
            if timestamp is not None:
                value = int(round(float(timestamp) * 1_000_000_000))
                if timestamps_ns[offset] in {-1, value}:
                    timestamps_ns[offset] = value
                else:
                    timestamps_ns[offset] = -1
            supplier_signals.append(
                {
                    "source_step": source_step,
                    "hand": hand,
                    "quality_tier": row["quality_tier"],
                    "trajectory_quality": row["trajectory_quality"],
                    "joint_mean_reprojection_error_px": row[
                        "joint_mean_reprojection_error_px"
                    ],
                    "joint_max_reprojection_error_px": row[
                        "joint_max_reprojection_error_px"
                    ],
                }
            )
        label = dict(text_label or {})
        opaque_keypoints = _topology_agnostic_keypoints(points)
        clip = ClipInputs(
            episode_idx=episode_idx,
            frame_indices=list(range(start_frame, end_frame + 1)),
            # QY has not supplied a confirmed 21-joint index topology.  Keep
            # the strict canonical [T,2,21,3] arrays for existence checks,
            # but do not invent anatomical names for topology-aware checks.
            keypoints=opaque_keypoints,
            instruction=str(label.get("task_name") or label.get("task") or ""),
            text_label=label,
            hand_keypoints_3d=points,
            hand_joint_valid_3d=valid,
            timestamps_ns=timestamps_ns,
            fps=float(fps),
        )
        setattr(clip, "asset_id", asset_id)
        setattr(clip, "supplier_id", "qy")
        assert self.trajectory_path is not None
        setattr(clip, "source_path", str(self.trajectory_path))
        setattr(clip, "clip_start_frame", start_frame)
        setattr(clip, "clip_end_frame", end_frame)
        setattr(clip, "source_frame_indices", tuple(range(start_frame, end_frame + 1)))
        setattr(clip, "source_frame_count", frame_count)
        setattr(clip, "primary_camera", primary_camera)
        setattr(clip, "sparse_trajectory", not valid.all())
        setattr(clip, "interpolation_applied", False)
        setattr(
            clip,
            "supplier_quality_signal",
            {"policy": "auxiliary_only", "rows": supplier_signals},
        )
        setattr(clip, "morphology_status", "not_ready_topology")
        setattr(clip, "joint_topology_status", "unverified")
        setattr(
            clip,
            "topology_agnostic_joint_names",
            tuple(sorted(opaque_keypoints)),
        )
        reference_status = str(
            (self._trajectory_audit or {}).get("reference_camera_status") or ""
        )
        metric_coordinate_status = (
            "verified_declared_camera_frame_meters_stable_reference"
            if coordinate_system_status
            == "contract_declared_camera_frame_meters_schema_unverified"
            and reference_status == "verified"
            else "mapping_unverified"
        )
        setattr(clip, "metric_coordinate_status", metric_coordinate_status)
        return clip


def load_qingyu_clip(
    row: Mapping[str, Any],
    *,
    episode_idx: int,
    session: QingyuHandPoseSession | None = None,
) -> ClipInputs:
    observations_path = Path(str(row["observations_2d_path"]))
    trajectory_path = Path(str(row["trajectory_3d_path"]))
    session = session or QingyuHandPoseSession(
        observations_path=observations_path,
        trajectory_path=trajectory_path,
    )
    start_frame = int(row["start_frame"])
    end_frame = int(row["end_frame"])
    primary_camera = _text(row.get("primary_camera"))
    if not primary_camera:
        raise ValueError("QY primary_camera is missing")
    fps = _finite_float(row.get("fps"))
    if fps is None or fps <= 0.0:
        raise ValueError("QY fps is missing or invalid")
    return session.build_clip(
        asset_id=str(row["asset_id"]),
        start_frame=start_frame,
        end_frame=end_frame,
        primary_camera=primary_camera,
        fps=fps,
        episode_idx=episode_idx,
        text_label={
            "category": row.get("category"),
            "task_name": row.get("task_name"),
            "episode_id": row.get("episode_id"),
            "scene": row.get("category"),
            "task": row.get("task_name"),
        },
        coordinate_system_status=str(
            row.get("coordinate_system_status") or "mapping_unverified"
        ),
    )


__all__ = [
    "QY_HANDS",
    "QingyuHandPoseSession",
    "QingyuSourceReadError",
    "load_qingyu_clip",
]
