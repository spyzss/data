"""Supplier-structured inventory audit independent of precheck/video outputs."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from qc_common.suppliers import normalize_supplier
from qc_common.manifest_metadata import manifest_metadata
from qc_pipeline.context import AssetContext

from acceptance_pull.supplier_adapters.structured_audit import (
    audit_csv_structure,
    audit_csv_timeline,
    audit_dr_calibration,
    audit_json_structure,
    audit_potentia_calibration,
    audit_video_metadata,
    read_mapped_json_values,
)


_REQUIRED_SOURCES = {
    "dr": (
        "hdf5",
        "lerobot_task",
        "head_video",
        "left_wrist_video",
        "right_wrist_video",
        "calibration",
        "trajectory",
    ),
    "potentia": (
        "video",
        "meta",
        "frames",
        "aligned",
        "imu",
        "calibration",
    ),
    "qy": (
        "video",
        "episode_manifest",
        "observations_2d",
        "trajectory_3d",
        "coordinate_system",
        "quality",
        "timebase",
        "semantic",
        "review_video",
        "qy_left_cam_left_video",
        "qy_left_cam_right_video",
        "qy_mid_cam_left_video",
        "qy_mid_cam_right_video",
        "qy_right_cam_left_video",
        "qy_right_cam_right_video",
    ),
}


def _inventory_entry(context: AssetContext, source_name: str) -> dict[str, Any]:
    declared = context.source_files.get(source_name)
    value = declared.get("path") if isinstance(declared, Mapping) else None
    if not isinstance(value, str) or not value.strip():
        return {"status": "missing", "path": None}
    path = context.batch_root / value
    supplier = str(context.metadata.get("supplier") or "").lower()
    status = "present" if path.exists() else "missing"
    if supplier in {"qy", "qingyu"} and path.exists() and not path.is_file():
        status = "wrong_type"
    return {
        "status": status,
        "path": value,
        "kind": "directory" if path.is_dir() else "file",
    }


def _manifest_inventory_entry(
    context: AssetContext,
    source_name: str,
) -> dict[str, Any]:
    value = context.metadata.get("file_inventory")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, Mapping):
        return {}
    entry = value.get(source_name)
    return dict(entry) if isinstance(entry, Mapping) else {}


def audit_supplier_data(
    context: AssetContext,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    source_supplier = (
        context.metadata.get("supplier")
        or context.metadata.get("supplier_id")
        or "unknown"
    )
    supplier, supplier_name, inferred_alias = normalize_supplier(source_supplier)
    supplier_alias = context.metadata.get("supplier_alias") or inferred_alias
    manifest = manifest_metadata(context.metadata)
    suppliers = parameters.get("suppliers")
    supplier_config = (
        suppliers.get(supplier, {}) if isinstance(suppliers, Mapping) else {}
    )
    mapping_config_identity = "sha256:" + hashlib.sha256(
        json.dumps(
            supplier_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if supplier not in _REQUIRED_SOURCES:
        return {
            "schema_version": "supplier_data_audit.raw.v2",
            "asset_id": context.asset_id,
            "supplier": supplier,
            "supplier_id": supplier,
            "supplier_name": supplier_name,
            "supplier_alias": supplier_alias,
            "manifest_metadata": manifest,
            "mapping_config_identity": mapping_config_identity,
            "task_metadata": {
                "task_id": manifest.get("task_id"),
                "task": manifest.get("task"),
                "task_name": manifest.get("task_name"),
            },
            "inventory": {},
            "missing_sources": [],
            "mapping_status": "not_applicable",
            "structured": {},
            "supplier_quality_signal": {
                "status": "not_applicable",
                "value": None,
                "source": None,
            },
            "issues": [],
            "decision": "skipped",
            "reason": "supplier_not_applicable",
        }
    required = _REQUIRED_SOURCES.get(supplier, ())
    inventory = {
        source_name: _inventory_entry(context, source_name)
        for source_name in required
    }
    derived_qy_timebase = bool(
        supplier == "qy"
        and context.metadata.get("timebase_source")
        == "derived_from_video_and_observations_2d"
        and context.metadata.get("timebase_status") == "derived"
        and context.metadata.get("frame_mapping_status") == "verified"
    )
    missing = [
        source_name
        for source_name, item in inventory.items()
        if item["status"] != "present"
        and not (source_name == "timebase" and derived_qy_timebase)
    ]
    if supplier == "qy":
        mapping_status = (
            "verified"
            if str(context.metadata.get("frame_mapping_status") or "")
            == "verified"
            else "unverified"
        )
    else:
        mapping_status = (
            str(supplier_config.get("mapping_status") or "unverified")
            if isinstance(supplier_config, Mapping)
            else "unverified"
        )
    issues: list[dict[str, Any]] = [
        {
            "code": "required_source_missing",
            "severity": "fail",
            "source_name": source_name,
            "observed_value": "missing",
        }
        for source_name in missing
    ]
    if supplier == "qy":
        supplier_timebase = _manifest_inventory_entry(context, "timebase")
        supplier_timebase_status = str(
            supplier_timebase.get("status") or ""
        )
        if supplier_timebase_status == "input_invalid":
            issues.append(
                {
                    "code": (
                        "supplier_timebase_invalid_derived"
                        if derived_qy_timebase
                        else "supplier_timebase_invalid"
                    ),
                    "severity": "warn" if derived_qy_timebase else "fail",
                    "source_name": "timebase",
                    "observed_value": supplier_timebase.get("reason"),
                }
            )
        elif derived_qy_timebase and inventory["timebase"]["status"] != "present":
            issues.append(
                {
                    "code": "supplier_timebase_missing_derived",
                    "severity": "warn",
                    "source_name": "timebase",
                    "observed_value": context.metadata.get("timebase_source"),
                }
            )
        for source_name, item in inventory.items():
            if item["status"] == "wrong_type":
                issues.append(
                    {
                        "code": "required_source_wrong_type",
                        "severity": "fail",
                        "source_name": source_name,
                        "observed_value": item.get("kind"),
                    }
                )
    if supplier == "dr" and str(
        context.metadata.get("primary_camera_status") or "present"
    ) == "primary_camera_missing":
        issues.append(
            {
                "code": "primary_camera_missing",
                "severity": "fail",
                "source_name": "video",
                "observed_value": context.metadata.get("primary_camera"),
            }
        )
    if supplier == "qy":
        if not str(context.metadata.get("primary_camera") or "").strip():
            issues.append(
                {
                    "code": "primary_camera_missing",
                    "severity": "fail",
                    "source_name": "video",
                    "observed_value": None,
                }
            )
        skeleton_status = str(
            context.metadata.get("skeleton_3d_status") or "input_missing"
        )
        if skeleton_status in {"no_valid_output", "input_missing"}:
            issues.append(
                {
                    "code": "required_skeleton_input_missing",
                    "severity": "fail",
                    "source_name": "trajectory_3d",
                    "observed_value": skeleton_status,
                }
            )
        elif skeleton_status == "input_invalid":
            issues.append(
                {
                    "code": "required_skeleton_input_invalid",
                    "severity": "fail",
                    "source_name": "trajectory_3d",
                    "observed_value": skeleton_status,
                }
            )
        if str(context.metadata.get("joint_topology_status") or "") != "verified":
            issues.append(
                {
                    "code": "joint_topology_unverified",
                    "severity": "warn",
                    "source_name": "trajectory_3d",
                    "observed_value": context.metadata.get(
                        "joint_topology_status"
                    ),
                }
            )
        coordinate_status = str(
            context.metadata.get("coordinate_system_status") or ""
        )
        if coordinate_status.endswith("schema_unverified"):
            issues.append(
                {
                    "code": "coordinate_system_schema_unverified",
                    "severity": "warn",
                    "source_name": "coordinate_system",
                    "observed_value": coordinate_status,
                }
            )
    if mapping_status != "verified":
        issues.append(
            {
                "code": "mapping_unverified",
                "severity": "warn",
                "source_name": "mapping_config",
                "observed_value": mapping_status,
            }
        )
    mapping = (
        supplier_config.get("mapping", {})
        if isinstance(supplier_config, Mapping)
        else {}
    )
    structured: dict[str, Any] = {}
    supplier_quality_signal = {
        "status": "not_provided",
        "value": None,
        "source": None,
    }
    effective_mapping = (
        mapping
        if mapping_status == "verified" and isinstance(mapping, Mapping)
        else {}
    )
    if supplier == "potentia":
        structured, supplier_quality_signal = _audit_potentia_structured(
            context,
            effective_mapping,
            supplier_config,
        )
    elif supplier == "dr":
        structured = _audit_dr_structured(
            context,
            effective_mapping,
            supplier_config,
        )
    elif supplier == "qy":
        structured = {
            "timebase_status": context.metadata.get("timebase_status"),
            "timebase_source": context.metadata.get("timebase_source"),
            "frame_mapping_status": context.metadata.get(
                "frame_mapping_status"
            ),
            "frame_mapping_source": context.metadata.get(
                "frame_mapping_source"
            ),
            "source_video_identity_assumed": context.metadata.get(
                "source_video_identity_assumed"
            ),
            "skeleton_2d_status": context.metadata.get("skeleton_2d_status"),
            "skeleton_3d_status": context.metadata.get("skeleton_3d_status"),
            "skeleton_3d_coverage_status": context.metadata.get(
                "skeleton_3d_coverage_status"
            ),
            "skeleton_3d_valid_row_count": context.metadata.get(
                "skeleton_3d_valid_row_count"
            ),
            "joint_topology_status": context.metadata.get(
                "joint_topology_status"
            ),
            "coordinate_system_status": context.metadata.get(
                "coordinate_system_status"
            ),
            "reference_camera_status": context.metadata.get(
                "reference_camera_status"
            ),
            "primary_camera": context.metadata.get("primary_camera"),
            "primary_camera_source": context.metadata.get(
                "primary_camera_source"
            ),
            "camera_coverage": context.metadata.get("camera_coverage"),
        }
        supplier_quality_signal = {
            "status": "auxiliary_only",
            "value": None,
            "source": "trajectory_3d supplier quality fields",
        }
    for name, result in _structured_status_results(structured):
        if not isinstance(result, Mapping):
            continue
        status = result.get("status")
        if status in {"invalid", "mapping_invalid"}:
            issue_code = {
                "calibration": "calibration_invalid",
                "calibration_structure": "calibration_invalid",
                "frames": "timeline_invalid",
                "aligned": "timeline_invalid",
                "imu": "timeline_invalid",
                "alignment": "timeline_invalid",
                "trajectory": "timeline_invalid",
            }.get(name, "structure_invalid")
            if status == "mapping_invalid":
                issue_code = "mapping_invalid"
            issues.append(
                {
                    "code": issue_code,
                    "severity": "fail",
                    "source_name": name,
                    "observed_value": result.get("reason") or status,
                }
            )
        elif status == "unverified":
            issue_code = (
                "calibration_unverified"
                if name == "calibration"
                else "mapping_unverified"
            )
            issues.append(
                {
                    "code": issue_code,
                    "severity": "warn",
                    "source_name": name,
                    "observed_value": result.get("reason") or status,
                }
            )
    verdict = (
        "fail"
        if any(item["severity"] == "fail" for item in issues)
        else "warn" if issues else "pass"
    )
    return {
        "schema_version": "supplier_data_audit.raw.v2",
        "asset_id": context.asset_id,
        "supplier": supplier,
        "supplier_id": supplier,
        "supplier_name": supplier_name,
        "supplier_alias": supplier_alias,
        "manifest_metadata": manifest,
        "mapping_config_identity": mapping_config_identity,
        "task_metadata": {
            "task_id": manifest.get("task_id"),
            "task": manifest.get("task"),
            "task_name": manifest.get("task_name"),
        },
        "inventory": inventory,
        "missing_sources": missing,
        "mapping_status": mapping_status,
        "structured": structured,
        "supplier_quality_signal": supplier_quality_signal,
        "issues": issues,
        "decision": verdict,
    }


def _path(context: AssetContext, source_name: str) -> Path:
    declared = context.source_files.get(source_name)
    value = declared.get("path") if isinstance(declared, Mapping) else None
    if not isinstance(value, str) or not value.strip():
        return context.batch_root / ".missing_supplier_source" / source_name
    return context.batch_root / value


def _structured_status_results(
    structured: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    results: list[tuple[str, Mapping[str, Any]]] = []
    for name, value in structured.items():
        if not isinstance(value, Mapping):
            continue
        if name == "video_metadata" and "status" not in value:
            for camera_name, camera_value in value.items():
                if isinstance(camera_value, Mapping):
                    results.append(
                        (f"video_metadata.{camera_name}", camera_value)
                    )
            continue
        results.append((name, value))
    return results


def _audit_potentia_structured(
    context: AssetContext,
    mapping: Mapping[str, Any],
    supplier_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    factor = float(supplier_config.get("max_timestamp_gap_factor", 3.0))
    meta_path = _path(context, "meta")
    frames_path = _path(context, "frames")
    aligned_path = _path(context, "aligned")
    imu_path = _path(context, "imu")
    calibration_path = _path(context, "calibration")
    video_path = _path(context, "video")
    frames = audit_csv_timeline(
        frames_path,
        mapping.get("frames", {}),
        max_gap_factor=factor,
    )
    aligned = audit_csv_timeline(
        aligned_path,
        mapping.get("aligned", {}),
        max_gap_factor=factor,
    )
    imu = audit_csv_timeline(
        imu_path,
        mapping.get("imu", {}),
        max_gap_factor=factor,
        require_frame_index=False,
    )
    if (
        frames.get("timestamp_min") is not None
        and frames.get("timestamp_max") is not None
        and imu.get("timestamp_min") is not None
        and imu.get("timestamp_max") is not None
    ):
        imu["covers_frame_timeline"] = (
            imu["timestamp_min"] <= frames["timestamp_min"]
            and imu["timestamp_max"] >= frames["timestamp_max"]
        )
    else:
        imu["covers_frame_timeline"] = None
    meta = read_mapped_json_values(meta_path, mapping.get("meta", {}))
    qc_value = meta.get("qc")
    supplier_signal = {
        "status": "provided" if qc_value is not None else "not_provided",
        "value": qc_value,
        "source": "meta.qc" if qc_value is not None else None,
    }
    video_metadata = audit_video_metadata(video_path)
    frames_count = frames.get("row_count")
    aligned_count = aligned.get("row_count")
    video_count = video_metadata.get("frame_count")
    aligned_covers: bool | None = None
    if (
        aligned.get("timestamp_min") is not None
        and aligned.get("timestamp_max") is not None
        and frames.get("timestamp_min") is not None
        and frames.get("timestamp_max") is not None
    ):
        aligned_covers = (
            aligned["timestamp_min"] <= frames["timestamp_min"]
            and aligned["timestamp_max"] >= frames["timestamp_max"]
        )
    comparable_counts = isinstance(frames_count, int) and isinstance(
        aligned_count, int
    ) and isinstance(video_count, int)
    comparable_timeline = aligned_covers is not None
    alignment = {
        "status": "pass" if comparable_counts and comparable_timeline else "unverified",
        "frames_vs_aligned_row_delta": (
            int(aligned_count) - int(frames_count)
            if isinstance(frames_count, int) and isinstance(aligned_count, int)
            else None
        ),
        "video_vs_frames_row_delta": (
            int(video_count) - int(frames_count)
            if isinstance(video_count, int) and isinstance(frames_count, int)
            else None
        ),
        "aligned_covers_frame_timeline": aligned_covers,
    }
    if (
        alignment["frames_vs_aligned_row_delta"] not in {0, None}
        or alignment["video_vs_frames_row_delta"] not in {0, None}
        or aligned_covers is False
    ):
        alignment["status"] = "invalid"
    if alignment["status"] == "unverified":
        alignment["reason"] = "mapped_alignment_output_unavailable"
    video_fps = video_metadata.get("fps")
    timestamp_fps = frames.get("timestamp_derived_fps")
    fps_delta = (
        float(video_fps) - float(timestamp_fps)
        if isinstance(video_fps, (int, float))
        and isinstance(timestamp_fps, (int, float))
        else None
    )
    video_timeline_comparison = {
        "video_fps": video_fps,
        "timestamp_derived_fps": timestamp_fps,
        "fps_delta": fps_delta,
        "fps_relative_delta": (
            abs(fps_delta) / float(timestamp_fps)
            if fps_delta is not None and float(timestamp_fps) > 0
            else None
        ),
        "effective_fps": (
            video_fps
            if video_metadata.get("status") == "pass"
            and isinstance(video_fps, (int, float))
            else timestamp_fps
        ),
        "effective_fps_source": (
            "video_metadata"
            if video_metadata.get("status") == "pass"
            and isinstance(video_fps, (int, float))
            else (
                "frames_timestamp"
                if isinstance(timestamp_fps, (int, float))
                else None
            )
        ),
    }
    video_resolution = None
    if isinstance(video_metadata.get("width"), int) and isinstance(
        video_metadata.get("height"), int
    ):
        video_resolution = [video_metadata["width"], video_metadata["height"]]
    calibration = audit_potentia_calibration(
        calibration_path,
        mapping.get("calibration", {}),
        video_resolution=video_resolution,
        max_scaled_intrinsics_relative_error=float(
            supplier_config.get("max_scaled_intrinsics_relative_error", 0.001)
        ),
        scaling_mismatch_action=str(
            supplier_config.get("scaling_mismatch_action", "review")
        ),
    )
    return {
        "meta_structure": audit_json_structure(meta_path),
        "frames_structure": audit_csv_structure(frames_path),
        "aligned_structure": audit_csv_structure(aligned_path),
        "imu_structure": audit_csv_structure(imu_path),
        "calibration_structure": audit_json_structure(calibration_path),
        "meta": meta,
        "frames": frames,
        "aligned": aligned,
        "imu": imu,
        "calibration": calibration,
        "video_metadata": video_metadata,
        "video_timeline_comparison": video_timeline_comparison,
        "alignment": alignment,
    }, supplier_signal


def _audit_dr_structured(
    context: AssetContext,
    mapping: Mapping[str, Any],
    supplier_config: Mapping[str, Any],
) -> dict[str, Any]:
    configured_trajectory = mapping.get("trajectory", {})
    trajectory_mapping = (
        dict(configured_trajectory)
        if isinstance(configured_trajectory, Mapping)
        else {}
    )
    pose_columns: list[str] = []
    for key in ("translation_columns", "quaternion_xyzw_columns"):
        values = trajectory_mapping.get(key)
        if isinstance(values, list):
            pose_columns.extend(
                str(value) for value in values if isinstance(value, str)
            )
    existing_numeric = trajectory_mapping.get("numeric_columns")
    if isinstance(existing_numeric, list):
        pose_columns.extend(
            str(value) for value in existing_numeric if isinstance(value, str)
        )
    trajectory_mapping["numeric_columns"] = list(dict.fromkeys(pose_columns))
    trajectory = audit_csv_timeline(
        _path(context, "trajectory"),
        trajectory_mapping,
        max_gap_factor=3.0,
        max_frame_step=int(
            supplier_config.get("max_trajectory_gap_frames", 1)
        ),
    )
    if isinstance(trajectory_mapping, Mapping):
        trajectory["transform_direction"] = trajectory_mapping.get(
            "transform_direction"
        )
        if trajectory["transform_direction"] not in {
            "world_to_camera",
            "camera_to_world",
        }:
            trajectory["status"] = "unverified"
            trajectory["reason"] = "transform_direction_missing_or_ambiguous"
    calibration = audit_dr_calibration(
        _path(context, "calibration"),
        mapping.get("calibration", {}),
    )
    video_metadata = {
        camera_name: audit_video_metadata(_path(context, source_name))
        for camera_name, source_name in (
            ("head", "head_video"),
            ("left_wrist", "left_wrist_video"),
            ("right_wrist", "right_wrist_video"),
        )
    }
    return {
        "video_metadata": video_metadata,
        "calibration_structure": audit_json_structure(
            _path(context, "calibration")
        ),
        "trajectory_structure": audit_csv_structure(
            _path(context, "trajectory")
        ),
        "trajectory": trajectory,
        "calibration": calibration,
    }


__all__ = ["audit_supplier_data"]
