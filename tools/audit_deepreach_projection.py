#!/usr/bin/env python3
"""Generate bounded, no-model DR projection records and overlay evidence."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import h5py
import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acceptance_pull.supplier_adapters.deepreach import SUPPORTED_CAMERAS  # noqa: E402
from acceptance_pull.supplier_adapters.deepreach_projection import (  # noqa: E402
    approximate_head_calibration,
    load_calibration,
    load_trajectory,
    project_hand,
    project_dr_hands_for_frame,
)
from qc_common.projection import scale_intrinsics  # noqa: E402
from tools.run_manifest_precheck import read_manifest  # noqa: E402


def _path(value: Any, manifest_dir: Path) -> Path | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    return (candidate if candidate.is_absolute() else manifest_dir / candidate).resolve()


def _integer(value: Any, name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def _mapping(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("projection mapping config must be a mapping")
    selected = payload.get("dr_projection")
    if not isinstance(selected, Mapping):
        raise ValueError("projection mapping config must define dr_projection")
    return selected


def _selected_frames(
    start: int,
    end: int,
    requested: Sequence[int],
    max_samples: int,
) -> tuple[int, ...]:
    if requested:
        selected = sorted({frame for frame in requested if start <= frame <= end})
    else:
        selected = sorted({start, (start + end) // 2, end})
    return tuple(selected[:max_samples])


def _read_hands(
    path: Path,
    *,
    source_frame: int,
    clip_start_frame: int,
) -> dict[str, np.ndarray]:
    local_frame = source_frame - clip_start_frame
    if local_frame < 0:
        raise ValueError("source frame precedes clip start")
    with h5py.File(path, "r") as handle:
        result: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            joints_path = f"hand/{side}/joints3d"
            valid_path = f"hand/{side}/valid"
            if joints_path not in handle or valid_path not in handle:
                raise ValueError(f"missing DR HDF5 dataset for {side} hand")
            joints = handle[joints_path]
            valid = handle[valid_path]
            if local_frame >= joints.shape[0] or local_frame >= valid.shape[0]:
                raise ValueError("source/local frame outside DR HDF5")
            points = np.asarray(joints[local_frame], dtype=np.float64)
            if points.shape != (21, 3):
                raise ValueError(f"{joints_path} frame must have shape (21, 3)")
            if not bool(np.asarray(valid[local_frame]).reshape(-1)[0]):
                points[:] = np.nan
            result[side] = points
        return result


def _read_video_frame(path: Path, local_frame: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def _video_resolution(path: Path | None) -> tuple[int, int] | None:
    if path is None or not path.is_file():
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    return (width, height) if width > 0 and height > 0 else None


def _video_metadata(path: Path | None) -> tuple[int, int, int] | None:
    if path is None or not path.is_file():
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if min(width, height, frame_count) <= 0:
        return None
    return width, height, frame_count


def _candidate_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    from tools.run_manifest_sam3_containment import read_records

    return [row for path in paths for row in read_records(path)]


def _approximate_sample_frames(
    *,
    asset_id: str,
    start: int,
    end: int,
    available_frame_count: int,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    available_end = min(end, start + available_frame_count - 1)
    if available_end < start:
        return ()
    selected = {start, (start + available_end) // 2, available_end}
    peaks = sorted(
        {
            _integer(row.get("peak_frame"), "peak_frame")
            for row in candidates
            if str(row.get("asset_id") or "") == asset_id
            and row.get("peak_frame") is not None
        }
    )
    selected.update(frame for frame in peaks[:1] if start <= frame <= available_end)
    return tuple(sorted(selected))


def _approximate_projection_metrics(
    *,
    hdf5_path: Path,
    clip_start: int,
    sampled_frames: Sequence[int],
    calibration: Any,
) -> tuple[dict[str, Any], dict[int, dict[str, dict[str, Any]]]]:
    total = positive_z = finite_projection = in_frame = 0
    projected_by_frame: dict[int, dict[str, dict[str, Any]]] = {}
    for source_frame in sampled_frames:
        projected = project_dr_hands_for_frame(
            hdf5_path,
            source_frame=source_frame,
            clip_start_frame=clip_start,
            calibration=calibration,
        )
        projected_by_frame[source_frame] = projected
        for hand in projected.values():
            points = np.asarray(hand["points_3d"], dtype=np.float64)
            finite_points = np.isfinite(points).all(axis=1)
            total += int(points.shape[0])
            positive_z += int(np.sum(finite_points & (points[:, 2] > 0.0)))
            finite_projection += int(np.sum(hand["valid"]))
            in_frame += int(np.sum(hand["in_frame"]))
    denominator = total if total else 1
    return (
        {
            "sampled_point_count": total,
            "positive_z_count": positive_z,
            "positive_z_ratio": positive_z / denominator,
            "finite_projection_count": finite_projection,
            "finite_projection_ratio": finite_projection / denominator,
            "in_frame_count": in_frame,
            "in_frame_ratio": in_frame / denominator,
        },
        projected_by_frame,
    )


def _write_approximate_contact_sheet(
    *,
    asset_id: str,
    video_path: Path,
    clip_start: int,
    sampled_frames: Sequence[int],
    candidates: Sequence[tuple[Mapping[str, Any], Mapping[int, Mapping[str, Mapping[str, Any]]]]],
    output_dir: Path,
) -> str | None:
    panels: list[np.ndarray] = []
    colors = {"left": (0, 255, 0), "right": (255, 0, 255)}
    for row, projected_by_frame in candidates:
        for source_frame in sampled_frames:
            frame = _read_video_frame(video_path, source_frame - clip_start)
            if frame is None:
                continue
            canvas = frame.copy()
            for side, hand in projected_by_frame.get(source_frame, {}).items():
                pixels = np.asarray(hand["pixels"], dtype=np.float64)
                valid = np.asarray(hand["valid"], dtype=bool)
                for pixel in pixels[valid]:
                    cv2.circle(
                        canvas,
                        (int(round(float(pixel[0]))), int(round(float(pixel[1])))),
                        2,
                        colors.get(side, (0, 255, 255)),
                        -1,
                        lineType=cv2.LINE_AA,
                    )
            labelled = cv2.copyMakeBorder(
                canvas,
                105,
                0,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(20, 20, 20),
            )
            lines = (
                f"asset={asset_id} frame={source_frame} hfov={row['head_hfov_deg']:g}",
                f"K fx={row['fx']:.3f} fy={row['fy']:.3f} cx={row['cx']:.3f} cy={row['cy']:.3f}",
                f"positive_z={row['positive_z_ratio']:.4f} finite={row['finite_projection_ratio']:.4f}",
                f"in_frame={row['in_frame_ratio']:.4f} heuristic; distortion=false",
            )
            for line_index, line in enumerate(lines):
                cv2.putText(
                    labelled,
                    line[:150],
                    (5, 19 + line_index * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
            if labelled.shape[1] > 640:
                scale = 640.0 / labelled.shape[1]
                labelled = cv2.resize(
                    labelled,
                    (640, max(1, int(round(labelled.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            panels.append(labelled)
    if not panels:
        return None
    width = max(panel.shape[1] for panel in panels)
    normalized = [
        cv2.copyMakeBorder(
            panel,
            0,
            0,
            0,
            width - panel.shape[1],
            cv2.BORDER_CONSTANT,
            value=(20, 20, 20),
        )
        for panel in panels
    ]
    safe_asset = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in asset_id
    )
    relative = Path("approx_head_projection_contact_sheets") / f"{safe_asset}.png"
    target = output_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), np.vstack(normalized)):
        raise OSError(f"failed to write contact sheet: {target}")
    return relative.as_posix()


def run_approximate_head_projection_audit(
    *,
    manifest: Path,
    output_dir: Path,
    candidate_hfov_deg: Sequence[float],
    candidate_windows: Sequence[Path] = (),
    asset_ids: Sequence[str] = (),
    max_assets: int = 5,
) -> dict[str, int]:
    """Compare explicit HFOV candidates without asserting head calibration."""
    if max_assets < 1:
        raise ValueError("max_assets must be >= 1")
    hfovs = tuple(float(value) for value in candidate_hfov_deg)
    if not hfovs:
        raise ValueError("at least one candidate HFOV is required")
    if len(set(hfovs)) != len(hfovs):
        raise ValueError("candidate HFOV values must be unique")
    manifest = Path(manifest)
    output_dir = Path(output_dir)
    manifest_dir = manifest.resolve().parent
    selected_assets = {str(value) for value in asset_ids}
    rows = [
        row
        for row in read_manifest(manifest)
        if str(row.get("supplier") or row.get("supplier_id") or "").lower()
        in {"dr", "deepreach"}
        and (
            not selected_assets
            or str(row.get("asset_id") or "") in selected_assets
        )
    ][:max_assets]
    candidate_paths = tuple(Path(path) for path in candidate_windows)
    candidates = _candidate_rows(candidate_paths)
    assets: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    contact_sheet_count = 0
    for source in rows:
        asset_id = str(source.get("asset_id") or "").strip()
        hdf5_path = _path(source.get("hdf5_path"), manifest_dir)
        video_path = _path(
            source.get("head_video_path") or source.get("primary_video_path"),
            manifest_dir,
        )
        if not asset_id or hdf5_path is None or not hdf5_path.is_file():
            raise ValueError("DR approximate audit requires asset_id and HDF5")
        metadata = _video_metadata(video_path)
        if video_path is None or metadata is None:
            raise ValueError(f"head video metadata unavailable for {asset_id}")
        width, height, video_frame_count = metadata
        reference_dataset = str(source.get("hdf5_reference_dataset") or "timestamp")
        with h5py.File(hdf5_path, "r") as handle:
            if reference_dataset not in handle or not handle[reference_dataset].shape:
                raise ValueError(f"reference dataset unavailable for {asset_id}")
            hdf5_frame_count = int(handle[reference_dataset].shape[0])
        start = _integer(source.get("start_frame", 0), "start_frame")
        end = _integer(
            source.get("end_frame", start + hdf5_frame_count - 1),
            "end_frame",
        )
        sampled_frames = _approximate_sample_frames(
            asset_id=asset_id,
            start=start,
            end=end,
            available_frame_count=min(video_frame_count, hdf5_frame_count),
            candidates=candidates,
        )
        render_inputs: list[
            tuple[Mapping[str, Any], Mapping[int, Mapping[str, Mapping[str, Any]]]]
        ] = []
        candidate_payloads: list[dict[str, Any]] = []
        for hfov in hfovs:
            calibration = approximate_head_calibration(
                width=width,
                height=height,
                horizontal_fov_deg=hfov,
            )
            metrics, projected = _approximate_projection_metrics(
                hdf5_path=hdf5_path,
                clip_start=start,
                sampled_frames=sampled_frames,
                calibration=calibration,
            )
            assert calibration.intrinsics is not None
            matrix = calibration.intrinsics
            row = {
                "asset_id": asset_id,
                "task_name": str(source.get("task_name") or asset_id),
                "projection_mode": "approx_pinhole_from_hfov",
                "head_hfov_deg": hfov,
                "fx": float(matrix[0, 0]),
                "fy": float(matrix[1, 1]),
                "cx": float(matrix[0, 2]),
                "cy": float(matrix[1, 2]),
                "image_width": width,
                "image_height": height,
                "video_frame_count": video_frame_count,
                "hdf5_frame_count": hdf5_frame_count,
                "source_frame_count": end - start + 1,
                "frame_count_match": video_frame_count == hdf5_frame_count,
                "source_range_match": end - start + 1 == hdf5_frame_count,
                "sampled_source_frames": list(sampled_frames),
                "calibration_status": "heuristic",
                "projection_validation_status": "pending_visual_validation",
                "distortion_applied": False,
                "camera_trajectory_applied": False,
                **metrics,
            }
            candidate_payloads.append(row)
            csv_rows.append(row)
            render_inputs.append((row, projected))
        contact_sheet = _write_approximate_contact_sheet(
            asset_id=asset_id,
            video_path=video_path,
            clip_start=start,
            sampled_frames=sampled_frames,
            candidates=render_inputs,
            output_dir=output_dir,
        )
        contact_sheet_count += int(contact_sheet is not None)
        assets.append(
            {
                "asset_id": asset_id,
                "task_name": str(source.get("task_name") or asset_id),
                "sampled_source_frames": list(sampled_frames),
                "contact_sheet_path": contact_sheet,
                "candidates": candidate_payloads,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "dr_approximate_head_projection_audit.v1",
        "projection_mode": "approx_pinhole_from_hfov",
        "calibration_status": "heuristic",
        "projection_validation_status": "pending_visual_validation",
        "distortion_applied": False,
        "camera_trajectory_applied": False,
        "manifest": str(manifest.resolve()),
        "candidate_hfov_deg": list(hfovs),
        "candidate_windows": [str(path.resolve()) for path in candidate_paths],
        "assets": assets,
    }
    (output_dir / "dr_approximate_head_projection_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _write_csv_records(
        output_dir / "dr_approximate_head_projection_audit.csv",
        csv_rows,
    )
    return {
        "asset_count": len(assets),
        "candidate_count": len(csv_rows),
        "contact_sheet_count": contact_sheet_count,
    }


def _write_csv_records(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not columns:
            return
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for source in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in source.items()
                }
            )


def _write_overlay(
    frame: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    colors = {"left": (0, 255, 0), "right": (255, 0, 255)}
    output = frame.copy()
    for row in records:
        if not row.get("projection_valid") or not row.get("in_frame"):
            continue
        cv2.circle(
            output,
            (int(round(float(row["projected_x"]))), int(round(float(row["projected_y"])))),
            2,
            colors.get(str(row.get("hand_side")), (0, 255, 255)),
            -1,
            lineType=cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), output):
        raise OSError(f"failed to write overlay: {path}")


def _blocked_row(
    *,
    asset_id: str,
    camera_name: str,
    source_frame: int,
    clip_start: int,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": "dr_projection_audit_row.v1",
        "asset_id": asset_id,
        "camera_name": camera_name,
        "source_frame": source_frame,
        "local_frame": source_frame - clip_start,
        "status": status,
        "reason": reason,
        "overlay_path": None,
    }


def run_projection_audit(
    *,
    manifest: Path,
    mapping_config: Path,
    output_dir: Path,
    asset_ids: Sequence[str] = (),
    cameras: Sequence[str] = ("head",),
    frames: Sequence[int] = (),
    max_assets: int = 5,
    max_samples: int = 5,
) -> dict[str, int]:
    if max_assets < 1:
        raise ValueError("max_assets must be >= 1")
    if max_samples < 1:
        raise ValueError("max_samples must be >= 1")
    invalid_cameras = sorted(set(cameras) - set(SUPPORTED_CAMERAS))
    if invalid_cameras:
        raise ValueError(f"unsupported DR camera: {invalid_cameras[0]}")
    rows = read_manifest(manifest)
    selected_assets = set(asset_ids)
    if selected_assets:
        rows = [row for row in rows if str(row.get("asset_id")) in selected_assets]
    rows = [
        row
        for row in rows
        if str(row.get("supplier") or row.get("supplier_id") or "").lower()
        in {"dr", "deepreach"}
    ][:max_assets]
    config = _mapping(mapping_config)
    trajectory_mapping = config.get("trajectory", {})
    calibration_root = config.get("calibration", {})
    camera_mappings = (
        calibration_root.get("cameras", {})
        if isinstance(calibration_root, Mapping)
        else {}
    )
    transform_chain = config.get("transform_chain", {})
    explicit_identity = (
        isinstance(transform_chain, Mapping)
        and transform_chain.get("calibration_extrinsic") == "identity"
    )
    manifest_dir = manifest.resolve().parent
    records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    overlays = 0
    blocked = 0
    sampled_assets: set[str] = set()
    for row in rows:
        supplier = str(row.get("supplier") or row.get("supplier_id") or "").lower()
        if supplier not in {"dr", "deepreach"}:
            continue
        asset_id = str(row.get("asset_id") or "").strip()
        if not asset_id:
            continue
        start = _integer(row.get("start_frame", 0), "start_frame")
        hdf5_path = _path(row.get("hdf5_path"), manifest_dir)
        if hdf5_path is None or not hdf5_path.is_file():
            continue
        with h5py.File(hdf5_path, "r") as handle:
            source_count = int(handle["timestamp"].shape[0])
        end = _integer(row.get("end_frame", start + source_count - 1), "end_frame")
        selected_frames = _selected_frames(start, end, frames, max_samples)
        trajectory_path = _path(row.get("camera_trajectory_path"), manifest_dir)
        trajectory = load_trajectory(
            trajectory_path or Path("__missing_trajectory__"),
            trajectory_mapping if isinstance(trajectory_mapping, Mapping) else {},
        )
        for camera_name in cameras:
            calibration_path = _path(row.get("calib_path"), manifest_dir)
            video_path = _path(row.get(f"{camera_name}_video_path"), manifest_dir)
            camera_mapping = (
                camera_mappings.get(camera_name, {})
                if isinstance(camera_mappings, Mapping)
                else {}
            )
            calibration = load_calibration(
                calibration_path or Path("__missing_calibration__"),
                camera_name,
                camera_mapping if isinstance(camera_mapping, Mapping) else {},
            )
            intrinsics_source_resolution = calibration.resolution
            projection_resolution = _video_resolution(video_path)
            intrinsics_scaled = False
            if (
                calibration.status == "verified"
                and calibration.intrinsics is not None
                and calibration.resolution is not None
                and projection_resolution is not None
                and projection_resolution != calibration.resolution
            ):
                calibration = replace(
                    calibration,
                    intrinsics=scale_intrinsics(
                        calibration.intrinsics,
                        calibration.resolution,
                        projection_resolution,
                    ),
                    resolution=projection_resolution,
                )
                intrinsics_scaled = True
            if projection_resolution is None:
                projection_resolution = calibration.resolution
            for source_frame in selected_frames:
                sampled_assets.add(asset_id)
                if not explicit_identity:
                    audit_rows.append(
                        _blocked_row(
                            asset_id=asset_id,
                            camera_name=camera_name,
                            source_frame=source_frame,
                            clip_start=start,
                            status="transform_ambiguous",
                            reason="calibration_extrinsic_not_explicit",
                        )
                    )
                    blocked += 1
                    continue
                if trajectory.status != "verified":
                    audit_rows.append(
                        _blocked_row(
                            asset_id=asset_id,
                            camera_name=camera_name,
                            source_frame=source_frame,
                            clip_start=start,
                            status=trajectory.status,
                            reason=trajectory.reason or trajectory.status,
                        )
                    )
                    blocked += 1
                    continue
                if calibration.status != "verified":
                    audit_rows.append(
                        _blocked_row(
                            asset_id=asset_id,
                            camera_name=camera_name,
                            source_frame=source_frame,
                            clip_start=start,
                            status=calibration.status,
                            reason=calibration.reason or calibration.status,
                        )
                    )
                    blocked += 1
                    continue
                pose = trajectory.poses.get(source_frame)
                if pose is None:
                    audit_rows.append(
                        _blocked_row(
                            asset_id=asset_id,
                            camera_name=camera_name,
                            source_frame=source_frame,
                            clip_start=start,
                            status="calibration_unverified",
                            reason="trajectory_frame_missing",
                        )
                    )
                    blocked += 1
                    continue
                hands = _read_hands(
                    hdf5_path,
                    source_frame=source_frame,
                    clip_start_frame=start,
                )
                sample_records: list[dict[str, Any]] = []
                for hand_side, points in hands.items():
                    projected = project_hand(
                        asset_id=asset_id,
                        source_frame=source_frame,
                        clip_start_frame=start,
                        camera_name=camera_name,
                        hand_side=hand_side,
                        points=points,
                        calibration=calibration,
                        pose=pose,
                    )
                    for record in projected:
                        record["schema_version"] = "dr_projection_record.v1"
                        record["trajectory_source"] = trajectory.source
                        record["intrinsics_source_resolution"] = (
                            list(intrinsics_source_resolution)
                            if intrinsics_source_resolution is not None
                            else None
                        )
                        record["projection_resolution"] = (
                            list(projection_resolution)
                            if projection_resolution is not None
                            else None
                        )
                        record["intrinsics_scaled"] = intrinsics_scaled
                        record["transform_chain"] = {
                            "trajectory": pose.transform_direction,
                            "calibration_extrinsic": "identity",
                        }
                    sample_records.extend(projected)
                records.extend(sample_records)
                overlay_relative: str | None = None
                if video_path is not None and video_path.is_file():
                    frame = _read_video_frame(video_path, source_frame - start)
                    if frame is not None:
                        overlay = (
                            output_dir
                            / "overlays"
                            / f"{asset_id}__{camera_name}__source_{source_frame:06d}.png"
                        )
                        _write_overlay(frame, sample_records, overlay)
                        overlay_relative = overlay.relative_to(output_dir).as_posix()
                        overlays += 1
                audit_rows.append(
                    {
                        "schema_version": "dr_projection_audit_row.v1",
                        "asset_id": asset_id,
                        "camera_name": camera_name,
                        "source_frame": source_frame,
                        "local_frame": source_frame - start,
                        "status": "projected_unverified",
                        "reason": "requires_manual_overlay_validation",
                        "overlay_path": overlay_relative,
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "projection_records.json").write_text(
        json.dumps(records, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "projection_audit_summary.json").write_text(
        json.dumps(audit_rows, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv_records(output_dir / "projection_records.csv", records)
    _write_csv_records(output_dir / "projection_audit_summary.csv", audit_rows)
    mapping_bytes = mapping_config.read_bytes()
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "schema_version": "dr_projection_audit_run.v1",
                "manifest": str(manifest.resolve()),
                "mapping_config": str(mapping_config.resolve()),
                "mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
                "asset_ids": list(asset_ids),
                "cameras": list(cameras),
                "frames": list(frames),
                "max_assets": max_assets,
                "max_samples_per_asset_camera": max_samples,
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "asset_count": len(sampled_assets),
        "sample_count": len(audit_rows),
        "record_count": len(records),
        "overlay_count": overlays,
        "blocked_count": blocked,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--mapping-config",
        type=Path,
        help="Required only for the legacy mapping-driven projection audit.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--projection-mode",
        choices=("mapping", "approx-pinhole-from-hfov"),
        help=(
            "Select the legacy explicit-mapping audit or the heuristic head "
            "HFOV comparison. Omitted mode keeps the existing flag-based behavior."
        ),
    )
    parser.add_argument(
        "--candidate-hfov-deg",
        help="Comma-separated head horizontal-FOV candidates for heuristic audit mode.",
    )
    parser.add_argument(
        "--candidate-windows",
        action="append",
        default=[],
        type=Path,
        help="Optional candidate windows used to add one peak source frame.",
    )
    parser.add_argument("--asset-id", action="append", default=[])
    parser.add_argument("--camera", action="append", choices=SUPPORTED_CAMERAS)
    parser.add_argument("--frame", action="append", type=int, default=[])
    parser.add_argument("--max-assets", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    approximate_mode = args.projection_mode == "approx-pinhole-from-hfov" or (
        args.projection_mode is None and bool(args.candidate_hfov_deg)
    )
    if approximate_mode:
        if not args.candidate_hfov_deg:
            raise ValueError(
                "--candidate-hfov-deg is required for "
                "--projection-mode approx-pinhole-from-hfov"
            )
        if args.mapping_config is not None or args.camera or args.frame:
            raise ValueError(
                "HFOV audit mode does not use --mapping-config, --camera, or --frame"
            )
        try:
            hfovs = tuple(
                float(value.strip())
                for value in args.candidate_hfov_deg.split(",")
                if value.strip()
            )
        except ValueError:
            raise ValueError("--candidate-hfov-deg must be comma-separated numbers") from None
        summary = run_approximate_head_projection_audit(
            manifest=args.manifest,
            output_dir=args.output_dir,
            candidate_hfov_deg=hfovs,
            candidate_windows=tuple(args.candidate_windows),
            asset_ids=tuple(args.asset_id),
            max_assets=args.max_assets,
        )
    else:
        if args.projection_mode == "mapping" and args.candidate_hfov_deg:
            raise ValueError(
                "--candidate-hfov-deg is not valid for --projection-mode mapping"
            )
        if args.mapping_config is None:
            raise ValueError(
                "--mapping-config is required unless --candidate-hfov-deg is used"
            )
        if args.candidate_windows:
            raise ValueError("--candidate-windows requires --candidate-hfov-deg")
        summary = run_projection_audit(
            manifest=args.manifest,
            mapping_config=args.mapping_config,
            output_dir=args.output_dir,
            asset_ids=tuple(args.asset_id),
            cameras=tuple(args.camera or ("head",)),
            frames=tuple(args.frame),
            max_assets=args.max_assets,
            max_samples=args.max_samples,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
