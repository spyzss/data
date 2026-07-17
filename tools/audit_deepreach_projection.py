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
    load_calibration,
    load_trajectory,
    project_hand,
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
    parser.add_argument("--mapping-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asset-id", action="append", default=[])
    parser.add_argument("--camera", action="append", choices=SUPPORTED_CAMERAS)
    parser.add_argument("--frame", action="append", type=int, default=[])
    parser.add_argument("--max-assets", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
