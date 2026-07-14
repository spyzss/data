from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


FRAME_STATUS_VISIBLE_OK = "visible_ok"
FRAME_STATUS_SHORT_TRUNCATION_CANDIDATE = "short_truncation_candidate"
FRAME_STATUS_MASK_MISSING_KEYPOINTS_INSIDE = "mask_missing_keypoints_inside"
FRAME_STATUS_MASK_TINY_KEYPOINTS_INSIDE = "mask_tiny_keypoints_inside"
FRAME_STATUS_CONTAINED_BUT_TRUNCATED_REVIEW = "contained_but_truncated_review"
FRAME_STATUS_COVERAGE_MISMATCH = "coverage_mismatch"
FRAME_STATUS_INSUFFICIENT_VISUAL_EVIDENCE = "insufficient_visual_evidence"

WINDOW_VERDICT_GOOD = "good"
WINDOW_VERDICT_ACCEPTABLE_FLAGGED = "acceptable_flagged"
WINDOW_VERDICT_REVIEW = "review"
WINDOW_VERDICT_COVERAGE_FAIL = "coverage_fail"


@dataclass(frozen=True)
class VisualCoverageConfig:
    projected_in_image_ratio_threshold: float = 0.8
    keypoints_inside_mask_ratio_threshold: float = 0.8
    mask_area_min_ratio: float = 0.002
    border_margin_px: int = 20
    short_accept_sec: float = 0.3
    fail_min_sec: float = 1.0
    bad_frame_ratio_threshold: float = 0.5
    min_bad_frames_for_review: int = 3
    seed_pre_context_sec: float = 0.5
    seed_post_context_sec: float = 0.5
    seed_merge_gap_sec: float = 0.2
    fps: float = 29.97


@dataclass(frozen=True)
class VisualCoverageFrameInput:
    frame_idx: int
    hand_side: str
    projected_keypoints_in_image_ratio: float
    hand_mask_present: bool
    hand_mask_area_ratio: float
    hand_mask_touches_border: bool = False
    mask_bbox_near_border: bool = False
    projected_keypoints_near_border_count: int = 0
    projected_keypoint_bbox_touches_border: bool = False
    hand_mask_area_px: float = 0.0
    keypoints_inside_hand_mask_ratio: float | None = None
    episode_idx: int | None = None
    asset_id: str | None = None


@dataclass(frozen=True)
class VisualCoverageFrameResult:
    frame_idx: int
    hand_side: str
    projected_keypoints_in_image_ratio: float
    projected_keypoints_near_border_count: int
    projected_keypoint_bbox_touches_border: bool
    hand_mask_present: bool
    hand_mask_area_px: float
    hand_mask_area_ratio: float
    hand_mask_touches_border: bool
    mask_bbox_near_border: bool
    keypoints_inside_hand_mask_ratio: float | None
    contained_but_mask_truncated: bool
    coverage_mismatch_frame: bool
    coverage_frame_status: str
    reason: str
    episode_idx: int | None = None
    asset_id: str | None = None


@dataclass(frozen=True)
class VisualCoverageWindowResult:
    hand_side: str
    start_frame: int
    end_frame: int
    bad_frame_count: int
    evaluated_frame_count: int
    bad_duration_sec: float
    bad_frame_ratio: float
    max_bad_run_frames: int
    max_bad_run_duration_sec: float
    verdict: str
    reason: str
    window_source: str
    projected_in_image_ratio_threshold: float
    mask_area_min_ratio: float
    short_accept_sec: float
    fail_min_sec: float
    bad_frame_ratio_threshold: float
    min_bad_frames_for_review: int
    seed_start_frame: int | None = None
    seed_end_frame: int | None = None
    episode_idx: int | None = None
    asset_id: str | None = None


def _as_frame_input(row: VisualCoverageFrameInput | dict[str, Any]) -> VisualCoverageFrameInput:
    if isinstance(row, VisualCoverageFrameInput):
        return row
    return VisualCoverageFrameInput(**row)


def classify_visual_coverage_frame(
    row: VisualCoverageFrameInput | dict[str, Any],
    config: VisualCoverageConfig | None = None,
) -> VisualCoverageFrameResult:
    config = config or VisualCoverageConfig()
    frame = _as_frame_input(row)

    keypoints_mostly_inside = (
        frame.projected_keypoints_in_image_ratio
        >= config.projected_in_image_ratio_threshold
    )
    mask_tiny = (
        not frame.hand_mask_present
        or frame.hand_mask_area_ratio < config.mask_area_min_ratio
    )
    mask_truncated = frame.hand_mask_touches_border or frame.mask_bbox_near_border
    keypoints_inside_mask = (
        frame.keypoints_inside_hand_mask_ratio is not None
        and frame.keypoints_inside_hand_mask_ratio
        >= config.keypoints_inside_mask_ratio_threshold
    )
    contained_but_truncated = (
        frame.hand_mask_present
        and mask_truncated
        and (keypoints_inside_mask or keypoints_mostly_inside)
    )
    visual_coverage_problem = mask_tiny or mask_truncated
    coverage_mismatch = bool(visual_coverage_problem and keypoints_mostly_inside)

    if not keypoints_mostly_inside and visual_coverage_problem:
        status = FRAME_STATUS_INSUFFICIENT_VISUAL_EVIDENCE
        reason = "visual hand coverage is weak, but projected keypoints are not mostly inside image"
    elif not frame.hand_mask_present and keypoints_mostly_inside:
        status = FRAME_STATUS_MASK_MISSING_KEYPOINTS_INSIDE
        reason = "hand mask missing while projected keypoints remain inside image"
    elif frame.hand_mask_area_ratio < config.mask_area_min_ratio and keypoints_mostly_inside:
        status = FRAME_STATUS_MASK_TINY_KEYPOINTS_INSIDE
        reason = "hand mask tiny while projected keypoints remain inside image"
    elif contained_but_truncated:
        status = FRAME_STATUS_CONTAINED_BUT_TRUNCATED_REVIEW
        reason = "keypoints are contained but visual hand mask is truncated near image border"
    elif coverage_mismatch:
        status = FRAME_STATUS_COVERAGE_MISMATCH
        reason = "visual coverage weak while projected keypoints remain inside image"
    else:
        status = FRAME_STATUS_VISIBLE_OK
        reason = "visual hand coverage and projected keypoints are consistent"

    return VisualCoverageFrameResult(
        episode_idx=frame.episode_idx,
        asset_id=frame.asset_id,
        frame_idx=frame.frame_idx,
        hand_side=frame.hand_side,
        projected_keypoints_in_image_ratio=frame.projected_keypoints_in_image_ratio,
        projected_keypoints_near_border_count=frame.projected_keypoints_near_border_count,
        projected_keypoint_bbox_touches_border=frame.projected_keypoint_bbox_touches_border,
        hand_mask_present=frame.hand_mask_present,
        hand_mask_area_px=frame.hand_mask_area_px,
        hand_mask_area_ratio=frame.hand_mask_area_ratio,
        hand_mask_touches_border=frame.hand_mask_touches_border,
        mask_bbox_near_border=frame.mask_bbox_near_border,
        keypoints_inside_hand_mask_ratio=frame.keypoints_inside_hand_mask_ratio,
        contained_but_mask_truncated=contained_but_truncated,
        coverage_mismatch_frame=coverage_mismatch,
        coverage_frame_status=status,
        reason=reason,
    )


def classify_visual_coverage_frames(
    rows: Iterable[VisualCoverageFrameInput | dict[str, Any]],
    config: VisualCoverageConfig | None = None,
) -> list[VisualCoverageFrameResult]:
    return [classify_visual_coverage_frame(row, config) for row in rows]


def aggregate_visual_coverage_windows(
    frame_results: Iterable[VisualCoverageFrameResult | dict[str, Any]],
    config: VisualCoverageConfig | None = None,
    candidate_windows: Iterable[dict[str, Any]] | None = None,
) -> list[VisualCoverageWindowResult]:
    config = config or VisualCoverageConfig()
    frames = [_as_frame_result(row) for row in frame_results]
    if not frames:
        return []
    if candidate_windows is not None:
        windows: list[VisualCoverageWindowResult] = []
        for window in candidate_windows:
            matched = frames_in_candidate(frames, window)
            for group in _group_by_hand(matched).values():
                windows.append(
                    _aggregate_window(
                        sorted(group, key=lambda item: item.frame_idx),
                        config,
                        window,
                        window_source="candidate_window",
                        seed_start_frame=None,
                        seed_end_frame=None,
                    )
                )
        return windows

    grouped: dict[tuple[str | None, int | None, str], list[VisualCoverageFrameResult]] = {}
    for frame in frames:
        key = (frame.asset_id, frame.episode_idx, frame.hand_side)
        grouped.setdefault(key, []).append(frame)
    windows: list[VisualCoverageWindowResult] = []
    for group in grouped.values():
        windows.extend(_seed_windows_for_group(sorted(group, key=lambda item: item.frame_idx), config))
    return windows


def _as_frame_result(
    row: VisualCoverageFrameResult | dict[str, Any],
) -> VisualCoverageFrameResult:
    if isinstance(row, VisualCoverageFrameResult):
        return row
    return VisualCoverageFrameResult(**row)


def frames_in_candidate(
    frames: list[VisualCoverageFrameResult],
    window: dict[str, Any],
) -> list[VisualCoverageFrameResult]:
    start = int(window["start_frame"])
    end = int(window["end_frame"])
    hand_side = str(window.get("hand_side", "both"))
    asset_id = window.get("asset_id")
    episode_idx = window.get("episode_idx")
    matched: list[VisualCoverageFrameResult] = []
    for frame in frames:
        if frame.frame_idx < start or frame.frame_idx > end:
            continue
        if hand_side != "both" and frame.hand_side != hand_side:
            continue
        if asset_id is not None and frame.asset_id is not None and frame.asset_id != asset_id:
            continue
        if episode_idx is not None and frame.episode_idx is not None and frame.episode_idx != episode_idx:
            continue
        matched.append(frame)
    return sorted(matched, key=lambda item: item.frame_idx)


def _group_by_hand(
    frames: Iterable[VisualCoverageFrameResult],
) -> dict[tuple[str | None, int | None, str], list[VisualCoverageFrameResult]]:
    grouped: dict[tuple[str | None, int | None, str], list[VisualCoverageFrameResult]] = {}
    for frame in frames:
        key = (frame.asset_id, frame.episode_idx, frame.hand_side)
        grouped.setdefault(key, []).append(frame)
    return grouped


def _seed_windows_for_group(
    frames: list[VisualCoverageFrameResult],
    config: VisualCoverageConfig,
) -> list[VisualCoverageWindowResult]:
    bad_frames = [frame for frame in frames if frame.coverage_mismatch_frame]
    if not bad_frames:
        return [
            _aggregate_window(
                frames,
                config,
                None,
                window_source="seed_window",
                seed_start_frame=None,
                seed_end_frame=None,
            )
        ]

    pre_frames = _seconds_to_frames(config.seed_pre_context_sec, config.fps)
    post_frames = _seconds_to_frames(config.seed_post_context_sec, config.fps)
    merge_gap_frames = _seconds_to_frames(config.seed_merge_gap_sec, config.fps)
    seed_windows = [
        {
            "start_frame": max(frames[0].frame_idx, frame.frame_idx - pre_frames),
            "end_frame": min(frames[-1].frame_idx, frame.frame_idx + post_frames),
            "seed_start_frame": frame.frame_idx,
            "seed_end_frame": frame.frame_idx,
        }
        for frame in bad_frames
    ]

    merged: list[dict[str, int]] = []
    active: dict[str, int] | None = None
    for window in seed_windows:
        if active is None or window["start_frame"] > active["end_frame"] + merge_gap_frames:
            if active is not None:
                merged.append(active)
            active = dict(window)
            continue
        active["end_frame"] = max(active["end_frame"], window["end_frame"])
        active["seed_end_frame"] = max(active["seed_end_frame"], window["seed_end_frame"])
    if active is not None:
        merged.append(active)

    results: list[VisualCoverageWindowResult] = []
    for window in merged:
        window_frames = [
            frame
            for frame in frames
            if window["start_frame"] <= frame.frame_idx <= window["end_frame"]
        ]
        if window_frames:
            results.append(
                _aggregate_window(
                    window_frames,
                    config,
                    window,
                    window_source="seed_window",
                    seed_start_frame=window["seed_start_frame"],
                    seed_end_frame=window["seed_end_frame"],
                )
            )
    return results


def _seconds_to_frames(seconds: float, fps: float) -> int:
    if fps <= 0.0:
        return 0
    return max(0, round(seconds * fps))


def _max_bad_run_frames(frames: list[VisualCoverageFrameResult]) -> int:
    max_run = 0
    current_run = 0
    previous_bad_frame: int | None = None
    for frame in sorted(frames, key=lambda item: item.frame_idx):
        if not frame.coverage_mismatch_frame:
            current_run = 0
            previous_bad_frame = None
            continue
        if previous_bad_frame is None or frame.frame_idx == previous_bad_frame + 1:
            current_run += 1
        else:
            current_run = 1
        previous_bad_frame = frame.frame_idx
        max_run = max(max_run, current_run)
    return max_run


def _aggregate_window(
    frames: list[VisualCoverageFrameResult],
    config: VisualCoverageConfig,
    candidate_window: dict[str, Any] | None,
    window_source: str,
    seed_start_frame: int | None,
    seed_end_frame: int | None,
) -> VisualCoverageWindowResult:
    first = frames[0]
    if candidate_window is None:
        start_frame = min(frame.frame_idx for frame in frames)
        end_frame = max(frame.frame_idx for frame in frames)
        hand_side = first.hand_side
    else:
        start_frame = int(candidate_window["start_frame"])
        end_frame = int(candidate_window["end_frame"])
        candidate_hand_side = str(candidate_window.get("hand_side", first.hand_side))
        hand_side = first.hand_side if candidate_hand_side == "both" else candidate_hand_side

    bad_count = sum(1 for frame in frames if frame.coverage_mismatch_frame)
    evaluated_count = len(frames)
    bad_duration_sec = bad_count / config.fps if config.fps > 0 else 0.0
    bad_ratio = bad_count / evaluated_count if evaluated_count else 0.0
    max_bad_run_frames = _max_bad_run_frames(frames)
    max_bad_run_duration_sec = (
        max_bad_run_frames / config.fps if config.fps > 0 else 0.0
    )

    if (
        max_bad_run_duration_sec >= config.fail_min_sec
        and bad_ratio >= config.bad_frame_ratio_threshold
    ):
        verdict = WINDOW_VERDICT_COVERAGE_FAIL
        reason = "sustained visual coverage mismatch"
    elif bad_count == 0:
        verdict = WINDOW_VERDICT_GOOD
        reason = "no visual coverage mismatch frames"
    elif max_bad_run_duration_sec < config.short_accept_sec:
        verdict = WINDOW_VERDICT_ACCEPTABLE_FLAGGED
        reason = "short visual coverage mismatch accepted for current MVP"
    elif bad_count >= config.min_bad_frames_for_review:
        verdict = WINDOW_VERDICT_REVIEW
        reason = "visual coverage mismatch needs review"
    else:
        verdict = WINDOW_VERDICT_ACCEPTABLE_FLAGGED
        reason = "limited visual coverage mismatch accepted for current MVP"

    return VisualCoverageWindowResult(
        episode_idx=first.episode_idx,
        asset_id=first.asset_id,
        hand_side=hand_side,
        start_frame=start_frame,
        end_frame=end_frame,
        bad_frame_count=bad_count,
        evaluated_frame_count=evaluated_count,
        bad_duration_sec=bad_duration_sec,
        bad_frame_ratio=bad_ratio,
        max_bad_run_frames=max_bad_run_frames,
        max_bad_run_duration_sec=max_bad_run_duration_sec,
        verdict=verdict,
        reason=reason,
        window_source=window_source,
        projected_in_image_ratio_threshold=config.projected_in_image_ratio_threshold,
        mask_area_min_ratio=config.mask_area_min_ratio,
        short_accept_sec=config.short_accept_sec,
        fail_min_sec=config.fail_min_sec,
        bad_frame_ratio_threshold=config.bad_frame_ratio_threshold,
        min_bad_frames_for_review=config.min_bad_frames_for_review,
        seed_start_frame=seed_start_frame,
        seed_end_frame=seed_end_frame,
    )


def result_to_record(result: Any) -> dict[str, Any]:
    return asdict(result)


def write_visual_coverage_outputs(
    frame_results: Iterable[VisualCoverageFrameResult],
    window_results: Iterable[VisualCoverageWindowResult],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_records = [result_to_record(result) for result in frame_results]
    window_records = [result_to_record(result) for result in window_results]
    frame_json = output_dir / "visual_coverage_frames.json"
    window_json = output_dir / "visual_coverage_windows.json"
    frame_json.write_text(json.dumps(frame_records, indent=2, sort_keys=True), encoding="utf-8")
    window_json.write_text(json.dumps(window_records, indent=2, sort_keys=True), encoding="utf-8")
    paths = {"frames_json": frame_json, "windows_json": window_json}
    try:
        import pandas as pd
    except ImportError:
        return paths
    frame_parquet = output_dir / "visual_coverage_frames.parquet"
    window_parquet = output_dir / "visual_coverage_windows.parquet"
    pd.DataFrame(frame_records).to_parquet(frame_parquet, index=False)
    pd.DataFrame(window_records).to_parquet(window_parquet, index=False)
    paths["frames_parquet"] = frame_parquet
    paths["windows_parquet"] = window_parquet
    return paths
