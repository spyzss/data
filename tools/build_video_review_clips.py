#!/usr/bin/env python3
"""Build short video clips and a static HTML review page for review queue items."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_manual_review_queue import (  # noqa: E402
    CONFIDENCE_ENUM,
    FAILURE_MODE_ENUM,
    MANUAL_OUTCOME_ENUM,
    MANUAL_TEMPLATE_COLUMNS,
    SEVERITY_ENUM,
)


LOGGER = logging.getLogger("build_video_review_clips")
DEFAULT_ASSET_CLIP_SEC = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build short video clips and a static HTML page for manual review."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--review-queue", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--padding-sec", type=float, default=1.0)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_clip_rows(
        read_manifest(args.manifest),
        read_review_queue(args.review_queue),
        output_dir=args.output_dir,
        padding_sec=args.padding_sec,
        max_items=args.max_items,
    )
    render_clips(rows, output_dir=args.output_dir, overwrite=args.overwrite)

    csv_path = args.output_dir / "review_queue_with_clips.csv"
    html_path = args.output_dir / "review_index_video.html"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    html_path.write_text(build_review_index_video_html(rows), encoding="utf-8")
    LOGGER.info("Wrote %s", csv_path)
    LOGGER.info("Wrote %s", html_path)
    return 0


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"supplier_id", "asset_id", "video_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    return normalize_dataframe(df)


def read_review_queue(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"supplier_id", "asset_id", "review_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"review queue missing required columns: {missing}")
    return normalize_dataframe(df)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.where(pd.notna(df), "")


def build_clip_rows(
    manifest_df: pd.DataFrame,
    review_df: pd.DataFrame,
    *,
    output_dir: Path,
    padding_sec: float,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    if max_items is not None:
        review_df = review_df.head(max_items)
    manifest_columns = [
        column
        for column in ("supplier_id", "asset_id", "video_path", "fps", "frame_count", "duration_sec")
        if column in manifest_df.columns
    ]
    merged = review_df.merge(
        manifest_df[manifest_columns],
        on=["supplier_id", "asset_id"],
        how="left",
        suffixes=("", "_manifest"),
    )
    rows: list[dict[str, Any]] = []
    clips_dir = output_dir / "clips"
    for index, row in enumerate(merged.to_dict(orient="records"), start=1):
        output = dict(row)
        review_id = str(output.get("review_id") or f"review_{index:04d}")
        fps = resolve_fps(output)
        duration_sec = float_or_none(output.get("duration_sec"))
        timing = compute_clip_timing(
            window_start_frame=int_or_none(output.get("window_start_frame")),
            window_end_frame=int_or_none(output.get("window_end_frame")),
            representative_frame=int_or_none(output.get("representative_frame")),
            fps=fps,
            asset_duration_sec=duration_sec,
            padding_sec=padding_sec,
        )
        display_clip_path = f"clips/{safe_filename(review_id)}.mp4"
        output["clip_path"] = str(clips_dir / f"{safe_filename(review_id)}.mp4")
        output["display_clip_path"] = display_clip_path
        output["clip_start_time_sec"] = round(timing["start_time_sec"], 6)
        output["clip_duration_sec"] = round(timing["duration_sec"], 6)
        output["clip_error"] = timing.get("error", "")
        rows.append(output)
    return rows


def compute_clip_timing(
    *,
    window_start_frame: int | None,
    window_end_frame: int | None,
    representative_frame: int | None,
    fps: float | None,
    asset_duration_sec: float | None,
    padding_sec: float,
) -> dict[str, Any]:
    padding_sec = max(0.0, padding_sec)
    if window_start_frame is not None and window_end_frame is not None:
        if fps is None or fps <= 0:
            return {"start_time_sec": 0.0, "duration_sec": 0.0, "error": "missing_or_invalid_fps"}
        start_frame = min(window_start_frame, window_end_frame)
        end_frame = max(window_start_frame, window_end_frame)
        start_time = max(0.0, start_frame / fps - padding_sec)
        duration = (end_frame - start_frame + 1) / fps + (2.0 * padding_sec)
        return clamp_timing(start_time, duration, asset_duration_sec)

    if representative_frame is not None and fps is not None and fps > 0:
        center = representative_frame / fps
        start_time = max(0.0, center - (DEFAULT_ASSET_CLIP_SEC / 2.0))
    else:
        start_time = 0.0
    return clamp_timing(start_time, DEFAULT_ASSET_CLIP_SEC, asset_duration_sec)


def clamp_timing(
    start_time_sec: float,
    duration_sec: float,
    asset_duration_sec: float | None,
) -> dict[str, Any]:
    start_time_sec = max(0.0, start_time_sec)
    duration_sec = max(0.0, duration_sec)
    if asset_duration_sec is not None and asset_duration_sec > 0:
        if start_time_sec >= asset_duration_sec:
            start_time_sec = max(0.0, asset_duration_sec - min(DEFAULT_ASSET_CLIP_SEC, asset_duration_sec))
        duration_sec = min(duration_sec, max(0.0, asset_duration_sec - start_time_sec))
    return {"start_time_sec": start_time_sec, "duration_sec": duration_sec, "error": ""}


def render_clips(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    overwrite: bool,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    for row in rows:
        if row.get("clip_error"):
            continue
        video_path = row.get("video_path")
        if not video_path:
            row["clip_error"] = "missing_video_path"
            continue
        source = Path(str(video_path))
        if not source.exists():
            row["clip_error"] = "video_path_not_found"
            continue
        target = Path(str(row["clip_path"]))
        if target.exists() and not overwrite:
            continue
        if ffmpeg is None:
            row["clip_error"] = "ffmpeg_not_found"
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        result = run_ffmpeg_clip(
            ffmpeg=ffmpeg,
            source=source,
            target=target,
            start_time_sec=float(row["clip_start_time_sec"]),
            duration_sec=float(row["clip_duration_sec"]),
            overwrite=overwrite,
        )
        if result:
            row["clip_error"] = result
            LOGGER.warning("Failed to write clip for %s: %s", row.get("review_id"), result)


def run_ffmpeg_clip(
    *,
    ffmpeg: str,
    source: Path,
    target: Path,
    start_time_sec: float,
    duration_sec: float,
    overwrite: bool,
) -> str:
    if duration_sec <= 0:
        return "non_positive_clip_duration"
    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-loglevel",
        "error",
        "-ss",
        f"{start_time_sec:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration_sec:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(target),
    ]
    try:
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    except OSError as exc:
        return f"ffmpeg_error: {exc}"
    if completed.returncode != 0:
        return completed.stderr.strip() or f"ffmpeg_exit_{completed.returncode}"
    return ""


def build_review_index_video_html(rows: list[dict[str, Any]]) -> str:
    rows_json = json.dumps(json_safe(rows), ensure_ascii=False).replace("</", "<\\/")
    enum_json = json.dumps(
        {
            "manual_outcome": MANUAL_OUTCOME_ENUM,
            "failure_mode": FAILURE_MODE_ENUM,
            "severity": SEVERITY_ENUM,
            "confidence": CONFIDENCE_ENUM,
        }
    )
    manual_columns_json = json.dumps(MANUAL_TEMPLATE_COLUMNS)
    return (
        "<!doctype html>\n"
        "<html><head><meta charset=\"utf-8\">\n"
        "<title>Video Manual Review</title>\n"
        "<style>\n"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;line-height:1.35;color:#202124;background:#fff}\n"
        ".toolbar{position:sticky;top:0;background:#fff;border-bottom:1px solid #dadce0;padding:12px 0;margin-bottom:16px;z-index:2}\n"
        "button{margin-right:8px;padding:7px 10px;border:1px solid #c7cdd4;background:#f8f9fa;border-radius:4px;cursor:pointer}button.primary{background:#1a73e8;color:white;border-color:#1a73e8}\n"
        ".item{border:1px solid #dadce0;border-radius:6px;margin:14px 0;padding:12px}.grid{display:grid;grid-template-columns:minmax(280px,420px) 1fr;gap:14px}.video{width:100%;max-height:280px;background:#111}.no-clip{height:160px;border:1px dashed #c7cdd4;color:#6b7280;display:flex;align-items:center;justify-content:center}\n"
        ".meta{display:grid;grid-template-columns:repeat(3,minmax(120px,1fr));gap:6px 12px;font-size:13px}.meta b{display:block;color:#5f6368;font-size:12px}.auto,.human{border:1px solid #eceff1;border-radius:6px;padding:10px;margin-top:10px}.auto{background:#fbfcfe}.human{background:#fffdf8}.title{font-weight:700;margin-bottom:8px}.metrics,.reason{white-space:pre-wrap;background:#f8f9fa;border:1px solid #eceff1;padding:8px;margin-top:8px;font-size:12px;overflow:auto}.controls{display:grid;grid-template-columns:repeat(3,minmax(140px,1fr));gap:8px}.controls label{font-size:12px;color:#5f6368}.controls select,.controls input,.controls textarea{width:100%;box-sizing:border-box;margin-top:3px;padding:6px;border:1px solid #c7cdd4;border-radius:4px;font:inherit}.controls textarea{min-height:58px;grid-column:span 2}.status{margin-left:8px;color:#188038;font-size:13px}\n"
        "</style></head><body>\n"
        "<h1>Video Manual Review</h1>\n"
        "<div class=\"toolbar\"><button class=\"primary\" onclick=\"exportManualLabelsCsv()\">Export manual_labels.csv</button><button onclick=\"saveProgress()\">Save progress to localStorage</button><button onclick=\"loadProgress()\">Load progress from localStorage</button><button onclick=\"clearProgress()\">Clear local saved progress</button><span id=\"status\" class=\"status\"></span></div>\n"
        "<div id=\"root\"></div>\n"
        "<script>\n"
        f"const REVIEW_ROWS = {rows_json};\n"
        f"const ENUMS = {enum_json};\n"
        f"const MANUAL_COLUMNS = {manual_columns_json};\n"
        "const STORAGE_KEY='video_review_progress_v1';\n"
        "function escapeHtml(value){return String(value ?? '').replace(/[&<>\"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[ch]));}\n"
        "function fieldId(index,field){return `field-${index}-${field}`;}\n"
        "function optionHtml(values,selected){return values.map(v=>`<option value=\"${escapeHtml(v)}\" ${v===selected?'selected':''}>${escapeHtml(v)}</option>`).join('');}\n"
        "function defaultFailureMode(row){return ENUMS.failure_mode.includes(row.suggested_issue_type) ? row.suggested_issue_type : 'unknown';}\n"
        "function defaultSeverity(row){return ENUMS.severity.includes(row.severity_suggestion) ? row.severity_suggestion : 'medium';}\n"
        "function render(){const root=document.getElementById('root'); let html=''; REVIEW_ROWS.forEach((row,index)=>{const clip=row.display_clip_path && !row.clip_error ? `<video class=\"video\" controls src=\"${escapeHtml(row.display_clip_path)}\"></video>` : `<div class=\"no-clip\">No clip${row.clip_error ? ': '+escapeHtml(row.clip_error) : ''}</div>`; html+=`<article class=\"item\"><div class=\"grid\"><div>${clip}</div><div><div class=\"meta\"><div><b>review_id</b>${escapeHtml(row.review_id)}</div><div><b>supplier_id</b>${escapeHtml(row.supplier_id)}</div><div><b>asset_id</b>${escapeHtml(row.asset_id)}</div><div><b>window</b>${escapeHtml(row.window_start_frame)}-${escapeHtml(row.window_end_frame)}</div><div><b>clip_start_time_sec</b>${escapeHtml(row.clip_start_time_sec)}</div><div><b>clip_duration_sec</b>${escapeHtml(row.clip_duration_sec)}</div></div><section class=\"auto\"><div class=\"title\">Auto result</div><div class=\"meta\"><div><b>suggested_issue_type</b>${escapeHtml(row.suggested_issue_type)}</div><div><b>auto_verdict</b>${escapeHtml(row.auto_verdict)}</div><div><b>severity_suggestion</b>${escapeHtml(row.severity_suggestion)}</div></div><div class=\"metrics\"><b>key_metrics_json</b>\\n${escapeHtml(row.key_metrics_json)}</div><div class=\"reason\"><b>reason</b>\\n${escapeHtml(row.reason)}</div></section><section class=\"human\"><div class=\"title\">Human label</div><div class=\"controls\"><label>manual_outcome<select id=\"${fieldId(index,'manual_outcome')}\">${optionHtml(ENUMS.manual_outcome,'review')}</select></label><label>failure_mode<select id=\"${fieldId(index,'failure_mode')}\">${optionHtml(ENUMS.failure_mode,defaultFailureMode(row))}</select></label><label>severity<select id=\"${fieldId(index,'severity')}\">${optionHtml(ENUMS.severity,defaultSeverity(row))}</select></label><label>confidence<select id=\"${fieldId(index,'confidence')}\">${optionHtml(ENUMS.confidence,'medium')}</select></label><label>reviewer<input id=\"${fieldId(index,'reviewer')}\"></label><label>comment optional<textarea id=\"${fieldId(index,'comment')}\"></textarea></label></div></section></div></div></article>`;}); root.innerHTML=html;}\n"
        "function getField(index,field){const el=document.getElementById(fieldId(index,field)); return el ? el.value : '';}\n"
        "function setField(index,field,value){const el=document.getElementById(fieldId(index,field)); if(el && value !== undefined && value !== null){el.value=value;}}\n"
        "function collectManualRows(){return REVIEW_ROWS.map((row,index)=>({review_id:row.review_id,supplier_id:row.supplier_id,asset_id:row.asset_id,window_start_frame:row.window_start_frame,window_end_frame:row.window_end_frame,representative_frame:row.representative_frame,auto_verdict:row.auto_verdict,suggested_issue_type:row.suggested_issue_type,severity_suggestion:row.severity_suggestion,key_metrics_json:row.key_metrics_json,reason:row.reason,manual_outcome:getField(index,'manual_outcome'),failure_mode:getField(index,'failure_mode'),severity:getField(index,'severity'),confidence:getField(index,'confidence'),comment:getField(index,'comment'),reviewer:getField(index,'reviewer')}));}\n"
        "function csvEscape(value){const text=String(value ?? ''); return /[\",\\n\\r]/.test(text) ? '\"' + text.replace(/\"/g,'\"\"') + '\"' : text;}\n"
        "function rowsToCsv(rows){return MANUAL_COLUMNS.join(',')+'\\n'+rows.map(row=>MANUAL_COLUMNS.map(col=>csvEscape(row[col])).join(',')).join('\\n')+'\\n';}\n"
        "function exportManualLabelsCsv(){const csv=rowsToCsv(collectManualRows()); const blob=new Blob([csv],{type:'text/csv;charset=utf-8'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='manual_labels.csv'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url); setStatus('Exported manual_labels.csv');}\n"
        "function saveProgress(){localStorage.setItem(STORAGE_KEY,JSON.stringify(collectManualRows())); setStatus('Saved progress locally');}\n"
        "function loadProgress(){const raw=localStorage.getItem(STORAGE_KEY); if(!raw){setStatus('No saved progress'); return;} JSON.parse(raw).forEach((row,index)=>['manual_outcome','failure_mode','severity','confidence','comment','reviewer'].forEach(field=>setField(index,field,row[field]))); setStatus('Loaded local progress');}\n"
        "function clearProgress(){localStorage.removeItem(STORAGE_KEY); setStatus('Cleared local progress');}\n"
        "function setStatus(text){document.getElementById('status').textContent=text;}\n"
        "render();\n"
        "</script></body></html>\n"
    )


def resolve_fps(row: dict[str, Any]) -> float | None:
    fps = float_or_none(row.get("fps"))
    if fps is not None and fps > 0:
        return fps
    frame_count = float_or_none(row.get("frame_count"))
    duration = float_or_none(row.get("duration_sec"))
    if frame_count is not None and duration is not None and duration > 0:
        return frame_count / duration
    return None


def int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
