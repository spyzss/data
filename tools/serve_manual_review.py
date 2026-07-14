#!/usr/bin/env python3
"""Serve manual review artifacts and persist autosaved labels."""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


LOGGER = logging.getLogger("serve_manual_review")
SAVE_ENDPOINT_SUFFIX = "/__manual_review_save__"
LEGACY_SAVE_ENDPOINT = "/api/manual-review/save"
MAX_REQUEST_BYTES = 10 * 1024 * 1024
MANUAL_LABEL_COLUMNS = (
    "review_id",
    "segment_id",
    "supplier_id",
    "asset_id",
    "window_start_frame",
    "window_end_frame",
    "representative_frame",
    "affected_start_frame",
    "affected_end_frame",
    "auto_verdict",
    "suggested_issue_type",
    "severity_suggestion",
    "key_metrics_json",
    "reason",
    "manual_outcome",
    "failure_mode",
    "severity",
    "confidence",
    "acceptance_status",
    "reviewer",
    "comment",
)
AUTHORITATIVE_FILENAMES = (
    "manual_labels.csv",
    "manual_review_progress.json",
    "manual_review_save_metadata.json",
)


class RequestBodyTooLarge(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve manual review UI with autosave support.")
    parser.add_argument("--review-dir", required=True, type=Path)
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8896)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    server = create_server(
        host=args.host,
        port=args.port,
        review_dir=args.review_dir,
        save_dir=args.save_dir,
    )
    LOGGER.info("Serving %s at http://%s:%s", args.review_dir, args.host, args.port)
    LOGGER.info("Manual review autosaves will be written to %s", args.save_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down")
    finally:
        server.server_close()
    return 0


def create_server(
    *,
    host: str,
    port: int,
    review_dir: Path,
    save_dir: Path,
) -> ThreadingHTTPServer:
    review_root = review_dir.resolve()
    save_root = save_dir.resolve()
    handler = partial(ManualReviewRequestHandler, directory=str(review_root), save_dir=save_root)
    return ThreadingHTTPServer((host, port), handler)


class ManualReviewRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, save_dir: Path, **kwargs: Any) -> None:
        self.save_dir = save_dir
        super().__init__(*args, **kwargs)

    def do_POST(self) -> None:  # noqa: N802 - http.server hook name.
        request_path = urlsplit(self.path).path
        if not is_save_endpoint(request_path):
            self.send_json({"ok": False, "error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self.read_json_payload()
            result = save_manual_review_payload(payload, self.save_dir)
        except RequestBodyTooLarge as exc:
            self.close_connection = True
            self.send_json(
                {"ok": False, "error": str(exc)},
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return
        except ValueError as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except OSError as exc:
            LOGGER.exception("Failed to save manual review payload")
            self.send_json(
                {"ok": False, "error": "write_failed"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self.send_json({"ok": True, **result})

    def read_json_payload(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError as exc:
            raise ValueError("invalid_content_length") from exc
        if length <= 0:
            raise ValueError("empty_request_body")
        if length > MAX_REQUEST_BYTES:
            raise RequestBodyTooLarge("request_body_too_large")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid_json: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("payload_must_be_object")
        return payload

    def send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def is_save_endpoint(path: str) -> bool:
    return path == LEGACY_SAVE_ENDPOINT or path.endswith(SAVE_ENDPOINT_SUFFIX)


def save_manual_review_payload(payload: dict[str, Any], save_dir: Path) -> dict[str, Any]:
    validated = validate_manual_review_payload(payload)
    manual_labels_csv = validated["manual_labels_csv"]
    progress_json = validated["progress_json"]
    source = validated["source"]
    reviewer = validated["reviewer"]
    run_label = validated["run_label"]
    saved_at = datetime.now(timezone.utc)
    timestamp = saved_at.strftime("%Y%m%d_%H%M%S")
    row_count = validated["manual_label_row_count"]
    progress_text = json.dumps(progress_json, indent=2, ensure_ascii=False) + "\n"
    meta = {
        "saved_at_utc": saved_at.isoformat(),
        "reviewer": reviewer,
        "run_label": run_label,
        "source": source,
        "manual_label_row_count": row_count,
        "review_item_count": validated["review_item_count"],
        "source_page": validated["source_page"],
        "schema_version": validated["schema_version"],
        "git_commit": git_commit(),
    }
    metadata_text = json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    files = {
        "manual_labels.csv": manual_labels_csv,
        "manual_review_progress.json": progress_text,
        "manual_review_save_metadata.json": metadata_text,
        # Preserve names consumed by older runbooks while the canonical files above
        # become the authoritative downstream contract.
        "manual_labels_autosave.csv": manual_labels_csv,
        "manual_review_progress_autosave.json": progress_text,
        "manual_review_autosave_meta.json": metadata_text,
    }
    if source == "explicit_save":
        files[f"manual_labels_{timestamp}.csv"] = manual_labels_csv
        files[f"manual_review_progress_{timestamp}.json"] = progress_text
    write_text_files_atomically(Path(save_dir), files)
    return {
        "saved_at_utc": meta["saved_at_utc"],
        "manual_label_row_count": row_count,
        "written_files": sorted(files),
        # Response aliases retain compatibility with existing callers.
        "saved_at": meta["saved_at_utc"],
        "row_count": row_count,
    }


def validate_manual_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    manual_labels_csv = payload.get("manual_labels_csv")
    progress_json = payload.get("progress_json")
    if not isinstance(manual_labels_csv, str):
        raise ValueError("manual_labels_csv_must_be_string")
    if not isinstance(progress_json, dict):
        raise ValueError("progress_json_must_be_object")
    segments = progress_json.get("segmentsByReviewId")
    if not isinstance(segments, dict):
        raise ValueError("progress_json.segmentsByReviewId_must_be_object")

    reader = csv.DictReader(io.StringIO(manual_labels_csv, newline=""))
    if reader.fieldnames != list(MANUAL_LABEL_COLUMNS):
        raise ValueError("manual_labels_csv_schema_mismatch")
    manual_rows = list(reader)
    if any(None in row for row in manual_rows):
        raise ValueError("manual_labels_csv_row_has_extra_columns")
    if any(value is None for row in manual_rows for value in row.values()):
        raise ValueError("manual_labels_csv_row_has_missing_columns")

    source = str(payload.get("source") or "autosave")
    if source not in {"autosave", "explicit_save"}:
        raise ValueError("source_must_be_autosave_or_explicit_save")
    review_item_count = payload.get("review_item_count", len(segments))
    if isinstance(review_item_count, bool) or not isinstance(review_item_count, int):
        raise ValueError("review_item_count_must_be_integer")
    if review_item_count < 0:
        raise ValueError("review_item_count_must_be_nonnegative")
    if review_item_count != len(segments):
        raise ValueError("review_item_count_does_not_match_progress")

    source_page = payload.get("source_page", "")
    schema_version = payload.get("schema_version", "manual_review_progress.v2")
    if not isinstance(source_page, str):
        raise ValueError("source_page_must_be_string")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("schema_version_must_be_nonempty_string")
    return {
        "manual_labels_csv": manual_labels_csv,
        "progress_json": progress_json,
        "source": source,
        "reviewer": str(payload.get("reviewer") or ""),
        "run_label": str(payload.get("run_label") or ""),
        "source_page": source_page,
        "schema_version": schema_version,
        "review_item_count": review_item_count,
        "manual_label_row_count": len(manual_rows),
    }


def write_text_atomic(path: Path, text: str) -> None:
    write_text_files_atomically(path.parent, {path.name: text})


def write_text_files_atomically(save_dir: Path, files: dict[str, str]) -> None:
    save_root = save_dir.resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    installed: set[str] = set()
    try:
        for filename, text in files.items():
            if Path(filename).name != filename:
                raise ValueError(f"unsafe_output_filename: {filename}")
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=save_root,
                delete=False,
            ) as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
                staged[filename] = Path(handle.name)

        for filename in files:
            target = save_root / filename
            if not target.exists():
                continue
            with tempfile.NamedTemporaryFile(dir=save_root, delete=False) as handle:
                backup = Path(handle.name)
            shutil.copy2(target, backup)
            backups[filename] = backup

        for filename, temp_path in staged.items():
            os.replace(temp_path, save_root / filename)
            installed.add(filename)
    except Exception:
        for filename in installed:
            target = save_root / filename
            backup = backups.get(filename)
            try:
                if backup is not None and backup.exists():
                    os.replace(backup, target)
                elif target.exists():
                    target.unlink()
            except OSError:
                LOGGER.exception("Failed to restore %s after save error", target)
        raise
    finally:
        for path in [*staged.values(), *backups.values()]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Could not remove temporary save file %s", path)


def count_csv_rows(value: str) -> int:
    return len(list(csv.DictReader(io.StringIO(value, newline=""))))


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
