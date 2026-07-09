#!/usr/bin/env python3
"""Serve manual review artifacts and persist autosaved labels."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import tempfile
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("serve_manual_review")
SAVE_ENDPOINT = "/api/manual-review/save"


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
        if self.path != SAVE_ENDPOINT:
            self.send_json({"ok": False, "error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self.read_json_payload()
            result = save_manual_review_payload(payload, self.save_dir)
        except ValueError as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except OSError as exc:
            LOGGER.exception("Failed to save manual review payload")
            self.send_json(
                {"ok": False, "error": f"write_failed: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self.send_json({"ok": True, **result})

    def read_json_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            raise ValueError("empty_request_body")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
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


def save_manual_review_payload(payload: dict[str, Any], save_dir: Path) -> dict[str, Any]:
    manual_labels_csv = payload.get("manual_labels_csv")
    progress_json = payload.get("progress_json")
    if not isinstance(manual_labels_csv, str):
        raise ValueError("manual_labels_csv_must_be_string")
    if not isinstance(progress_json, dict):
        raise ValueError("progress_json_must_be_object")

    source = str(payload.get("source") or "autosave")
    reviewer = str(payload.get("reviewer") or "")
    run_label = str(payload.get("run_label") or "")
    saved_at = datetime.now(timezone.utc)
    timestamp = saved_at.strftime("%Y%m%d_%H%M%S")
    row_count = count_csv_rows(manual_labels_csv)

    save_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(save_dir / "manual_labels_autosave.csv", manual_labels_csv)
    write_text_atomic(
        save_dir / "manual_review_progress_autosave.json",
        json.dumps(progress_json, indent=2, ensure_ascii=False) + "\n",
    )
    meta = {
        "saved_at": saved_at.isoformat(),
        "reviewer": reviewer,
        "run_label": run_label,
        "source": source,
        "row_count": row_count,
        "git_commit": git_commit(),
    }
    write_text_atomic(
        save_dir / "manual_review_autosave_meta.json",
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
    )
    if source == "explicit_save":
        write_text_atomic(save_dir / f"manual_labels_{timestamp}.csv", manual_labels_csv)
        write_text_atomic(
            save_dir / f"manual_review_progress_{timestamp}.json",
            json.dumps(progress_json, indent=2, ensure_ascii=False) + "\n",
        )
    return {"saved_at": meta["saved_at"], "row_count": row_count}


def write_text_atomic(path: Path, text: str) -> None:
    target = path.resolve()
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=parent, delete=False) as handle:
        handle.write(text)
        tmp_path = Path(handle.name)
    tmp_path.replace(target)


def count_csv_rows(value: str) -> int:
    reader = csv.reader(value.splitlines())
    rows = list(reader)
    return max(0, len(rows) - 1)


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
