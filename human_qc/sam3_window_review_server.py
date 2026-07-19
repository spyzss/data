"""Safe HTTP transport for the standalone SAM3 window-review workbench."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .sam3_window_review import (
    ReviewBundle,
    ReviewConflictError,
    ReviewValidationError,
    Sam3WindowReviewStore,
)


MAX_REQUEST_BYTES = 1024 * 1024
STATIC_ROOT = Path(__file__).with_name("static")
STATIC_FILES = {
    "/": "sam3_window_review.html",
    "/static/sam3_window_review.js": "sam3_window_review.js",
    "/static/sam3_window_review.css": "sam3_window_review.css",
}


class Sam3WindowReviewHttpServer(ThreadingHTTPServer):
    bundle: ReviewBundle
    store: Sam3WindowReviewStore


def create_sam3_window_review_server(
    host: str,
    port: int,
    *,
    bundle: ReviewBundle,
    store: Sam3WindowReviewStore,
) -> Sam3WindowReviewHttpServer:
    server = Sam3WindowReviewHttpServer((host, port), Sam3WindowReviewRequestHandler)
    server.bundle = bundle
    server.store = store
    return server


class Sam3WindowReviewRequestHandler(BaseHTTPRequestHandler):
    server: Sam3WindowReviewHttpServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/review-bundle":
            state = self.server.store.snapshot()
            self._send_json(
                {
                    "bundle": self.server.bundle.to_dict(state.get("reviews", {})),
                    "state": state,
                }
            )
            return
        if path.startswith("/assets/"):
            if self._serve_asset(path):
                return
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "evidence file not found")
            return
        if path in STATIC_FILES:
            if self._serve_static(path):
                return
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "static file not found")
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")

    def do_POST(self) -> None:  # noqa: N802
        raw_path = urlsplit(self.path).path
        prefix = "/api/reviews/"
        if not raw_path.startswith(prefix) or not raw_path[len(prefix) :]:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")
            return
        review_id = unquote(raw_path[len(prefix) :])
        if not review_id or "/" in review_id or "\\" in review_id:
            self._send_error(HTTPStatus.BAD_REQUEST, "bad_request", "invalid review_id path")
            return
        try:
            payload = self._read_json()
            review = self.server.store.save(
                review_id,
                payload.get("verdict"),
                payload.get("reviewer"),
                expected_revision=payload.get("expected_revision"),
            )
        except ReviewConflictError as exc:
            self._send_error(HTTPStatus.CONFLICT, "state_conflict", str(exc))
            return
        except KeyError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", str(exc))
            return
        except (ReviewValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, "bad_request", str(exc))
            return
        except OSError:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed", "unable to save review")
            return
        self._send_json({"review": review})

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError as exc:
            raise ReviewValidationError("invalid Content-Length") from exc
        if length <= 0:
            raise ReviewValidationError("empty request body")
        if length > MAX_REQUEST_BYTES:
            raise ReviewValidationError("request body too large")
        raw = self.rfile.read(length)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ReviewValidationError("payload must be an object")
        return value

    def _serve_asset(self, raw_path: str) -> bool:
        encoded = raw_path[len("/assets/") :]
        relative_text = unquote(encoded)
        if not relative_text or "\\" in relative_text or "\x00" in relative_text:
            return False
        relative = Path(relative_text)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            return False
        normalized = relative.as_posix()
        if normalized not in self.server.bundle.allowed_asset_paths:
            return False
        root = self.server.bundle.assets_root.resolve()
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        if not candidate.is_file():
            return False
        self._send_file(candidate)
        return True

    def _serve_static(self, path: str) -> bool:
        filename = STATIC_FILES[path]
        candidate = STATIC_ROOT / filename
        if not candidate.is_file():
            return False
        self._send_file(candidate)
        return True

    def _send_file(self, path: Path) -> None:
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: MappingLike, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, code: str, message: str) -> None:
        self._send_json({"error": {"code": code, "message": message}}, status=status)


MappingLike = dict[str, Any]


__all__ = ["create_sam3_window_review_server"]
