"""Standard-library HTTP transport for the independent semantic application."""

from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlsplit

from qc_common.report import StaleReportRevisionError
from qc_common.reviewer_lease import LeaseConflictError, LeaseTokenError

from .application import SemanticCalibrationApplication, jsonable
from .service import (
    LeaseError,
    PendingEditError,
    SemanticEligibilityError,
    StaleSemanticRevisionError,
    TaskStateError,
)


MAX_REQUEST_BYTES = 10 * 1024 * 1024
VIDEO_CHUNK_SIZE = 64 * 1024

SEMANTIC_ROUTES = {
    "GET /api/semantic/assets",
    "GET /api/semantic/assets/{asset_id}/task",
    "GET /api/semantic/assets/{asset_id}/video",
    "POST /api/semantic/assets/{asset_id}/lease/acquire",
    "POST /api/semantic/assets/{asset_id}/lease/renew",
    "POST /api/semantic/assets/{asset_id}/lease/release",
    "POST /api/semantic/assets/{asset_id}/boundary/pending",
    "POST /api/semantic/assets/{asset_id}/text/pending",
    "POST /api/semantic/assets/{asset_id}/pending/confirm",
    "POST /api/semantic/assets/{asset_id}/pending/cancel",
    "POST /api/semantic/assets/{asset_id}/complete",
}

_MUTATIONS = {
    "boundary/pending": "semantic_boundary_pending",
    "text/pending": "semantic_text_pending",
    "pending/confirm": "semantic_pending_confirm",
    "pending/cancel": "semantic_pending_cancel",
    "complete": "semantic_complete",
}


class ApiRequestError(ValueError):
    """Malformed public request."""


class UnsupportedMediaTypeError(ApiRequestError):
    """A state-changing route was not sent as JSON."""


def _parts(path: str) -> tuple[str | None, str]:
    values = [unquote(part) for part in urlsplit(path).path.strip("/").split("/") if part]
    if values == ["api", "semantic", "assets"]:
        return None, "assets"
    if len(values) >= 5 and values[:3] == ["api", "semantic", "assets"]:
        return values[3], "/".join(values[4:])
    return None, ""


def _error(exc: Exception) -> tuple[int, str, str]:
    if isinstance(exc, SemanticEligibilityError):
        message = "semantic task is not ready"
        if "skipped_due_to_fail" in str(exc):
            message = "semantic task was skipped due to review failure"
        return HTTPStatus.CONFLICT, "semantic_not_ready", message
    if isinstance(exc, LeaseConflictError):
        return HTTPStatus.LOCKED, "lease_held", "asset is held by another reviewer"
    if isinstance(exc, (LeaseTokenError, LeaseError)):
        return HTTPStatus.LOCKED, "lease_invalid", "lease token is stale or expired"
    if isinstance(exc, (StaleReportRevisionError, StaleSemanticRevisionError)):
        return HTTPStatus.CONFLICT, "stale_revision", "report revision is stale"
    if isinstance(exc, (PendingEditError, TaskStateError)):
        return HTTPStatus.CONFLICT, "state_conflict", "semantic task state conflicts with this operation"
    if isinstance(exc, (KeyError, FileNotFoundError)):
        return HTTPStatus.NOT_FOUND, "not_found", "semantic asset was not found"
    if isinstance(exc, UnsupportedMediaTypeError):
        return HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type", "Content-Type must be application/json"
    if isinstance(exc, ApiRequestError):
        return HTTPStatus.BAD_REQUEST, "bad_request", str(exc)
    if isinstance(exc, (ValueError, TypeError, json.JSONDecodeError)):
        return HTTPStatus.BAD_REQUEST, "bad_request", "request payload is invalid"
    return HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", "internal server error"


class SemanticCalibrationRequestHandler(BaseHTTPRequestHandler):
    server: "SemanticCalibrationHttpServer"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        asset_id, route = _parts(self.path)
        try:
            if asset_id is None and route == "assets":
                self._send_json({"assets": self.server.application.list_assets()})
                return
            if asset_id is not None and route == "task":
                self._send_json({"task": self.server.application.get_task(asset_id)})
                return
            if asset_id is not None and route == "video":
                self._serve_video(asset_id)
                return
            if self._allowed_methods():
                self._method_not_allowed()
                return
            if not urlsplit(self.path).path.startswith("/api/") and not urlsplit(self.path).path.startswith("/evidence/"):
                if self._serve_static():
                    return
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint", asset_id)
        except Exception as exc:
            self._handle(exc, asset_id)

    def do_POST(self) -> None:  # noqa: N802
        asset_id, route = _parts(self.path)
        if asset_id is None or not route:
            if self._allowed_methods():
                self._method_not_allowed()
                return
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint", asset_id)
            return
        known_routes = {"lease/acquire", "lease/renew", "lease/release", *_MUTATIONS}
        if route not in known_routes:
            if self._allowed_methods():
                self._method_not_allowed()
                return
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint", asset_id)
            return
        try:
            payload = self._read_json()
            if route == "lease/acquire":
                reviewer = payload.get("reviewer")
                if not isinstance(reviewer, str) or not reviewer.strip():
                    raise ApiRequestError("reviewer is required")
                lease = self.server.application.acquire_lease(asset_id, reviewer, payload.get("ttl_seconds"))
                self._send_json({"lease": jsonable(lease)})
                return
            if route == "lease/release":
                self._require_lease_token(payload)
                self._send_json(
                    self.server.application.release_lease(asset_id, payload["lease_token"])
                )
                return
            if route == "lease/renew":
                self._require_write_fields(payload)
                self._validate_revision(asset_id, payload["expected_revision"])
                lease = self.server.application.renew_lease(
                    asset_id,
                    payload["lease_token"],
                    payload.get("ttl_seconds"),
                )
                self._send_json({"lease": jsonable(lease)})
                return
            operation = _MUTATIONS.get(route)
            self._require_write_fields(payload)
            self._validate_revision(asset_id, payload["expected_revision"])
            method = getattr(self.server.application, "mutate", None)
            if not callable(method):
                raise KeyError(operation)
            result = method(operation, asset_id, payload)
            self._send_json({"task": jsonable(result)})
        except Exception as exc:
            self._handle(exc, asset_id)

    def _read_json(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            raise UnsupportedMediaTypeError("Content-Type must be application/json")
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "0")
        except ValueError as exc:
            raise ApiRequestError("invalid Content-Length") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ApiRequestError("request body is too large")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiRequestError("invalid JSON body") from exc
        if not isinstance(value, Mapping):
            raise ApiRequestError("payload must be a JSON object")
        return dict(value)

    @staticmethod
    def _require_lease_token(payload: Mapping[str, Any]) -> None:
        token = payload.get("lease_token")
        if not isinstance(token, str) or not token:
            raise ApiRequestError("lease_token must be a non-empty string")

    @staticmethod
    def _require_write_fields(payload: Mapping[str, Any]) -> None:
        missing = [field for field in ("expected_revision", "lease_token") if field not in payload]
        if missing:
            raise ApiRequestError("missing required field(s): " + ", ".join(missing))
        revision = payload["expected_revision"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ApiRequestError("expected_revision must be a non-negative integer")
        token = payload["lease_token"]
        if not isinstance(token, str) or not token:
            raise ApiRequestError("lease_token must be a non-empty string")

    def _validate_revision(self, asset_id: str, expected_revision: int) -> None:
        getter = getattr(self.server.application, "current_revision", None)
        if not callable(getter):
            return
        current = getter(asset_id)
        if isinstance(current, int) and current != expected_revision:
            raise StaleReportRevisionError("stale report revision")

    def _serve_static(self) -> bool:
        root = self.server.static_root
        if root is None:
            return False
        relative = Path(unquote(urlsplit(self.path).path.lstrip("/")) or "index.html")
        if any(part in {"", ".", ".."} for part in relative.parts):
            return False
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            return False
        if not candidate.is_file():
            return False
        body = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        return True

    def _serve_video(self, asset_id: str) -> None:
        resolver = getattr(self.server.application, "semantic_video_path", None)
        if not callable(resolver):
            raise KeyError(asset_id)
        path = Path(resolver(asset_id))
        size = path.stat().st_size
        start = 0
        end = size - 1
        partial = False
        header = self.headers.get("Range")
        if header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
            if match is None or size == 0:
                self._send_range_error(size, asset_id)
                return
            raw_start, raw_end = match.groups()
            if not raw_start and not raw_end:
                self._send_range_error(size, asset_id)
                return
            try:
                if raw_start:
                    start = int(raw_start)
                    end = int(raw_end) if raw_end else end
                else:
                    length = int(raw_end)
                    if length <= 0:
                        raise ValueError
                    start = max(0, size - length)
            except ValueError:
                self._send_range_error(size, asset_id)
                return
            if start < 0 or end < start or start >= size:
                self._send_range_error(size, asset_id)
                return
            end = min(end, size - 1)
            partial = True
        length = 0 if size == 0 else end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if length == 0:
            return
        remaining = length
        with path.open("rb") as handle:
            handle.seek(start)
            while remaining > 0:
                chunk = handle.read(min(VIDEO_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _send_range_error(self, size: int, asset_id: str) -> None:
        self._send_error(
            HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
            "invalid_range",
            "video range is invalid",
            asset_id,
            extra_headers={"Content-Range": f"bytes */{size}"},
        )

    def _allowed_methods(self) -> tuple[str, ...]:
        asset_id, route = _parts(self.path)
        if asset_id is None and route == "assets":
            return ("GET",)
        if asset_id is not None and route in {"task", "video"}:
            return ("GET",)
        if asset_id is not None and route in {"lease/acquire", "lease/renew", "lease/release", *_MUTATIONS}:
            return ("POST",)
        return ()

    def _method_not_allowed(self, *, send_body: bool = True) -> None:
        asset_id, _ = _parts(self.path)
        allowed = self._allowed_methods()
        if not allowed:
            self._send_error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                "unknown endpoint",
                asset_id,
                send_body=send_body,
            )
            return
        self._send_error(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "method_not_allowed",
            "method is not allowed for this endpoint",
            asset_id,
            extra_headers={"Allow": ", ".join(allowed)} if allowed else None,
            send_body=send_body,
        )

    def do_HEAD(self) -> None:  # noqa: N802
        self._method_not_allowed(send_body=False)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PUT(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PATCH(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_TRACE(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_CONNECT(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """Keep BaseHTTPRequestHandler's unsupported-method fallback JSON-only."""

        if code == HTTPStatus.NOT_IMPLEMENTED:
            self._method_not_allowed()
            return
        self._send_error(code, "http_error", "request could not be processed", None)

    def _handle(self, exc: Exception, asset_id: str | None) -> None:
        status, code, message = _error(exc)
        self._send_error(status, code, message, asset_id)

    def _send_error(
        self,
        status: int,
        code: str,
        message: str,
        asset_id: str | None,
        *,
        extra_headers: Mapping[str, str] | None = None,
        send_body: bool = True,
    ) -> None:
        revision = None
        if asset_id is not None:
            getter = getattr(self.server.application, "current_revision", None)
            if callable(getter):
                try:
                    value = getter(asset_id)
                    revision = value if isinstance(value, int) else None
                except Exception:
                    revision = None
        self._send_json(
            {"error": {"code": code, "message": message, "current_revision": revision}},
            status=status,
            extra_headers=extra_headers,
            send_body=send_body,
        )

    def _send_json(
        self,
        value: Any,
        *,
        status: int = HTTPStatus.OK,
        extra_headers: Mapping[str, str] | None = None,
        send_body: bool = True,
    ) -> None:
        body = json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, header_value in (extra_headers or {}).items():
            self.send_header(name, header_value)
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class SemanticCalibrationHttpServer(ThreadingHTTPServer):
    application: Any
    static_root: Path | None


def create_http_server(
    host: str,
    port: int,
    application: SemanticCalibrationApplication | Any,
    *,
    static_root: str | Path | None = None,
) -> SemanticCalibrationHttpServer:
    server = SemanticCalibrationHttpServer((host, port), SemanticCalibrationRequestHandler)
    server.application = application
    default_root = Path(__file__).resolve().parent / "static"
    server.static_root = Path(static_root).resolve() if static_root is not None else (
        default_root if default_root.is_dir() else None
    )
    return server


__all__ = [
    "MAX_REQUEST_BYTES",
    "VIDEO_CHUNK_SIZE",
    "SEMANTIC_ROUTES",
    "SemanticCalibrationHttpServer",
    "SemanticCalibrationRequestHandler",
    "create_http_server",
]
