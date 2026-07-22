"""Standard-library HTTP transport for the independent semantic application."""

from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
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

SEMANTIC_ROUTES = {
    "GET /api/semantic/assets",
    "GET /api/semantic/assets/{asset_id}/task",
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
            if not urlsplit(self.path).path.startswith("/api/") and not urlsplit(self.path).path.startswith("/evidence/"):
                if self._serve_static():
                    return
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint", asset_id)
        except Exception as exc:
            self._handle(exc, asset_id)

    def do_POST(self) -> None:  # noqa: N802
        asset_id, route = _parts(self.path)
        if asset_id is None or not route:
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
            if route in {"lease/renew", "lease/release"}:
                self._require_write_fields(payload)
                self._validate_revision(asset_id, payload["expected_revision"])
                if route == "lease/renew":
                    lease = self.server.application.renew_lease(
                        asset_id,
                        payload["lease_token"],
                        payload.get("ttl_seconds"),
                    )
                    self._send_json({"lease": jsonable(lease)})
                else:
                    self._send_json(
                        self.server.application.release_lease(asset_id, payload["lease_token"])
                    )
                return
            operation = _MUTATIONS.get(route)
            if operation is None:
                self._send_error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint", asset_id)
                return
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

    def _handle(self, exc: Exception, asset_id: str | None) -> None:
        status, code, message = _error(exc)
        self._send_error(status, code, message, asset_id)

    def _send_error(self, status: int, code: str, message: str, asset_id: str | None) -> None:
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
        )

    def _send_json(self, value: Any, *, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
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
    "SEMANTIC_ROUTES",
    "SemanticCalibrationHttpServer",
    "SemanticCalibrationRequestHandler",
    "create_http_server",
]
