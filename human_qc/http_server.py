"""Warn-only HTTP transport with sanitized errors and bounded media streaming."""

from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from qc_common.report import StaleReportRevisionError
from qc_common.reviewer_lease import Lease, LeaseConflictError, LeaseTokenError

from .media import (
    MediaNotFoundError,
    MediaResource,
    MediaUnavailableError,
    RangeNotSatisfiable,
    iter_file_chunks,
    parse_byte_range,
)
from .warn_service import WarnLeaseError, WarnRevisionError, WarnStateError
from .warn_workbench_service import InvalidIssueRangeError, WarnWorkbenchService


MAX_REQUEST_BYTES = 10 * 1024 * 1024


class ApiRequestError(ValueError):
    """A malformed request which should be returned as HTTP 400."""


_SAFE_MESSAGES = {
    "bad_request": "the request is invalid",
    "internal_error": "the request could not be completed",
    "invalid_issue_range": "a selected issue has an invalid frame range",
    "lease_held": "the asset is currently read-only",
    "lease_invalid": "the reviewer lease is invalid or expired",
    "not_found": "the requested resource was not found",
    "range_not_satisfiable": "the requested byte range is not satisfiable",
    "source_video_unavailable": "the source video is unavailable",
    "stale_revision": "the report revision is stale",
    "state_conflict": "the task cannot accept this operation",
}


def _decoded_parts(path: str) -> tuple[str, ...] | None:
    raw_path = urlsplit(path).path
    if raw_path == "/":
        return ()
    raw_parts = raw_path.strip("/").split("/")
    if not raw_parts or any(part == "" for part in raw_parts):
        return None
    result: list[str] = []
    for raw in raw_parts:
        value = unquote(raw)
        if value in {"", ".", ".."} or "/" in value or "\\" in value:
            return None
        result.append(value)
    return tuple(result)


def _error_details(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, LeaseConflictError):
        return HTTPStatus.LOCKED, "lease_held"
    if isinstance(exc, (LeaseTokenError, WarnLeaseError)):
        return HTTPStatus.LOCKED, "lease_invalid"
    if isinstance(exc, (StaleReportRevisionError, WarnRevisionError)):
        return HTTPStatus.CONFLICT, "stale_revision"
    if isinstance(exc, InvalidIssueRangeError):
        return HTTPStatus.CONFLICT, "invalid_issue_range"
    if isinstance(exc, MediaUnavailableError):
        return HTTPStatus.NOT_FOUND, "source_video_unavailable"
    if isinstance(exc, (MediaNotFoundError, KeyError, FileNotFoundError)):
        return HTTPStatus.NOT_FOUND, "not_found"
    if isinstance(exc, WarnStateError):
        return HTTPStatus.CONFLICT, "state_conflict"
    if isinstance(exc, ApiRequestError):
        return HTTPStatus.BAD_REQUEST, "bad_request"
    if isinstance(exc, (ValueError, TypeError, json.JSONDecodeError)):
        return HTTPStatus.BAD_REQUEST, "bad_request"
    return HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error"


def _lease_dict(lease: Lease) -> dict[str, object]:
    return {"token": lease.token, "expires_at": lease.expires_at}


class HumanQcRequestHandler(BaseHTTPRequestHandler):
    """Transport-only handler for one fixed-reviewer Warn service."""

    server: "HumanQcHttpServer"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook name.
        parts = _decoded_parts(self.path)
        if parts == ("api", "warn", "assets"):
            try:
                self._send_json({"assets": list(self.server.service.list_assets())})
            except Exception as exc:
                self._handle_exception(exc, None)
            return
        if (
            parts is not None
            and len(parts) == 5
            and parts[:3] == ("api", "warn", "assets")
            and parts[4] == "task"
        ):
            asset_id = parts[3]
            query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
            if any("token" in key.lower() for key in query):
                self._send_error(HTTPStatus.BAD_REQUEST, "bad_request", asset_id)
                return
            token = self.headers.get("X-Reviewer-Lease")
            try:
                task = self.server.service.get_asset_task(asset_id, lease_token=token)
            except Exception as exc:
                self._handle_exception(exc, asset_id)
                return
            self._send_json(self._task_response(task))
            return
        if (
            parts is not None
            and len(parts) == 4
            and parts[:2] == ("media", "assets")
            and parts[3] == "source"
        ):
            asset_id = parts[2]
            try:
                resource = self.server.service.source_media(asset_id)
                self._serve_media(resource)
            except Exception as exc:
                self._handle_exception(exc, asset_id)
            return
        if (
            parts is not None
            and len(parts) == 5
            and parts[:2] == ("media", "assets")
            and parts[3] == "overlays"
        ):
            asset_id, overlay_id = parts[2], parts[4]
            try:
                resource = self.server.service.overlay_media(asset_id, overlay_id)
                self._serve_media(resource)
            except Exception as exc:
                self._handle_exception(exc, asset_id)
            return

        raw_path = urlsplit(self.path).path
        if not raw_path.startswith(("/api/", "/media/", "/evidence/")):
            if self._serve_static():
                return
        self._send_error(HTTPStatus.NOT_FOUND, "not_found", None)

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook name.
        parts = _decoded_parts(self.path)
        asset_id = (
            parts[3]
            if parts is not None
            and len(parts) >= 5
            and parts[:3] == ("api", "warn", "assets")
            else None
        )
        if asset_id is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", None)
            return
        route = parts[4:]
        is_verdict = (
            len(route) == 3 and route[0] == "issues" and route[2] == "verdict"
        )
        if route not in {
            ("lease", "acquire"),
            ("lease", "renew"),
            ("lease", "release"),
            ("complete",),
        } and not is_verdict:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", asset_id)
            return
        try:
            payload = self._read_json()
            if route == ("lease", "acquire"):
                ttl = payload.get("ttl_seconds")
                lease = self.server.service.acquire_lease(asset_id, ttl_seconds=ttl)
                self._send_json({"lease": _lease_dict(lease)})
                return

            expected_revision, token = self._mutation_credentials(payload)
            self._validate_revision(asset_id, expected_revision)
            if route == ("lease", "renew"):
                ttl = payload.get("ttl_seconds")
                lease = self.server.service.renew_lease(
                    asset_id, token, ttl_seconds=ttl
                )
                self._send_json({"lease": _lease_dict(lease)})
                return
            if route == ("lease", "release"):
                self.server.service.release_lease(asset_id, token)
                self._send_json({"released": True})
                return

            self.server.service.validate_lease(asset_id, token)
            if is_verdict:
                issue_id = route[1]
                body_issue_id = payload.get("issue_id")
                if body_issue_id is not None and body_issue_id != issue_id:
                    raise ApiRequestError("issue_id does not match URL")
                result = self.server.service.warn_verdict(
                    asset_id,
                    issue_id=issue_id,
                    verdict=payload.get("verdict"),
                    reason=payload.get("reason"),
                    failure_reason=payload.get("failure_reason"),
                    expected_revision=expected_revision,
                    lease_token=token,
                )
            else:
                result = self.server.service.warn_complete(
                    asset_id,
                    completion_mode=payload.get("completion_mode"),
                    failure_reason=payload.get("failure_reason"),
                    expected_revision=expected_revision,
                    lease_token=token,
                )
            self._send_json(self._task_response(result))
        except Exception as exc:
            self._handle_exception(exc, asset_id)

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
            raise ApiRequestError("payload must be an object")
        return dict(value)

    def _mutation_credentials(self, payload: Mapping[str, Any]) -> tuple[int, str]:
        revision = payload.get("expected_revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ApiRequestError("expected_revision must be a non-negative integer")
        body_token = payload.get("lease_token", payload.get("token"))
        header_token = self.headers.get("X-Reviewer-Lease")
        if header_token is not None and body_token is not None and header_token != body_token:
            raise ApiRequestError("lease credentials do not match")
        token = header_token if header_token is not None else body_token
        if not isinstance(token, str) or not token:
            raise ApiRequestError("lease_token is required")
        return revision, token

    def _validate_revision(self, asset_id: str, expected_revision: int) -> None:
        current = self.server.service.current_revision(asset_id)
        if current is None:
            raise KeyError("unknown_asset")
        if current != expected_revision:
            raise StaleReportRevisionError("stale_revision")

    def _serve_media(self, resource: MediaResource) -> None:
        try:
            byte_range = parse_byte_range(self.headers.get("Range"), resource.size)
        except RangeNotSatisfiable:
            self._send_error(
                HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                "range_not_satisfiable",
                None,
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes */{resource.size}",
                },
            )
            return
        if byte_range is None:
            start, length, status = 0, resource.size, HTTPStatus.OK
            content_range = None
        else:
            start = byte_range.start
            length = byte_range.length
            status = HTTPStatus.PARTIAL_CONTENT
            content_range = (
                f"bytes {byte_range.start}-{byte_range.end_inclusive}/{resource.size}"
            )
        self.send_response(status)
        self.send_header("Content-Type", resource.mime_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if content_range is not None:
            self.send_header("Content-Range", content_range)
        if resource.etag is not None:
            self.send_header("ETag", json.dumps(resource.etag))
        self.end_headers()
        try:
            with resource.path.open("rb") as source:
                for chunk in iter_file_chunks(source, start=start, length=length):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _serve_static(self) -> bool:
        root = self.server.static_root
        if root is None:
            return False
        raw_path = urlsplit(self.path).path
        relative = Path(unquote(raw_path.lstrip("/")) or "index.html")
        if any(part in {"", ".", ".."} for part in relative.parts):
            return False
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        if not candidate.is_file():
            return False
        try:
            body = candidate.read_bytes()
        except OSError:
            return False
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        return True

    def _handle_exception(self, exc: Exception, asset_id: str | None) -> None:
        status, code = _error_details(exc)
        self._send_error(status, code, asset_id)

    def _send_error(
        self,
        status: int | HTTPStatus,
        code: str,
        asset_id: str | None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        current_revision: int | None = None
        if asset_id is not None:
            try:
                value = self.server.service.current_revision(asset_id)
                current_revision = value if isinstance(value, int) else None
            except Exception:
                current_revision = None
        self._send_json(
            {
                "error": {
                    "code": code,
                    "message": _SAFE_MESSAGES.get(code, _SAFE_MESSAGES["internal_error"]),
                    "current_revision": current_revision,
                }
            },
            status=status,
            headers=headers,
        )

    def _send_json(
        self,
        value: Any,
        *,
        status: int | HTTPStatus = HTTPStatus.OK,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        body = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, header_value in (headers or {}).items():
            self.send_header(name, header_value)
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _task_response(task: Mapping[str, Any]) -> dict[str, Any]:
        return {"task": dict(task), "revision": task.get("report_revision")}

    def log_message(self, format: str, *args: Any) -> None:
        return


class HumanQcHttpServer(ThreadingHTTPServer):
    service: Any
    static_root: Path | None


def create_http_server(
    host: str,
    port: int,
    service: WarnWorkbenchService | Any,
    *,
    static_root: str | Path | None = None,
    evidence_root: str | Path | None = None,
    evidence_allowed_prefixes: tuple[str, ...] | None = None,
) -> HumanQcHttpServer:
    """Create a bound Warn server; legacy evidence arguments are ignored."""

    del evidence_root, evidence_allowed_prefixes
    server = HumanQcHttpServer((host, port), HumanQcRequestHandler)
    server.service = service
    default_root = Path(__file__).resolve().parent / "static"
    server.static_root = (
        Path(static_root).resolve()
        if static_root is not None
        else (default_root if default_root.is_dir() else None)
    )
    return server


__all__ = [
    "ApiRequestError",
    "HumanQcHttpServer",
    "HumanQcRequestHandler",
    "MAX_REQUEST_BYTES",
    "create_http_server",
]
