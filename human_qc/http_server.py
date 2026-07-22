"""Warn-only HTTP transport with sanitized errors and bounded media streaming."""

from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import secrets
from threading import RLock
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
    open_verified_media,
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
    "method_not_allowed": "the requested method is not allowed",
    "not_found": "the requested resource was not found",
    "range_not_satisfiable": "the requested byte range is not satisfiable",
    "source_video_unavailable": "the source video is unavailable",
    "stale_revision": "the report revision is stale",
    "state_conflict": "the task cannot accept this operation",
}

_SESSION_COOKIE = "warn_reviewer_session"
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{20,128}").fullmatch


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


def _allowed_methods(parts: tuple[str, ...] | None) -> tuple[str, ...] | None:
    if parts == ("api", "warn", "assets"):
        return ("GET",)
    if (
        parts is not None
        and len(parts) == 5
        and parts[:3] == ("api", "warn", "assets")
        and parts[4] == "task"
    ):
        return ("GET",)
    if (
        parts is not None
        and len(parts) == 4
        and parts[:2] == ("media", "assets")
        and parts[3] == "source"
    ):
        return ("GET", "HEAD")
    if (
        parts is not None
        and len(parts) == 5
        and parts[:2] == ("media", "assets")
        and parts[3] == "overlays"
    ):
        return ("GET", "HEAD")
    if (
        parts is not None
        and len(parts) >= 5
        and parts[:3] == ("api", "warn", "assets")
    ):
        route = parts[4:]
        if route in {
            ("lease", "acquire"),
            ("lease", "renew"),
            ("lease", "release"),
            ("complete",),
        } or (
            len(route) == 3
            and route[0] == "issues"
            and route[2] == "verdict"
        ):
            return ("POST",)
    return None


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
            token, session_id, from_session = self._task_lease_credentials(asset_id)
            try:
                task = self.server.service.get_asset_task(asset_id, lease_token=token)
                lease = task.get("lease")
                if (
                    from_session
                    and isinstance(lease, Mapping)
                    and lease.get("code") == "lease_invalid"
                ):
                    task = self.server.service.get_asset_task(asset_id)
            except Exception as exc:
                self._handle_exception(exc, asset_id)
                return
            session_header = self._remember_task_lease(
                asset_id, task, session_id=session_id
            )
            self._send_json(
                self._task_response(task),
                headers=(
                    {"Set-Cookie": session_header}
                    if session_header is not None
                    else None
                ),
            )
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
        self._send_route_error(parts)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib hook name.
        parts = _decoded_parts(self.path)
        if (
            parts is not None
            and len(parts) == 4
            and parts[:2] == ("media", "assets")
            and parts[3] == "source"
        ):
            asset_id = parts[2]
            try:
                self._serve_media(
                    self.server.service.source_media(asset_id), send_body=False
                )
            except Exception as exc:
                self._handle_exception(exc, asset_id, head_only=True)
            return
        if (
            parts is not None
            and len(parts) == 5
            and parts[:2] == ("media", "assets")
            and parts[3] == "overlays"
        ):
            asset_id, overlay_id = parts[2], parts[4]
            try:
                self._serve_media(
                    self.server.service.overlay_media(asset_id, overlay_id),
                    send_body=False,
                )
            except Exception as exc:
                self._handle_exception(exc, asset_id, head_only=True)
            return
        self._send_route_error(parts, head_only=True)

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
            self._send_route_error(parts)
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
            self._send_route_error(parts, asset_id=asset_id)
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

    def do_PUT(self) -> None:  # noqa: N802 - stdlib hook name.
        self._send_route_error(_decoded_parts(self.path))

    def do_PATCH(self) -> None:  # noqa: N802 - stdlib hook name.
        self._send_route_error(_decoded_parts(self.path))

    def do_DELETE(self) -> None:  # noqa: N802 - stdlib hook name.
        self._send_route_error(_decoded_parts(self.path))

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib hook name.
        self._send_route_error(_decoded_parts(self.path))

    def _task_lease_credentials(
        self, asset_id: str
    ) -> tuple[str | None, str | None, bool]:
        header_token = self.headers.get("X-Reviewer-Lease")
        if header_token is not None:
            return header_token, self._session_id(), False
        session_id = self._session_id()
        if session_id is None:
            return None, None, False
        with self.server.lease_sessions_lock:
            token = self.server.lease_sessions.get(session_id, {}).get(asset_id)
        return token, session_id, token is not None

    def _session_id(self) -> str | None:
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        cookie = SimpleCookie()
        try:
            cookie.load(raw)
        except CookieError:
            return None
        morsel = cookie.get(_SESSION_COOKIE)
        value = morsel.value if morsel is not None else None
        return value if isinstance(value, str) and _SESSION_ID(value) else None

    def _remember_task_lease(
        self,
        asset_id: str,
        task: Mapping[str, Any],
        *,
        session_id: str | None,
    ) -> str | None:
        lease = task.get("lease")
        if not isinstance(lease, Mapping) or lease.get("read_only") is not False:
            return None
        token = lease.get("token")
        if not isinstance(token, str) or not token:
            return None
        with self.server.lease_sessions_lock:
            if session_id is None or session_id not in self.server.lease_sessions:
                session_id = secrets.token_urlsafe(32)
            self.server.lease_sessions.setdefault(session_id, {})[asset_id] = token
        return (
            f"{_SESSION_COOKIE}={session_id}; HttpOnly; SameSite=Strict; "
            "Path=/api/warn/"
        )

    def send_error(
        self,
        code: int,
        message: str | None = None,
        explain: str | None = None,
    ) -> None:
        """Replace BaseHTTPRequestHandler's HTML 501 for unknown methods."""

        del message, explain
        if code == HTTPStatus.NOT_IMPLEMENTED:
            self._send_route_error(_decoded_parts(self.path))
            return
        self._send_error(code, "internal_error", None)

    def _send_route_error(
        self,
        parts: tuple[str, ...] | None,
        *,
        asset_id: str | None = None,
        head_only: bool = False,
    ) -> None:
        allowed = _allowed_methods(parts)
        if allowed is None:
            self._send_error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                asset_id,
                head_only=head_only,
            )
            return
        self._send_error(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "method_not_allowed",
            asset_id,
            headers={"Allow": ", ".join(allowed)},
            head_only=head_only,
        )

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

    def _serve_media(
        self, resource: MediaResource, *, send_body: bool = True
    ) -> None:
        with open_verified_media(resource) as source:
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
                    head_only=not send_body,
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
            if not send_body:
                return
            try:
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

    def _handle_exception(
        self,
        exc: Exception,
        asset_id: str | None,
        *,
        head_only: bool = False,
    ) -> None:
        status, code = _error_details(exc)
        self._send_error(status, code, asset_id, head_only=head_only)

    def _send_error(
        self,
        status: int | HTTPStatus,
        code: str,
        asset_id: str | None,
        *,
        headers: Mapping[str, str] | None = None,
        head_only: bool = False,
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
            head_only=head_only,
        )

    def _send_json(
        self,
        value: Any,
        *,
        status: int | HTTPStatus = HTTPStatus.OK,
        headers: Mapping[str, str] | None = None,
        head_only: bool = False,
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
        if not head_only:
            self.wfile.write(body)

    @staticmethod
    def _task_response(task: Mapping[str, Any]) -> dict[str, Any]:
        return {"task": dict(task), "revision": task.get("report_revision")}

    def log_message(self, format: str, *args: Any) -> None:
        return


class HumanQcHttpServer(ThreadingHTTPServer):
    service: Any
    static_root: Path | None
    lease_sessions: dict[str, dict[str, str]]
    lease_sessions_lock: RLock


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
    server.lease_sessions = {}
    server.lease_sessions_lock = RLock()
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
