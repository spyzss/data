"""Revision-aware HTTP transport for :mod:`human_qc.workbench_service`.

The server intentionally uses only the Python standard library.  It is a
small local workbench API, not a general-purpose public web service; callers
must still provide the expected report revision and current lease token for
every state-changing request.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from qc_common.report import StaleReportRevisionError

from .lease import LeaseConflictError, LeaseTokenError
from .semantic_service import (
    LeaseError as SemanticLeaseError,
    PendingEditError as SemanticPendingEditError,
    StaleSemanticRevisionError,
    TaskStateError as SemanticTaskStateError,
)
from .warn_service import (
    WarnLeaseError,
    WarnRevisionError,
    WarnStateError,
)
from .workbench_service import WorkbenchService, jsonable


MAX_REQUEST_BYTES = 10 * 1024 * 1024


class ApiRequestError(ValueError):
    """A malformed request which should be returned as HTTP 400."""


_MUTATION_ROUTES: dict[str, str] = {
    "semantic/boundary/pending": "semantic_boundary_pending",
    "semantic/boundary": "semantic_boundary_pending",
    "semantic/text/pending": "semantic_text_pending",
    "semantic/text": "semantic_text_pending",
    "semantic/pending/confirm": "semantic_pending_confirm",
    "semantic/confirm": "semantic_pending_confirm",
    "semantic/pending/cancel": "semantic_pending_cancel",
    "semantic/cancel": "semantic_pending_cancel",
    "semantic/complete": "semantic_complete",
    "warn/verdict": "warn_verdict",
    "warn/issue/verdict": "warn_verdict",
    "warn/complete": "warn_complete",
}


def _path_parts(path: str) -> tuple[str, str | None]:
    parts = [unquote(part) for part in urlsplit(path).path.strip("/").split("/") if part]
    if len(parts) < 3 or parts[:2] != ["api", "assets"]:
        return "", None
    return "/".join(parts[3:]), parts[2]


def _as_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ApiRequestError("payload must be a JSON object")
    return dict(value)


def _normalize_payload(route: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Accept the UI's short aliases while keeping service contracts typed."""

    normalized = dict(payload)
    if route in {"semantic/boundary/pending", "semantic/boundary"}:
        normalized.setdefault("boundary_index", normalized.get("boundary", normalized.get("boundary_id")))
        normalized.setdefault(
            "new_frame_exclusive",
            normalized.get(
                "new_boundary",
                normalized.get("new_frame", normalized.get("target_frame")),
            ),
        )
        normalized.setdefault(
            "actor_segment_id",
            normalized.get("actor", normalized.get("segment_id")),
        )
    elif route in {"semantic/text/pending", "semantic/text"}:
        normalized.setdefault("segment_id", normalized.get("segment"))
        normalized.setdefault("text_cn", normalized.get("cn", normalized.get("text")))
        normalized.setdefault("text_en", normalized.get("en", ""))
    return normalized


def _error_details(exc: Exception) -> tuple[int, str, str]:
    message = str(exc) or exc.__class__.__name__
    if isinstance(exc, (LeaseConflictError,)):
        return HTTPStatus.LOCKED, "lease_held", message
    if isinstance(exc, (LeaseTokenError, SemanticLeaseError, WarnLeaseError)):
        return HTTPStatus.LOCKED, "lease_invalid", message
    if isinstance(exc, (StaleReportRevisionError, StaleSemanticRevisionError, WarnRevisionError)):
        return HTTPStatus.CONFLICT, "stale_revision", message
    if isinstance(
        exc,
        (
            SemanticPendingEditError,
            SemanticTaskStateError,
            WarnStateError,
        ),
    ):
        if isinstance(exc, WarnStateError) and "issue" in message and any(
            marker in message
            for marker in ("not a candidate", "not selected", "missing from report")
        ):
            return HTTPStatus.NOT_FOUND, "not_found", message
        return HTTPStatus.CONFLICT, "state_conflict", message
    if isinstance(exc, (KeyError, FileNotFoundError)):
        return HTTPStatus.NOT_FOUND, "not_found", message
    if isinstance(exc, ApiRequestError):
        return HTTPStatus.BAD_REQUEST, "bad_request", message
    if isinstance(exc, (ValueError, TypeError, json.JSONDecodeError)):
        return HTTPStatus.BAD_REQUEST, "bad_request", message
    return HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", message


class HumanQcRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler whose only stateful dependency is ``server.service``."""

    server: "HumanQcHttpServer"

    # The default BaseHTTPRequestHandler logging is useful when running the
    # local CLI and can be disabled by embedding applications as usual.
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook name.
        route, asset_id = _path_parts(self.path)
        if urlsplit(self.path).path.startswith("/evidence/"):
            if self._serve_evidence():
                return
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "evidence file not found", None)
            return
        if asset_id is None and not urlsplit(self.path).path.startswith("/api/"):
            if self._serve_static():
                return
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "static file not found", None)
            return
        if asset_id is None or route != "task":
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint", asset_id)
            return
        try:
            result = self.server.service.get_asset_task(asset_id)
        except Exception as exc:  # map service errors consistently
            self._handle_exception(exc, asset_id)
            return
        self._send_json(self._task_response(result))

    def _serve_static(self) -> bool:
        root = getattr(self.server, "static_root", None)
        if root is None:
            return False
        raw_path = urlsplit(self.path).path
        relative = Path(unquote(raw_path.lstrip("/")) or "index.html")
        if any(part in {"", ".", ".."} for part in relative.parts):
            return False
        candidate = (Path(root) / relative).resolve()
        try:
            candidate.relative_to(Path(root).resolve())
        except ValueError:
            return False
        if not candidate.is_file():
            return False
        try:
            body = candidate.read_bytes()
        except OSError:
            return False
        self.send_response(HTTPStatus.OK)
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        return True

    def _serve_evidence(self) -> bool:
        root = getattr(self.server, "evidence_root", None)
        if root is None:
            return False
        prefix = "/evidence/"
        relative = Path(unquote(urlsplit(self.path).path[len(prefix) :]))
        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            return False
        allowed = getattr(self.server, "evidence_allowed_prefixes", frozenset())
        if allowed and relative.parts[0] not in allowed:
            return False
        candidate = (Path(root) / relative).resolve()
        try:
            candidate.relative_to(Path(root).resolve())
        except ValueError:
            return False
        if not candidate.is_file():
            return False
        try:
            body = candidate.read_bytes()
        except OSError:
            return False
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        return True

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook name.
        route, asset_id = _path_parts(self.path)
        if asset_id is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint", None)
            return
        try:
            payload = self._read_json()
        except Exception as exc:
            self._handle_exception(exc, asset_id)
            return

        try:
            if route == "lease/acquire":
                self._ensure_asset_exists(asset_id)
                reviewer = payload.get("reviewer")
                if not isinstance(reviewer, str) or not reviewer.strip():
                    raise ApiRequestError("reviewer is required")
                ttl = payload.get("ttl_seconds")
                result = self.server.service.acquire_lease(asset_id, reviewer, ttl)
                self._send_json({"lease": jsonable(result)})
                return

            if route == "lease/renew":
                self._ensure_asset_exists(asset_id)
                token = payload.get("lease_token", payload.get("token"))
                if not isinstance(token, str) or not token:
                    raise ApiRequestError("lease_token is required")
                # Lease renewal is a state-changing endpoint too.  Requiring
                # the caller's revision prevents a stale browser tab from
                # keeping a lease alive after its task view has diverged.
                self._require_mutation_fields(
                    {**payload, "lease_token": token}
                )
                current_revision = getattr(self.server.service, "current_revision", None)
                if callable(current_revision):
                    current = current_revision(asset_id)
                    if isinstance(current, int) and payload["expected_revision"] != current:
                        raise StaleReportRevisionError(
                            f"expected revision {payload['expected_revision']}, found {current}"
                        )
                ttl = payload.get("ttl_seconds")
                result = self.server.service.renew_lease(asset_id, token, ttl)
                self._send_json({"lease": jsonable(result)})
                return

            operation = _MUTATION_ROUTES.get(route)
            issue_id_from_path: str | None = None
            route_parts = route.split("/") if route else []
            if len(route_parts) == 3 and route_parts[0] == "warn" and route_parts[2] == "verdict":
                # The stable public route keeps issue_id in the path so a UI
                # cannot accidentally submit a verdict for the selected row.
                operation = "warn_verdict"
                issue_id_from_path = route_parts[1]
            if operation is None:
                raise KeyError(f"unknown endpoint: {route}")
            payload = _normalize_payload(route, payload)
            if issue_id_from_path is not None:
                body_issue_id = payload.get("issue_id")
                if body_issue_id is not None and body_issue_id != issue_id_from_path:
                    raise ApiRequestError("issue_id does not match the URL path")
                payload["issue_id"] = issue_id_from_path
            self._require_mutation_fields(payload)
            self._validate_revision(asset_id, payload["expected_revision"])
            self._validate_lease(asset_id, payload["lease_token"])
            result = self._invoke_mutation(operation, asset_id, payload)
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
        return _as_payload(value)

    @staticmethod
    def _require_mutation_fields(payload: Mapping[str, Any]) -> None:
        missing = [
            field
            for field in ("expected_revision", "lease_token")
            if field not in payload
        ]
        if missing:
            raise ApiRequestError("missing required field(s): " + ", ".join(missing))
        revision = payload["expected_revision"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ApiRequestError("expected_revision must be a non-negative integer")
        if not isinstance(payload["lease_token"], str) or not payload["lease_token"]:
            raise ApiRequestError("lease_token must be a non-empty string")

    def _ensure_asset_exists(self, asset_id: str) -> None:
        getter = getattr(self.server.service, "get_asset_task", None)
        if callable(getter):
            getter(asset_id)

    def _validate_lease(self, asset_id: str, token: str) -> None:
        validator = getattr(self.server.service, "validate_lease", None)
        if callable(validator):
            validator(asset_id, token)
            return
        store = getattr(self.server.service, "lease_store", None)
        validator = getattr(store, "validate", None)
        if callable(validator):
            validator(asset_id, token)

    def _validate_revision(self, asset_id: str, expected_revision: int) -> None:
        getter = getattr(self.server.service, "current_revision", None)
        if not callable(getter):
            return
        current = getter(asset_id)
        if current is None:
            raise KeyError(asset_id)
        if isinstance(current, int) and current != expected_revision:
            raise StaleReportRevisionError(
                f"expected revision {expected_revision}, found {current}"
            )

    def _invoke_mutation(self, operation: str, asset_id: str, payload: dict[str, Any]) -> Any:
        method = getattr(self.server.service, operation, None)
        if callable(method):
            kwargs = dict(payload)
            return method(asset_id, **kwargs)
        # A deliberately small compatibility seam lets an embedding facade
        # centralize mutation dispatch while the HTTP contract stays stable.
        mutate = getattr(self.server.service, "mutate", None)
        if callable(mutate):
            return mutate(operation, asset_id, payload)
        raise KeyError(f"service does not implement {operation}")

    def _handle_exception(self, exc: Exception, asset_id: str | None) -> None:
        status, code, message = _error_details(exc)
        self._send_error(status, code, message, asset_id)

    def _send_error(
        self,
        status: int | HTTPStatus,
        code: str,
        message: str,
        asset_id: str | None,
    ) -> None:
        current_revision: int | None = None
        if asset_id is not None:
            getter = getattr(self.server.service, "current_revision", None)
            if callable(getter):
                try:
                    value = getter(asset_id)
                    current_revision = value if isinstance(value, int) else None
                except Exception:
                    current_revision = None
        self._send_json(
            {
                "error": {
                    "code": code,
                    "message": message,
                    "current_revision": current_revision,
                }
            },
            status=status,
        )

    def _send_json(self, value: Any, *, status: int | HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _task_response(result: Any) -> dict[str, Any]:
        task = jsonable(result)
        revision = task.get("revision") if isinstance(task, Mapping) else None
        return {"task": task, "revision": revision}

    def log_message(self, format: str, *args: Any) -> None:
        # Keep library use quiet; the CLI can install its own server logging.
        return


class HumanQcHttpServer(ThreadingHTTPServer):
    service: Any
    static_root: Path | None
    evidence_root: Path | None
    evidence_allowed_prefixes: frozenset[str]


def create_http_server(
    host: str,
    port: int,
    service: WorkbenchService | Any,
    *,
    static_root: str | Path | None = None,
    evidence_root: str | Path | None = None,
    evidence_allowed_prefixes: tuple[str, ...] | None = None,
) -> HumanQcHttpServer:
    """Create a bound server; pass ``port=0`` to let the OS choose a port."""

    server = HumanQcHttpServer((host, port), HumanQcRequestHandler)
    server.service = service
    default_root = Path(__file__).resolve().parent / "static"
    server.static_root = Path(static_root).resolve() if static_root is not None else (
        default_root if default_root.is_dir() else None
    )
    if evidence_root is not None:
        server.evidence_root = Path(evidence_root).resolve()
    else:
        contexts = getattr(service, "asset_contexts", {})
        first_context = next(iter(contexts.values()), None) if isinstance(contexts, Mapping) else None
        inferred = getattr(first_context, "batch_root", None)
        server.evidence_root = Path(inferred).resolve() if inferred is not None else None
    server.evidence_allowed_prefixes = frozenset(
        evidence_allowed_prefixes
        if evidence_allowed_prefixes is not None
        else (
            ".human_qc_evidence",
            "evidence",
            "clips",
            "overlays",
            "qc_evidence",
            "sam3",
        )
    )
    return server


__all__ = [
    "HumanQcHttpServer",
    "HumanQcRequestHandler",
    "MAX_REQUEST_BYTES",
    "create_http_server",
]
