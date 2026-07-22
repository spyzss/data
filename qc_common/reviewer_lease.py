"""Small process-local reviewer lease store used by the workbench server."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import secrets
from threading import RLock
from typing import Callable


class LeaseError(RuntimeError):
    """Base lease failure."""


class LeaseConflictError(LeaseError):
    """Another reviewer holds a live lease."""


class LeaseTokenError(LeaseError):
    """The lease token is missing, stale, or expired."""


@dataclass(frozen=True)
class Lease:
    asset_id: str
    reviewer: str
    token: str
    expires_at: str

    @property
    def expires_at_datetime(self) -> datetime:
        return datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))


class LeaseStore:
    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._leases: dict[str, Lease] = {}
        self._lock = RLock()

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value

    @staticmethod
    def _ttl(ttl_seconds: int) -> int:
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer")
        return ttl_seconds

    def acquire(self, asset_id: str, reviewer: str, ttl_seconds: int) -> Lease:
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise ValueError("asset_id must be a non-empty string")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("reviewer must be a non-empty string")
        ttl = self._ttl(ttl_seconds)
        with self._lock:
            current = self._leases.get(asset_id)
            if current is not None and current.expires_at_datetime > self._now():
                raise LeaseConflictError(f"asset {asset_id} is leased by {current.reviewer}")
            lease = Lease(
                asset_id=asset_id,
                reviewer=reviewer,
                token=secrets.token_urlsafe(32),
                expires_at=(self._now() + timedelta(seconds=ttl)).isoformat(),
            )
            self._leases[asset_id] = lease
            return lease

    def renew(self, asset_id: str, token: str, ttl_seconds: int) -> Lease:
        ttl = self._ttl(ttl_seconds)
        if not isinstance(token, str) or not token:
            raise LeaseTokenError("lease token is required")
        with self._lock:
            current = self._leases.get(asset_id)
            if current is None or current.token != token or current.expires_at_datetime <= self._now():
                raise LeaseTokenError("lease token is stale or expired")
            renewed = Lease(
                asset_id=asset_id,
                reviewer=current.reviewer,
                token=current.token,
                expires_at=(self._now() + timedelta(seconds=ttl)).isoformat(),
            )
            self._leases[asset_id] = renewed
            return renewed

    def validate(self, asset_id: str, token: str) -> Lease:
        with self._lock:
            current = self._leases.get(asset_id)
            if current is None or current.token != token or current.expires_at_datetime <= self._now():
                raise LeaseTokenError("lease token is stale or expired")
            return current

    def release(self, asset_id: str, token: str) -> Lease:
        """Release the current lease only when ``token`` still owns it."""

        if not isinstance(token, str) or not token:
            raise LeaseTokenError("lease token is required")
        with self._lock:
            current = self._leases.get(asset_id)
            if current is None or current.token != token or current.expires_at_datetime <= self._now():
                raise LeaseTokenError("lease token is stale or expired")
            del self._leases[asset_id]
            return current


__all__ = [
    "Lease",
    "LeaseConflictError",
    "LeaseError",
    "LeaseStore",
    "LeaseTokenError",
]
