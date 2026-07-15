"""Durable atomic commit for validated Curated LeRobot v3 releases."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Iterator

from .contracts import (
    CanonicalDiagnostic,
    PublishPlan,
    PublishPrerequisiteError,
    PublishRequest,
    PublishResult,
    ReleaseManifest,
    StagedRelease,
)
from .prerequisites import revalidate_publish_plan, validate_publish_request
from .validation import validate_staged_release
from .writer import write_staging


def _directory_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _placeholder_manifest(plan: PublishPlan) -> ReleaseManifest:
    return ReleaseManifest(
        schema_version="curated_lerobot_v3_release_manifest.v1",
        release_id=plan.release_id,
        publisher_version=plan.publisher_version,
        asset_id=plan.request.episode.identity.asset_id,
        canonical_revision=plan.request.canonical_revision,
        semantic_fingerprint=plan.semantic_fingerprint,
        source_fingerprint=plan.source_fingerprint,
        qc_report_revision=plan.qc_report_revision,
        qc_report_sha256=plan.qc_report_sha256,
    )


def _release_candidate(
    plan: PublishPlan,
    root: Path,
    identity: tuple[int, int] | None = None,
) -> StagedRelease:
    if identity is None:
        metadata = os.stat(root, follow_symlinks=False)
        identity = (metadata.st_dev, metadata.st_ino)
    return StagedRelease(
        plan=plan,
        root=root,
        transaction_id="committed-release",
        manifest=_placeholder_manifest(plan),
        manifest_sha256="",
        checksums_sha256="",
        root_device=identity[0],
        root_inode=identity[1],
    )


def _remove_tree_at(parent_fd: int, name: str) -> None:
    try:
        child_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except FileNotFoundError:
        return
    try:
        with os.scandir(child_fd) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    _remove_tree_at(child_fd, entry.name)
                else:
                    os.unlink(entry.name, dir_fd=child_fd)
    finally:
        os.close(child_fd)
    os.rmdir(name, dir_fd=parent_fd)


@dataclass(slots=True)
class _CommitDirectories:
    release_root: Path
    root_fd: int
    staging_fd: int
    releases_fd: int
    lock_fd: int
    root_identity: tuple[int, int]
    staging_identity: tuple[int, int]
    releases_identity: tuple[int, int]

    @staticmethod
    def _identity(value: os.stat_result) -> tuple[int, int]:
        return value.st_dev, value.st_ino

    def verify(self) -> None:
        for path, identity in (
            (self.release_root, self.root_identity),
            (self.release_root / ".staging", self.staging_identity),
            (self.release_root / "releases", self.releases_identity),
        ):
            observed = os.stat(path, follow_symlinks=False)
            if self._identity(observed) != identity or not stat.S_ISDIR(observed.st_mode):
                raise OSError(errno.ESTALE, f"commit directory identity changed: {path}")

    def release_identity(self, release_id: str) -> tuple[int, int] | None:
        try:
            value = os.stat(release_id, dir_fd=self.releases_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(value.st_mode):
            raise OSError(errno.ENOTDIR, "release target is not a directory")
        return self._identity(value)

    def staging_identity_for(self, transaction_id: str) -> tuple[int, int] | None:
        try:
            value = os.stat(
                transaction_id, dir_fd=self.staging_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(value.st_mode):
            raise OSError(errno.ENOTDIR, "staging transaction is not a directory")
        return self._identity(value)

    def cleanup(self, staged: StagedRelease) -> None:
        identity = self.staging_identity_for(staged.transaction_id)
        if identity is None:
            return
        if identity != (staged.root_device, staged.root_inode):
            raise OSError(errno.ESTALE, "refusing to clean a replaced staging transaction")
        _remove_tree_at(self.staging_fd, staged.transaction_id)

    def close(self) -> None:
        fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
        for descriptor in (
            self.lock_fd,
            self.releases_fd,
            self.staging_fd,
            self.root_fd,
        ):
            os.close(descriptor)


def _cleanup_staging(
    staged: StagedRelease, directories: _CommitDirectories | None = None
) -> None:
    if directories is not None:
        directories.cleanup(staged)
        return
    with _publish_lock(staged.plan.request.release_root) as locked:
        locked.cleanup(staged)


def _fsync_tree(
    root: Path,
    *,
    parent_fd: int | None = None,
    name: str | None = None,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    descriptor = os.open(
        root if parent_fd is None else name,
        _directory_flags(),
        dir_fd=parent_fd,
    )
    observed_identity = (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
    if expected_identity is not None and observed_identity != expected_identity:
        os.close(descriptor)
        raise OSError(errno.ESTALE, "fsync target identity changed")

    def sync_directory(directory_fd: int) -> None:
        children: list[int] = []
        try:
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        raise OSError(errno.ELOOP, "symlink in release tree")
                    if entry.is_dir(follow_symlinks=False):
                        child = os.open(entry.name, _directory_flags(), dir_fd=directory_fd)
                        children.append(child)
                    elif entry.is_file(follow_symlinks=False):
                        file_fd = os.open(
                            entry.name,
                            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                            dir_fd=directory_fd,
                        )
                        try:
                            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                                raise OSError(errno.EINVAL, "non-regular release artifact")
                            os.fsync(file_fd)
                        finally:
                            os.close(file_fd)
                    else:
                        raise OSError(errno.EINVAL, "non-regular release artifact")
            for child in children:
                sync_directory(child)
            os.fsync(directory_fd)
        finally:
            for child in children:
                os.close(child)

    try:
        sync_directory(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _publish_lock(release_root: Path) -> Iterator[_CommitDirectories]:
    root_fd = os.open(release_root, _directory_flags())
    staging_fd = -1
    releases_fd = -1
    lock_fd = -1
    try:
        for name in (".staging", "releases"):
            try:
                os.mkdir(name, mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
        staging_fd = os.open(".staging", _directory_flags(), dir_fd=root_fd)
        releases_fd = os.open("releases", _directory_flags(), dir_fd=root_fd)
        for attempt in range(3):
            try:
                lock_fd = os.open(
                    ".publish.lock",
                    os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=root_fd,
                )
                break
            except FileNotFoundError:
                current = os.stat(release_root, follow_symlinks=False)
                if _CommitDirectories._identity(current) != _CommitDirectories._identity(
                    os.fstat(root_fd)
                ):
                    raise OSError(errno.ESTALE, "release root changed during lock creation")
                if attempt == 2:
                    raise
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        directories = _CommitDirectories(
            release_root=release_root,
            root_fd=root_fd,
            staging_fd=staging_fd,
            releases_fd=releases_fd,
            lock_fd=lock_fd,
            root_identity=_CommitDirectories._identity(os.fstat(root_fd)),
            staging_identity=_CommitDirectories._identity(os.fstat(staging_fd)),
            releases_identity=_CommitDirectories._identity(os.fstat(releases_fd)),
        )
        directories.verify()
        yield directories
    finally:
        if lock_fd >= 0:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        for descriptor in (lock_fd, releases_fd, staging_fd, root_fd):
            if descriptor >= 0:
                os.close(descriptor)


def _rename_noreplace(
    source: Path,
    target: Path,
    directories: _CommitDirectories | None = None,
) -> None:
    source_fd = -100 if directories is None else directories.staging_fd
    target_fd = -100 if directories is None else directories.releases_fd
    source_name = os.fsencode(source if directories is None else source.name)
    target_name = os.fsencode(target if directories is None else target.name)
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.renameatx_np(
            ctypes.c_int(source_fd),
            source_name,
            ctypes.c_int(target_fd),
            target_name,
            ctypes.c_uint(0x00000004),
        )
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(target))
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        result = renameat2(
            ctypes.c_int(source_fd),
            source_name,
            ctypes.c_int(target_fd),
            target_name,
            ctypes.c_uint(1),
        )
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(target))
        return
    if directories is not None and directories.release_identity(target.name) is not None:
        raise FileExistsError(errno.EEXIST, "target exists", str(target))
    if directories is None and target.exists():
        raise FileExistsError(errno.EEXIST, "target exists", str(target))
    os.rename(
        source.name if directories is not None else source,
        target.name if directories is not None else target,
        src_dir_fd=None if directories is None else directories.staging_fd,
        dst_dir_fd=None if directories is None else directories.releases_fd,
    )


def _current_payload(release_id: str, manifest_sha256: str) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": "curated_lerobot_v3_current.v1",
                "release_id": release_id,
                "manifest_sha256": manifest_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _replace_current(
    plan: PublishPlan,
    manifest_sha256: str,
    directories: _CommitDirectories | None = None,
) -> None:
    root_fd = (
        os.open(plan.request.release_root, _directory_flags())
        if directories is None
        else os.dup(directories.root_fd)
    )
    temp_name = f".CURRENT-{secrets.token_hex(16)}.tmp"
    temp_fd = -1
    try:
        expected_payload = _current_payload(plan.release_id, manifest_sha256)
        try:
            current_fd = os.open(
                "CURRENT.json",
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_fd,
            )
        except FileNotFoundError:
            current_fd = -1
        if current_fd >= 0:
            try:
                observed = bytearray()
                while True:
                    chunk = os.read(current_fd, 4096)
                    if not chunk:
                        break
                    observed.extend(chunk)
                if bytes(observed) == expected_payload:
                    os.fsync(root_fd)
                    return
            finally:
                os.close(current_fd)
        temp_fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_fd,
        )
        view = memoryview(expected_payload)
        while view:
            written = os.write(temp_fd, view)
            view = view[written:]
        os.fsync(temp_fd)
        os.close(temp_fd)
        temp_fd = -1
        os.replace(temp_name, "CURRENT.json", src_dir_fd=root_fd, dst_dir_fd=root_fd)
        os.fsync(root_fd)
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        try:
            os.unlink(temp_name, dir_fd=root_fd)
        except FileNotFoundError:
            pass
        os.close(root_fd)


def _sync_commit_directories(
    plan: PublishPlan, directories: _CommitDirectories | None = None
) -> None:
    if directories is None:
        sources: tuple[Path | int, ...] = (
            plan.request.release_root / ".staging",
            plan.request.release_root / "releases",
            plan.request.release_root,
        )
    else:
        sources = (
            directories.staging_fd,
            directories.releases_fd,
            directories.root_fd,
        )
    for source in sources:
        descriptor = (
            os.dup(source) if isinstance(source, int) else os.open(source, _directory_flags())
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _initialize_release_root(release_root: Path) -> None:
    release_root.mkdir(parents=True, exist_ok=True)
    root_fd = os.open(release_root, _directory_flags())
    lock_fd = -1
    try:
        for name in (".staging", "releases"):
            try:
                os.mkdir(name, mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
        for attempt in range(3):
            try:
                lock_fd = os.open(
                    ".publish.lock",
                    os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=root_fd,
                )
                break
            except FileNotFoundError:
                current = os.stat(release_root, follow_symlinks=False)
                if (current.st_dev, current.st_ino) != (
                    os.fstat(root_fd).st_dev,
                    os.fstat(root_fd).st_ino,
                ):
                    raise OSError(errno.ESTALE, "release root changed during initialization")
                if attempt == 2:
                    raise
        os.fsync(lock_fd)
        os.fsync(root_fd)
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        os.close(root_fd)


def _validate_existing(
    plan: PublishPlan, directories: _CommitDirectories | None = None
):
    try:
        identity = (
            None
            if directories is None
            else directories.release_identity(plan.release_id)
        )
        if directories is not None and identity is None:
            raise FileNotFoundError(plan.release_id)
        return validate_staged_release(
            _release_candidate(plan, plan.release_path, identity), plan.request.episode
        )
    except (OSError, PublishPrerequisiteError) as exc:
        raise PublishPrerequisiteError(
            CanonicalDiagnostic(
                "commit_conflict",
                "publish_commit",
                "release_path",
                f"existing deterministic release is invalid or different: {exc}",
                False,
            )
        ) from exc


def publish(request: PublishRequest) -> PublishResult:
    """Validate, durably commit, and atomically select one immutable release."""

    plan = validate_publish_request(request)
    _initialize_release_root(request.release_root)

    with _publish_lock(request.release_root) as directories:
        if directories.release_identity(plan.release_id) is not None:
            revalidate_publish_plan(plan)
            report = _validate_existing(plan, directories)
            _fsync_tree(
                plan.release_path,
                parent_fd=directories.releases_fd,
                name=plan.release_id,
                expected_identity=directories.release_identity(plan.release_id),
            )
            _sync_commit_directories(plan, directories)
            revalidate_publish_plan(plan)
            directories.verify()
            _replace_current(plan, report.manifest_sha256, directories)
            directories.verify()
            return PublishResult("already_published", plan, report.manifest)

    staged = write_staging(plan, request.release_root / ".staging")
    renamed = False
    try:
        report = validate_staged_release(staged, request.episode)
        with _publish_lock(request.release_root) as directories:
            directories.verify()
            if directories.staging_identity_for(staged.transaction_id) != (
                staged.root_device,
                staged.root_inode,
            ):
                raise OSError(errno.ESTALE, "staging transaction identity changed")
            revalidate_publish_plan(plan)
            report = validate_staged_release(staged, request.episode)
            _fsync_tree(
                staged.root,
                parent_fd=directories.staging_fd,
                name=staged.transaction_id,
                expected_identity=(staged.root_device, staged.root_inode),
            )
            report = validate_staged_release(staged, request.episode)
            revalidate_publish_plan(plan)
            directories.verify()
            if directories.release_identity(plan.release_id) is not None:
                existing = _validate_existing(plan, directories)
                _fsync_tree(
                    plan.release_path,
                    parent_fd=directories.releases_fd,
                    name=plan.release_id,
                    expected_identity=directories.release_identity(plan.release_id),
                )
                directories.cleanup(staged)
                _sync_commit_directories(plan, directories)
                revalidate_publish_plan(plan)
                directories.verify()
                _replace_current(plan, existing.manifest_sha256, directories)
                directories.verify()
                return PublishResult("already_published", plan, existing.manifest)
            try:
                _rename_noreplace(staged.root, plan.release_path, directories)
                renamed = True
            except FileExistsError:
                existing = _validate_existing(plan, directories)
                directories.cleanup(staged)
                _fsync_tree(
                    plan.release_path,
                    parent_fd=directories.releases_fd,
                    name=plan.release_id,
                    expected_identity=directories.release_identity(plan.release_id),
                )
                _sync_commit_directories(plan, directories)
                revalidate_publish_plan(plan)
                directories.verify()
                _replace_current(plan, existing.manifest_sha256, directories)
                directories.verify()
                return PublishResult("already_published", plan, existing.manifest)
            _sync_commit_directories(plan, directories)
            revalidate_publish_plan(plan)
            directories.verify()
            _replace_current(plan, report.manifest_sha256, directories)
            directories.verify()
        return PublishResult("published", plan, report.manifest)
    except BaseException:
        if not renamed:
            _cleanup_staging(staged)
        raise


__all__ = ["publish"]
