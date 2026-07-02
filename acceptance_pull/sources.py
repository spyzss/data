from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import oss2
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from acceptance_pull.config import SourceConfig
from acceptance_pull.models import PullResult, SampleRow


OSS_BROWSER_LOGIN_UNAVAILABLE = "oss-browser2 login context is unavailable; open oss-browser2 and sign in first"
OSS_BROWSER_AES_KEY = b"NsGVeVW7BM7BBPqizlJv8fPz0hQFU4Dn"


@dataclass(frozen=True)
class BatchInputFiles:
    manifest: Path
    readme: Path


class Source(ABC):
    file_type: str

    @abstractmethod
    def resolve(self, asset_id: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def pull(self, asset_id: str, target: Path) -> PullResult:
        raise NotImplementedError


def filename_for(asset_id: str, file_type: str) -> str:
    if file_type == "hdf5":
        return f"{asset_id}_hdf5.hdf5"
    if file_type == "video":
        return f"{asset_id}_video.mp4"
    raise ValueError(f"unsupported file type: {file_type}")


class LocalSource(Source):
    def __init__(self, root: Path, file_type: str) -> None:
        self.root = root
        self.file_type = file_type

    def resolve(self, asset_id: str) -> str:
        return str(self.root / filename_for(asset_id, self.file_type))

    def pull(self, asset_id: str, target: Path) -> PullResult:
        source_path = Path(self.resolve(asset_id))
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source_path, target)
            return PullResult(asset_id, self.file_type, str(source_path), target, "success")
        except Exception as exc:
            return PullResult(asset_id, self.file_type, str(source_path), target, "failed", type(exc).__name__)


class OssBrowserSource(Source):
    def __init__(self, config: SourceConfig, file_type: str, app_support_dir: Path | None = None) -> None:
        self.config = config
        self.file_type = file_type
        self.app_support_dir = app_support_dir or Path.home() / "Library" / "Application Support" / "oss-browser2"

    def _load_options(self) -> dict[str, Any]:
        session = load_oss_browser2_current_session(self.app_support_dir)
        try:
            options = session["loginSession"]["options"]
        except KeyError as exc:
            raise RuntimeError(OSS_BROWSER_LOGIN_UNAVAILABLE) from exc
        if not isinstance(options, dict):
            raise RuntimeError(OSS_BROWSER_LOGIN_UNAVAILABLE)
        if not options.get("accessKeyId") or not options.get("accessKeySecret"):
            raise RuntimeError(OSS_BROWSER_LOGIN_UNAVAILABLE)
        return options

    def _bucket(self) -> oss2.Bucket:
        options = self._load_options()
        token = options.get("stsToken") or options.get("securityToken") or options.get("security_token")
        if token:
            auth = oss2.StsAuth(options["accessKeyId"], options["accessKeySecret"], token)
        else:
            auth = oss2.Auth(options["accessKeyId"], options["accessKeySecret"])
        return oss2.Bucket(auth, self.config.endpoint, self.config.bucket)

    def resolve(self, asset_id: str) -> str:
        self._load_options()
        key = "/".join(part for part in [self.config.prefix, filename_for(asset_id, self.file_type)] if part)
        return f"oss://{self.config.bucket}/{key}"

    def pull(self, asset_id: str, target: Path) -> PullResult:
        source = self.resolve(asset_id)
        key = "/".join(part for part in [self.config.prefix, filename_for(asset_id, self.file_type)] if part)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._bucket().get_object_to_file(key, str(target))
            return PullResult(asset_id, self.file_type, source, target, "success")
        except Exception as exc:
            return PullResult(asset_id, self.file_type, source, target, "failed", type(exc).__name__)

    def list_direct_object_keys(self, prefix: str) -> list[str]:
        normalized = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
        result = self._bucket().list_objects(prefix=normalized, delimiter="/", max_keys=1000)
        return [obj.key for obj in result.object_list if not obj.key.endswith("/")]

    def download_key(self, key: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        self._bucket().get_object_to_file(key, str(target))


def load_oss_browser2_current_session(app_support_dir: Path) -> dict[str, Any]:
    config_path = app_support_dir / "config.json"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(OSS_BROWSER_LOGIN_UNAVAILABLE) from exc

    session = raw.get("currentSession")
    if isinstance(session, dict):
        return session
    if isinstance(session, str) and session.strip().startswith("{"):
        try:
            parsed = json.loads(session)
        except json.JSONDecodeError as exc:
            raise RuntimeError(OSS_BROWSER_LOGIN_UNAVAILABLE) from exc
        if isinstance(parsed, dict):
            return parsed
    if isinstance(session, str) and _is_hex(session):
        try:
            parsed = json.loads(_decrypt_oss_browser_hex(session))
        except Exception as exc:
            raise RuntimeError(OSS_BROWSER_LOGIN_UNAVAILABLE) from exc
        if isinstance(parsed, dict):
            return parsed
    raise RuntimeError(OSS_BROWSER_LOGIN_UNAVAILABLE)


def _is_hex(value: str) -> bool:
    stripped = value.strip()
    return bool(stripped) and len(stripped) % 2 == 0 and all(char in "0123456789abcdefABCDEF" for char in stripped)


def _decrypt_oss_browser_hex(value: str) -> str:
    encrypted = bytes.fromhex(value.strip())
    decryptor = Cipher(algorithms.AES(OSS_BROWSER_AES_KEY), modes.CBC(bytes(16))).decryptor()
    padded = decryptor.update(encrypted) + decryptor.finalize()
    pad_len = padded[-1]
    if pad_len < 1 or pad_len > 16:
        raise ValueError("invalid padding")
    return padded[:-pad_len].decode("utf-8")


def build_source(config: SourceConfig, file_type: str) -> Source:
    if config.kind == "local":
        if config.root is None:
            raise ValueError("local source requires root")
        return LocalSource(config.root, file_type)
    if config.kind in {"oss", "oss-browser2"}:
        return OssBrowserSource(config, file_type)
    raise ValueError(f"unsupported source kind: {config.kind}")


def download_oss_batch_inputs(config: SourceConfig, target_dir: Path) -> BatchInputFiles:
    if config.kind not in {"oss", "oss-browser2"}:
        raise ValueError("batch inputs can only be downloaded from OSS source")

    source = OssBrowserSource(config, "hdf5")
    keys = source.list_direct_object_keys(config.prefix or "")
    readme_key = _first_key_with_basename(keys, "README.txt")
    manifest_key = _first_key_with_suffix(keys, ".xlsx")
    if readme_key is None:
        raise FileNotFoundError(f"README.txt not found under oss://{config.bucket}/{config.prefix}")
    if manifest_key is None:
        raise FileNotFoundError(f"xlsx manifest not found under oss://{config.bucket}/{config.prefix}")

    readme_path = target_dir / Path(readme_key).name
    manifest_path = target_dir / Path(manifest_key).name
    source.download_key(readme_key, readme_path)
    source.download_key(manifest_key, manifest_path)
    return BatchInputFiles(manifest=manifest_path, readme=readme_path)


def _first_key_with_basename(keys: list[str], basename: str) -> str | None:
    expected = basename.lower()
    for key in sorted(keys):
        if Path(key).name.lower() == expected:
            return key
    return None


def _first_key_with_suffix(keys: list[str], suffix: str) -> str | None:
    expected = suffix.lower()
    for key in sorted(keys):
        if Path(key).name.lower().endswith(expected):
            return key
    return None


def pull_pairs(
    rows: list[SampleRow],
    hdf5_source: Source,
    video_source: Source,
    output: Path,
    workers: int,
) -> list[PullResult]:
    tasks: list[tuple[Source, str, Path]] = []
    for row in rows:
        tasks.append((hdf5_source, row.asset_id, output / "hdf5" / filename_for(row.asset_id, "hdf5")))
        tasks.append((video_source, row.asset_id, output / "video" / filename_for(row.asset_id, "video")))

    results: list[PullResult] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(source.pull, asset_id, target) for source, asset_id, target in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: (item.asset_id, item.file_type))
