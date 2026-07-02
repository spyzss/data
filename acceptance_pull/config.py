from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


DEFAULT_WORKERS = 8
DEFAULT_SAMPLE_RATIO = 0.01


def default_seed(today: date | None = None) -> int:
    value = today or date.today()
    return int(value.strftime("%Y%m%d"))


def endpoint_from_region(region: str) -> str:
    value = str(region).strip().lower().rstrip("/")
    if not value:
        raise ValueError("region is required for OSS batch source")
    if value.startswith(("http://", "https://")):
        return value
    if value.startswith("oss-"):
        region_id = value
    elif value.startswith("cn-"):
        region_id = f"oss-{value}"
    else:
        region_id = f"oss-cn-{value}"
    return f"https://{region_id}.aliyuncs.com"


def parse_oss_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(str(uri).strip())
    if parsed.scheme != "oss":
        raise ValueError("batch_uri must start with oss://")
    if not parsed.netloc:
        raise ValueError("batch_uri must include bucket")
    return parsed.netloc, parsed.path.strip("/")


def join_oss_prefix(*parts: str | None) -> str:
    return "/".join(str(part).strip("/") for part in parts if part is not None and str(part).strip("/"))


@dataclass(frozen=True)
class SourceConfig:
    kind: str
    root: Path | None = None
    endpoint: str | None = None
    bucket: str | None = None
    prefix: str | None = None

    @classmethod
    def from_mapping(cls, base_dir: Path, data: dict[str, Any], label: str) -> "SourceConfig":
        kind = str(data.get("kind", "")).strip().lower()
        if kind not in {"local", "oss", "oss-browser2"}:
            raise ValueError(f"{label}.kind must be local, oss, or oss-browser2")

        root = data.get("root")
        if kind == "local":
            if not root:
                raise ValueError(f"{label}.root is required for local source")
            return cls(kind=kind, root=(base_dir / str(root)).resolve())

        endpoint = data.get("endpoint")
        bucket = data.get("bucket")
        prefix = data.get("prefix")
        if not endpoint or not bucket or prefix is None:
            raise ValueError(f"{label} OSS source requires endpoint, bucket, and prefix")
        return cls(kind=kind, endpoint=str(endpoint), bucket=str(bucket), prefix=str(prefix).strip("/"))


@dataclass(frozen=True)
class PullConfig:
    hdf5: SourceConfig
    video: SourceConfig
    output: Path
    manifest: Path | None = None
    readme: Path | None = None
    batch_source: SourceConfig | None = None
    workers: int = DEFAULT_WORKERS
    seed: int = field(default_factory=default_seed)
    sample_ratio: float = DEFAULT_SAMPLE_RATIO

    @classmethod
    def from_file(cls, path: Path) -> "PullConfig":
        base_dir = path.resolve().parent
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        workers = int(raw.get("workers", DEFAULT_WORKERS))
        if workers < 1:
            raise ValueError("workers must be >= 1")
        sample_ratio = float(raw.get("sample_ratio", DEFAULT_SAMPLE_RATIO))
        if sample_ratio <= 0 or sample_ratio > 1:
            raise ValueError("sample_ratio must be > 0 and <= 1")

        output = raw.get("output")
        if not output:
            raise ValueError("output is required")

        seed = int(raw["seed"]) if "seed" in raw else default_seed()
        output_path = (base_dir / str(output)).resolve()
        manifest = raw.get("manifest")
        readme = raw.get("readme")
        batch_uri = raw.get("batch_uri")

        if batch_uri:
            region = raw.get("region") or raw.get("endpoint")
            endpoint = endpoint_from_region(str(region or ""))
            bucket, prefix = parse_oss_uri(str(batch_uri))
            batch_source = SourceConfig(kind="oss", endpoint=endpoint, bucket=bucket, prefix=prefix)
            return cls(
                hdf5=SourceConfig(
                    kind="oss",
                    endpoint=endpoint,
                    bucket=bucket,
                    prefix=join_oss_prefix(prefix, "hdf5"),
                ),
                video=SourceConfig(
                    kind="oss",
                    endpoint=endpoint,
                    bucket=bucket,
                    prefix=join_oss_prefix(prefix, "video"),
                ),
                output=output_path,
                manifest=(base_dir / str(manifest)).resolve() if manifest else None,
                readme=(base_dir / str(readme)).resolve() if readme else None,
                batch_source=batch_source,
                workers=workers,
                seed=seed,
                sample_ratio=sample_ratio,
            )

        if not manifest:
            raise ValueError("manifest is required")
        return cls(
            hdf5=SourceConfig.from_mapping(base_dir, raw.get("hdf5") or {}, "hdf5"),
            video=SourceConfig.from_mapping(base_dir, raw.get("video") or {}, "video"),
            output=output_path,
            manifest=(base_dir / str(manifest)).resolve(),
            readme=(base_dir / str(readme)).resolve() if readme else None,
            workers=workers,
            seed=seed,
            sample_ratio=sample_ratio,
        )
