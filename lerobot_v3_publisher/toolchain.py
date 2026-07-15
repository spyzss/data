"""Stable identity for tools that affect Curated LeRobot v3 bytes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
import subprocess

import numpy as np
import pandas as pd
import pyarrow as pa


TOOLCHAIN_SCHEMA_VERSION = "curated_lerobot_v3_toolchain.v1"


@dataclass(frozen=True, slots=True)
class WriterToolchain:
    schema_version: str
    python_version: str
    numpy_version: str
    pyarrow_version: str
    pandas_version: str
    ffmpeg_version: str
    ffmpeg_signature_sha256: str
    libx264_signature_sha256: str
    fingerprint: str


def _command_signature(argv: list[str]) -> str:
    completed = subprocess.run(
        argv,
        shell=False,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"toolchain command failed: {' '.join(argv)}")
    return "\n".join(
        line.rstrip() for line in (completed.stdout + completed.stderr).splitlines()
    ).strip()


def current_toolchain() -> WriterToolchain:
    ffmpeg_output = _command_signature(["ffmpeg", "-version"])
    build_configuration = _command_signature(["ffmpeg", "-buildconf"])
    encoder = _command_signature(["ffmpeg", "-hide_banner", "-h", "encoder=libx264"])
    payload = {
        "schema_version": TOOLCHAIN_SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pyarrow_version": pa.__version__,
        "pandas_version": pd.__version__,
        "ffmpeg_version": ffmpeg_output.splitlines()[0],
        "ffmpeg_signature_sha256": hashlib.sha256(
            f"{ffmpeg_output}\n{build_configuration}".encode("utf-8")
        ).hexdigest(),
        "libx264_signature_sha256": hashlib.sha256(encoder.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return WriterToolchain(**payload, fingerprint=hashlib.sha256(encoded).hexdigest())


def publisher_version_for(base: str) -> str:
    return f"{base}+toolchain.{current_toolchain().fingerprint[:16]}"


__all__ = [
    "TOOLCHAIN_SCHEMA_VERSION",
    "WriterToolchain",
    "current_toolchain",
    "publisher_version_for",
]
