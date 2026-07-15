"""Versioned top-level configuration for Canonical ingest, QC, and publish."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import yaml

from .adapters import StandardHdf5Adapter, StandardLeRobotAdapter


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class LoadedCanonicalQcConfig:
    path: Path
    raw: dict[str, Any]
    sha256: str
    qc_config_path: Path

    @property
    def config_version(self) -> str:
        return str(self.raw["config_version"])

    @property
    def timestamp_tolerance_ns(self) -> int:
        return int(self.raw["source"]["timestamp_tolerance_ns"])

    @property
    def profiles(self) -> tuple[str, ...]:
        return tuple(self.raw["qc"]["profiles"])

    @property
    def exit_codes(self) -> dict[str, int]:
        return {key: int(value) for key, value in self.raw["cli"]["exit_codes"].items()}

    @property
    def source_gate_rule_id(self) -> str:
        source_gate = self.raw.get("source_gate")
        if not isinstance(source_gate, dict):
            raise ValueError("Canonical QC config has no Source Gate rule registry")
        return str(source_gate["rules"]["contract_failure"]["rule_id"])


def _validate_schema(raw: dict[str, Any]) -> None:
    schema = json.loads(
        (_repo_root() / "schemas/canonical_qc_config.v1.schema.json").read_text()
    )
    errors = sorted(
        Draft202012Validator(schema).iter_errors(raw), key=lambda item: list(item.path)
    )
    if errors:
        error = errors[0]
        field = ".".join(str(item) for item in error.absolute_path) or "$"
        raise ValueError(f"Canonical QC config validation failed at {field}: {error.message}")


def load_canonical_qc_config(path: Path | None = None) -> LoadedCanonicalQcConfig:
    from lerobot_v3_publisher.contracts import PUBLISHER_VERSION
    from lerobot_v3_publisher.toolchain import TOOLCHAIN_SCHEMA_VERSION
    from lerobot_v3_publisher.validation import OFFICIAL_READER_VERSION

    root = _repo_root()
    resolved = (path or root / "configs/canonical_qc.yaml").resolve()
    payload = resolved.read_bytes()
    raw = yaml.safe_load(payload)
    if not isinstance(raw, dict):
        raise ValueError("Canonical QC config must be an object")
    _validate_schema(raw)
    if "source_gate" not in raw:
        raise ValueError(
            "Canonical QC configs before v1.1.0 are not executable: "
            "the Source Gate rule registry is required"
        )
    if path is None:
        snapshot = root / "configs/canonical_qc" / f"{raw['config_version']}.yaml"
        if not snapshot.is_file() or snapshot.read_bytes() != payload:
            raise ValueError(
                f"active Canonical QC config does not match immutable snapshot: {snapshot.name}"
            )
    else:
        registered = root / "configs/canonical_qc" / f"{raw['config_version']}.yaml"
        if not registered.is_file() or registered.read_bytes() != payload:
            raise ValueError(
                f"explicit Canonical QC config does not match immutable snapshot: {registered.name}"
            )

    qc_configured = Path(raw["qc"]["config_path"]).expanduser()
    if qc_configured.is_absolute():
        qc_path = qc_configured.resolve()
    else:
        qc_path = (root / qc_configured).resolve()
    qc_payload = qc_path.read_bytes()
    qc_hash = f"sha256:{hashlib.sha256(qc_payload).hexdigest()}"
    if qc_hash != raw["qc"]["config_sha256"]:
        raise ValueError("configured QC snapshot hash does not match qc.config_sha256")
    qc_raw = yaml.safe_load(qc_payload)
    if not isinstance(qc_raw, dict) or qc_raw.get("config_version") != raw["qc"]["config_version"]:
        raise ValueError("configured QC version does not match qc.config_version")

    runtime = raw["runtime_contract"]
    if StandardHdf5Adapter.adapter_version != runtime["standard_hdf5_adapter_version"]:
        raise ValueError("StandardHdf5Adapter runtime version drift")
    if StandardLeRobotAdapter.adapter_version != runtime["standard_lerobot_adapter_version"]:
        raise ValueError("StandardLeRobotAdapter runtime version drift")
    if not PUBLISHER_VERSION.startswith(raw["publisher"]["publisher_version_prefix"]):
        raise ValueError("LeRobot v3 publisher runtime version drift")
    if TOOLCHAIN_SCHEMA_VERSION != raw["publisher"]["toolchain_schema_version"]:
        raise ValueError("LeRobot v3 toolchain schema version drift")
    if OFFICIAL_READER_VERSION != raw["publisher"]["official_reader_version"]:
        raise ValueError("official LeRobot reader version drift")

    return LoadedCanonicalQcConfig(
        path=resolved,
        raw=raw,
        sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        qc_config_path=qc_path,
    )


__all__ = ["LoadedCanonicalQcConfig", "load_canonical_qc_config"]
