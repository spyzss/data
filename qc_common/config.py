from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from qc_common.schema import validate_qc_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LoadedQcConfig:
    path: Path
    raw: dict[str, Any]
    sha256: str

    @property
    def schema_version(self) -> str:
        return str(self.raw["schema_version"])

    @property
    def config_version(self) -> str:
        return str(self.raw["config_version"])

    @property
    def config_name(self) -> str:
        return str(self.raw["config_name"])

    def module_parameters(self, module_name: str) -> dict[str, Any]:
        module = self.raw["modules"][module_name]
        parameters = module.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"module {module_name} has no parameters mapping")
        return copy.deepcopy(parameters)

    def module_rules(self, module_name: str) -> dict[str, Any]:
        rules = self.raw["modules"][module_name].get("rules", {})
        if not isinstance(rules, dict):
            raise ValueError(f"module {module_name} rules must be a mapping")
        return copy.deepcopy(rules)

    def json_reference(self) -> dict[str, str]:
        try:
            config_path = str(self.path.relative_to(_repo_root()))
        except ValueError:
            config_path = str(self.path)
        return {
            "schema_version": self.schema_version,
            "config_version": self.config_version,
            "config_name": self.config_name,
            "config_path": config_path,
            "config_hash": self.sha256,
        }


def load_qc_acceptance_config(path: Path | None = None) -> LoadedQcConfig:
    resolved = (path or _repo_root() / "configs" / "qc_acceptance.yaml").resolve()
    payload = resolved.read_bytes()
    raw = yaml.safe_load(payload) or {}
    if not isinstance(raw, dict) or "schema_version" not in raw:
        raise ValueError("expected unified qc_acceptance config")
    validate_qc_config(raw)

    modules = raw["modules"]
    missing_modules = [name for name in raw["pipeline"]["modules"] if name not in modules]
    if missing_modules:
        raise ValueError(f"pipeline module missing config: {missing_modules[0]}")

    seen_rule_ids: set[str] = set()
    for module_name, module in modules.items():
        rules = module.get("rules", {})
        if not isinstance(rules, dict):
            raise ValueError(f"module {module_name} rules must be a mapping")
        for rule in rules.values():
            if not isinstance(rule, dict):
                raise ValueError(f"module {module_name} rule must be a mapping")
            rule_id = rule.get("rule_id")
            if rule_id in seen_rule_ids:
                raise ValueError(f"duplicate rule_id: {rule_id}")
            if rule_id is not None:
                seen_rule_ids.add(str(rule_id))

    return LoadedQcConfig(
        path=resolved,
        raw=raw,
        sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )
