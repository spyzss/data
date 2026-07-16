from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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

    @property
    def pipeline_modules(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.raw["pipeline"]["modules"])

    @property
    def default_profile(self) -> str:
        return str(self.raw["pipeline"]["default_profile"])

    def execution_profile(self, name: str) -> dict[str, str]:
        try:
            return copy.deepcopy(self.raw["execution_profiles"][name])
        except KeyError as exc:
            raise ValueError(f"unknown execution profile: {name}") from exc

    def frame_survival_policy(self, profile: str) -> dict[str, Any]:
        """Return the versioned acceptance-only frame-survival policy."""
        if profile != "acceptance":
            return {"enabled": False}
        acceptance_policy = self.raw.get("acceptance_policy")
        if not isinstance(acceptance_policy, Mapping):
            return {"enabled": False}
        policy = acceptance_policy.get("frame_survival")
        if not isinstance(policy, Mapping):
            return {"enabled": False}
        return copy.deepcopy(dict(policy))

    def module_config(self, name: str) -> dict[str, Any]:
        try:
            return copy.deepcopy(self.raw["modules"][name])
        except KeyError as exc:
            raise ValueError(f"unknown pipeline module: {name}") from exc

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
        policy = self.frame_survival_policy("acceptance")
        policy_payload = json.dumps(
            policy,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "schema_version": self.schema_version,
            "config_version": self.config_version,
            "config_name": self.config_name,
            "config_path": config_path,
            "config_hash": self.sha256,
            "acceptance_policy_version": str(policy.get("version", "disabled")),
            "acceptance_policy_hash": "sha256:"
            + hashlib.sha256(policy_payload).hexdigest(),
        }

    def assert_same_reference(self, reference: Mapping[str, str]) -> None:
        expected = self.json_reference()
        for key in (
            "schema_version",
            "config_version",
            "config_hash",
            "acceptance_policy_version",
            "acceptance_policy_hash",
        ):
            if reference.get(key) != expected[key]:
                raise ValueError(
                    f"QC config drift at {key}: {reference.get(key)} != {expected[key]}"
                )


def load_qc_acceptance_config(path: Path | None = None) -> LoadedQcConfig:
    resolved = (path or _repo_root() / "configs" / "qc_acceptance.yaml").resolve()
    payload = resolved.read_bytes()
    raw = yaml.safe_load(payload) or {}
    if not isinstance(raw, dict) or "schema_version" not in raw:
        raise ValueError("expected unified qc_acceptance config")
    validate_qc_config(raw)

    if path is None:
        snapshot = (
            _repo_root()
            / "configs"
            / "qc_acceptance"
            / f"{raw['config_version']}.yaml"
        )
        if not snapshot.is_file() or snapshot.read_bytes() != payload:
            raise ValueError(
                f"active QC config does not match immutable snapshot: {snapshot.name}"
            )

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
            if not isinstance(rule_id, str) or not rule_id.strip():
                raise ValueError(f"module {module_name} rule must have non-empty rule_id")
            if rule_id in seen_rule_ids:
                raise ValueError(f"duplicate rule_id: {rule_id}")
            seen_rule_ids.add(rule_id)

    if raw["schema_version"] == "qc_acceptance_config_schema.v2":
        profiles = raw["execution_profiles"]
        default_profile = raw["pipeline"]["default_profile"]
        if default_profile not in profiles:
            raise ValueError(f"unknown default execution profile: {default_profile}")

        for module_name, module in modules.items():
            if module["enabled"]:
                has_implementation = bool(str(module.get("implementation", "")).strip())
                is_external = module.get("execution_kind") == "external"
                if has_implementation == is_external:
                    raise ValueError(
                        f"enabled module {module_name} must define exactly one of implementation "
                        "or execution_kind: external"
                    )
            elif not str(module.get("disabled_reason", "")).strip():
                raise ValueError(f"disabled module {module_name} must define disabled_reason")

    return LoadedQcConfig(
        path=resolved,
        raw=raw,
        sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )
