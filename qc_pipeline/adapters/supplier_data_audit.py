"""Adapt raw supplier inventory/audit evidence into a QC module result."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qc_common.config import LoadedQcConfig
from qc_common.contracts import Issue, ModuleResult, build_issue_id


def adapt_supplier_data_audit(
    raw: Mapping[str, Any],
    config: LoadedQcConfig,
) -> ModuleResult:
    module = "supplier_data_audit"
    asset_id = str(raw["asset_id"])
    configured_rules = config.module_rules(module)
    issues: list[Issue] = []
    for index, item in enumerate(raw.get("issues", ())):
        code = str(item["code"])
        rule = configured_rules.get(code, {})
        rule_id = str(rule.get("rule_id") or f"{module}.{code}")
        source_name = str(item.get("source_name") or "supplier_source")
        issues.append(
            Issue(
                issue_id=build_issue_id(
                    asset_id=asset_id,
                    module=module,
                    rule_id=rule_id,
                    source_relative_path=source_name,
                    coordinate_system="asset",
                    start_frame=None,
                    end_frame=None,
                    hand_side=None,
                    evidence_kind=f"supplier_audit_{index}",
                ),
                code=code,
                severity=item["severity"],
                module=module,
                issue_type="supplier_data_audit",
                metric=f"supplier_data_audit.{source_name}.status",
                observed_value=item.get("observed_value"),
                operator="==",
                boundary_value="present" if code.endswith("missing") else "verified",
                rule_id=rule_id,
                needs_manual_review=code == "mapping_unverified",
                context={"source_name": source_name},
            )
        )
    verdict = str(raw.get("decision") or "warn")
    evaluation = {
        "decision": verdict,
        "mapping_status": raw.get("mapping_status"),
    }
    if raw.get("reason") is not None:
        evaluation["reason"] = raw.get("reason")
    return ModuleResult(
        module=module,
        verdict=verdict,
        evaluation=evaluation,
        metrics={
            "supplier": raw.get("supplier"),
            "supplier_id": raw.get("supplier_id"),
            "supplier_name": raw.get("supplier_name"),
            "supplier_alias": raw.get("supplier_alias"),
            "required_source_count": len(raw.get("inventory", {})),
            "missing_source_count": len(raw.get("missing_sources", ())),
            "inventory": dict(raw.get("inventory", {})),
            "structured": dict(raw.get("structured", {})),
            "supplier_quality_signal": dict(
                raw.get("supplier_quality_signal", {})
            ),
        },
        issues=tuple(issues),
        runtime={"source": "supplier_data_audit"},
    )


__all__ = ["adapt_supplier_data_audit"]
