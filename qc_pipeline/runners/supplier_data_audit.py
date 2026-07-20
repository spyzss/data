"""Durable supplier-data audit producer and report adapter bridge."""

from __future__ import annotations

from dataclasses import replace
import json
from time import perf_counter

from qc_common.config import LoadedQcConfig
from qc_common.contracts import ModuleResult
from qc_common.manifest_metadata import manifest_metadata
from qc_pipeline.context import AssetContext


_IMPLEMENTATION_VERSION = "supplier-data-audit-producer-v3"
_RAW_OUTPUT_SCHEMA_VERSION = "supplier_data_audit.raw.v2"


def _with_runtime(
    result: ModuleResult,
    *,
    state: str,
    elapsed_seconds: float,
    fingerprint_sha256: str,
) -> ModuleResult:
    return replace(
        result,
        runtime={
            **dict(result.runtime),
            "artifact_state": state,
            "elapsed_seconds": float(elapsed_seconds),
            "fingerprint_sha256": fingerprint_sha256,
        },
    )


def run(context: AssetContext, config: LoadedQcConfig) -> ModuleResult:
    from acceptance_pull.supplier_audit import audit_supplier_data
    from qc_pipeline.adapters.supplier_data_audit import adapt_supplier_data_audit
    from qc_pipeline.artifacts import (
        artifact_for,
        build_run_fingerprint,
        canonical_sha256,
        module_result_from_dict,
        promote_artifact,
        reusable_artifact,
        staged_artifact,
        write_run_config,
    )

    started = perf_counter()
    present_source_names: list[str] = []
    declared_missing_sources: dict[str, str] = {}
    for source_name, declared in context.source_files.items():
        value = declared.get("path") if hasattr(declared, "get") else None
        if isinstance(value, str) and (context.batch_root / value).exists():
            present_source_names.append(source_name)
        else:
            declared_missing_sources[source_name] = str(value or "")
    fingerprint = build_run_fingerprint(
        context=context,
        producer="supplier_data_audit",
        config=config,
        module_names=("supplier_data_audit",),
        source_names=tuple(sorted(present_source_names)),
        implementation_version=_IMPLEMENTATION_VERSION,
        extra={
            "manifest_metadata": manifest_metadata(context.metadata),
            "declared_missing_sources": declared_missing_sources,
        },
    )
    supplier = str(
        context.metadata.get("supplier")
        or context.metadata.get("supplier_id")
        or ""
    ).lower()
    if supplier in {"qy", "qingyu"}:
        fingerprint["supplier_contract"] = {
            "quality_source_policy": (
                "inventory_only_ignored_for_acceptance_v1"
            )
        }
    fingerprint["output_schema_version"] = _RAW_OUTPUT_SCHEMA_VERSION
    fingerprint_sha256 = canonical_sha256(fingerprint)
    artifact = artifact_for(context, "supplier_data_audit")
    if bool(context.metadata.get("reuse_artifacts", True)) and reusable_artifact(
        artifact, fingerprint
    ):
        payload = json.loads(
            (artifact.directory / "supplier_data_audit_result.json").read_text(
                encoding="utf-8"
            )
        )
        return _with_runtime(
            module_result_from_dict(payload["module_result"]),
            state="reused",
            elapsed_seconds=perf_counter() - started,
            fingerprint_sha256=fingerprint_sha256,
        )

    raw = audit_supplier_data(
        context,
        config.module_parameters("supplier_data_audit"),
    )
    if raw.get("schema_version") != _RAW_OUTPUT_SCHEMA_VERSION:
        raise RuntimeError(
            "supplier_data_audit raw schema does not match producer identity"
        )
    result = adapt_supplier_data_audit(raw, config)
    elapsed = perf_counter() - started
    with staged_artifact(artifact) as staging:
        (staging / "supplier_data_audit_result.json").write_text(
            json.dumps(
                {"raw_result": raw, "module_result": result.to_dict()},
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        write_run_config(
            staging,
            producer="supplier_data_audit",
            outcome="completed",
            fingerprint=fingerprint,
            elapsed_seconds=elapsed,
        )
        promote_artifact(staging, artifact)
    return _with_runtime(
        result,
        state="computed",
        elapsed_seconds=elapsed,
        fingerprint_sha256=fingerprint_sha256,
    )


__all__ = ["run"]
