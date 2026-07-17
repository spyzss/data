#!/usr/bin/env python3
"""Ingest one explicit Canonical source and run or resume automatic QC."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canonical_qc.cli_support import (  # noqa: E402
    CliUsageError,
    JsonArgumentParser,
    emit,
    error_payload,
)
from canonical_qc.errors import CanonicalInputError  # noqa: E402
from canonical_qc.workflow import run_canonical_source_qc  # noqa: E402


COMMAND = "run_canonical_qc"


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-format", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--quality-archive", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=("acceptance", "supplier_evaluation"))
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--supplier-id", required=True)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--batch-metadata", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _result_payload(result: object) -> dict[str, object]:
    return {
        "asset_id": result.episode.identity.asset_id,
        "source": str(result.source),
        "source_format": result.source_format,
        "state": result.status,
        "status": result.status,
        "overall_decision": result.overall_decision,
        "report_path": str(result.report_path),
        "report_revision": result.report_revision,
        "executed_modules": list(result.executed_modules),
        "resumed": result.resumed,
        "dry_run": result.dry_run,
        "canonical_config": {
            "path": str(result.canonical_config_path),
            "version": result.canonical_config_version,
            "hash": result.canonical_config_hash,
        },
        "qc_config": {
            "path": str(result.qc_config_path),
            "version": result.qc_config_version,
            "hash": result.qc_config_hash,
        },
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        result = run_canonical_source_qc(
            source=args.source,
            source_format=args.source_format,
            source_root=args.source_root,
            batch_root=args.batch_root,
            quality_archive=args.quality_archive,
            profile=args.profile,
            expected_asset_id=args.asset_id,
            expected_batch_id=args.batch_id,
            expected_supplier_id=args.supplier_id,
            canonical_config_path=args.config,
            batch_metadata_path=args.batch_metadata,
            episode_index=args.episode_index,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    except CliUsageError as exc:
        emit(
            error_payload(
                command=COMMAND,
                category="input_contract",
                code="cli_usage_error",
                stage="cli",
                field=None,
                message=str(exc),
                retryable=False,
            ),
            stream=sys.stderr,
        )
        return 2
    except CanonicalInputError as exc:
        if exc.report_path is not None and exc.overall_decision == "fail":
            emit(
                {
                    "ok": False,
                    "command": COMMAND,
                    "category": "quality_fail",
                    "error": {
                        "category": "input_contract",
                        "code": exc.code,
                        "stage": "source_ingest",
                        "field": exc.field,
                        "message": exc.detail,
                        "retryable": False,
                    },
                    "result": {
                        "asset_id": args.asset_id,
                        "source": str(args.source.resolve()),
                        "source_format": args.source_format,
                        "state": "stopped",
                        "status": "stopped",
                        "overall_decision": "fail",
                        "report_path": str(exc.report_path),
                        "report_revision": exc.report_revision,
                    },
                }
            )
            return 2
        category = "qc_runtime" if exc.retryable else "input_contract"
        payload = error_payload(
                command=COMMAND,
                category=category,
                code=exc.code,
                stage="source_ingest",
                field=exc.field,
                message=exc.detail,
                retryable=exc.retryable,
            )
        if exc.report_path is not None:
            payload["report_path"] = str(exc.report_path)
            payload["report_revision"] = exc.report_revision
        emit(payload, stream=sys.stderr)
        return 3 if exc.retryable else 2
    except ValueError as exc:
        emit(
            error_payload(
                command=COMMAND,
                category="input_contract",
                code="input_contract_error",
                stage="source_ingest",
                field=None,
                message=str(exc),
                retryable=False,
            ),
            stream=sys.stderr,
        )
        return 2
    except OSError as exc:
        emit(
            error_payload(
                command=COMMAND,
                category="qc_runtime",
                code="qc_runtime_error",
                stage="qc",
                field=None,
                message=str(exc),
                retryable=True,
            ),
            stream=sys.stderr,
        )
        return 3
    except Exception as exc:
        emit(
            error_payload(
                command=COMMAND,
                category="qc_runtime",
                code="qc_runtime_error",
                stage="qc",
                field=None,
                message=str(exc),
                retryable=True,
            ),
            stream=sys.stderr,
        )
        return 3

    payload = _result_payload(result)
    if result.status == "error":
        runtime = result.runtime_error or {}
        emit(
            error_payload(
                command=COMMAND,
                category="qc_runtime",
                code=str(runtime.get("error_type", "qc_runtime_error")),
                stage="qc",
                field=str(runtime.get("module")) if runtime.get("module") else None,
                message=str(runtime.get("message", "QC orchestrator returned error")),
                retryable=True,
            ),
            stream=sys.stderr,
        )
        return 3
    if result.overall_decision == "fail":
        emit({"ok": False, "command": COMMAND, "category": "quality_fail", "result": payload})
        return 2
    emit({"ok": True, "command": COMMAND, "result": payload})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
