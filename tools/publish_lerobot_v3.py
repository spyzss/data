#!/usr/bin/env python3
"""Validate or atomically publish one final Canonical episode as LeRobot v3."""

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


COMMAND = "publish_lerobot_v3"


def publish_from_paths(**kwargs: object) -> object:
    from lerobot_v3_publisher.workflow import publish_from_paths as implementation

    return implementation(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-format", required=True, choices=("hdf5", "lerobot"))
    parser.add_argument("--canonical-source-root", required=True, type=Path)
    parser.add_argument("--qc-report", required=True, type=Path)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--batch-metadata", type=Path)
    parser.add_argument("--revision-artifact", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _publisher_error(exc: BaseException) -> tuple[int, str, dict[str, object]] | None:
    diagnostic = getattr(exc, "diagnostic", None)
    if diagnostic is None:
        return None
    code = str(diagnostic.code)
    if code == "commit_conflict":
        exit_code, category = 6, "commit_conflict"
    elif code == "validation_failed":
        exit_code, category = 5, "publish_validation"
    elif code == "staging_failed":
        exit_code, category = 3, "publish_runtime"
    elif code == "source_integrity_error" and diagnostic.retryable:
        exit_code, category = 3, "publish_runtime"
    else:
        exit_code, category = 4, "publish_prerequisite"
    payload = error_payload(
        command=COMMAND,
        category=category,
        code=code,
        stage=str(diagnostic.stage),
        field=diagnostic.field,
        message=str(diagnostic.message),
        retryable=bool(diagnostic.retryable),
    )
    return exit_code, category, payload


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        result = publish_from_paths(
            source=args.source,
            source_format=args.source_format,
            canonical_source_root=args.canonical_source_root,
            qc_report_path=args.qc_report,
            release_root=args.release_root,
            episode_index=args.episode_index,
            canonical_config_path=args.config,
            batch_metadata_path=args.batch_metadata,
            revision_artifact_path=args.revision_artifact,
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
        category = "publish_runtime" if exc.retryable else "input_contract"
        emit(
            error_payload(
                command=COMMAND,
                category=category,
                code=exc.code,
                stage="source_ingest",
                field=exc.field,
                message=exc.detail,
                retryable=exc.retryable,
            ),
            stream=sys.stderr,
        )
        return 3 if exc.retryable else 2
    except Exception as exc:
        classified = _publisher_error(exc)
        if classified is not None:
            exit_code, _category, payload = classified
            emit(payload, stream=sys.stderr)
            return exit_code
        category = "input_contract" if isinstance(exc, ValueError) and not isinstance(exc, OSError) else "publish_runtime"
        emit(
            error_payload(
                command=COMMAND,
                category=category,
                code="publish_input_error" if category == "input_contract" else "publish_runtime_error",
                stage="publish",
                field=None,
                message=str(exc),
                retryable=category == "publish_runtime",
            ),
            stream=sys.stderr,
        )
        return 2 if category == "input_contract" else 3

    plan = result if args.dry_run else result.plan
    state = "validated" if args.dry_run else result.state
    emit(
        {
            "ok": True,
            "command": COMMAND,
            "result": {
                "state": state,
                "asset_id": plan.request.episode.identity.asset_id,
                "release_id": plan.release_id,
                "release_path": str(plan.release_path),
                "current_path": str(plan.current_path),
                "qc_report_path": str(plan.request.qc_report_path),
                "qc_report_revision": plan.qc_report_revision,
                "canonical_revision": plan.request.canonical_revision,
                "dry_run": bool(args.dry_run),
            },
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
