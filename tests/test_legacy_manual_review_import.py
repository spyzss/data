from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from human_qc.legacy_import import import_legacy_manual_review
from qc_common.report import StaleReportRevisionError, load_asset_qc_report, write_asset_qc_report
from tests.qc_report_fixtures import make_manual_block, make_semantic_block, make_v2_report
from tools import import_legacy_manual_review as legacy_cli
from tools import serve_manual_review as legacy_server


ASSET_ID = "asset-001"


def _issue(issue_id: str) -> dict:
    return {
        "issue_id": issue_id,
        "code": "legacy_warn",
        "severity": "warn",
        "module": "video_quality",
        "needs_manual_review": True,
        "context": {},
    }


def _report(tmp_path: Path, *, issue_ids: tuple[str, ...] = ("warn-1", "warn-2")) -> Path:
    report = make_v2_report(status="awaiting_external")
    report["asset_id"] = ASSET_ID
    report["issues"] = [_issue(issue_id) for issue_id in issue_ids]
    report["semantic_calibration"] = make_semantic_block(state="completed")
    report["manual_review"] = make_manual_block(
        state="queued",
        candidate_issue_ids=list(dict.fromkeys(issue_ids)),
        selected_issue_ids=list(dict.fromkeys(issue_ids)),
    )
    report["pipeline_state"] = {
        "status": "awaiting_external",
        "last_completed_module": "semantic_consistency",
        "next_module": "manual_review",
        "stop_reason": None,
    }
    path = tmp_path / "quality_archive" / f"{ASSET_ID}.json"
    write_asset_qc_report(path, report, expected_revision=0, profile="acceptance")
    return path


def _csv(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "manual_labels.csv"
    fieldnames = [
        "asset_id",
        "issue_id",
        "review_id",
        "manual_outcome",
        "comment",
        "reviewer",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(
    issue_id: str,
    outcome: str,
    *,
    asset_id: str = ASSET_ID,
    comment: str = "legacy note",
) -> dict[str, str]:
    return {
        "asset_id": asset_id,
        "issue_id": issue_id,
        "review_id": "",
        "manual_outcome": outcome,
        "comment": comment,
        "reviewer": "old-reviewer",
    }


def test_dry_run_reports_exact_matches_without_writing(tmp_path: Path) -> None:
    report_path = _report(tmp_path)
    csv_path = _csv(
        tmp_path,
        [
            _row("warn-1", "false_positive"),
            _row("unknown", "true_positive"),
            _row("warn-2", "true_positive", asset_id="another-asset"),
        ],
    )
    before = report_path.read_bytes()

    result = import_legacy_manual_review(
        report_path,
        csv_path,
        None,
        expected_revision=1,
        reviewer="migration-bot",
        dry_run=True,
    )

    assert result.matched_count == 1
    assert result.unmatched_count == 2
    assert result.conflict_count == 0
    assert result.matched_issue_ids == ("warn-1",)
    assert {problem.code for problem in result.problems} == {
        "asset_id_mismatch",
        "unknown_issue_id",
    }
    assert result.written is False
    assert result.report_revision == 1
    assert report_path.read_bytes() == before


def test_import_writes_manual_block_audit_and_completes_selected_reviews(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path)
    csv_path = _csv(
        tmp_path,
        [
            _row("warn-1", "false_positive", comment="not a real defect"),
            _row("warn-2", "true_positive", comment="confirmed defect"),
        ],
    )
    progress_path = tmp_path / "manual_review_progress.json"
    progress_path.write_text(
        json.dumps({"segmentsByReviewId": {"warn-1": [], "warn-2": []}}),
        encoding="utf-8",
    )
    machine_before = copy.deepcopy(load_asset_qc_report(report_path)["issues"])

    result = import_legacy_manual_review(
        report_path,
        csv_path,
        progress_path,
        expected_revision=1,
        reviewer="migration-bot",
    )

    assert result.written is True
    assert result.report_revision == 2
    persisted = load_asset_qc_report(report_path)
    assert persisted is not None
    assert persisted["issues"] == machine_before
    manual = persisted["manual_review"]
    assert manual["state"] == "completed"
    assert manual["completion_mode"] == "early_fail"
    assert manual["failure_reason"] is None
    assert manual["issue_reviews"]["warn-1"]["verdict"] == "pass"
    assert manual["issue_reviews"]["warn-2"]["verdict"] == "fail"
    assert manual["issue_reviews"]["warn-1"]["source"] == "legacy_manual_review_import"
    assert manual["import_audit"][-1]["action"] == "legacy_manual_review_import"
    assert manual["import_audit"][-1]["csv_sha256"].startswith("sha256:")
    assert manual["import_audit"][-1]["progress_sha256"].startswith("sha256:")


def test_import_pass_preserves_existing_failure_reason_while_fail_remains(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path)
    report = load_asset_qc_report(report_path)
    assert report is not None
    reason = {
        "mode": "manual",
        "reason_codes": ["occlusion"],
        "other_text": None,
    }
    original_audit = [
        {
            "action": "failure_reason_changed",
            "issue_id": "warn-1",
            "previous_failure_reason": None,
            "failure_reason": copy.deepcopy(reason),
            "reviewed_at": "2026-07-21T00:00:00Z",
        }
    ]
    report["manual_review"].update(
        {
            "state": "in_progress",
            "issue_reviews": {
                "warn-1": {
                    "verdict": "fail",
                    "effective_verdict": "fail",
                    "machine_verdict": "warn",
                    "reason": "confirmed defect",
                    "reviewer": "alice",
                    "reviewed_at": "2026-07-21T00:00:00Z",
                }
            },
            "failure_reason": copy.deepcopy(reason),
            "review_audit": copy.deepcopy(original_audit),
        }
    )
    report["report_revision"] = 2
    write_asset_qc_report(
        report_path,
        report,
        expected_revision=1,
        profile="acceptance",
    )

    import_legacy_manual_review(
        report_path,
        _csv(tmp_path, [_row("warn-2", "false_positive")]),
        None,
        expected_revision=2,
        reviewer="migration-bot",
    )

    persisted = load_asset_qc_report(report_path)
    assert persisted is not None
    manual = persisted["manual_review"]
    assert manual["completion_mode"] == "early_fail"
    assert manual["failure_reason"] == reason
    assert manual["review_audit"] == original_audit


def test_import_audits_cleared_failure_reason_when_no_fail_remains(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path, issue_ids=("warn-1",))
    report = load_asset_qc_report(report_path)
    assert report is not None
    reason = {
        "mode": "manual",
        "reason_codes": ["occlusion"],
        "other_text": None,
    }
    report["manual_review"]["failure_reason"] = copy.deepcopy(reason)
    report["report_revision"] = 2
    write_asset_qc_report(
        report_path,
        report,
        expected_revision=1,
        profile="acceptance",
    )

    import_legacy_manual_review(
        report_path,
        _csv(tmp_path, [_row("warn-1", "false_positive")]),
        None,
        expected_revision=2,
        reviewer="migration-bot",
    )

    persisted = load_asset_qc_report(report_path)
    assert persisted is not None
    manual = persisted["manual_review"]
    assert manual["failure_reason"] is None
    assert manual["review_audit"][-1] == {
        "action": "failure_reason_changed",
        "issue_id": None,
        "previous_failure_reason": reason,
        "failure_reason": None,
        "reviewed_at": manual["completed_at"],
    }


def test_unknown_duplicate_and_ambiguous_rows_are_reported_without_guessing(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path, issue_ids=("warn-1", "ambiguous", "ambiguous"))
    csv_path = _csv(
        tmp_path,
        [
            _row("warn-1", "false_positive"),
            _row("warn-1", "true_positive"),
            _row("missing", "true_positive"),
            _row("ambiguous", "true_positive"),
        ],
    )

    result = import_legacy_manual_review(
        report_path,
        csv_path,
        None,
        expected_revision=1,
        reviewer="migration-bot",
    )

    assert result.matched_count == 0
    assert result.unmatched_count == 1
    assert result.conflict_count == 3
    assert [problem.code for problem in result.problems].count("duplicate_csv_key") == 2
    assert any(problem.code == "ambiguous_report_issue_id" for problem in result.problems)
    assert load_asset_qc_report(report_path)["report_revision"] == 1


def test_repeat_import_is_content_idempotent_and_legacy_files_are_not_read_later(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path, issue_ids=("warn-1",))
    csv_path = _csv(tmp_path, [_row("warn-1", "false_positive")])
    progress_path = tmp_path / "progress.json"
    progress_path.write_text('{"segmentsByReviewId": {}}', encoding="utf-8")

    first = import_legacy_manual_review(
        report_path,
        csv_path,
        progress_path,
        expected_revision=1,
        reviewer="migration-bot",
    )
    second = import_legacy_manual_review(
        report_path,
        csv_path,
        progress_path,
        expected_revision=first.report_revision,
        reviewer="migration-bot",
    )
    persisted_before_source_change = load_asset_qc_report(report_path)

    csv_path.write_text("asset_id,issue_id\ncorrupt,other\n", encoding="utf-8")
    progress_path.write_text('{"segmentsByReviewId": "changed"}', encoding="utf-8")

    assert first.written is True
    assert second.written is False
    assert second.idempotent_count == 1
    assert second.report_revision == first.report_revision == 2
    assert load_asset_qc_report(report_path) == persisted_before_source_change


def test_changed_import_requires_current_expected_revision(tmp_path: Path) -> None:
    report_path = _report(tmp_path, issue_ids=("warn-1",))
    csv_path = _csv(tmp_path, [_row("warn-1", "false_positive")])

    import_legacy_manual_review(
        report_path,
        csv_path,
        None,
        expected_revision=1,
        reviewer="migration-bot",
        dry_run=True,
    )
    report = load_asset_qc_report(report_path)
    report["overall_decision"] = None
    report["report_revision"] = 2
    write_asset_qc_report(report_path, report, expected_revision=1, profile="acceptance")

    with pytest.raises(StaleReportRevisionError):
        import_legacy_manual_review(
            report_path,
            csv_path,
            None,
            expected_revision=1,
            reviewer="migration-bot",
        )


def test_generated_review_id_is_rejected_without_authoritative_issue_mapping(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path, issue_ids=("warn-1",))
    row = _row("", "false_positive")
    row["review_id"] = "rq_001"
    result = import_legacy_manual_review(
        report_path,
        _csv(tmp_path, [row]),
        None,
        expected_revision=1,
        reviewer="migration-bot",
        dry_run=True,
    )

    assert result.matched_count == 0
    assert result.matched_issue_ids == ()
    assert result.conflict_count == 1
    assert result.problems[0].code == "review_id_requires_mapping"


def test_authoritative_mapping_resolves_generated_review_id_exactly(tmp_path: Path) -> None:
    report_path = _report(tmp_path, issue_ids=("warn-1",))
    row = _row("", "false_positive")
    row["review_id"] = "rq_001"
    mapping_path = tmp_path / "review_issue_mapping.csv"
    mapping_path.write_text(
        "asset_id,review_id,issue_id\nasset-001,rq_001,warn-1\n",
        encoding="utf-8",
    )

    result = import_legacy_manual_review(
        report_path,
        _csv(tmp_path, [row]),
        None,
        expected_revision=1,
        reviewer="migration-bot",
        dry_run=True,
        issue_mapping_path=mapping_path,
    )

    assert result.matched_issue_ids == ("warn-1",)
    assert result.conflict_count == 0


def test_matched_issue_ids_exclude_existing_review_conflicts(tmp_path: Path) -> None:
    report_path = _report(tmp_path, issue_ids=("warn-1",))
    report = load_asset_qc_report(report_path)
    assert report is not None
    report["manual_review"]["issue_reviews"] = {
        "warn-1": {"verdict": "fail", "reviewer": "human", "reviewed_at": "earlier"}
    }
    report["report_revision"] = 2
    write_asset_qc_report(report_path, report, expected_revision=1, profile="acceptance")

    result = import_legacy_manual_review(
        report_path,
        _csv(tmp_path, [_row("warn-1", "false_positive")]),
        None,
        expected_revision=2,
        reviewer="migration-bot",
        dry_run=True,
    )

    assert result.matched_count == 0
    assert result.matched_issue_ids == ()
    assert result.conflict_count == 1
    assert result.problems[0].code == "existing_review_differs"


def test_import_cli_dry_run_prints_counts_and_does_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = _report(tmp_path, issue_ids=("warn-1",))
    csv_path = _csv(tmp_path, [_row("warn-1", "false_positive")])
    monkeypatch.setattr(
        "sys.argv",
        [
            "import_legacy_manual_review.py",
            "--quality-archive",
            str(report_path.parent),
            "--csv",
            str(csv_path),
            "--reviewer",
            "migration-bot",
            "--dry-run",
        ],
    )

    assert legacy_cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["totals"] == {"matched": 1, "unmatched": 0, "conflicts": 0}
    assert output["dry_run"] is True
    assert load_asset_qc_report(report_path)["report_revision"] == 1


def test_batch_dry_run_counts_each_unknown_asset_source_row_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "quality_archive"
    first = _report(tmp_path, issue_ids=("warn-1",))
    second_report = load_asset_qc_report(first)
    assert second_report is not None
    second_report["asset_id"] = "asset-002"
    second_path = archive / "asset-002.json"
    write_asset_qc_report(second_path, second_report, expected_revision=0, profile="acceptance")
    csv_path = _csv(
        tmp_path,
        [
            _row("warn-1", "false_positive", asset_id="asset-001"),
            _row("warn-1", "false_positive", asset_id="asset-002"),
            _row("warn-1", "false_positive", asset_id="missing-asset"),
            _row("warn-2", "true_positive", asset_id="missing-asset"),
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "import_legacy_manual_review.py",
            "--quality-archive",
            str(archive),
            "--csv",
            str(csv_path),
            "--reviewer",
            "migration-bot",
            "--dry-run",
        ],
    )

    assert legacy_cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["totals"] == {"matched": 2, "unmatched": 2, "conflicts": 0}


def test_batch_write_preflights_all_scalar_expected_revisions_before_mutating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "quality_archive"
    first = _report(tmp_path, issue_ids=("warn-1",))
    second_report = load_asset_qc_report(first)
    assert second_report is not None
    second_report["asset_id"] = "asset-002"
    second_path = archive / "asset-002.json"
    write_asset_qc_report(second_path, second_report, expected_revision=0, profile="acceptance")
    second_report = load_asset_qc_report(second_path)
    assert second_report is not None
    second_report["report_revision"] = 2
    write_asset_qc_report(second_path, second_report, expected_revision=1, profile="acceptance")
    csv_path = _csv(
        tmp_path,
        [
            _row("warn-1", "false_positive", asset_id="asset-001"),
            _row("warn-1", "false_positive", asset_id="asset-002"),
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "import_legacy_manual_review.py",
            "--quality-archive",
            str(archive),
            "--csv",
            str(csv_path),
            "--reviewer",
            "migration-bot",
            "--expected-revision",
            "1",
        ],
    )

    with pytest.raises(StaleReportRevisionError):
        legacy_cli.main()

    assert load_asset_qc_report(first)["report_revision"] == 1
    assert load_asset_qc_report(second_path)["report_revision"] == 2


def test_exact_cli_batch_repeat_is_idempotent_with_original_expected_revision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = _report(tmp_path, issue_ids=("warn-1",))
    csv_path = _csv(tmp_path, [_row("warn-1", "false_positive")])
    argv = [
        "--quality-archive",
        str(report_path.parent),
        "--csv",
        str(csv_path),
        "--reviewer",
        "migration-bot",
        "--expected-revision",
        "1",
    ]

    assert legacy_cli.main(argv) == 0
    first_output = json.loads(capsys.readouterr().out)
    assert load_asset_qc_report(report_path)["report_revision"] == 2

    assert legacy_cli.main(argv) == 0
    second_output = json.loads(capsys.readouterr().out)

    assert first_output["results"][0]["written"] is True
    assert second_output["results"][0]["matched"] == 1
    assert second_output["results"][0]["idempotent"] == 1
    assert second_output["results"][0]["written"] is False
    assert second_output["results"][0]["report_revision"] == 2
    assert load_asset_qc_report(report_path)["report_revision"] == 2


def test_audit_hash_uses_same_csv_byte_snapshot_as_imported_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import human_qc.legacy_import as legacy_import_module

    report_path = _report(tmp_path, issue_ids=("warn-1",))
    csv_path = _csv(tmp_path, [_row("warn-1", "false_positive")])
    progress_path = tmp_path / "progress.json"
    progress_path.write_text('{"segmentsByReviewId": {}}', encoding="utf-8")
    original_bytes = csv_path.read_bytes()
    original_progress_bytes = progress_path.read_bytes()
    original_reader = legacy_import_module._read_source_bytes

    def mutate_after_snapshot(path: Path) -> bytes:
        snapshot = original_reader(path)
        if Path(path) == csv_path:
            csv_path.write_text(
                "asset_id,issue_id,manual_outcome\nasset-001,warn-1,true_positive\n",
                encoding="utf-8",
            )
        if Path(path) == progress_path:
            progress_path.write_text('{"segmentsByReviewId": "changed"}', encoding="utf-8")
        return snapshot

    monkeypatch.setattr(legacy_import_module, "_read_source_bytes", mutate_after_snapshot)
    result = import_legacy_manual_review(
        report_path,
        csv_path,
        progress_path,
        expected_revision=1,
        reviewer="migration-bot",
    )

    assert result.written is True
    persisted = load_asset_qc_report(report_path)
    review = persisted["manual_review"]["issue_reviews"]["warn-1"]
    audit = persisted["manual_review"]["import_audit"][-1]
    assert review["verdict"] == "pass"
    assert audit["csv_sha256"] == "sha256:" + hashlib.sha256(original_bytes).hexdigest()
    assert audit["progress_sha256"] == (
        "sha256:" + hashlib.sha256(original_progress_bytes).hexdigest()
    )


def test_legacy_server_startup_logs_non_authoritative_deprecation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FakeServer:
        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            return None

    args = type(
        "Args",
        (),
        {
            "review_dir": tmp_path,
            "save_dir": tmp_path / "save",
            "host": "127.0.0.1",
            "port": 0,
            "log_level": "INFO",
        },
    )()
    monkeypatch.setattr(legacy_server, "parse_args", lambda: args)
    monkeypatch.setattr(legacy_server, "create_server", lambda **_kwargs: FakeServer())

    with caplog.at_level("WARNING"):
        assert legacy_server.main() == 0

    assert "not the authoritative source of truth" in caplog.text
    assert "import_legacy_manual_review.py" in caplog.text
