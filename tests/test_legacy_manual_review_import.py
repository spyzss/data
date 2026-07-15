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
    assert manual["issue_reviews"]["warn-1"]["verdict"] == "pass"
    assert manual["issue_reviews"]["warn-2"]["verdict"] == "fail"
    assert manual["issue_reviews"]["warn-1"]["source"] == "legacy_manual_review_import"
    assert manual["import_audit"][-1]["action"] == "legacy_manual_review_import"
    assert manual["import_audit"][-1]["csv_sha256"].startswith("sha256:")
    assert manual["import_audit"][-1]["progress_sha256"].startswith("sha256:")


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


def test_review_id_is_a_supported_exact_legacy_issue_id_alias(tmp_path: Path) -> None:
    report_path = _report(tmp_path, issue_ids=("warn-1",))
    row = _row("", "false_positive")
    row["review_id"] = "warn-1"
    result = import_legacy_manual_review(
        report_path,
        _csv(tmp_path, [row]),
        None,
        expected_revision=1,
        reviewer="migration-bot",
        dry_run=True,
    )

    assert result.matched_issue_ids == ("warn-1",)


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
