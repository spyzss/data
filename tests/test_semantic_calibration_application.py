from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_calibration.application import SemanticCalibrationApplication
from semantic_calibration.service import SemanticEligibilityError
from tests.qc_report_fixtures import make_v2_report


class DomainStub:
    def __init__(self, reports: dict[str, Path]) -> None:
        self._reports = reports
        self.calls: list[tuple[str, str]] = []
        self.bound_tokens: dict[str, str] = {}

    def asset_ids(self):
        return tuple(self._reports)

    def report_path(self, asset_id: str) -> Path:
        return self._reports[asset_id]

    def bind_lease(self, asset_id: str, token: str) -> None:
        self.bound_tokens[asset_id] = token

    def get_task(self, asset_id: str):
        self.calls.append(("get", asset_id))
        report = json.loads(self._reports[asset_id].read_text(encoding="utf-8"))
        return {
            "asset_id": asset_id,
            "report_revision": report["report_revision"],
            "report_state": report.get("semantic_calibration", {}).get("state", "not_started"),
        }

    def begin_boundary_edit(self, asset_id: str, request):
        self.calls.append(("boundary", asset_id))
        return self.get_task(asset_id)

    def begin_text_edit(self, asset_id: str, request):
        self.calls.append(("text", asset_id))
        return self.get_task(asset_id)

    def confirm_pending(self, asset_id: str, expected_revision: int, lease_token: str):
        self.calls.append(("confirm", asset_id))
        return self.get_task(asset_id)

    def cancel_pending(self, asset_id: str, expected_revision: int, lease_token: str):
        self.calls.append(("cancel", asset_id))
        return self.get_task(asset_id)

    def complete(self, asset_id: str, expected_revision: int, lease_token: str, *, advance_pipeline: bool = True):
        self.calls.append(("complete", asset_id))
        return self.get_task(asset_id)


def _report(tmp_path: Path, asset_id: str, manual_state: str, completion_mode: str | None = None) -> Path:
    value = make_v2_report(status="awaiting_external")
    value["asset_id"] = asset_id
    value["report_revision"] = 4
    value["pipeline_state"]["next_module"] = "semantic_consistency"
    value["manual_review"].update(
        {
            "state": manual_state,
            "completion_mode": completion_mode,
            "required": manual_state != "not_required",
            "selected_issue_ids": [] if manual_state == "not_required" else ["warn-1"],
            "issue_reviews": {},
            "completed_at": None,
        }
    )
    if completion_mode == "early_fail":
        value["semantic_calibration"] = {"state": "skipped_due_to_fail"}
    path = tmp_path / f"{asset_id}.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_list_and_direct_task_read_recompute_latest_persisted_eligibility(tmp_path: Path) -> None:
    ready = _report(tmp_path, "ready", "completed", "all_reviewed")
    not_required = _report(tmp_path, "not-required", "not_required")
    blocked = _report(tmp_path, "blocked", "queued")
    failed = _report(tmp_path, "failed", "completed", "early_fail")
    domain = DomainStub(
        {"ready": ready, "not-required": not_required, "blocked": blocked, "failed": failed}
    )
    application = SemanticCalibrationApplication(domain)

    assert [row["asset_id"] for row in application.list_assets()] == ["not-required", "ready"]
    assert application.get_task("ready")["asset_id"] == "ready"
    for asset_id in ("blocked", "failed"):
        with pytest.raises(SemanticEligibilityError, match="semantic_not_ready|skipped"):
            application.get_task(asset_id)

    value = json.loads(ready.read_text(encoding="utf-8"))
    value["manual_review"]["state"] = "in_progress"
    value["manual_review"]["completion_mode"] = None
    ready.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SemanticEligibilityError, match="semantic_not_ready"):
        application.get_task("ready")


def test_every_write_rechecks_eligibility_after_task_was_opened(tmp_path: Path) -> None:
    report = _report(tmp_path, "asset", "completed", "all_reviewed")
    domain = DomainStub({"asset": report})
    application = SemanticCalibrationApplication(domain)
    application.get_task("asset")
    lease = application.acquire_lease("asset", "alice", 60)

    value = json.loads(report.read_text(encoding="utf-8"))
    value["manual_review"]["state"] = "queued"
    value["manual_review"]["completion_mode"] = None
    report.write_text(json.dumps(value), encoding="utf-8")

    operations = (
        lambda: application.begin_boundary_edit(
            "asset",
            boundary_index=1,
            new_frame_exclusive=42,
            expected_revision=4,
            lease_token=lease.token,
        ),
        lambda: application.begin_text_edit(
            "asset",
            segment_id="s1",
            text_cn="甲",
            text_en="a",
            expected_revision=4,
            lease_token=lease.token,
        ),
        lambda: application.confirm_pending("asset", expected_revision=4, lease_token=lease.token),
        lambda: application.cancel_pending("asset", expected_revision=4, lease_token=lease.token),
        lambda: application.complete("asset", expected_revision=4, lease_token=lease.token),
        lambda: application.renew_lease("asset", lease.token, 60),
        lambda: application.release_lease("asset", lease.token),
    )
    for operation in operations:
        with pytest.raises(SemanticEligibilityError, match="semantic_not_ready"):
            operation()
    assert not any(name in {"boundary", "text", "confirm", "cancel", "complete"} for name, _ in domain.calls)


def test_reportless_preview_can_be_read_but_never_mutated(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    domain = DomainStub({"preview": missing})
    domain.get_task = lambda asset_id: {"asset_id": asset_id, "report_revision": 0}
    application = SemanticCalibrationApplication(domain)

    assert application.get_task("preview")["asset_id"] == "preview"
    with pytest.raises(SemanticEligibilityError, match="persisted report"):
        application.acquire_lease("preview", "alice", 60)
