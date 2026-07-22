from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

from semantic_calibration.application import SemanticCalibrationApplication
from semantic_calibration.service import SemanticCalibrationService, SemanticEligibilityError
from qc_common.report import write_asset_qc_report
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
        self.last_boundary_request = request
        self.calls.append(("boundary", asset_id))
        return self.get_task(asset_id)

    def begin_text_edit(self, asset_id: str, request):
        self.last_text_request = request
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
    )
    for operation in operations:
        with pytest.raises(SemanticEligibilityError, match="semantic_not_ready"):
            operation()
    assert application.release_lease("asset", lease.token) == {"released": True}
    assert not any(name in {"boundary", "text", "confirm", "cancel", "complete"} for name, _ in domain.calls)


def test_reportless_preview_can_be_read_but_never_mutated(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    domain = DomainStub({"preview": missing})
    domain.get_task = lambda asset_id: {"asset_id": asset_id, "report_revision": 0}
    application = SemanticCalibrationApplication(domain)

    preview = application.get_task("preview")
    assert preview["asset_id"] == "preview"
    assert preview["editable"] is False
    with pytest.raises(SemanticEligibilityError, match="persisted report"):
        application.acquire_lease("preview", "alice", 60)


def test_queue_contains_only_live_persisted_tasks_while_deep_links_remain_read_only(
    tmp_path: Path,
) -> None:
    live = _report(tmp_path, "live", "completed", "all_reviewed")
    completed = _report(tmp_path, "completed", "completed", "all_reviewed")
    completed_value = json.loads(completed.read_text(encoding="utf-8"))
    completed_value["semantic_calibration"] = {"state": "completed"}
    completed_value["pipeline_state"].update({"status": "completed", "next_module": None})
    completed.write_text(json.dumps(completed_value), encoding="utf-8")
    preview = tmp_path / "preview.json"
    domain = DomainStub({"live": live, "completed": completed, "preview": preview})
    original_get_task = domain.get_task

    def get_task(asset_id: str):
        if asset_id == "preview":
            return {"asset_id": asset_id, "report_revision": 0}
        return original_get_task(asset_id)

    domain.get_task = get_task
    application = SemanticCalibrationApplication(domain)

    assert application.list_assets() == [
        {"asset_id": "live", "state": "not_started", "report_revision": 4, "editable": True}
    ]
    assert application.get_task("live")["editable"] is True
    assert application.get_task("completed")["editable"] is False
    assert application.get_task("preview")["editable"] is False


def test_video_capability_is_encoded_rechecks_report_and_stays_inside_explicit_root(
    tmp_path: Path,
) -> None:
    asset_id = "asset/a?b"
    report = _report(tmp_path, "video-report", "completed", "all_reviewed")
    domain = DomainStub({asset_id: report})
    video_root = tmp_path / "videos"
    video_root.mkdir()
    video = video_root / "source.mp4"
    video.write_bytes(b"video")

    with pytest.raises(ValueError, match="video root"):
        SemanticCalibrationApplication(domain, video_paths={asset_id: video})

    application = SemanticCalibrationApplication(
        domain,
        video_paths={asset_id: video},
        video_roots={asset_id: video_root},
    )
    assert application.get_task(asset_id)["video_url"] == "/api/semantic/assets/asset%2Fa%3Fb/video"
    assert application.semantic_video_path(asset_id) == video.resolve()

    value = json.loads(report.read_text(encoding="utf-8"))
    value["manual_review"].update({"state": "queued", "completion_mode": None})
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SemanticEligibilityError, match="semantic_not_ready"):
        application.semantic_video_path(asset_id)

    missing_report = tmp_path / "missing-video-report.json"
    reportless_domain = DomainStub({"preview": missing_report})
    reportless = SemanticCalibrationApplication(
        reportless_domain,
        video_paths={"preview": video},
        video_roots={"preview": video_root},
    )
    with pytest.raises(SemanticEligibilityError, match="persisted report"):
        reportless.semantic_video_path("preview")


def test_video_root_rejects_symlink_escape_at_construction_and_request_time(tmp_path: Path) -> None:
    report = _report(tmp_path, "asset", "completed", "all_reviewed")
    domain = DomainStub({"asset": report})
    root = tmp_path / "videos"
    root.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"secret")
    link = root / "source.mp4"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="video.*root"):
        SemanticCalibrationApplication(
            domain,
            video_paths={"asset": link},
            video_roots={"asset": root},
        )

    link.unlink()
    link.write_bytes(b"safe")
    application = SemanticCalibrationApplication(
        domain,
        video_paths={"asset": link},
        video_roots={"asset": root},
    )
    link.unlink()
    link.symlink_to(outside)
    with pytest.raises(KeyError):
        application.semantic_video_path("asset")


def test_public_task_dto_is_an_explicit_safe_allowlist_after_real_lease_acquire(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path, "asset", "completed", "all_reviewed")

    class LeakyDomain(DomainStub):
        def get_task(self, asset_id: str):
            return SimpleNamespace(
                asset_id=asset_id,
                report_revision=4,
                report_state="in_progress",
                pipeline_state="awaiting_external",
                semantic_consistency_state="not_started",
                timeline_edit_count=0,
                subtask_text_edit_count=0,
                lease_token=self.bound_tokens.get(asset_id),
                hdf5_path="/private/secret/source.hdf5",
                source_path="/private/secret/source.hdf5",
                source_dataset_path="/label/private",
                hdf5_sha256="secret-hash",
                pending_edit=None,
                timeline=SimpleNamespace(
                    frame_count=10,
                    fps=30.0,
                    asset_id=asset_id,
                    segments=(
                        SimpleNamespace(
                            internal_id="s1",
                            start_frame=0,
                            end_frame_exclusive=10,
                            text_cn="动作",
                            text_en="action",
                            canonical_record={"private_domain_field": "secret"},
                        ),
                    ),
                ),
            )

    domain = LeakyDomain({"asset": report})
    application = SemanticCalibrationApplication(domain)
    application.acquire_lease("asset", "alice", 60)
    task = application.get_task("asset")
    encoded = json.dumps(task, ensure_ascii=False)

    assert task["semantic"]["timeline"]["segments"] == [
        {
            "internal_id": "s1",
            "start_frame": 0,
            "end_frame_exclusive": 10,
            "text_cn": "动作",
            "text_en": "action",
        }
    ]
    for forbidden in (
        "lease_token",
        "hdf5_path",
        "source_path",
        "source_dataset_path",
        "canonical_record",
        "private_domain_field",
        "/private/",
        "secret-hash",
    ):
        assert forbidden not in encoded


def test_real_semantic_service_does_not_leak_paths_or_lease_after_acquire(
    tmp_path: Path,
) -> None:
    asset_id = "real-asset"
    hdf5_path = tmp_path / "private-source.hdf5"
    payload = {
        "id": asset_id,
        "scene": "demo",
        "task": "demo",
        "fps": 30.0,
        "frame_count": 10,
        "annotations": [
            {
                "start_frame": 0,
                "end_frame": 9,
                "subtask_cn": "动作",
                "subtask_en": "action",
                "start_time_sec": 0.0,
                "end_time_sec": 0.3,
                "verb": "act",
                "object": "object",
                "target": "target",
                "hand": "right",
                "phase": "demo",
                "evidence_frames": [0, 9],
                "confidence": 1.0,
                "status": "confirmed",
            }
        ],
    }
    with h5py.File(hdf5_path, "w") as handle:
        group = handle.create_group("label")
        dataset = group.create_dataset("subtask_label", shape=(), dtype="S65536")
        dataset[()] = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    report_path = tmp_path / "quality_archive" / f"{asset_id}.json"
    report = make_v2_report(status="awaiting_external")
    report["asset_id"] = asset_id
    report["pipeline_state"]["next_module"] = "semantic_consistency"
    report["manual_review"].update(
        {
            "state": "not_required",
            "required": False,
            "selected_issue_ids": [],
            "selected_issue_id": None,
            "issue_reviews": {},
            "completed_at": None,
        }
    )
    write_asset_qc_report(report_path, report, expected_revision=0, profile="acceptance")
    domain = SemanticCalibrationService(
        assets={asset_id: hdf5_path}, reports={asset_id: report_path}
    )
    application = SemanticCalibrationApplication(domain)

    application.acquire_lease(asset_id, "alice", 60)
    task = application.get_task(asset_id)
    encoded = json.dumps(task, ensure_ascii=False)

    assert task["semantic"]["timeline"]["segments"][0]["text_cn"] == "动作"
    assert "lease_token" not in encoded
    assert str(hdf5_path) not in encoded
    assert "hdf5_path" not in encoded
    assert "source_dataset_path" not in encoded


def test_release_is_allowed_after_terminal_transition_but_mutations_are_not(tmp_path: Path) -> None:
    report = _report(tmp_path, "asset", "completed", "all_reviewed")
    domain = DomainStub({"asset": report})
    application = SemanticCalibrationApplication(domain)
    lease = application.acquire_lease("asset", "alice", 60)

    value = json.loads(report.read_text(encoding="utf-8"))
    value["semantic_calibration"] = {"state": "completed"}
    value["pipeline_state"].update({"status": "completed", "next_module": None})
    report.write_text(json.dumps(value), encoding="utf-8")

    assert application.release_lease("asset", lease.token) == {"released": True}
    with pytest.raises(Exception):
        application.renew_lease("asset", lease.token, 60)


def test_boundary_and_text_audit_reviewer_is_always_the_lease_owner(tmp_path: Path) -> None:
    report = _report(tmp_path, "asset", "completed", "all_reviewed")
    domain = DomainStub({"asset": report})
    application = SemanticCalibrationApplication(domain)
    lease = application.acquire_lease("asset", "alice", 60)

    application.begin_boundary_edit(
        "asset",
        boundary_index=1,
        new_frame_exclusive=42,
        expected_revision=4,
        lease_token=lease.token,
        reviewer="mallory",
    )
    boundary_request = domain.last_boundary_request
    assert boundary_request.reviewer == "alice"

    application.begin_text_edit(
        "asset",
        segment_id="s1",
        text_cn="甲",
        text_en="a",
        expected_revision=4,
        lease_token=lease.token,
        reviewer="mallory",
    )
    assert domain.last_text_request.reviewer == "alice"


def test_single_get_resumes_after_domain_recovers_finalizing_state(tmp_path: Path) -> None:
    report = _report(tmp_path, "asset", "completed", "all_reviewed")
    value = json.loads(report.read_text(encoding="utf-8"))
    value["semantic_calibration"] = {"state": "finalizing"}
    report.write_text(json.dumps(value), encoding="utf-8")

    class RecoveringDomain(DomainStub):
        def get_task(self, asset_id: str):
            self.calls.append(("get", asset_id))
            current = json.loads(self._reports[asset_id].read_text(encoding="utf-8"))
            if current["semantic_calibration"]["state"] == "finalizing":
                current["semantic_calibration"].update(
                    {"state": "completed", "orchestrator_resume_required": True}
                )
                self._reports[asset_id].write_text(json.dumps(current), encoding="utf-8")
            return {
                "asset_id": asset_id,
                "report_revision": current["report_revision"],
                "report_state": current["semantic_calibration"]["state"],
                "timeline": {"frame_count": 0, "fps": 0, "segments": []},
            }

    class RecoveringApplication(SemanticCalibrationApplication):
        def __init__(self, domain):
            super().__init__(domain, config=object())
            self.resume_calls = 0

        def _resume(self, asset_id: str) -> None:
            self.resume_calls += 1
            current = json.loads(report.read_text(encoding="utf-8"))
            current["semantic_calibration"].pop("orchestrator_resume_required", None)
            current["pipeline_state"].update({"status": "completed", "next_module": None})
            report.write_text(json.dumps(current), encoding="utf-8")

    domain = RecoveringDomain({"asset": report})
    application = RecoveringApplication(domain)
    task = application.get_task("asset")

    assert task["semantic"]["report_state"] == "completed"
    assert application.resume_calls == 1
    assert domain.calls == [("get", "asset"), ("get", "asset")]
