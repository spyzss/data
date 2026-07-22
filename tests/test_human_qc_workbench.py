from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from human_qc.warn_service import WarnStateError
from qc_common.reviewer_lease import LeaseStore
from qc_pipeline.context import AssetContext
from tests.qc_report_fixtures import make_manual_block, make_v2_report


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 22, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def _report(asset_id: str = "asset-1") -> dict:
    report = make_v2_report(status="awaiting_external")
    report.update(
        {
            "asset_id": asset_id,
            "report_revision": 3,
            "source_path": "/private/batch/secret.mp4",
            "task_type": "shared-workbench",
            "profile": "internal",
            "semantic": {"secret": True},
            "issues": [
                {
                    "issue_id": "warn-1",
                    "code": "blur",
                    "display_name": "画面模糊",
                    "severity": "warn",
                    "module": "video_quality",
                    "operator": "<",
                    "boundary_value": 0.5,
                    "observed_value": 0.2,
                    "default_reason": "清晰度低于阈值",
                    "source_path": "/private/batch/secret.mp4",
                    "command": "ffmpeg --secret",
                    "traceback": "private traceback",
                    "context": {
                        "start_frame": 120,
                        "end_frame": 168,
                        "debug_path": "/private/context",
                    },
                    "evidence_ids": ["internal-evidence-id"],
                },
                {
                    "issue_id": "warn-2",
                    "code": "hand_occlusion",
                    "severity": "warn",
                    "module": "sam3_containment",
                    "message": "/private/batch/renderer command=ffmpeg",
                    "start_frame": -10,
                    "end_frame": 20,
                },
                {
                    "issue_id": "not-selected",
                    "code": "internal-candidate",
                    "severity": "warn",
                    "context": {"start_frame": 1, "end_frame": 2},
                },
            ],
        }
    )
    report["pipeline_state"].update(
        {"status": "awaiting_external", "next_module": "manual_review"}
    )
    report["manual_review"] = make_manual_block(
        state="in_progress",
        candidate_issue_ids=["warn-1", "warn-2", "not-selected"],
        selected_issue_ids=["warn-2", "warn-1"],
        issue_reviews={
            "warn-1": {
                "verdict": "fail",
                "effective_verdict": "fail",
                "machine_verdict": "warn",
                "reason": "occlusion",
                "reviewer": "alice",
                "reviewed_at": "2026-07-22T00:00:00+00:00",
                "debug": "/private/review",
            }
        },
    )
    report["manual_review"]["failure_reason"] = {
        "mode": "manual",
        "reason_codes": ["occlusion"],
        "other_text": None,
        "internal": "/private/failure",
    }
    return report


def _context(tmp_path: Path, report: dict) -> AssetContext:
    asset_id = report["asset_id"]
    video = tmp_path / "videos" / f"{asset_id}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(bytes(range(250)) * 8)
    report_path = tmp_path / "quality_archive" / f"{asset_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["source_files"] = {
        "video": {
            "path": video.relative_to(tmp_path).as_posix(),
            "sha256": "sha256:" + "a" * 64,
        }
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return AssetContext(
        asset_id=asset_id,
        batch_root=tmp_path,
        report_path=report_path,
        source_files=report["source_files"],
    )


def _service(tmp_path: Path, *, clock: Clock | None = None):
    from human_qc.media import MediaCatalog
    from human_qc.warn_workbench_service import WarnWorkbenchService

    report = _report()
    context = _context(tmp_path, report)
    probe_calls: list[Path] = []

    def probe(path: Path):
        probe_calls.append(path)
        return {"fps": 30.0, "total_frames": 1800}

    lease_store = LeaseStore(clock=clock) if clock is not None else LeaseStore()
    media = MediaCatalog({context.asset_id: context}, probe=probe)
    service = WarnWorkbenchService(
        reviewer="alice",
        asset_contexts={context.asset_id: context},
        lease_store=lease_store,
        media_catalog=media,
        lease_ttl_seconds=60,
    )
    return service, lease_store, probe_calls, context


def test_warn_task_is_an_explicit_safe_projection_in_selected_order(tmp_path: Path) -> None:
    service, _, probe_calls, _ = _service(tmp_path)

    task = service.get_asset_task("asset-1")
    token = task["lease"]["token"]
    reloaded = service.get_asset_task("asset-1", lease_token=token)

    assert set(task) == {
        "asset_id",
        "report_revision",
        "manual_review_state",
        "completion_mode",
        "failure_reason",
        "review_audit",
        "can_complete",
        "video",
        "issues",
        "reason_options",
        "lease",
    }
    assert task["video"] == {
        "url": "/media/assets/asset-1/source",
        "fps": 30.0,
        "total_frames": 1800,
    }
    assert [issue["id"] for issue in task["issues"]] == ["warn-2", "warn-1"]
    assert task["issues"][0]["frame_range"] == {
        "start_frame": 0,
        "end_frame_exclusive": 20,
    }
    assert task["issues"][1]["frame_range"] == {
        "start_frame": 120,
        "end_frame_exclusive": 169,
    }
    assert set(task["issues"][1]) == {
        "id",
        "display_name",
        "frame_range",
        "default_reason",
        "threshold",
        "evidence_type",
        "review",
        "overlay",
    }
    assert task["issues"][1]["threshold"] == {"operator": "<", "value": 0.5}
    assert task["issues"][0]["overlay"] == {
        "status": "pending",
        "frame_range": {"start_frame": 0, "end_frame_exclusive": 20},
        "url": None,
        "code": None,
    }
    assert task["failure_reason"] == {
        "mode": "manual",
        "reason_codes": ["occlusion"],
        "other_text": None,
    }
    assert task["reason_options"] == [
        {"code": "occlusion", "display_name": "遮挡", "requires_text": False},
        {
            "code": "action_unrecognizable",
            "display_name": "动作不可辨",
            "requires_text": False,
        },
        {
            "code": "inaccurate_interval",
            "display_name": "标注区间不准确",
            "requires_text": False,
        },
        {"code": "other", "display_name": "其他", "requires_text": True},
    ]
    serialized = json.dumps(task, ensure_ascii=False)
    for forbidden in (
        "source_path",
        "/private/",
        "observed_value",
        "command",
        "traceback",
        "evidence_ids",
        "semantic",
        "task_type",
        "profile",
        "next_module",
        "not-selected",
    ):
        assert forbidden not in serialized
    assert reloaded["lease"]["token"] == token
    assert len(probe_calls) == 1


def test_warn_task_rejects_unknown_public_state_values(tmp_path: Path) -> None:
    from human_qc.media import MediaCatalog
    from human_qc.warn_workbench_service import WarnWorkbenchService

    report = _report()
    report["manual_review"]["state"] = "/private/run traceback command=ffmpeg"
    context = _context(tmp_path, report)
    service = WarnWorkbenchService(
        reviewer="alice",
        asset_contexts={"asset-1": context},
        media_catalog=MediaCatalog(
            {"asset-1": context},
            probe=lambda _path: {"fps": 30.0, "total_frames": 1800},
        ),
    )

    with pytest.raises(WarnStateError, match="invalid manual review state"):
        service.get_asset_task("asset-1")


def test_warn_task_replaces_untrusted_issue_text_with_stable_codes(
    tmp_path: Path,
) -> None:
    report = _report()
    report["issues"][0].update(
        {
            "display_name": "/private/display traceback",
            "default_reason": "command=ffmpeg /private/source.mp4",
            "operator": "command=cat /private/token",
            "boundary_value": "traceback /private/value",
        }
    )
    report["manual_review"]["issue_reviews"]["warn-1"].update(
        {
            "reason": "command=ffmpeg /private/reason",
            "reviewer": "/private/reviewer",
            "reviewed_at": "traceback /private/time",
        }
    )
    report["manual_review"]["failure_reason"] = {
        "mode": "command=ffmpeg",
        "reason_codes": ["/private/failure"],
        "other_text": "traceback /private/other",
    }
    service, _, _, context = _service(tmp_path)
    context.report_path.write_text(json.dumps(report), encoding="utf-8")

    task = service.get_asset_task("asset-1")

    issue = next(item for item in task["issues"] if item["id"] == "warn-1")
    assert issue["display_name"] == "blur"
    assert issue["default_reason"] == "blur"
    assert issue["threshold"] is None
    assert issue["review"] == {
        "verdict": "fail",
        "effective_verdict": "fail",
        "machine_verdict": "warn",
    }
    assert task["failure_reason"] is None
    serialized = json.dumps(task, ensure_ascii=False).lower()
    for forbidden in ("/private", "command=", "ffmpeg", "traceback"):
        assert forbidden not in serialized


def test_warn_task_projects_minimal_audit_and_server_completion_gate(
    tmp_path: Path,
) -> None:
    report = _report()
    report["manual_review"]["review_audit"] = [
        {
            "action": "resubmitted",
            "issue_id": "warn-1",
            "previous": {"reason": "/private/secret", "command": "ffmpeg"},
            "previous_failure_reason": {"traceback": "/private/trace"},
            "reviewed_at": "2026-07-22T00:01:00+00:00",
        },
        {
            "action": "/private/unknown",
            "issue_id": "warn-2",
            "reviewed_at": "traceback",
        },
    ]
    service, _, _, context = _service(tmp_path)
    context.report_path.write_text(json.dumps(report), encoding="utf-8")

    task = service.get_asset_task("asset-1")

    assert task["can_complete"] is True
    assert task["review_audit"] == [
        {
            "action": "resubmitted",
            "issue_id": "warn-1",
            "reviewed_at": "2026-07-22T00:01:00+00:00",
        }
    ]
    serialized = json.dumps(task, ensure_ascii=False).lower()
    for forbidden in ("/private", "command", "ffmpeg", "traceback", "previous"):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "raw_issue_id",
    [
        "/private/source.mp4",
        "command",
        "ffmpeg",
        "traceback",
    ],
)
def test_untrusted_selected_issue_id_is_opaque_in_task_and_audit(
    tmp_path: Path, raw_issue_id: str,
) -> None:
    from human_qc.media import MediaCatalog
    from human_qc.warn_workbench_service import WarnWorkbenchService

    report = _report()
    report["issues"] = [{**report["issues"][0], "issue_id": raw_issue_id}]
    report["manual_review"]["candidate_issue_ids"] = [raw_issue_id]
    report["manual_review"]["selected_issue_ids"] = [raw_issue_id]
    report["manual_review"]["issue_reviews"] = {
        raw_issue_id: report["manual_review"]["issue_reviews"]["warn-1"]
    }
    report["manual_review"]["review_audit"] = [
        {
            "action": "resubmitted",
            "issue_id": raw_issue_id,
            "reviewed_at": "2026-07-22T00:01:00+00:00",
        }
    ]
    context = _context(tmp_path, report)
    service = WarnWorkbenchService(
        reviewer="alice",
        asset_contexts={"asset-1": context},
        media_catalog=MediaCatalog(
            {"asset-1": context},
            probe=lambda _path: {"fps": 30.0, "total_frames": 1800},
        ),
    )

    task = service.get_asset_task("asset-1")
    public_issue_id = task["issues"][0]["id"]
    reloaded = service.get_asset_task(
        "asset-1", lease_token=task["lease"]["token"]
    )

    assert public_issue_id != raw_issue_id
    assert public_issue_id.startswith("issue-")
    assert public_issue_id.replace("-", "").isalnum()
    assert reloaded["issues"][0]["id"] == public_issue_id
    assert task["review_audit"] == [
        {
            "action": "resubmitted",
            "issue_id": public_issue_id,
            "reviewed_at": "2026-07-22T00:01:00+00:00",
        }
    ]
    serialized = json.dumps(task, ensure_ascii=False).lower()
    for forbidden in (raw_issue_id, "/private", "command", "ffmpeg", "traceback"):
        assert forbidden not in serialized


def test_opaque_issue_id_is_not_reused_as_display_fallback(tmp_path: Path) -> None:
    from human_qc.media import MediaCatalog
    from human_qc.warn_workbench_service import WarnWorkbenchService

    raw_issue_id = "command"
    report = _report()
    issue = {**report["issues"][0], "issue_id": raw_issue_id}
    for field in ("code", "display_name", "title", "default_reason"):
        issue.pop(field, None)
    report["issues"] = [issue]
    report["manual_review"]["candidate_issue_ids"] = [raw_issue_id]
    report["manual_review"]["selected_issue_ids"] = [raw_issue_id]
    report["manual_review"]["issue_reviews"] = {}
    context = _context(tmp_path, report)
    service = WarnWorkbenchService(
        reviewer="alice",
        asset_contexts={"asset-1": context},
        media_catalog=MediaCatalog(
            {"asset-1": context},
            probe=lambda _path: {"fps": 30.0, "total_frames": 1800},
        ),
    )

    task = service.get_asset_task("asset-1")

    assert task["issues"][0]["id"] != raw_issue_id
    assert task["issues"][0]["display_name"] == "issue"
    assert task["issues"][0]["default_reason"] == "issue"
    assert raw_issue_id not in json.dumps(task, ensure_ascii=False).lower()


def test_opaque_selected_issue_id_preserves_internal_write_semantics(
    tmp_path: Path,
) -> None:
    from human_qc.media import MediaCatalog
    from human_qc.warn_workbench_service import WarnWorkbenchService

    raw_issue_id = "/private/traceback-command=ffmpeg"
    report = _report()
    report["issues"] = [{**report["issues"][0], "issue_id": raw_issue_id}]
    report["manual_review"]["candidate_issue_ids"] = [raw_issue_id]
    report["manual_review"]["selected_issue_ids"] = [raw_issue_id]
    report["manual_review"]["issue_reviews"] = {}
    context = _context(tmp_path, report)
    warn = RecordingWarnService(context.report_path)
    service = WarnWorkbenchService(
        reviewer="alice",
        warn_service=warn,
        asset_contexts={"asset-1": context},
        media_catalog=MediaCatalog(
            {"asset-1": context},
            probe=lambda _path: {"fps": 30.0, "total_frames": 1800},
        ),
    )
    task = service.get_asset_task("asset-1")
    public_issue_id = task["issues"][0]["id"]

    service.warn_verdict(
        "asset-1",
        issue_id=public_issue_id,
        verdict="pass",
        expected_revision=3,
        lease_token=task["lease"]["token"],
    )

    assert public_issue_id != raw_issue_id
    assert warn.calls[-1][0] == "submit_verdict"
    assert warn.calls[-1][1][1] == raw_issue_id


def test_untrusted_selected_issue_id_error_does_not_echo_raw_value(
    tmp_path: Path,
) -> None:
    raw_issue_id = "/private/traceback-command=ffmpeg"
    report = _report()
    report["manual_review"]["candidate_issue_ids"] = [raw_issue_id]
    report["manual_review"]["selected_issue_ids"] = [raw_issue_id]
    service, _, _, context = _service(tmp_path)
    context.report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(WarnStateError) as caught:
        service.get_asset_task("asset-1")

    assert raw_issue_id not in str(caught.value)
    assert "/private" not in str(caught.value).lower()
    assert "traceback" not in str(caught.value).lower()
    assert "ffmpeg" not in str(caught.value).lower()


def test_task_load_lease_collision_is_safe_read_only(tmp_path: Path) -> None:
    service, lease_store, _, _ = _service(tmp_path)
    held = lease_store.acquire("asset-1", "bob", 60)

    task = service.get_asset_task("asset-1")

    assert task["lease"] == {
        "read_only": True,
        "token": None,
        "expires_at": None,
        "code": "lease_held",
    }
    assert held.token not in json.dumps(task)
    assert "bob" not in json.dumps(task)
    assert task["video"]["url"] == "/media/assets/asset-1/source"


def test_task_reload_renews_current_header_token(tmp_path: Path) -> None:
    clock = Clock()
    service, _, _, _ = _service(tmp_path, clock=clock)
    first = service.get_asset_task("asset-1")
    token = first["lease"]["token"]
    first_expiry = first["lease"]["expires_at"]
    clock.advance(20)

    renewed = service.get_asset_task("asset-1", lease_token=token)

    assert renewed["lease"]["token"] == token
    assert renewed["lease"]["expires_at"] > first_expiry


def test_invalid_clamped_issue_range_is_rejected(tmp_path: Path) -> None:
    from human_qc.media import MediaCatalog
    from human_qc.warn_workbench_service import InvalidIssueRangeError, WarnWorkbenchService

    report = _report()
    report["manual_review"]["selected_issue_ids"] = ["warn-1"]
    report["issues"][0]["context"] = {"start_frame": 2000, "end_frame": 2100}
    context = _context(tmp_path, report)
    service = WarnWorkbenchService(
        reviewer="alice",
        asset_contexts={"asset-1": context},
        media_catalog=MediaCatalog(
            {"asset-1": context},
            probe=lambda _path: {"fps": 30.0, "total_frames": 1800},
        ),
    )

    with pytest.raises(InvalidIssueRangeError, match="invalid_issue_range"):
        service.get_asset_task("asset-1")


class RecordingWarnService:
    def __init__(self, report_path: Path) -> None:
        self._report_path = report_path
        self.calls: list[tuple[str, tuple, dict]] = []

    def report_path(self, asset_id: str) -> Path:
        if asset_id != "asset-1":
            raise KeyError(asset_id)
        return self._report_path

    def submit_verdict(self, *args, **kwargs):
        self.calls.append(("submit_verdict", args, kwargs))

    def complete(self, *args, **kwargs):
        self.calls.append(("complete", args, kwargs))


def test_mutations_forward_atomic_warn_payload_and_session_reviewer(tmp_path: Path) -> None:
    from human_qc.media import MediaCatalog
    from human_qc.warn_workbench_service import WarnWorkbenchService

    report = _report()
    context = _context(tmp_path, report)
    warn = RecordingWarnService(context.report_path)
    service = WarnWorkbenchService(
        reviewer="alice",
        warn_service=warn,
        asset_contexts={"asset-1": context},
        media_catalog=MediaCatalog(
            {"asset-1": context},
            probe=lambda _path: {"fps": 30.0, "total_frames": 1800},
        ),
    )
    task = service.get_asset_task("asset-1")
    token = task["lease"]["token"]
    failure_reason = {"reason_codes": ["other"], "other_text": "遮挡严重"}

    service.warn_verdict(
        "asset-1",
        issue_id="warn-1",
        verdict="fail",
        reason="occlusion",
        failure_reason=failure_reason,
        expected_revision=3,
        lease_token=token,
    )
    service.warn_complete(
        "asset-1",
        completion_mode="early_fail",
        failure_reason=failure_reason,
        expected_revision=3,
        lease_token=token,
    )

    verdict = warn.calls[0]
    assert verdict[0] == "submit_verdict"
    assert verdict[1] == ("asset-1", "warn-1", "fail", "occlusion", 3, token)
    assert verdict[2] == {"reviewer": "alice", "failure_reason": failure_reason}
    complete = warn.calls[1]
    assert complete[0] == "complete"
    assert complete[1] == ("asset-1", 3, token)
    assert complete[2] == {
        "completion_mode": "early_fail",
        "failure_reason": failure_reason,
    }


def test_selected_issue_missing_from_report_fails_closed(tmp_path: Path) -> None:
    from human_qc.media import MediaCatalog
    from human_qc.warn_workbench_service import WarnWorkbenchService

    report = _report()
    report["manual_review"]["selected_issue_ids"] = ["missing"]
    context = _context(tmp_path, report)
    service = WarnWorkbenchService(
        reviewer="alice",
        asset_contexts={"asset-1": context},
        media_catalog=MediaCatalog(
            {"asset-1": context},
            probe=lambda _path: {"fps": 30.0, "total_frames": 1800},
        ),
    )

    with pytest.raises(WarnStateError, match="missing from report"):
        service.get_asset_task("asset-1")


def test_public_package_exports_only_the_warn_facade() -> None:
    import human_qc

    assert human_qc.WarnWorkbenchService is not None
    assert "WarnWorkbenchService" in human_qc.__all__
    assert "WorkbenchService" not in human_qc.__all__
    assert "jsonable" not in human_qc.__all__
