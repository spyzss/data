from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from qc_common import EvidenceRef, Issue, ModuleResult
from qc_common.contracts import build_issue_id, relative_evidence_path


BASE = {
    "asset_id": "asset-1",
    "module": "keypoint_temporal",
    "rule_id": "keypoint_temporal.strong_temporal_failure",
    "source_relative_path": "video/a.mp4",
    "coordinate_system": "source_inclusive",
    "start_frame": 10,
    "end_frame": 20,
    "hand_side": "left",
    "evidence_kind": "clip",
}


def make_issue(*, severity: Any = "warn") -> Issue:
    return Issue(
        issue_id="issue-1",
        code="review",
        severity=severity,
        module="semantic_review",
        issue_type="semantic",
        metric="external_payload",
        observed_value=None,
        operator="manual_review",
        boundary_value=None,
        rule_id="semantic_review.external_payload",
        needs_manual_review=True,
    )


def make_module_result(
    *, verdict: Any = "pass", metrics: Any = None
) -> ModuleResult:
    return ModuleResult(
        module="semantic_review",
        verdict=verdict,
        evaluation={},
        metrics={} if metrics is None else metrics,
    )


def test_issue_id_is_repeatable_and_reason_independent() -> None:
    first = build_issue_id(**BASE)
    second = build_issue_id(**dict(reversed(list(BASE.items()))))

    assert first == second
    assert first.startswith("keypoint_temporal:strong_temporal_failure:")
    assert len(first.rsplit(":", 1)[1]) == 20


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("asset_id", "asset-2"),
        ("module", "semantic_review"),
        ("rule_id", "keypoint_temporal.weak_temporal_failure"),
        ("source_relative_path", "video/b.mp4"),
        ("coordinate_system", "source_exclusive"),
        ("start_frame", 11),
        ("end_frame", 21),
        ("hand_side", "right"),
        ("evidence_kind", "image"),
    ],
)
def test_issue_id_changes_with_each_identity_field(
    field: str, changed_value: object
) -> None:
    assert build_issue_id(**BASE) != build_issue_id(
        **{**BASE, field: changed_value}
    )


def test_evidence_path_cannot_escape_batch(tmp_path: Path) -> None:
    inside = tmp_path / "evidence" / "a.png"
    inside.parent.mkdir()
    inside.write_bytes(b"x")

    assert relative_evidence_path(inside, tmp_path) == "evidence/a.png"
    with pytest.raises(ValueError, match="outside batch root"):
        relative_evidence_path(tmp_path.parent / "secret.png", tmp_path)


def test_evidence_path_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    target = outside / "secret.png"
    target.write_bytes(b"x")
    link = tmp_path / "linked-evidence"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="outside batch root"):
        relative_evidence_path(link / "secret.png", tmp_path)


def test_contracts_are_frozen_and_serialize_to_json_native_values() -> None:
    evidence = EvidenceRef(
        evidence_id="evidence-1",
        kind="clip",
        path="evidence/a.mp4",
        coordinate_system="source_inclusive",
        start_frame=10,
        end_frame=20,
        hand_side="left",
        checksum="sha256:abc",
        mime_type="video/mp4",
        generator_version="keypoint-qc/1",
    )
    issue = Issue(
        issue_id="issue-1",
        code="strong_temporal_failure",
        severity="fail",
        module="keypoint_temporal",
        issue_type="temporal",
        metric="jump_distance",
        observed_value=MappingProxyType({"samples": (1.5, 2.5)}),
        operator=">",
        boundary_value=1.0,
        rule_id="keypoint_temporal.strong_temporal_failure",
        needs_manual_review=True,
        context=MappingProxyType({"labels": ("left", "right")}),
        evidence_ids=("evidence-1", "evidence-2"),
    )
    result = ModuleResult(
        module="keypoint_temporal",
        verdict="fail",
        evaluation=MappingProxyType({"states": ("evaluated", "failed")}),
        metrics=MappingProxyType({"scores": (0.2, 1.5)}),
        issues=(issue,),
        evidence=(evidence,),
        runtime=MappingProxyType({"versions": ("1", "2")}),
    )

    payload = result.to_dict()

    assert payload == {
        "module": "keypoint_temporal",
        "verdict": "fail",
        "evaluation": {"states": ["evaluated", "failed"]},
        "metrics": {"scores": [0.2, 1.5]},
        "issues": [
            {
                "issue_id": "issue-1",
                "code": "strong_temporal_failure",
                "severity": "fail",
                "module": "keypoint_temporal",
                "issue_type": "temporal",
                "metric": "jump_distance",
                "observed_value": {"samples": [1.5, 2.5]},
                "operator": ">",
                "boundary_value": 1.0,
                "rule_id": "keypoint_temporal.strong_temporal_failure",
                "needs_manual_review": True,
                "context": {"labels": ["left", "right"]},
                "evidence_ids": ["evidence-1", "evidence-2"],
            }
        ],
        "evidence": [
            {
                "evidence_id": "evidence-1",
                "kind": "clip",
                "path": "evidence/a.mp4",
                "coordinate_system": "source_inclusive",
                "start_frame": 10,
                "end_frame": 20,
                "hand_side": "left",
                "checksum": "sha256:abc",
                "mime_type": "video/mp4",
                "generator_version": "keypoint-qc/1",
            }
        ],
        "runtime": {"versions": ["1", "2"]},
    }
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    with pytest.raises(FrozenInstanceError):
        evidence.path = "evidence/changed.mp4"


def test_optional_contract_fields_remain_explicit() -> None:
    evidence = EvidenceRef(
        evidence_id="evidence-1",
        kind="frame",
        path="evidence/a.png",
        coordinate_system="source_inclusive",
    )
    issue = Issue(
        issue_id="issue-1",
        code="review",
        severity="warn",
        module="semantic_review",
        issue_type="semantic",
        metric="external_payload",
        observed_value={"provider_field": {"future_shape": [1, 2]}},
        operator="manual_review",
        boundary_value=None,
        rule_id="semantic_review.external_payload",
        needs_manual_review=True,
    )

    assert evidence.to_dict() == {
        "evidence_id": "evidence-1",
        "kind": "frame",
        "path": "evidence/a.png",
        "coordinate_system": "source_inclusive",
        "start_frame": None,
        "end_frame": None,
        "hand_side": None,
        "checksum": None,
        "mime_type": None,
        "generator_version": None,
    }
    assert issue.to_dict()["observed_value"] == {
        "provider_field": {"future_shape": [1, 2]}
    }
    assert issue.to_dict()["context"] == {}
    assert issue.to_dict()["evidence_ids"] == []


@pytest.mark.parametrize("verdict", ["pass", "warn", "fail", "skipped"])
def test_module_result_accepts_every_valid_verdict(verdict: str) -> None:
    result = make_module_result(verdict=verdict)

    assert result.verdict == verdict


def test_module_result_rejects_invalid_verdict() -> None:
    with pytest.raises(ValueError, match="verdict must be one of"):
        make_module_result(verdict="unknown")


@pytest.mark.parametrize("severity", ["warn", "fail"])
def test_issue_accepts_every_valid_severity(severity: str) -> None:
    issue = make_issue(severity=severity)

    assert issue.severity == severity


def test_issue_rejects_invalid_severity() -> None:
    with pytest.raises(ValueError, match="severity must be one of"):
        make_issue(severity="pass")


@pytest.mark.parametrize(
    ("bad_value", "error_type", "message"),
    [
        (
            Path("evidence/a.png"),
            TypeError,
            "unsupported JSON value type: PosixPath",
        ),
        ({"unordered"}, TypeError, "unsupported JSON value type: set"),
        (b"binary", TypeError, "unsupported JSON value type: bytes"),
        ({1: "not-a-string-key"}, TypeError, "mapping keys must be strings"),
        (float("nan"), ValueError, "non-finite float"),
        (float("inf"), ValueError, "non-finite float"),
        (float("-inf"), ValueError, "non-finite float"),
    ],
)
def test_to_dict_rejects_nested_non_json_values(
    bad_value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    result = make_module_result(metrics={"outer": [{"bad": bad_value}]})

    with pytest.raises(error_type, match=message):
        result.to_dict()


def test_to_dict_accepts_all_json_native_values_recursively() -> None:
    result = make_module_result(
        metrics={
            "native": [None, False, 7, 1.25, "value"],
            "ordered_tuple": ("first", "second"),
        }
    )

    payload = result.to_dict()

    assert payload["metrics"] == {
        "native": [None, False, 7, 1.25, "value"],
        "ordered_tuple": ["first", "second"],
    }
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
