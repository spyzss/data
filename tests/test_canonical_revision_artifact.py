from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from canonical_qc import StandardHdf5Adapter, Subtask
from canonical_qc.provenance import semantic_fingerprint
from canonical_qc.revision import apply_revision_artifact, load_revision_artifact
from lerobot_v3_publisher import PublishRequest, validate_publish_request
from tests.fixtures import write_standard_hdf5_episode
from tests.test_lerobot_v3_publish_prerequisites import (
    _rewrite_report,
    _write_publish_fixture,
)


def _artifact_payload(episode, *, edits: list[dict[str, object]]) -> dict[str, object]:
    text_count = sum(
        not str(row["path"]).endswith(("start_frame", "end_frame_exclusive"))
        for row in edits
    )
    timeline_rows = len(edits) - text_count
    timeline_count = timeline_rows // 2
    semantics = episode.semantics
    updated = semantics
    for edit in edits:
        path = str(edit["path"])
        if path == "semantics.task_en":
            updated = replace(updated, task_en=edit["after"])
    after_episode = replace(episode, semantics=updated)
    return {
        "schema_version": "canonical_revision_artifact.v1",
        "asset_id": episode.identity.asset_id,
        "parent_canonical_revision": 2,
        "canonical_revision": 3,
        "source_fingerprint": episode.provenance.source_fingerprint,
        "before_semantic_fingerprint": semantic_fingerprint(episode),
        "after_semantic_fingerprint": semantic_fingerprint(after_episode),
        "qc_report_revision": 9,
        "reviewer": "reviewer-001",
        "reviewed_at": "2026-07-17T10:00:00Z",
        "reason": "semantic consistency correction",
        "edit_counts": {
            "timeline": timeline_count,
            "subtask_text": text_count,
        },
        "edits": edits,
    }


def _write_artifact(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_revision_applies_task_text_without_mutating_base(tmp_path: Path) -> None:
    source_root = tmp_path / "asset-001"
    write_standard_hdf5_episode(source_root)
    episode = StandardHdf5Adapter().load(source_root)
    payload = _artifact_payload(
        episode,
        edits=[
            {
                "path": "semantics.task_en",
                "before": episode.semantics.task_en,
                "after": "Pick up and open the bottle cap",
            }
        ],
    )
    artifact = load_revision_artifact(_write_artifact(tmp_path / "revision.json", payload))

    revised = apply_revision_artifact(
        episode,
        artifact,
        expected_canonical_revision=3,
        expected_qc_report_revision=9,
        expected_timeline_edit_count=0,
        expected_subtask_text_edit_count=1,
    )

    assert episode.semantics.task_en != revised.semantics.task_en
    assert revised.semantics.task_en == "Pick up and open the bottle cap"
    assert semantic_fingerprint(revised) == payload["after_semantic_fingerprint"]


def test_revision_applies_shared_subtask_boundary_atomically(tmp_path: Path) -> None:
    source_root = tmp_path / "asset-001"
    write_standard_hdf5_episode(source_root)
    episode = StandardHdf5Adapter().load(source_root)
    original = episode.semantics.subtask_sequence[0]
    episode = replace(
        episode,
        semantics=replace(
            episode.semantics,
            subtask_sequence=(
                Subtask(
                    subtask_id="subtask-001",
                    start_frame=0,
                    end_frame_exclusive=1,
                    description_cn=original.description_cn,
                    description_en=original.description_en,
                ),
                Subtask(
                    subtask_id="subtask-002",
                    start_frame=1,
                    end_frame_exclusive=episode.time_axis.frame_count,
                    description_cn=original.description_cn,
                    description_en=original.description_en,
                ),
            ),
        ),
    )
    first, second = episode.semantics.subtask_sequence[:2]
    old_boundary = first.end_frame_exclusive
    new_boundary = old_boundary + 1
    revised_subtasks = (
        replace(first, end_frame_exclusive=new_boundary),
        replace(second, start_frame=new_boundary),
        *episode.semantics.subtask_sequence[2:],
    )
    revised_episode = replace(
        episode,
        semantics=replace(
            episode.semantics,
            subtask_sequence=revised_subtasks,
        ),
    )
    payload = _artifact_payload(
        episode,
        edits=[
            {
                "path": "semantics.subtask_sequence[0].end_frame_exclusive",
                "before": old_boundary,
                "after": new_boundary,
            },
            {
                "path": "semantics.subtask_sequence[1].start_frame",
                "before": old_boundary,
                "after": new_boundary,
            },
        ],
    )
    payload["after_semantic_fingerprint"] = semantic_fingerprint(revised_episode)
    artifact = load_revision_artifact(_write_artifact(tmp_path / "revision.json", payload))

    revised = apply_revision_artifact(
        episode,
        artifact,
        expected_canonical_revision=3,
        expected_qc_report_revision=9,
        expected_timeline_edit_count=1,
        expected_subtask_text_edit_count=0,
    )

    assert revised.semantics.subtask_sequence[0].end_frame_exclusive == new_boundary
    assert revised.semantics.subtask_sequence[1].start_frame == new_boundary


def test_revision_rejects_unpaired_boundary_patch(tmp_path: Path) -> None:
    source_root = tmp_path / "asset-001"
    write_standard_hdf5_episode(source_root)
    episode = StandardHdf5Adapter().load(source_root)
    boundary = episode.semantics.subtask_sequence[0].end_frame_exclusive
    payload = _artifact_payload(
        episode,
        edits=[
            {
                "path": "semantics.subtask_sequence[0].end_frame_exclusive",
                "before": boundary,
                "after": boundary - 1,
            }
        ],
    )
    payload["edit_counts"]["timeline"] = 1

    with pytest.raises(ValueError, match="atomically pair"):
        load_revision_artifact(_write_artifact(tmp_path / "revision.json", payload))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload["edits"][0].update(before="stale"), "before value"),
        (lambda payload: payload["edits"][0].update(path="observation.state"), "path"),
        (lambda payload: payload.update(canonical_revision=4), "canonical revision"),
        (
            lambda payload: payload.update(after_semantic_fingerprint="0" * 64),
            "after semantic fingerprint",
        ),
    ],
)
def test_revision_rejects_stale_or_out_of_scope_patch(
    tmp_path: Path, mutate, match: str
) -> None:
    source_root = tmp_path / "asset-001"
    write_standard_hdf5_episode(source_root)
    episode = StandardHdf5Adapter().load(source_root)
    payload = _artifact_payload(
        episode,
        edits=[
            {
                "path": "semantics.task_en",
                "before": episode.semantics.task_en,
                "after": "Corrected task",
            }
        ],
    )
    mutate(payload)

    with pytest.raises(ValueError, match=match):
        artifact = load_revision_artifact(
            _write_artifact(tmp_path / "revision.json", payload)
        )
        apply_revision_artifact(
            episode,
            artifact,
            expected_canonical_revision=3,
            expected_qc_report_revision=9,
            expected_timeline_edit_count=0,
            expected_subtask_text_edit_count=1,
        )


def test_publisher_applies_revision_and_binds_artifact_hash(tmp_path: Path) -> None:
    request, report = _write_publish_fixture(tmp_path)
    episode = request.episode
    corrected = "Corrected task for training"
    payload = _artifact_payload(
        episode,
        edits=[
            {
                "path": "semantics.task_en",
                "before": episode.semantics.task_en,
                "after": corrected,
            }
        ],
    )
    artifact_path = _write_artifact(tmp_path / "revision.json", payload)
    report["semantic_calibration"]["subtask_text_edit_count"] = 1
    report["canonical_binding"]["semantic_fingerprint"] = payload[
        "after_semantic_fingerprint"
    ]
    _rewrite_report(request, report)
    revised_request = PublishRequest(
        episode=request.episode,
        canonical_revision=request.canonical_revision,
        canonical_source_root=request.canonical_source_root,
        qc_report_path=request.qc_report_path,
        expected_report_revision=request.expected_report_revision,
        release_root=request.release_root,
        revision_artifact_path=artifact_path,
    )

    plan = validate_publish_request(revised_request)

    assert plan.episode.semantics.task_en == corrected
    assert plan.revision_artifact_sha256 == hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()


def test_revision_artifact_hash_participates_in_release_identity(tmp_path: Path) -> None:
    request, report = _write_publish_fixture(tmp_path)
    payload = _artifact_payload(
        request.episode,
        edits=[
            {
                "path": "semantics.task_en",
                "before": request.episode.semantics.task_en,
                "after": "Corrected task for training",
            }
        ],
    )
    first_path = _write_artifact(tmp_path / "first.json", payload)
    second_payload = dict(payload)
    second_payload["reviewer"] = "reviewer-002"
    second_path = _write_artifact(tmp_path / "second.json", second_payload)
    report["semantic_calibration"]["subtask_text_edit_count"] = 1
    report["canonical_binding"]["semantic_fingerprint"] = payload[
        "after_semantic_fingerprint"
    ]
    _rewrite_report(request, report)

    first = validate_publish_request(
        replace(request, revision_artifact_path=first_path)
    )
    second = validate_publish_request(
        replace(request, revision_artifact_path=second_path)
    )

    assert first.semantic_fingerprint == second.semantic_fingerprint
    assert first.release_id != second.release_id
