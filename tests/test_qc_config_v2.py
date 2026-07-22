from pathlib import Path
import copy
import hashlib

import pytest
import yaml

from qc_common.config import load_qc_acceptance_config
from qc_common.schema import validate_qc_config


V1_SHA256 = "0ef58453f381651711ec84490975a1af5e274a179647f6ea2cc83e42404ea41d"
V20_SHA256 = "747dc605a066eb992d89346611b1421f4225dbeeb37431d85e532e39fd94c37b"


def _write_config(tmp_path: Path, raw: dict[str, object]) -> Path:
    path = tmp_path / "qc_acceptance.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_default_config_is_v2_snapshot_with_two_profiles() -> None:
    loaded = load_qc_acceptance_config()
    assert loaded.schema_version == "qc_acceptance_config_schema.v2"
    assert loaded.config_version == "qc_acceptance_v2.4.0"
    assert loaded.default_profile == "acceptance"
    assert loaded.execution_profile("acceptance")["fail_action"] == "stop"
    assert loaded.execution_profile("supplier_evaluation")["fail_action"] == "record_and_continue"
    assert loaded.execution_profile("acceptance")["runtime_error_action"] == "stop_incomplete"
    assert (
        loaded.execution_profile("supplier_evaluation")["runtime_error_action"]
        == "record_and_continue"
    )
    assert loaded.pipeline_modules == (
        "hdf5_text_info",
        "quality_hand",
        "keypoint_presence",
        "keypoint_morphology",
        "keypoint_temporal",
        "video_quality",
        "supplier_data_audit",
        "sam3_containment",
        "manual_review",
        "semantic_consistency",
        "duplicate_check",
        "content_validity",
        "effective_duration",
    )
    assert loaded.module_config("semantic_consistency")["execution_kind"] == "external"
    assert loaded.module_config("manual_review")["execution_kind"] == "external"
    assert loaded.module_config("supplier_data_audit")["implementation"] == (
        "supplier_data_audit.v1"
    )
    assert loaded.module_parameters("supplier_data_audit")["suppliers"]["dr"][
        "mapping_status"
    ] == "unverified"
    potentia_audit = loaded.module_parameters("supplier_data_audit")["suppliers"][
        "potentia"
    ]
    assert potentia_audit["max_scaled_intrinsics_relative_error"] == 0.001
    assert potentia_audit["scaling_mismatch_action"] == "review"
    assert loaded.module_parameters("video_quality")["supplier_overrides"] == {
        "potentia": {"hdf5_alignment": {"enabled": False}}
    }
    assert loaded.module_config("duplicate_check") == {
        **loaded.module_config("duplicate_check"),
        "enabled": False,
        "disabled_reason": "no_registered_implementation",
    }


def test_active_config_matches_immutable_v2_snapshot() -> None:
    assert Path("configs/qc_acceptance.yaml").read_bytes() == Path(
        "configs/qc_acceptance/qc_acceptance_v2.4.0.yaml"
    ).read_bytes()
    assert (
        hashlib.sha256(Path("configs/qc_acceptance/qc_acceptance_v1.1.0.yaml").read_bytes()).hexdigest()
        == V1_SHA256
    )
    assert (
        hashlib.sha256(
            Path("configs/qc_acceptance/qc_acceptance_v2.0.0.yaml").read_bytes()
        ).hexdigest()
        == V20_SHA256
    )


def test_v23_preserves_v22_acceptance_frame_survival_policy() -> None:
    active = load_qc_acceptance_config()
    previous = load_qc_acceptance_config(
        Path("configs/qc_acceptance/qc_acceptance_v2.2.0.yaml")
    )

    assert active.frame_survival_policy("acceptance") == previous.frame_survival_policy(
        "acceptance"
    )
    assert active.frame_survival_policy("supplier_evaluation") == {"enabled": False}


def test_temporal_no_valid_output_rule_is_versioned() -> None:
    rule = load_qc_acceptance_config().module_rules("keypoint_temporal")[
        "no_valid_output"
    ]

    assert rule == {
        "rule_id": "keypoint_temporal.no_valid_output",
        "verdict": "warn",
    }


def test_v2_schema_requires_supplier_data_audit_module() -> None:
    raw = copy.deepcopy(load_qc_acceptance_config().raw)
    raw["modules"].pop("supplier_data_audit")
    raw["pipeline"]["modules"].remove("supplier_data_audit")

    with pytest.raises(ValueError, match="supplier_data_audit"):
        validate_qc_config(raw)


def test_v2_schema_rejects_invalid_supplier_audit_mapping_status(
    tmp_path: Path,
) -> None:
    raw = copy.deepcopy(load_qc_acceptance_config().raw)
    raw["modules"]["supplier_data_audit"]["parameters"]["suppliers"]["dr"][
        "mapping_status"
    ] = "guessed"

    with pytest.raises(ValueError, match="mapping_status"):
        load_qc_acceptance_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scaling_mismatch_action", "pass"),
        ("max_scaled_intrinsics_relative_error", -0.1),
    ],
)
def test_v2_schema_rejects_invalid_potentia_calibration_policy(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    raw = copy.deepcopy(load_qc_acceptance_config().raw)
    raw["modules"]["supplier_data_audit"]["parameters"]["suppliers"][
        "potentia"
    ][field] = value

    with pytest.raises(ValueError, match=field):
        load_qc_acceptance_config(_write_config(tmp_path, raw))


def test_v2_schema_rejects_invalid_timestamp_mapping_unit(tmp_path: Path) -> None:
    raw = copy.deepcopy(load_qc_acceptance_config().raw)
    potentia = raw["modules"]["supplier_data_audit"]["parameters"]["suppliers"][
        "potentia"
    ]
    potentia["mapping_status"] = "verified"
    potentia["mapping"]["frames"] = {
        "frame_index_column": "frame_index",
        "timestamp_column": "timestamp",
        "timestamp_unit": "minutes",
    }

    with pytest.raises(ValueError, match="timestamp_unit"):
        load_qc_acceptance_config(_write_config(tmp_path, raw))


def test_v2_schema_rejects_ambiguous_dr_projection_chain(tmp_path: Path) -> None:
    raw = copy.deepcopy(load_qc_acceptance_config().raw)
    dr = raw["modules"]["supplier_data_audit"]["parameters"]["suppliers"]["dr"]
    dr["mapping_status"] = "verified"
    dr["mapping"]["projection"] = {
        "camera_name": "head",
        "joints3d_coordinate_frame": "head_camera",
        "joints3d_unit": "meter",
        "projection_direction": "direct_camera",
        "trajectory_usage": "apply",
        "resolution_policy": "exact",
    }

    with pytest.raises(ValueError, match="trajectory_usage"):
        load_qc_acceptance_config(_write_config(tmp_path, raw))


def test_v2_rejects_enabled_module_without_implementation(tmp_path: Path) -> None:
    raw = copy.deepcopy(load_qc_acceptance_config().raw)
    raw["modules"]["duplicate_check"].update({"enabled": True})
    raw["modules"]["duplicate_check"].pop("implementation", None)
    path = _write_config(tmp_path, raw)
    with pytest.raises(ValueError, match="enabled module duplicate_check"):
        load_qc_acceptance_config(path)


def test_v2_rejects_enabled_module_with_two_execution_kinds(tmp_path: Path) -> None:
    raw = copy.deepcopy(load_qc_acceptance_config().raw)
    raw["modules"]["video_quality"]["execution_kind"] = "external"

    with pytest.raises(ValueError, match="video_quality"):
        load_qc_acceptance_config(_write_config(tmp_path, raw))


def test_v2_rejects_disabled_module_without_reason(tmp_path: Path) -> None:
    raw = copy.deepcopy(load_qc_acceptance_config().raw)
    raw["modules"]["duplicate_check"].pop("disabled_reason")

    with pytest.raises(ValueError, match="duplicate_check"):
        load_qc_acceptance_config(_write_config(tmp_path, raw))


def test_v2_rejects_empty_and_duplicate_rule_ids(tmp_path: Path) -> None:
    raw = copy.deepcopy(load_qc_acceptance_config().raw)
    raw["modules"]["quality_hand"]["rules"]["invalid_quality_hand_shape"]["rule_id"] = ""
    with pytest.raises(ValueError, match="non-empty rule_id"):
        load_qc_acceptance_config(_write_config(tmp_path, raw))

    raw = copy.deepcopy(load_qc_acceptance_config().raw)
    duplicate = raw["modules"]["hdf5_text_info"]["rules"]["missing_required_field"]["rule_id"]
    raw["modules"]["quality_hand"]["rules"]["invalid_quality_hand_shape"]["rule_id"] = duplicate
    with pytest.raises(ValueError, match=f"duplicate rule_id: {duplicate}"):
        load_qc_acceptance_config(_write_config(tmp_path, raw))


def test_v2_rejects_unknown_pipeline_module(tmp_path: Path) -> None:
    raw = copy.deepcopy(load_qc_acceptance_config().raw)
    raw["pipeline"]["modules"].append("unknown_module")

    with pytest.raises(ValueError, match="pipeline module missing config: unknown_module"):
        load_qc_acceptance_config(_write_config(tmp_path, raw))


def test_v2_rejects_invalid_profile_action(tmp_path: Path) -> None:
    raw = copy.deepcopy(load_qc_acceptance_config().raw)
    raw["execution_profiles"]["acceptance"]["fail_action"] = "ignore"

    with pytest.raises(ValueError, match="fail_action"):
        load_qc_acceptance_config(_write_config(tmp_path, raw))


def test_loaded_config_accessors_are_defensive_and_report_unknown_names() -> None:
    loaded = load_qc_acceptance_config()

    profile = loaded.execution_profile("acceptance")
    profile["fail_action"] = "record_and_continue"
    assert loaded.execution_profile("acceptance")["fail_action"] == "stop"

    module = loaded.module_config("video_quality")
    module["enabled"] = False
    assert loaded.module_config("video_quality")["enabled"] is True

    with pytest.raises(ValueError, match="unknown execution profile: missing"):
        loaded.execution_profile("missing")
    with pytest.raises(ValueError, match="unknown pipeline module: missing"):
        loaded.module_config("missing")


def test_loaded_config_detects_reference_drift() -> None:
    loaded = load_qc_acceptance_config()
    reference = loaded.json_reference()
    loaded.assert_same_reference(reference)

    for key in ("schema_version", "config_version", "config_hash"):
        drifted = dict(reference)
        drifted[key] = "changed"
        with pytest.raises(ValueError, match=f"QC config drift at {key}"):
            loaded.assert_same_reference(drifted)


def test_schema_dispatch_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="unknown QC config schema_version: future"):
        validate_qc_config({"schema_version": "future"})
