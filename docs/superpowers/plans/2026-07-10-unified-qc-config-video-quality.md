# Unified QC Config And Video Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the unified QC config the only video-QC runtime configuration, remove hand ROI completely, and emit schema-validated asset JSON with separate pipeline, final-decision, and manual-review states.

**Architecture:** Add a small shared loader that validates `configs/qc_acceptance.yaml`, hashes the exact loaded bytes, and exposes typed module parameters. Video QC consumes `modules.video_quality.parameters` and carries the resulting config reference into report generation. Asset reports use canonical issues plus module flow fields, while JSON Schema enforces state combinations and field types.

**Tech Stack:** Python 3.12, dataclasses, PyYAML, jsonschema Draft 2020-12, pytest, OpenCV, h5py.

## Global Constraints

- `configs/qc_acceptance.yaml` is the only supported runtime config entry point.
- Target config version is `qc_acceptance_v1.1.0`.
- `pending` is only a pipeline state; it is never an `overall_decision`.
- `overall_decision` is `null` while running, `fail` when stopped, and `pass|warn` when completed.
- Manual-review routing is represented by `manual_review.required` and `manual_review.state`.
- New JSON does not contain module `thresholds`, per-issue `config_version`, or hand ROI fields.
- Video rule IDs remain stable, but all video parameters and thresholds live under `modules.video_quality.parameters`.
- HDF5 keypoints may still support freeze motion-conflict checks; they must not produce a hand ROI.
- Production config and report writes fail on schema violations instead of downgrading them to quality warnings.

---

## File Structure

**Create:**

- `qc_common/__init__.py`: public exports for shared QC config and schema helpers.
- `qc_common/config.py`: unified config loading, validation, module extraction, and provenance hashing.
- `qc_common/report.py`: revision-aware, atomic one-asset JSON updates.
- `qc_common/schema.py`: schema path resolution and JSON/config validation.
- `schemas/qc_acceptance_config.v1.schema.json`: machine contract for unified YAML.
- `schemas/asset_qc_report.v1.schema.json`: machine contract for one-asset-one-JSON reports.
- `configs/qc_acceptance/qc_acceptance_v1.1.0.yaml`: immutable archive of the released unified config.
- `tests/test_qc_config.py`: shared loader and config-schema tests.
- `tests/test_asset_qc_schema.py`: report state and issue-schema tests.
- `tests/test_qc_report_store.py`: atomic update and stale-revision tests.
- `tests/qc_report_fixtures.py`: reusable schema-valid report factory for report and writer tests.

**Modify:**

- `requirements.txt`: add `jsonschema>=4.23.0`.
- `configs/qc_acceptance.yaml`: bump config version, add complete video parameters, enrich video rule metadata, and delete hand ROI rules.
- `acceptance_pull/video_quality.py`: consume unified config, remove hand ROI, build canonical issues and module flow, and validate reports.
- `tests/test_acceptance_video_quality.py`: replace legacy video-YAML tests, delete ROI tests, and assert the new report contract.
- `ACCEPTANCE.md`: document the single config entry point and new state model.
- `docs/asset-qc-json-format.md`: replace old summary/ROI examples with the schema-valid report shape.
- `docs/PRD-qc-gated-json.md`: align state, issue, and config-version rules.
- `docs/PRD-qc-unified-config.md`: mark the loader and parameter layout as implemented.
- `docs/batch-sampling-pull.md`: remove video-only YAML and hand ROI examples.

---

### Task 1: Shared Unified Config Loader

**Files:**

- Create: `qc_common/__init__.py`
- Create: `qc_common/config.py`
- Create: `qc_common/schema.py`
- Create: `schemas/qc_acceptance_config.v1.schema.json`
- Create: `tests/test_qc_config.py`
- Modify: `requirements.txt`

**Interfaces:**

- Produces: `LoadedQcConfig`, `load_qc_acceptance_config(path: Path | None) -> LoadedQcConfig`, `validate_qc_config(data: dict[str, Any]) -> None`.
- Produces: `LoadedQcConfig.module_parameters(module_name: str) -> dict[str, Any]` and `LoadedQcConfig.json_reference() -> dict[str, str]`.
- Consumes: repository root `configs/qc_acceptance.yaml` and `schemas/qc_acceptance_config.v1.schema.json`.

- [ ] **Step 1: Add the schema-validation dependency**

Add this exact line under config management in `requirements.txt`:

```text
jsonschema>=4.23.0
```

Install it into the existing environment:

```bash
uv pip install --python .venv/bin/python "jsonschema>=4.23.0"
```

Expected: installation exits with code 0 and `.venv/bin/python -c "import jsonschema"` succeeds.

- [ ] **Step 2: Write failing loader tests**

Create `tests/test_qc_config.py` with tests covering the public contract:

```python
from pathlib import Path

import pytest
import yaml

from qc_common.config import load_qc_acceptance_config


def test_default_qc_config_loads_and_hashes() -> None:
    loaded = load_qc_acceptance_config()

    assert loaded.config_version == "qc_acceptance_v1.0.0"
    assert loaded.module_rules("video_quality")["fps_below_min"]["verdict"] == "fail"
    assert loaded.json_reference()["config_hash"].startswith("sha256:")


def test_video_only_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "video-only.yaml"
    path.write_text("decode:\n  max_sample_frames: 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unified qc_acceptance config"):
        load_qc_acceptance_config(path)

```

- [ ] **Step 3: Run the loader tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_qc_config.py -q
```

Expected: collection fails because `qc_common.config` does not exist.

- [ ] **Step 4: Add the config schema**

Create a Draft 2020-12 schema whose required contract includes:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "qc_acceptance_config.v1.schema.json",
  "type": "object",
  "required": ["schema_version", "config_version", "config_name", "pipeline", "modules"],
  "properties": {
    "schema_version": {"const": "qc_acceptance_config_schema.v1"},
    "config_version": {"type": "string", "pattern": "^qc_acceptance_v[0-9]+\\.[0-9]+\\.[0-9]+$"},
    "config_name": {"const": "acceptance_gate"},
    "pipeline": {
      "type": "object",
      "required": ["stop_on_fail", "modules"],
      "properties": {
        "stop_on_fail": {"const": true},
        "modules": {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": true}
      }
    },
    "modules": {
      "type": "object",
      "required": ["video_quality", "manual_review"],
      "properties": {
        "video_quality": {
          "type": "object",
          "required": ["enabled", "rules"],
          "properties": {
            "enabled": {"type": "boolean"},
            "rules": {"type": "object", "minProperties": 1}
          }
        },
        "manual_review": {"type": "object", "required": ["enabled", "selection_policy"]}
      }
    }
  }
}
```

At this task, keep unrelated colleague module objects and the existing video module open. Task 2 tightens `video_quality.parameters` after the canonical YAML contains those fields. The custom loader still requires every module named in `pipeline.modules` to exist.

- [ ] **Step 5: Implement the shared loader**

Implement these exact public objects in `qc_common/config.py`:

```python
@dataclass(frozen=True)
class LoadedQcConfig:
    path: Path
    raw: dict[str, Any]
    sha256: str

    @property
    def schema_version(self) -> str:
        return str(self.raw["schema_version"])

    @property
    def config_version(self) -> str:
        return str(self.raw["config_version"])

    @property
    def config_name(self) -> str:
        return str(self.raw["config_name"])

    def module_parameters(self, module_name: str) -> dict[str, Any]:
        module = self.raw["modules"][module_name]
        parameters = module.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"module {module_name} has no parameters mapping")
        return copy.deepcopy(parameters)

    def module_rules(self, module_name: str) -> dict[str, Any]:
        rules = self.raw["modules"][module_name].get("rules", {})
        if not isinstance(rules, dict):
            raise ValueError(f"module {module_name} rules must be a mapping")
        return copy.deepcopy(rules)

    def json_reference(self) -> dict[str, str]:
        repo_root = Path(__file__).resolve().parents[1]
        try:
            config_path = str(self.path.relative_to(repo_root))
        except ValueError:
            config_path = str(self.path)
        return {
            "schema_version": self.schema_version,
            "config_version": self.config_version,
            "config_name": self.config_name,
            "config_path": config_path,
            "config_hash": self.sha256,
        }


def load_qc_acceptance_config(path: Path | None = None) -> LoadedQcConfig:
    repo_root = Path(__file__).resolve().parents[1]
    resolved = (path or repo_root / "configs" / "qc_acceptance.yaml").resolve()
    payload = resolved.read_bytes()
    raw = yaml.safe_load(payload) or {}
    if not isinstance(raw, dict) or "schema_version" not in raw:
        raise ValueError("expected unified qc_acceptance config")
    validate_qc_config(raw)

    modules = raw["modules"]
    missing_modules = [name for name in raw["pipeline"]["modules"] if name not in modules]
    if missing_modules:
        raise ValueError(f"pipeline module missing config: {missing_modules[0]}")

    seen_rule_ids: set[str] = set()
    for module_name, module in modules.items():
        for rule in module.get("rules", {}).values():
            rule_id = rule.get("rule_id")
            if rule_id in seen_rule_ids:
                raise ValueError(f"duplicate rule_id: {rule_id}")
            if rule_id is not None:
                seen_rule_ids.add(rule_id)

    return LoadedQcConfig(
        path=resolved,
        raw=raw,
        sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )
```

Implementation rules:

- Default path is `<repo>/configs/qc_acceptance.yaml`.
- Hash `path.read_bytes()` before parsing and store `sha256:<hex>`.
- Reject a root without `schema_version` using `ValueError("expected unified qc_acceptance config")`.
- Convert `jsonschema.ValidationError` into `ValueError` containing the dotted failing path.
- Verify every `pipeline.modules` name exists under `modules`.
- Walk all module `rules` and reject duplicate `rule_id` values.
- `json_reference()` uses the loaded path and values, not module constants.

Implement `validate_qc_config()` in `qc_common/schema.py` using this shared validator body:

```python
def _validate_with_schema(instance: dict[str, Any], schema_path: Path, label: str) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda item: list(item.path))
    if not errors:
        return
    error = errors[0]
    dotted_path = ".".join(str(part) for part in error.absolute_path) or "$"
    raise ValueError(f"{label} validation failed at {dotted_path}: {error.message}")


def validate_qc_config(data: dict[str, Any]) -> None:
    root = Path(__file__).resolve().parents[1]
    _validate_with_schema(data, root / "schemas" / "qc_acceptance_config.v1.schema.json", "QC config")
```

- [ ] **Step 6: Run the loader tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_qc_config.py -q
```

Expected: all tests in `tests/test_qc_config.py` pass.

- [ ] **Step 7: Commit the loader**

```bash
git add requirements.txt qc_common schemas/qc_acceptance_config.v1.schema.json tests/test_qc_config.py
git commit -m "feat: add unified QC config loader"
```

---

### Task 2: Move Video Parameters Into Unified Config

**Files:**

- Modify: `configs/qc_acceptance.yaml`
- Modify: `acceptance_pull/video_quality.py`
- Modify: `tests/test_acceptance_video_quality.py`

**Interfaces:**

- Consumes: `load_qc_acceptance_config()` and `LoadedQcConfig.module_parameters("video_quality")` from Task 1.
- Produces: `load_video_quality_config(path: Path | None) -> VideoQualityConfig` backed only by unified config.
- Produces: `VideoQualityConfig.qc_config_reference: dict[str, str]` and `VideoQualityConfig.module_version: str`.

- [ ] **Step 1: Replace legacy-config tests with unified-config tests**

Update the beginning of `tests/test_acceptance_video_quality.py`:

```python
def test_default_video_quality_config_comes_from_unified_config() -> None:
    config = load_video_quality_config(None)

    assert config.module_version == "video_prefilter_v0.3.2"
    assert config.qc_config_reference["config_version"] == "qc_acceptance_v1.1.0"
    assert config.fps.min_fps_pass == 24
    assert not hasattr(config, "threshold_version")
    assert not hasattr(config, "hand_roi")


def test_unified_video_threshold_override_changes_runtime_config(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/qc_acceptance.yaml").read_text(encoding="utf-8"))
    raw["config_version"] = "qc_acceptance_v1.1.1"
    raw["modules"]["video_quality"]["parameters"]["fps"]["min_fps_pass"] = 26
    path = tmp_path / "qc_acceptance.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_video_quality_config(path)

    assert config.fps.min_fps_pass == 26
    assert config.qc_config_reference["config_version"] == "qc_acceptance_v1.1.1"


def test_unified_config_missing_video_parameters_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/qc_acceptance.yaml").read_text(encoding="utf-8"))
    del raw["modules"]["video_quality"]["parameters"]["freeze"]
    path = tmp_path / "missing-freeze.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="freeze"):
        load_video_quality_config(path)
```

Delete the test that accepts a video-only YAML. Its rejection is covered by Task 1.

Add this helper and replace every test-local video-only YAML writer with it:

```python
def write_unified_video_config(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    raw = yaml.safe_load(Path("configs/qc_acceptance.yaml").read_text(encoding="utf-8"))
    raw["config_version"] = "qc_acceptance_v1.1.1"
    mutate(raw["modules"]["video_quality"]["parameters"])
    path = tmp_path / "qc_acceptance.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
```

Import `Any` and `Callable` from `typing`. For example, replace `config_path.write_text("freeze:\n  enabled: false\n")` with:

```python
config_path = write_unified_video_config(tmp_path, lambda parameters: parameters["freeze"].update(enabled=False))
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_acceptance_video_quality.py::test_default_video_quality_config_comes_from_unified_config tests/test_acceptance_video_quality.py::test_unified_video_threshold_override_changes_runtime_config -q
```

Expected: failures show missing `module_version`, missing `qc_config_reference`, or rejection of the unified config root.

- [ ] **Step 3: Populate `modules.video_quality.parameters`**

Set `config_version: qc_acceptance_v1.1.0`, replace `threshold_profile` with `module_version`, and add every active runtime field using the current accepted values:

```yaml
parameters:
  pipeline:
    stop_before_mask_if_fail: true
    do_keypoint_quality_check: false
    do_keypoint_mask_matching: false
    do_keypoint_temporal_check: false
  fps:
    expected_fps: null
    min_fps_pass: 24.0
    min_fps_warn: 20.0
    min_fps_fail: 20.0
  resolution:
    exact_resolution_required: false
    use_display_size_after_rotation: true
    min_short_side_fail: 720
    min_long_side_fail: 1280
    min_short_side_warn: 720
    min_long_side_warn: 1280
  timeline:
    use_actual_fps: true
    drop_interval_factor: 1.35
    drop_interval_extra_ms: 10.0
    drop_frame_ratio_pass: 0.05
    drop_frame_ratio_warn: 0.10
    max_gap_factor: 3.0
    max_gap_floor_ms: 100.0
    pts_monotonic_required: true
  decode:
    sample_decode_ratio_pass: 0.995
    sample_decode_ratio_warn: 0.98
    max_sample_frames: 300
    sample_interval_sec: 0.5
    include_head_tail_frames: 10
  exposure:
    black:
      mean_y_max: 10.0
      dark_pixel_ratio_min: 0.98
      ratio_pass: 0.01
      ratio_warn: 0.90
      max_frame_count_fail: 10
    over_dark:
      mean_y_max: 35.0
      dark_pixel_ratio_min: 0.75
      ratio_pass: 0.05
      ratio_warn: 0.90
    over_exposed:
      mean_y_min: 235.0
      over_exposed_pixel_ratio_min: 0.35
      ratio_pass: 0.05
      ratio_warn: 0.90
  sharpness_global:
    normalize_before_compute: true
    target_short_side: 720
    no_upscale: true
    laplacian_p10_pass: 15.0
    laplacian_p10_warn: 0.0
    laplacian_median_pass: 20.0
    laplacian_median_warn: 0.0
    laplacian_under_100_ratio_pass: 1.0
    laplacian_under_100_ratio_warn: 1.0
    tenengrad_p10_pass: 6.0
    tenengrad_p10_warn: 4.0
    tenengrad_median_pass: 7.0
    tenengrad_median_warn: 4.0
  freeze:
    enabled: true
    downscale_short_side: 360
    frame_diff_mean_abs_max: 1.0
    hist_diff_max: 0.01
    ssim_min: 0.995
    phash_hamming_max: 4
    freeze_candidate_window_sec: 0.5
    confirmed_freeze_window_sec: 1.0
    adjacent_near_duplicate_ratio_warn: 0.90
    frozen_frame_ratio_pass: 0.05
    frozen_frame_ratio_warn: 0.10
    min_interval_frames: 6
    min_interval_duration_ms: 100.0
    max_consecutive_frozen_sec_pass: 0.5
    max_consecutive_frozen_sec_fail: 1.0
    motion_conflict_enabled: true
    motion_keypoint_delta_normalized_min: 0.02
    motion_keypoint_delta_px_min: 12.0
    motion_numeric_delta_min: 0.01
    critical_window_enabled: true
    critical_window_keywords: [grasp, place, contact, hand-object, hand object, interaction, 抓取, 放置, 接触, 交互]
    critical_window_interval_duration_ms_fail: 100.0
    video_state_conflict_noncritical_duration_ms_fail: 1000.0
    video_state_conflict_critical_duration_ms_fail: 500.0
  defects:
    max_duration_ratio_fail: 0.10
    duration_ratio_warn: 0.05
  hdf5_alignment:
    enabled: true
    mode: fail
    max_delta_frames_pass: 2
    max_delta_frames_warn: 5
    max_delta_ratio_pass: 0.001
    max_delta_ratio_warn: 0.005
```

Tighten `schemas/qc_acceptance_config.v1.schema.json` at the same time. Require `module_version` and `parameters` under `modules.video_quality`, set `additionalProperties: false` inside `parameters` and every nested video subsection, and require every key shown above. Apply these exact type rules:

- booleans: every `enabled`, `use_*`, `no_upscale`, `normalize_before_compute`, `stop_before_mask_if_fail`, `do_*`, `pts_monotonic_required`, `motion_conflict_enabled`, and `critical_window_enabled` field
- integers with minimum 0: side sizes, frame counts, sample counts, pHash Hamming distance, and HDF5 delta-frame limits
- numbers with minimum 0: ratios, time values, pixel thresholds, sharpness thresholds, and motion deltas
- ratio fields: number with minimum 0 and maximum 1
- `fps.expected_fps`: number greater than 0 or `null`
- `hdf5_alignment.mode`: enum `ignore|warn|fail`
- `freeze.critical_window_keywords`: non-empty array of unique strings

Add `issue_type` to every active video rule using these exact groups:

```text
video_unreadable: video_not_opened, cannot_open_video, video_stream_missing, codec_unreadable, metadata_unreadable, no_sample_frames_decoded
low_fps: fps_below_min, fps_below_pass, fps_below_expected
low_resolution: short_side_below_min, long_side_below_min
timeline_discontinuity: pts_monotonic_invalid, drop_frame_ratio_above_max, drop_frame_ratio_warn, frame_interval_p99_ms_above_max, frame_interval_p99_ms_warn, max_frame_gap_ms_above_max
decode_incomplete: sample_decode_ratio_below_min, sample_decode_ratio_warn
black_frame: black_frame_ratio_above_max, black_frame_ratio_warn, black_frame_count_above_max
visual_defect_duration: defect_duration_ratio_above_max, defect_duration_ratio_warn
over_dark: mean_over_dark_ratio_above_max, mean_over_dark_ratio_warn
over_exposed: mean_over_exposed_ratio_above_max, mean_over_exposed_ratio_warn
low_sharpness: all laplacian_* and tenengrad_* rules
low_motion: adjacent_near_duplicate_ratio_warn
freeze: frozen_frame_ratio_above_max, frozen_frame_ratio_warn, max_consecutive_frozen_sec_above_max, max_consecutive_frozen_sec_warn
video_state_conflict: video_state_conflict, video_state_conflict_warn
hdf5_alignment: hdf5_missing, hdf5_unreadable, hdf5_frame_count_mismatch, hdf5_frame_count_mismatch_warn
```

Set `needs_manual_review: true` on warn rules and `false` on fail rules. Rules with runtime-configured severity (`hdf5_missing`, `hdf5_unreadable`) set it from the emitted severity rather than a static YAML boolean.

- [ ] **Step 4: Archive the released config and verify identical bytes**

Create `configs/qc_acceptance/qc_acceptance_v1.1.0.yaml` as an exact byte-for-byte copy of the completed canonical config. Add this test to `tests/test_qc_config.py`:

```python
def test_released_config_archive_matches_canonical() -> None:
    canonical = Path("configs/qc_acceptance.yaml").read_bytes()
    archived = Path("configs/qc_acceptance/qc_acceptance_v1.1.0.yaml").read_bytes()
    assert archived == canonical
```

- [ ] **Step 5: Replace the video-only loader**

Change `VideoQualityConfig` to include:

```python
@dataclass(frozen=True)
class VideoQualityConfig:
    module_version: str
    qc_config_reference: dict[str, str]
    pipeline: PipelineConfig
    fps: FpsConfig
    resolution: ResolutionConfig
    timeline: TimelineConfig
    decode: DecodeConfig
    exposure: ExposureConfig
    sharpness_global: SharpnessGlobalConfig
    freeze: FreezeConfig
    defects: DefectDurationConfig
    hdf5_alignment: Hdf5AlignmentConfig
```

Make each nested config constructor receive values extracted from the validated `parameters`; missing fields must fail schema validation before construction. Remove `_apply_legacy_config()` and do not merge arbitrary video-only YAML roots.

Use this flow:

```python
def load_video_quality_config(path: Path | None) -> VideoQualityConfig:
    loaded = load_qc_acceptance_config(path)
    module = loaded.raw["modules"]["video_quality"]
    parameters = loaded.module_parameters("video_quality")
    return _video_config_from_parameters(
        parameters,
        module_version=module["module_version"],
        qc_config_reference=loaded.json_reference(),
    )
```

- [ ] **Step 6: Make report provenance use the loaded config**

Delete `QC_CONFIG_SCHEMA_VERSION`, `QC_CONFIG_VERSION`, `QC_CONFIG_NAME`, `QC_CONFIG_RELATIVE_PATH`, `_qc_config_hash()`, and `_qc_config_json()`.

In `asset_qc_result_to_json()`, write:

```python
"qc_config": dict(config.qc_config_reference),
```

No report code may reopen `configs/qc_acceptance.yaml` or compute a hash from a file other than the one used to build `config`.

- [ ] **Step 7: Run focused video-loader tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_qc_config.py tests/test_acceptance_video_quality.py::test_default_video_quality_config_comes_from_unified_config tests/test_acceptance_video_quality.py::test_unified_video_threshold_override_changes_runtime_config tests/test_acceptance_video_quality.py::test_unified_config_missing_video_parameters_is_rejected -q
```

Expected: all selected config and video-loader tests pass.

- [ ] **Step 8: Commit unified video configuration**

```bash
git add configs/qc_acceptance.yaml configs/qc_acceptance/qc_acceptance_v1.1.0.yaml schemas/qc_acceptance_config.v1.schema.json acceptance_pull/video_quality.py tests/test_qc_config.py tests/test_acceptance_video_quality.py
git commit -m "feat: load video QC thresholds from unified config"
```

---

### Task 3: Remove Hand ROI Completely

**Files:**

- Modify: `acceptance_pull/video_quality.py`
- Modify: `configs/qc_acceptance.yaml`
- Modify: `tests/test_acceptance_video_quality.py`
- Modify: `tests/fixtures.py`

**Interfaces:**

- Consumes: typed video configuration from Task 2.
- Produces: video metrics and JSON with no `hand_roi`, `HandRoi*`, or `hand_roi_*` surface.

- [ ] **Step 1: Add an absence regression test**

Add this test before deleting implementation:

```python
def test_video_qc_has_no_hand_roi_surface(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    video = video_dir / "408817_video.mp4"
    write_test_video(video, [textured_frame(0), textured_frame(10)], fps=30.0)
    write_quality_hdf5(batch / "hdf5" / "408817_hdf5.hdf5", 2)

    assert run_video_quality_check(batch) == 0
    report = json.loads((batch / "quality_archive" / "408817.json").read_text(encoding="utf-8"))
    serialized = json.dumps(report)

    assert "hand_roi" not in serialized
    assert "hand_roi_metrics" not in report["video_quality"]["metrics"]
    assert "hand_roi" not in load_video_quality_config(None).to_dict()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_acceptance_video_quality.py::test_video_qc_has_no_hand_roi_surface -q
```

Expected: FAIL because the report still contains `hand_roi_metrics` or the config still exposes hand ROI.

- [ ] **Step 3: Delete hand ROI types and computation**

Remove all of these symbols and their call sites:

```text
PipelineConfig.run_hand_roi
PipelineConfig.hand_roi_source
HandRoiSevereFailConfig
HandRoiConfig
HandRoiMetrics
VideoMetrics.hand_roi
_compute_hand_roi_metrics
_hand_roi_json
```

Delete the hand ROI branch in `_reason_details_for_codes()` and `evaluate_video_quality()`. Delete ROI-only HDF5 projection helpers only when `rg` confirms they are not used by freeze motion-conflict logic.

- [ ] **Step 4: Delete ROI rules and tests**

Remove every `hand_roi_*` rule from `modules.video_quality.rules`. Delete ROI-specific tests and the `HandRoiMetrics` import from `tests/test_acceptance_video_quality.py`. Remove `write_hand_keypoint_hdf5` from `tests/fixtures.py` only if no remaining test uses it.

- [ ] **Step 5: Run the absence test and all video tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_acceptance_video_quality.py -q
```

Expected: all video tests pass, `rg -n "hand_roi|HandRoi" acceptance_pull configs` returns no matches, and `rg -n "HandRoiMetrics|test_hand_roi_" tests` returns no matches. The intentional absence regression test may retain the lowercase string in its name and assertions.

- [ ] **Step 6: Commit the removal**

```bash
git add acceptance_pull/video_quality.py configs/qc_acceptance.yaml tests/test_acceptance_video_quality.py tests/fixtures.py
git commit -m "refactor: remove hand ROI video checks"
```

---

### Task 4: Add Report State, Canonical Issues, And Report Schema

**Files:**

- Create: `schemas/asset_qc_report.v1.schema.json`
- Create: `tests/test_asset_qc_schema.py`
- Create: `tests/qc_report_fixtures.py`
- Modify: `qc_common/schema.py`
- Modify: `qc_common/__init__.py`
- Modify: `acceptance_pull/video_quality.py`
- Modify: `tests/test_acceptance_video_quality.py`

**Interfaces:**

- Produces: `validate_asset_qc_report(report: dict[str, Any]) -> None`.
- Produces: canonical `IssueDetail` records with `issue_id`, observed value, boundary value, and rule ID.
- Produces: `pipeline_state`, `overall_decision`, `issues`, `manual_review`, and `video_quality.flow`.

- [ ] **Step 1: Write state-schema tests**

Create `tests/qc_report_fixtures.py` with this factory:

```python
from typing import Any


def make_valid_running_report(asset_id: str = "408817") -> dict[str, Any]:
    return {
        "schema_version": "asset_qc_report.v1",
        "qc_config": {
            "schema_version": "qc_acceptance_config_schema.v1",
            "config_version": "qc_acceptance_v1.1.0",
            "config_name": "acceptance_gate",
            "config_path": "configs/qc_acceptance.yaml",
            "config_hash": "sha256:" + "0" * 64,
        },
        "asset_id": asset_id,
        "report_revision": 1,
        "pipeline_state": {
            "status": "running",
            "current_module": "video_quality",
            "next_module": "sam3_containment",
        },
        "overall_decision": None,
        "issues": [],
        "manual_review": {
            "required": None,
            "state": "not_evaluated",
            "candidate_issue_ids": [],
            "failures_for_batch_stats_issue_ids": [],
        },
        "video_quality": {
            "flow": {
                "entry_gate": {
                    "state": "ready",
                    "eligible": True,
                    "blocked_by_module": None,
                    "required_inputs": ["source_files.video.path"],
                    "missing_inputs": [],
                    "upstream_continue": True,
                },
                "result_gate": {"verdict": "pass", "has_fail": False, "has_warn": False},
                "exit_gate": {
                    "state": "continue",
                    "continue_to_next_module": True,
                    "next_module": "sam3_containment",
                },
            },
            "evaluation": {
                "decision": "pass",
                "reasons": [],
                "warn_reasons": [],
                "issue_ids": [],
                "should_run_mask_qc": True,
            },
            "metrics": {},
        },
    }
```

Create `tests/test_asset_qc_schema.py` with these cases and a local fixture that calls the shared factory:

```python
def test_running_report_requires_null_overall_decision(valid_running_report: dict) -> None:
    valid_running_report["pipeline_state"]["status"] = "running"
    valid_running_report["overall_decision"] = "warn"
    with pytest.raises(ValueError, match="overall_decision"):
        validate_asset_qc_report(valid_running_report)


def test_stopped_report_requires_fail(valid_running_report: dict) -> None:
    valid_running_report["pipeline_state"]["status"] = "stopped"
    valid_running_report["pipeline_state"]["current_module"] = "batch_statistics"
    valid_running_report["pipeline_state"]["next_module"] = "batch_statistics"
    valid_running_report["overall_decision"] = "fail"
    valid_running_report["video_quality"]["flow"]["result_gate"] = {
        "verdict": "fail",
        "has_fail": True,
        "has_warn": False,
    }
    valid_running_report["video_quality"]["flow"]["exit_gate"] = {
        "state": "stop_qc",
        "continue_to_next_module": False,
        "next_module": "batch_statistics",
    }
    validate_asset_qc_report(valid_running_report)


def test_pending_does_not_require_manual_review(valid_running_report: dict) -> None:
    valid_running_report["pipeline_state"]["status"] = "pending"
    valid_running_report["overall_decision"] = None
    valid_running_report["manual_review"] = {"required": None, "state": "not_evaluated", "candidate_issue_ids": []}
    validate_asset_qc_report(valid_running_report)
```

At the top of `tests/test_asset_qc_schema.py`, import `make_valid_running_report` and define:

```python
@pytest.fixture
def valid_running_report() -> dict:
    return make_valid_running_report()
```

- [ ] **Step 2: Run schema tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_asset_qc_schema.py -q
```

Expected: collection fails because `validate_asset_qc_report` and the report schema do not exist.

- [ ] **Step 3: Create the report schema**

The schema must require:

```json
{
  "required": [
    "schema_version",
    "qc_config",
    "asset_id",
    "report_revision",
    "pipeline_state",
    "overall_decision",
    "issues",
    "manual_review",
    "video_quality"
  ]
}
```

Define these conditions with `if`/`then`:

- `pending|running` requires `overall_decision: null`.
- `stopped` requires `overall_decision: fail`.
- `completed` requires `overall_decision: pass|warn`.
- A `fail` video verdict requires `continue_to_next_module: false` and `next_module: batch_statistics`.
- A `pass|warn` video verdict requires `continue_to_next_module: true`.

Define the issue schema with required fields:

```text
issue_id, code, severity, module, issue_type, metric,
observed_value, operator, boundary_value, rule_id,
needs_manual_review, context
```

Do not define `config_version`, `pass_threshold`, `fail_threshold`, `hand_roi`, or `hand_roi_metrics` as allowed new issue/video fields.

- [ ] **Step 4: Implement report validation**

Add to `qc_common/schema.py`:

```python
def validate_asset_qc_report(report: dict[str, Any]) -> None:
    _validate_with_schema(report, _repo_root() / "schemas" / "asset_qc_report.v1.schema.json", "asset QC report")
```

Return `None` on success and raise `ValueError` with the dotted JSON path on failure.

- [ ] **Step 5: Replace `ReasonDetail` with canonical issue fields**

Use this dataclass contract:

```python
@dataclass(frozen=True)
class IssueDetail:
    issue_id: str
    code: str
    severity: str
    module: str
    issue_type: str
    metric: str | None
    observed_value: Any | None
    operator: str | None
    boundary_value: Any | None
    rule_id: str
    needs_manual_review: bool
    context: dict[str, Any] = field(default_factory=dict)
```

Generate deterministic IDs as `video_quality:<code>:<zero-padded-occurrence>`. Select `boundary_value` according to the triggering severity: warn uses the pass boundary and fail uses the fail boundary. Boolean state checks use the required boolean as `boundary_value`; infrastructure errors without a threshold use `operator: null` and `boundary_value: null`.

- [ ] **Step 6: Add video module flow and top-level state**

For video fail, emit:

```json
{
  "pipeline_state": {"status": "stopped", "current_module": "batch_statistics", "next_module": "batch_statistics"},
  "overall_decision": "fail",
  "manual_review": {"required": false, "state": "skipped_due_to_fail", "candidate_issue_ids": [], "failures_for_batch_stats_issue_ids": ["video_quality:cannot_open_video:001"]},
  "video_quality": {
    "flow": {
      "result_gate": {"verdict": "fail", "has_fail": true, "has_warn": false},
      "exit_gate": {"continue_to_next_module": false, "next_module": "batch_statistics"}
    }
  }
}
```

For video pass or warn, emit `pipeline_state.status: running`, `overall_decision: null`, and route to the next configured module. Warn issue IDs are appended to `manual_review.candidate_issue_ids`, while `manual_review.required` remains `null` and state remains `not_evaluated`.

Keep `should_run_mask_qc` only as a compatibility alias inside video evaluation for this release. The flow exit gate is authoritative.

- [ ] **Step 7: Validate before each JSON write**

Call `validate_asset_qc_report(report)` in `write_per_asset_qc_json_reports()` before `json.dumps()`. A validation error must propagate and prevent the invalid report from replacing an existing file.

- [ ] **Step 8: Run report and video tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_asset_qc_schema.py tests/test_acceptance_video_quality.py -q
```

Expected: all tests pass; generated pass/warn reports are running with `overall_decision: null`, while fail reports are stopped with `overall_decision: fail`.

- [ ] **Step 9: Commit the report contract**

```bash
git add schemas/asset_qc_report.v1.schema.json qc_common tests/test_asset_qc_schema.py tests/test_acceptance_video_quality.py acceptance_pull/video_quality.py
git commit -m "feat: enforce gated asset QC report schema"
```

---

### Task 5: Atomic Revision-Aware Asset Report Updates

**Files:**

- Create: `qc_common/report.py`
- Create: `tests/test_qc_report_store.py`
- Modify: `tests/qc_report_fixtures.py`
- Modify: `qc_common/__init__.py`
- Modify: `acceptance_pull/video_quality.py`
- Modify: `tests/test_acceptance_video_quality.py`

**Interfaces:**

- Produces: `load_asset_qc_report(path: Path) -> dict[str, Any] | None`.
- Produces: `write_asset_qc_report(path: Path, report: dict[str, Any], expected_revision: int) -> None`.
- Produces: `StaleReportRevisionError` for concurrent or stale updates.
- Consumes: `validate_asset_qc_report()` from Task 4.

- [ ] **Step 1: Write failing atomic-update tests**

Create `tests/test_qc_report_store.py` and import `make_valid_running_report` from `tests.qc_report_fixtures`:

```python
import json
from pathlib import Path

import pytest

from qc_common.report import StaleReportRevisionError, load_asset_qc_report, write_asset_qc_report
from tests.qc_report_fixtures import make_valid_running_report


@pytest.fixture
def valid_running_report() -> dict:
    return make_valid_running_report()


def test_write_asset_report_rejects_stale_revision(tmp_path: Path, valid_running_report: dict) -> None:
    path = tmp_path / "quality_archive" / "408817.json"
    valid_running_report["report_revision"] = 1
    write_asset_qc_report(path, valid_running_report, expected_revision=0)

    stale = dict(valid_running_report)
    stale["report_revision"] = 2
    write_asset_qc_report(path, stale, expected_revision=1)

    competing = dict(stale)
    competing["report_revision"] = 2
    with pytest.raises(StaleReportRevisionError, match="expected revision 1, found 2"):
        write_asset_qc_report(path, competing, expected_revision=1)


def test_failed_validation_does_not_replace_existing_report(tmp_path: Path, valid_running_report: dict) -> None:
    path = tmp_path / "quality_archive" / "408817.json"
    valid_running_report["report_revision"] = 1
    write_asset_qc_report(path, valid_running_report, expected_revision=0)
    original = path.read_bytes()

    invalid = dict(valid_running_report)
    invalid["report_revision"] = 2
    invalid["overall_decision"] = "warn"
    with pytest.raises(ValueError, match="overall_decision"):
        write_asset_qc_report(path, invalid, expected_revision=1)

    assert path.read_bytes() == original


def test_load_missing_asset_report_returns_none(tmp_path: Path) -> None:
    assert load_asset_qc_report(tmp_path / "missing.json") is None
```

- [ ] **Step 2: Run store tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_qc_report_store.py -q
```

Expected: collection fails because `qc_common.report` does not exist.

- [ ] **Step 3: Implement the atomic report store**

Implement this contract in `qc_common/report.py`:

```python
class StaleReportRevisionError(RuntimeError):
    pass


def load_asset_qc_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"asset QC report root must be an object: {path}")
    return loaded


def write_asset_qc_report(path: Path, report: dict[str, Any], expected_revision: int) -> None:
    current = load_asset_qc_report(path)
    current_revision = 0 if current is None else int(current.get("report_revision", 0))
    if current_revision != expected_revision:
        raise StaleReportRevisionError(
            f"expected revision {expected_revision}, found {current_revision}: {path}"
        )
    next_revision = int(report.get("report_revision", 0))
    if next_revision != expected_revision + 1:
        raise ValueError(
            f"report_revision must be {expected_revision + 1}, got {next_revision}"
        )

    validate_asset_qc_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
```

Import `json`, `os`, `tempfile`, `Any`, `Path`, and `validate_asset_qc_report` explicitly.

- [ ] **Step 4: Run store tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_qc_report_store.py -q
```

Expected: all report-store tests pass.

- [ ] **Step 5: Add a video upsert regression test**

Add to `tests/test_acceptance_video_quality.py`:

```python
def test_video_qc_preserves_existing_module_blocks(tmp_path: Path) -> None:
    batch = tmp_path
    video_dir = batch / "video"
    video_dir.mkdir()
    write_test_video(video_dir / "408817_video.mp4", [textured_frame(0), textured_frame(10)], fps=30.0)
    write_quality_hdf5(batch / "hdf5" / "408817_hdf5.hdf5", 2)
    archive = batch / "quality_archive"
    archive.mkdir()
    existing = make_valid_running_report(asset_id="408817")
    existing["report_revision"] = 1
    existing["quality_hand"] = {"owner": "quality_hand", "metrics": {"score": 1.0}}
    (archive / "408817.json").write_text(json.dumps(existing), encoding="utf-8")

    assert run_video_quality_check(batch) == 0
    updated = json.loads((archive / "408817.json").read_text(encoding="utf-8"))

    assert updated["report_revision"] == 2
    assert updated["quality_hand"] == existing["quality_hand"]
    assert updated["video_quality"]["flow"]["result_gate"]["verdict"] == "pass"
```

- [ ] **Step 6: Run the upsert test and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_acceptance_video_quality.py::test_video_qc_preserves_existing_module_blocks -q
```

Expected: FAIL because the current video writer replaces the whole report or resets revision state.

- [ ] **Step 7: Update video report generation to merge owned fields**

Change report creation to accept `existing_report: dict[str, Any] | None`. Start from `copy.deepcopy(existing_report)` when present. Replace only these owned fields:

```text
qc_config
source_files.video
video_quality
video-quality issues in top-level issues
video-quality issue references in manual_review
pipeline_state and overall_decision produced by the video exit gate
```

Preserve all other top-level and module fields. Remove prior issues where `issue.module == "video_quality"`, append the current video issues, and rebuild only the corresponding video candidate/failure references. Set `report_revision` to the existing revision plus one.

In `write_per_asset_qc_json_reports()`:

```python
existing = load_asset_qc_report(path)
expected_revision = 0 if existing is None else int(existing.get("report_revision", 0))
report = asset_qc_result_to_json(result, config, existing_report=existing)
write_asset_qc_report(path, report, expected_revision=expected_revision)
```

- [ ] **Step 8: Run video, schema, and store tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_qc_report_store.py tests/test_asset_qc_schema.py tests/test_acceptance_video_quality.py -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit atomic report updates**

```bash
git add qc_common/report.py qc_common/__init__.py acceptance_pull/video_quality.py tests/test_qc_report_store.py tests/test_acceptance_video_quality.py
git commit -m "feat: write asset QC reports atomically"
```

---

### Task 6: Documentation Cleanup And Full Verification

**Files:**

- Modify: `ACCEPTANCE.md`
- Modify: `docs/asset-qc-json-format.md`
- Modify: `docs/PRD-qc-gated-json.md`
- Modify: `docs/PRD-qc-unified-config.md`
- Modify: `docs/batch-sampling-pull.md`
- Modify: `tests/test_qc_config.py`
- Modify: `tests/test_asset_qc_schema.py`

**Interfaces:**

- Consumes: final config/report schemas and video writer behavior.
- Produces: one consistent reviewer-facing contract with executable examples.

- [ ] **Step 1: Add documentation consistency tests**

Add tests that scan active docs and config:

```python
def test_active_contract_has_no_hand_roi_terms() -> None:
    paths = [
        Path("configs/qc_acceptance.yaml"),
        Path("ACCEPTANCE.md"),
        Path("docs/asset-qc-json-format.md"),
        Path("docs/batch-sampling-pull.md"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "hand_roi" not in text
        assert "HandRoi" not in text


def test_active_contract_has_no_threshold_version() -> None:
    for path in [Path("configs/qc_acceptance.yaml"), Path("ACCEPTANCE.md"), Path("docs/asset-qc-json-format.md")]:
        assert "threshold_version" not in path.read_text(encoding="utf-8")
```

The approved design and historical migration explanations may mention removed terms; exclude `docs/superpowers/specs` and migration-history sections from this active-doc scan.

- [ ] **Step 2: Run consistency tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_qc_config.py tests/test_asset_qc_schema.py -q
```

Expected: FAIL with current hand ROI and old state/config examples in active docs.

- [ ] **Step 3: Update active documentation**

Apply these exact contract changes:

- Config examples use only `configs/qc_acceptance.yaml` and `modules.video_quality.parameters`.
- Delete all hand ROI sections and reason-code tables.
- Explain `pipeline_state.status` separately from `overall_decision`.
- Explain that `manual_review.required` is independent and is not inferred from `pending`.
- Replace duplicated reason details with canonical top-level `issues` and issue-ID references.
- Show `observed_value`, `operator`, and `boundary_value` in issue examples.
- Remove per-issue `config_version`; point to top-level `qc_config.config_version`.
- Link both schema files from `ACCEPTANCE.md` and `docs/asset-qc-json-format.md`.

- [ ] **Step 4: Run consistency and schema tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_qc_config.py tests/test_asset_qc_schema.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run the complete test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass with no test failures.

- [ ] **Step 6: Run static contract checks**

Run:

```bash
rg -n "hand_roi|HandRoi|threshold_version|QC_CONFIG_VERSION" acceptance_pull configs ACCEPTANCE.md docs/asset-qc-json-format.md docs/batch-sampling-pull.md
git diff --check
```

Expected: `rg` returns no active-contract matches and `git diff --check` exits 0.

- [ ] **Step 7: Commit documentation and cleanup**

```bash
git add ACCEPTANCE.md docs tests configs acceptance_pull qc_common schemas requirements.txt
git commit -m "docs: align QC config and report contracts"
```

- [ ] **Step 8: Verify final branch state**

Run:

```bash
git status -sb
git log -6 --oneline
```

Expected: clean worktree on `codex/integrate-acceptance-modules`, with the implementation commits ahead of the remote until explicitly pushed.
