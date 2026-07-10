# Unified QC Config And Video Quality Design

Date: 2026-07-10

Status: approved for planning

Target configuration version: `qc_acceptance_v1.1.0`

## 1. Decisions

This design records the decisions confirmed for the asset QC pipeline:

1. `pending` is a pipeline lifecycle state, not a quality decision.
2. `overall_decision` only uses `pass`, `warn`, `fail`, or `null` before a final decision exists.
3. Manual review routing is represented only by `manual_review.required` and `manual_review.state`.
4. `configs/qc_acceptance.yaml` is the only accepted runtime configuration entry point.
5. Video quality thresholds and algorithm parameters are loaded from `modules.video_quality.parameters`.
6. Rule IDs remain in config for stable issue identification, but do not replace threshold configuration.
7. Hand ROI quality analysis is removed completely.
8. JSON Schema validates asset QC reports; a second schema validates the unified YAML config.

## 2. Goals

- Make the unified QC config the actual runtime source of truth.
- Ensure the config reference written to each asset JSON describes the configuration that was really used.
- Keep pipeline state, quality verdict, and manual review state independent.
- Let downstream automation and manual review consume one asset JSON without reconstructing module-specific outputs.
- Remove inactive hand ROI behavior, configuration, output, documentation, and tests.
- Detect malformed config and JSON reports before they continue through the pipeline.

## 3. Non-Goals

- This change does not add new video quality metrics.
- This change does not reintroduce VMAF or reference-video comparison.
- This change does not perform keypoint quality validation inside video QC.
- This change does not define the batch statistics implementation.
- This change does not preserve the old video-only YAML as a supported CLI input format.

## 4. State Model

### 4.1 Pipeline lifecycle

`pipeline_state.status` describes execution progress:

```text
pending -> running -> completed
                   -> stopped
```

Allowed values:

- `pending`: the QC pipeline has not started.
- `running`: at least one module is running or more modules remain.
- `stopped`: a module returned `fail`, so later QC modules were not executed.
- `completed`: all eligible QC modules have finished.

### 4.2 Overall quality decision

`overall_decision` describes the final asset result:

- While `pipeline_state.status` is `pending` or `running`, `overall_decision` is `null`.
- When the pipeline is `stopped`, `overall_decision` is `fail`.
- When the pipeline is `completed`, `overall_decision` is `pass` or `warn`.

`pending` is not an allowed `overall_decision` value. Module warnings are stored in module verdicts and issues while the pipeline is still running; the final `warn` is produced only when the pipeline reaches a terminal state without a fail.

### 4.3 Manual review state

Manual review is independent of both fields above:

```json
{
  "manual_review": {
    "required": null,
    "state": "not_evaluated"
  }
}
```

Allowed states:

- `not_evaluated`
- `not_required`
- `required`
- `queued`
- `in_progress`
- `completed`
- `skipped_due_to_fail`

`required` is `null` until the manual review routing module evaluates accumulated warn candidates. A pipeline `pending` state never implies manual review.

## 5. Unified Configuration

### 5.1 Single accepted format

The CLI accepts only the unified QC config schema:

```bash
run_acceptance_video_quality.py --batch <batch> --config configs/qc_acceptance.yaml
```

When `--config` is omitted, the loader uses the repository's canonical `configs/qc_acceptance.yaml`. A video-only YAML is rejected with a clear migration error.

### 5.2 Video quality section

The unified config contains both runtime parameters and rule metadata:

```yaml
modules:
  video_quality:
    enabled: true
    module_version: video_prefilter_v0.3.2

    parameters:
      fps: {}
      resolution: {}
      timeline: {}
      decode: {}
      exposure: {}
      sharpness_global: {}
      freeze: {}
      defects: {}
      hdf5_alignment: {}

    rules:
      fps_below_min:
        rule_id: video_quality.fps_below_min
        verdict: fail
      fps_below_pass:
        rule_id: video_quality.fps_below_pass
        verdict: warn
```

Every field currently used by `VideoQualityConfig` must exist under `parameters`, except removed hand ROI fields. Production threshold defaults must not live only in Python code.

### 5.3 Loader behavior

A shared config loader performs these steps:

1. Read the unified YAML file.
2. Validate it against `qc_acceptance_config.v1.schema.json`.
3. Verify `config_version`, module order, required modules, and unique rule IDs.
4. Extract `modules.video_quality.parameters` into the typed video configuration object.
5. Build the top-level JSON `qc_config` from the same loaded file and bytes.

The loader must fail before video decoding when the config is missing, malformed, incomplete, or has an unsupported schema version.

### 5.4 Config provenance

The top-level report reference is derived from the loaded config, never from hard-coded constants:

```json
{
  "qc_config": {
    "schema_version": "qc_acceptance_config_schema.v1",
    "config_version": "qc_acceptance_v1.1.0",
    "config_name": "acceptance_gate",
    "config_path": "configs/qc_acceptance.yaml",
    "config_hash": "sha256:<actual-loaded-bytes>"
  }
}
```

Released config versions are archived as immutable files under `configs/qc_acceptance/`. The canonical `configs/qc_acceptance.yaml` may select the active version, but reports must retain enough path and hash information to retrieve and verify the exact historical file.

Changing any threshold, rule severity, module order, routing policy, or metric computation parameter requires a `config_version` bump.

## 6. Issue Contract

There is no bulk `thresholds` snapshot in module JSON. Each triggered issue stores only the boundary used for that concrete decision so the report remains understandable by itself:

```json
{
  "issue_id": "video_quality:laplacian_p10_warn:001",
  "code": "laplacian_p10_warn",
  "severity": "warn",
  "module": "video_quality",
  "issue_type": "low_sharpness",
  "metric": "sharpness_global.laplacian_p10",
  "observed_value": 12.6,
  "operator": "<",
  "boundary_value": 15.0,
  "rule_id": "video_quality.laplacian_p10_warn",
  "needs_manual_review": false,
  "context": {}
}
```

The top-level `qc_config.config_version` is authoritative, so individual issues do not repeat `config_version`.

Issues are stored once in a canonical top-level `issues` array. Module blocks, manual review candidates, and summary fields reference `issue_id` values instead of copying complete issue objects.

## 7. Module Flow

Every executable QC module writes:

```json
{
  "flow": {
    "entry_gate": {},
    "result_gate": {
      "verdict": "pass | warn | fail | skipped"
    },
    "exit_gate": {
      "continue_to_next_module": true,
      "next_module": "next_module_name"
    }
  }
}
```

Rules:

- `pass`: continue.
- `warn`: add issue references to manual review candidates and continue.
- `fail`: set pipeline status to `stopped`, set `overall_decision` to `fail`, and route directly to batch statistics.
- `skipped`: record the blocking reason and do not treat it as a quality warning.

The legacy video-specific `should_run_mask_qc` may be read during migration, but the authoritative routing field is `flow.exit_gate.continue_to_next_module`. New code must not use the legacy field to represent the whole pipeline.

## 8. Hand ROI Removal

The following are removed:

- `PipelineConfig.run_hand_roi`
- `PipelineConfig.hand_roi_source`
- `HandRoiConfig`
- `HandRoiSevereFailConfig`
- `HandRoiMetrics`
- ROI extraction and ROI sharpness computation
- hand ROI evaluation reason codes and rule IDs
- `hand_roi_metrics` in asset JSON
- hand ROI configuration examples, documentation, and tests

HDF5 keypoints remain available only where another active feature needs them, including video-state conflict checks during confirmed freezes. No hand ROI bbox or hand ROI sharpness metric is computed.

## 9. Schemas

Two machine-readable schemas are added:

```text
schemas/asset_qc_report.v1.schema.json
schemas/qc_acceptance_config.v1.schema.json
```

The config schema validates:

- required top-level version fields
- module order and module configuration shape
- video parameter types and required values
- rule ID format and severity enums
- manual review routing fields

The report schema validates:

- pipeline lifecycle and final decision combinations
- required module flow fields
- issue field types and enums
- fail routing invariants
- manual review state and required flag combinations
- absence of removed hand ROI fields in newly written reports

Schema validation runs when config is loaded and after each module updates the asset JSON. Validation failure stops that asset with an explicit infrastructure error; it must not be converted into a normal quality `warn`.

## 10. Safe JSON Updates

Multiple modules update the same asset report. Writers must:

- preserve unknown blocks owned by other modules
- update only their owned module block and shared flow fields
- increment `report_revision`
- write to a temporary file and atomically replace the target JSON
- reject stale revisions rather than silently overwriting newer data

## 11. Migration

This is a deliberate breaking configuration change:

1. Move current video thresholds into `modules.video_quality.parameters`.
2. Change the video loader to accept only unified config files.
3. Derive JSON config provenance from the loaded config.
4. Remove hand ROI implementation and output.
5. Change `overall_decision` pending representation to `null`.
6. Add schemas and validation.
7. Update docs and examples.

Existing historical JSON remains readable. New writers do not emit module `thresholds`, per-issue `config_version`, or hand ROI fields.

## 12. Test Strategy

Tests must cover:

- unified config loads successfully and supplies all video parameters
- a legacy video-only config is rejected with a migration message
- changing a video threshold in unified config changes the decision
- JSON `qc_config` version and hash match the file actually loaded
- missing required video parameters fail config validation
- `pending/running` reports require `overall_decision: null`
- stopped reports require `fail`; completed reports require `pass` or `warn`
- `pending` never implies manual review
- issue objects contain observed value and boundary value
- issue objects do not repeat `config_version`
- hand ROI configuration, rules, metrics, and output are absent
- fail routes to batch statistics and prevents later modules from running
- report and config examples validate against their schemas
- atomic update logic rejects stale report revisions

## 13. Acceptance Criteria

The change is complete when:

1. Video QC has one runtime config source: unified `qc_acceptance.yaml`.
2. No active video threshold exists only as a Python default.
3. The report config reference matches the actual loaded config bytes.
4. Hand ROI is absent from code, active config, output JSON, docs, and tests.
5. Pipeline lifecycle, final decision, and manual review state are independent.
6. All new config and report fixtures pass schema validation.
7. Existing non-ROI video QC behavior remains covered by tests.
