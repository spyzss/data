# DR and Potentia Supplier Evaluation Design

**Date:** 2026-07-17

## Objective

Add DeepReach (canonical supplier name `DR`, identifier `dr`) and Potentia to the existing unified `supplier_evaluation` workflow without coupling supplier file audits into precheck or creating supplier-specific pipelines.

## Invariants

- Keep the behavior introduced by commit `47bad93f7a9ee319fa01419de8d1fb39c315239e`.
- Do not change `frame_survival_v2`, the acceptance 90% threshold, or supplier-evaluation continue-on-failure semantics.
- Keep `hdf5_text_info` as the sole canonical text verdict. Supplier audit may report metadata presence and parseability but must not emit a competing text verdict.
- Preserve raw module outputs. A final verdict or supplier signal must not overwrite them.
- Keep precheck free of SAM3, DA3, VLM, video-quality, supplier CSV, IMU, calibration, and trajectory orchestration.
- Treat all manifest frame bounds as source-inclusive and convert to half-open coordinates exactly once in `AssetContext`.
- Do not infer undocumented supplier field names or external-transform direction.

## Architecture

The existing manifest-to-`AssetContext` flow remains the only pipeline entry. The actual automatic order is `precheck -> video_quality -> supplier_data_audit -> sam3_containment`. `video_quality` and `supplier_data_audit` are independent producers: ordering does not create a data dependency, and neither reads or mutates the other's artifact. Supplier audit consumes only declared manifest sources and supplier-specific mapping configuration, writes a durable JSON artifact, and adapts that artifact into the same v2 QC report.

DR continues to use the existing HDF5 precheck adapter. Potentia has no HDF5 or hand keypoints, so its five precheck modules report `input_missing` independently while supplier evaluation proceeds to supplier audit and video quality. SAM3 checks supplier prerequisites before looking for candidate artifacts so DR and Potentia receive explicit blocked reasons.

## Canonical Supplier Identity

- New DR manifests and reports use `supplier=dr`, `supplier_id=dr`, and `supplier_name=DR`.
- Input aliases `deepreach` and `DeepReach` are normalized to `dr` in the runtime context.
- The unmodified source manifest row and the original supplier spelling remain under source/alias metadata for lineage.
- Potentia uses `supplier=potentia`, `supplier_id=potentia`, and `supplier_name=Potentia`.

## DR Manifest Contract

Default granularity is one logical task per asset. `asset_id` is the task identifier without a camera suffix. Each row records:

- HDF5 and LeRobot v2 sources;
- `head_video_path`, `left_wrist_video_path`, and `right_wrist_video_path`;
- `primary_camera` and `primary_video_path`;
- `calib_path` and `camera_trajectory_path`;
- per-source inventory status and overall adapter status.

Primary camera comes from CLI/config and defaults to `head`. If the selected camera is absent, the row remains bound to that camera and reports `primary_camera_missing`; it never falls back. All three views are inventoried, while `video_quality` consumes only `primary_video_path`.

The old task-camera format remains available only through explicit `granularity=task_camera`. Its rows retain camera-suffixed asset identifiers and cannot be mixed with task-level rows in one builder invocation.

The task-level HDF5 frame contract requires an explicit reference dataset selected from the five participating datasets. This repository does not assume a default until the supplier contract is confirmed; cloud smoke may pass `timestamp` only as an explicit, reviewed choice. The manifest records `reference_dataset`, `expected_frame_count`, every dataset length, mismatch ranges, mismatch count, and source path lineage. Any unequal length is `inconsistent_frame_count`: precheck reports input-invalid and emits no official partial-prefix result, while independent supplier-evaluation modules continue.

## Potentia Manifest Contract

Transport package directories are source partitions only. Discovery descends through packages and emits exactly one `potentia__<task_id>` asset for each task directory containing or expected to contain the named deliverables:

- `video.mp4`
- `meta.json`
- `frames.csv`
- `aligned.csv`
- `imu.csv`
- `calibration.json`

The manifest records source partition, task directory, all six expected paths, per-file presence, source granularity, source-coordinate convention, and adapter status. Duplicate task identifiers across packages are input-invalid instead of silently overwritten.

## Supplier Audit Artifact

The producer writes:

```text
module_outputs/<asset_id>/supplier_data_audit/
├── supplier_data_audit_result.json
└── run_config.json
```

The JSON contains the raw inventory/audit payload and its adapted `ModuleResult`. It is fingerprinted by supplier sources, supplier mapping config, implementation version, and manifest metadata. It never reads or modifies precheck or video-quality artifacts.

DR audit records three-camera inventory, primary-camera selection, HDF5/LeRobot/calibration/trajectory presence, JSON parseability, and mapping-driven trajectory range, monotonicity, duplicate, and gap metrics.

Potentia audit records file completeness, meta structure, configured metadata paths, frame/aligned/IMU row and timestamp metrics, IMU coverage and sampling statistics, calibration representations, and actual-video metadata supplied by the video source. `meta.qc` is retained only as `supplier_quality_signal`.

Missing required files are fail evidence in this module but do not stop `supplier_evaluation`. Missing or incomplete mapping produces `unverified`/warn evidence rather than guessed values or pass.

## Mapping Configuration

All supplier-specific field interpretation lives under the `supplier_data_audit` module configuration. Mapping uses explicit column names and JSON dot paths. Transform direction is a required enum when a DR pose mapping is enabled. Every mapped timestamp column must declare exactly one of `timestamp_unit` (`s`, `ms`, `us`, `ns`) or a positive `timestamp_scale_to_seconds`; derived fps, Hz, gap, and coverage values use normalized seconds only. Missing units are unverified, while invalid or conflicting mappings are config/mapping-invalid rather than guessed. The raw artifact records the mapping config identity/hash. Active config ships with empty/unverified real-data mappings; tests use small synthetic configs with explicit mappings.

Potentia calibration distinguishes mathematical invalidity from unresolved scaling. Invalid JSON, mapped matrix shape, resolution, focal length, or principal point can fail. Valid parameters whose raw/scaled/video resolution relationship cannot be explained are `unverified`/warn by default. `scaling_mismatch_action` is restricted to `review` or `fail` and defaults to `review`; only explicit `fail` upgrades a mathematically valid scaling mismatch.

Potentia's HDF5-alignment exemption is an exact `potentia` supplier override under `video_quality.parameters.supplier_overrides`. No default or wildcard override exists, so JD, XJGT, and DR retain current alignment behavior.

## DR Projection Audit

Pure geometry utilities implement rigid transforms, calibrated projection, validity masks, and resolution-aware intrinsic scaling in `qc_common`. Mapping-driven calibration and trajectory parsing stays in the DR supplier adapter. A standalone tool samples requested assets/frames/cameras and writes projection records plus combined left/right overlays.

Every projection record preserves source frame, local frame, camera, hand side, projected coordinates, depth, in-frame state, calibration/trajectory sources, and the declared transform chain. Missing mappings, missing/gapped trajectories, or unspecified transform direction produce `calibration_unverified` or `transform_ambiguous` and no invented overlay.

## SAM3 States

- Candidate rows require `temporal_output_valid is True` to seed temporal windows. Manual rotation/side-view lineage remains available but is marked `sam3_eligible=false`; SAM3 consumes only explicit `sam3_eligible=true` rows.
- DR: calibration gates precede all candidate checks. Unverified calibration is `blocked/calibration_unverified`; ambiguous transforms are `blocked/transform_ambiguous`; once projection is validated, the still-unimplemented DR/head adapter is always `blocked/adapter_missing`, even with zero candidates.
- Potentia: `blocked` with reason `no_keypoint_input`.
- JDT: no valid temporal output is `blocked/no_valid_temporal_output`; valid temporal output with zero eligible candidates is `skipped/no_candidates`.
- Neither state may be converted to pass or an unexplained skipped state.

## Producer Identity

The outer artifact contract remains `qc_producer_run_config.v1`. Temporal output uses `keypoint_temporal.output.v2` with producer identity `precheck-session-v7-calibrated-temporal-validity`. Supplier audit keeps raw schema `supplier_data_audit.raw.v2` with producer identity `supplier-data-audit-producer-v3`. Reuse requires matching implementation, output schema, config, manifest/source identity, so temporal v5/v6 and supplier-audit v1/v2 artifacts are stale.

## Verification

Development follows test-first red/green cycles. Directed tests cover manifest contracts, path normalization, producer isolation, continue-on-failure, mapping-driven audit metrics, video alignment isolation, projection math/lineage, and precise SAM3 blocked states. Final verification includes compileall, directed tests, the full pytest suite, `git diff --check`, and a complete worktree inventory. Cloud smoke commands are documented but not executed locally.
