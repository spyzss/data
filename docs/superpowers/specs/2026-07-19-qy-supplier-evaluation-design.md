# QY Supplier Evaluation Design

## Scope

Add an independent `qy` (`supplier_name=QY`, source alias `qingyu`) adapter for
Qingyu episode directories. One episode is one canonical asset. The adapter
integrates with the existing unified QC producers without moving SAM3 into
precheck and without changing JD, JDT, DR, or Potentia semantics.

Local development uses synthetic fixtures only. The cloud root
`/mnt/oss/擎羽/Hugging_face` is not available in the local environment.

## Authoritative timebase and frame mapping

`timestamps/episode_timebase.json` is authoritative only for per-camera video
metadata:

- `videos[*].frames` is the physical video frame count.
- `source_start_frame` and `source_end_frame` define an inclusive source range.
- `source_frame_count = source_end_frame - source_start_frame + 1`.
- The source range is never inferred from physical frame count or Parquet row
  count.

The timebase has no per-frame video-to-source mapping. Each 2D observation must
therefore provide valid `camera`, `video_frame`, `source_frame_index`, and
`timestamp` values. `video_frame == source_frame_index` is never assumed.
`timebase_status` and `frame_mapping_status` are separate manifest fields.
Missing or conflicting mappings produce `mapping_unverified` and make the
camera ineligible; no maximum row index or video length is used as a fallback.

2D and 3D timestamps use different origins. Cross-table alignment uses
`source_frame_index` and `source_step`; timestamps remain diagnostic evidence.

## Camera recommendation

An explicit primary camera always wins if it is eligible. An explicitly
requested but missing or ineligible camera does not fall back.

Without an explicit camera, the adapter audits every official camera. A camera
is eligible only when its video and timebase are valid and its 2D rows contain
finite `[21, 2]` keypoints with legal explicit mappings. The recommendation
score combines minimum left/right coverage, both-hands coverage, valid-joint
ratio, and timeline-match ratio. Only a unique best eligible camera outside the
configured tie tolerance becomes primary; camera name ordering never breaks a
tie.

The manifest records the complete per-camera audit, best and second-best
scores, recommendation reason, and an immutable camera-selection configuration
snapshot. The built-in versioned defaults are schema-validated and can be
overridden explicitly. Failure to choose a qualified camera produces
`primary_camera=null`, `adapter_status=input_missing`, and
`reason=primary_camera_missing`.

## Shared hand-pose session

A QY hand-pose session reads `observations_2d.parquet` and
`trajectory_3d.parquet` at most once per asset. The precheck adapter indexes 3D
rows by `(source_step, hand)` and materializes the selected authoritative source
range as a dense tensor. Missing source steps remain `NaN` with validity
`False`; no interpolation or synthesized points are allowed.

Rows with fewer than 21 points, invalid shape, NaN, infinity, duplicate
hand-step identity, or missing required hand data are hard invalid. Empty or
all-null 3D input produces `skeleton_3d_status=no_valid_output`, zero valid rows,
and required-skeleton failure while supplier evaluation continues independent
modules.

Every valid 3D row also requires an official, non-empty `reference_camera`.
The reference camera must remain stable across the episode. Any valid row
outside the authoritative source range, an unknown/missing reference camera,
or reference-camera drift produces `input_invalid`/`mapping_invalid`; rows are
never silently trimmed. A corrupt Parquet is isolated to its episode and
becomes an auditable invalid manifest row instead of aborting discovery of the
remaining episodes.

The supplier has not yet provided a confirmed anatomical index order for the
21-point arrays. The adapter therefore does not apply the JD/DR MANO remap or
claim acceptance joint names. `joint_topology_status=unverified` is preserved.
Morphology returns review/no-valid-output. Temporal displacement and
acceleration may run on stable opaque indices because those metrics are
permutation-invariant; joint-angle, bone, rotation, and anatomical-orientation
metrics remain explicitly uncalibrated. The coordinate-system JSON is checked
for readable object structure, while its field schema remains recorded as
unverified; the externally confirmed camera-frame/meters contract plus a stable
row-level reference camera is required before metric temporal runs.

Supplier `quality_tier`, `trajectory_quality`, and reprojection errors are
preserved as `supplier_quality_signal` diagnostics only. They do not create a
second hard-pass policy.

## Unified precheck and SAM3

The existing `PrecheckSession` remains the owner of the five cheap modules and
loads one QY clip per asset. QY source files participate in producer
fingerprints. Required-hand invalidity is normalized at the supplier boundary
without changing the DR rule.

SAM3 remains a separate producer. For QY it consumes direct 2D points for the
chosen official camera and resolves each requested source frame to the explicit
video frame. Missing/ambiguous primary-camera 2D mappings block with a precise
reason before model loading. It never projects QY 3D or falls back to another
camera. Until the anatomical order is confirmed, SAM3 uses opaque
`qy_<hand>_joint_00..20` labels and point-only overlays; it does not draw
inferred finger bones or apply anatomical wrist labels. Containment ratios are
unchanged because they are invariant to point ordering. Candidate, verdict,
artifact, producer-version, and source-frame semantics remain unchanged.

## Validation

Tests cover valid left/right 2D and 3D, separate multi-camera/trajectory axes,
explicit frame mapping, single-hand and sparse 3D, empty/all-null input,
nonfinite and malformed keypoints, duplicate hand-step rows, auxiliary supplier
quality, out-of-range rows, reference-camera mapping, corrupt-source isolation,
required-file type checks, camera recommendation/ties/overrides, symlink-safe
identities, topology-agnostic temporal metrics, direct 2D SAM3 selection and
blocking, plus JD and DR regressions.
