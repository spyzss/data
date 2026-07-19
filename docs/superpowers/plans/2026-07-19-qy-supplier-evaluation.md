# QY Supplier Evaluation Implementation Plan

1. Add synthetic QY episode fixtures and failing adapter tests for discovery,
   authoritative timebase parsing, 2D/3D validation, explicit frame mapping,
   and camera recommendation.
2. Implement `qingyu.py` manifest discovery and `qingyu_hand_pose.py` shared
   immutable load session; expose QY through the supplier manifest CLI and
   supplier identity normalizer.
3. Add failing unified precheck tests for one-load reuse, sparse/no-valid 3D,
   hard presence invalidity, source-frame lineage, and fingerprints; implement
   the isolated QY loader route.
4. Add failing QY SAM3 tests for direct primary-camera 2D lookup, source-to-video
   mapping, no fallback, and lazy blocked gates; implement the dedicated QY
   path without changing standalone/JD/DR behavior.
5. Extend optional QY camera-selection config schema and manifest/source-path
   plumbing, then run QY targeted tests, existing JD/DR regressions, QC smoke,
   compileall, full relevant pytest, and `git diff --check`.
6. Harden the adapter after read-only review: preserve unverified 21-point
   topology, run only topology-invariant temporal metrics, use point-only SAM3
   labels, validate stable reference-camera lineage, reject out-of-range 3D,
   isolate corrupt Parquet per episode, and require regular-file inventory.

No stage, commit, push, real cloud scan, or real full supplier run is part of
this plan.
