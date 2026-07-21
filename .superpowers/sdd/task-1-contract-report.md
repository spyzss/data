# Task 1 Report: Warn Early-Fail Report Contract

## Status

DONE_WITH_CONCERNS. Commit: `3a52716bda630f5e2740dba80b19aaafba8f8654`.

## Implementation

- Added `manual_review.completion_mode` validation for `all_reviewed` and `early_fail`, while keeping historical completed blocks without this field readable.
- Added manual `failure_reason` validation, unique non-empty reason codes and non-blank `other_text` when `other` is selected.
- Enforced `all_reviewed` coverage and at least one Fail for `early_fail`.
- Updated v1-to-v2 migration to add the new fields only to fully reviewed historical completed reports.
- Added `qc_common.projection.project_manual_review_counts(...)` returning reviewed, confirmed-Fail and unreviewed-selected counts.

## RED

Command: `uv run pytest -q tests/test_human_qc_report_schema.py tests/test_human_qc_aggregation.py tests/test_qc_migration_reconciliation.py`.

- Initial collection failed because `project_manual_review_counts` did not exist.
- After adding the minimal projection interface: `2 failed, 46 passed`; early-fail was rejected by the old all-selected requirement and migration emitted no completion mode.
- Focused migration test then failed because an incomplete historical completed report was not readable.

## GREEN

Command: `uv run pytest -q tests/test_human_qc_report_schema.py tests/test_human_qc_aggregation.py tests/test_qc_migration_reconciliation.py`.

Result: `48 passed in 0.65s`. `git diff --check` was clean before commit.

## Changed files

- `qc_common/schema.py`
- `qc_common/report_migration.py`
- `qc_common/projection.py`
- `tests/test_human_qc_report_schema.py`
- `tests/test_human_qc_aggregation.py`
- `tests/test_qc_migration_reconciliation.py`

## Concern

The existing batch-wide aggregator is `qc_reporting/aggregate.py`, outside this task's allowed modification list. This commit exposes the required stable counts from `qc_common/projection.py` but intentionally does not rewire the existing aggregate output.

---

# Task 1 Contract Repair: Completion-Mode Invariants

## Status

READY_TO_COMMIT. Repair commit: `fix(human-qc): enforce completion mode invariants`.

## Fresh RED

Command: `.venv/bin/pytest -q tests/test_human_qc_report_schema.py tests/test_qc_migration_reconciliation.py tests/test_report_mutation.py`.

Result: `6 failed, 62 passed`. The failures proved that `all_reviewed` accepted a Fail, legacy completed reports skipped existing structure checks, completed v2 reports without canonical fields were writable, and fully reviewed Pass+Fail v1 history migrated to `all_reviewed`. The subsequent parameterized write-boundary test also failed as expected: `2 failed`.

## GREEN

Required Task 1 command: `.venv/bin/pytest -q tests/test_human_qc_report_schema.py tests/test_human_qc_aggregation.py tests/test_qc_migration_reconciliation.py`.

Result: `55 passed in 0.58s`.

Additional write-path regression: `.venv/bin/pytest -q tests/test_report_mutation.py`.

Result: `21 passed in 0.53s`. `git diff --check` is clean.

## Changed files

- `qc_common/schema.py`
- `qc_common/report_migration.py`
- `qc_common/report.py`
- `tests/test_human_qc_report_schema.py`
- `tests/test_qc_migration_reconciliation.py`
- `.superpowers/sdd/task-1-contract-report.md`

## Self-review

- `all_reviewed` now requires coverage of every selected issue and only Pass verdicts.
- Fully reviewed historical completed reports migrate to `all_reviewed` only when all selected verdicts are Pass; any selected Fail gives `early_fail`.
- Historical completed reports missing `completion_mode` remain readable, but candidate subset, selected-current, review verdict, and `completed_at` validation now still runs.
- The writer rejects completed v2 manual-review blocks that omit either `completion_mode` or `failure_reason`, keeping legacy compatibility read-only.
- No projection or batch-aggregator code changed.

## Concern

The current `human_qc/warn_service.py` finalization path does not yet populate the two canonical completion fields. It will therefore be rejected by the deliberately strict write boundary until its owning task is updated; this repair was constrained not to modify that service.
