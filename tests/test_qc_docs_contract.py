from pathlib import Path
import subprocess
import sys


PRD = Path("docs/PRD-asset-qc-pipeline.md")
REVIEWER_GUIDE = Path("docs/reviewer-guide.md")
JSON_FORMAT = Path("docs/asset-qc-json-format.md")
ACCEPTANCE = Path("ACCEPTANCE.md")
WORKFLOW = Path("WORKFLOW_INTERFACE.md")
DATAFLOW_MIGRATION = Path("docs/qc-dataflow-migration.md")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_reviewer_docs_name_v2_profiles_and_single_source() -> None:
    assert "asset_qc_report.v2" in _text(JSON_FORMAT)
    assert "qc_acceptance_v2.1.0" in _text(Path("docs/PRD-qc-gated-json.md"))
    assert "acceptance" in _text(PRD)
    assert "supplier_evaluation" in _text(PRD)
    assert "quality_archive/*.json" in _text(ACCEPTANCE)
    assert "sidecar 只作证据" in _text(WORKFLOW)


def test_docs_do_not_advertise_pass_sample_review() -> None:
    text = _text(ACCEPTANCE)
    assert "正常 Pass 样本抽检" not in text
    assert "仅累计 warn 进入人工质检" in text


def _assert_warn_gate_contract(text: str) -> None:
    assert "`not_required`：`manual_review.state=not_required`，不写 `completion_mode`；Warn Gate 已满足，语义为 ready。" in text
    assert "`all_reviewed`：`manual_review.state=completed` 且 `completion_mode=all_reviewed`；Warn Gate 已满足，语义为 ready。" in text
    assert "`early_fail`：`manual_review.state=completed` 且 `completion_mode=early_fail`；资产 `pipeline_state.status=stopped`，`semantic_calibration.state=skipped_due_to_fail`。" in text
    assert "点击 Fail 立即写入当前 issue 的 `manual_review.issue_reviews[issue_id].verdict=fail`。" in text
    assert "人工原因预选、多选、取消或填写 Other 只更新本地草稿，不改变 issue verdict 或资产状态。" in text
    assert "点击“完成复核”才写资产级 `completion_mode=early_fail`，停止资产并自动跳转下一条。" in text
    assert "not_required 并完成为 pass" not in text
    assert "Fail 只在“完成复核”时提交" not in text
    assert "检测分数" not in text


def test_prd_publishes_precise_warn_gate_and_fail_contract() -> None:
    text = _text(PRD)
    _assert_warn_gate_contract(text)
    assert "自动/SAM3 → Warn 人工复核 → 语义" in text
    assert "manual_review.reviews[]" in text
    assert "正文不显示机器指标或阈值字段" in text


def test_reviewer_guide_publishes_precise_warn_gate_and_fail_contract() -> None:
    text = _text(REVIEWER_GUIDE)
    _assert_warn_gate_contract(text)
    assert "整条视频" in text
    assert "popover" in text
    assert "短片" not in text


def test_acceptance_publishes_precise_warn_gate_and_fail_contract() -> None:
    text = _text(ACCEPTANCE)
    _assert_warn_gate_contract(text)
    assert "manual_review=all_reviewed or not_required -> semantic_consistency external" in text
    assert "8897" in text
    assert "8898" in text


def test_workflow_interface_publishes_precise_warn_gate_and_fail_contract() -> None:
    text = _text(WORKFLOW)
    _assert_warn_gate_contract(text)
    assert "Lease acquisition is automatic." in text


def test_json_format_publishes_precise_warn_gate_and_fail_contract() -> None:
    text = _text(JSON_FORMAT)
    _assert_warn_gate_contract(text)
    assert "manual_review.issue_reviews" in text
    assert "manual_review.reviews[]" in text
    assert "当前代码已实现 `video_quality` 的写入合同；其余模块按" not in text


def test_dataflow_migration_does_not_publish_the_obsolete_semantic_first_route() -> None:
    text = _text(DATAFLOW_MIGRATION)
    assert "Warn 人工复核在语义校准之前" in text
    assert "语义校准是人工质检之前的 external 阶段" not in text


def test_warn_service_help_describes_required_inputs_and_optional_sam3() -> None:
    completed = subprocess.run(
        [sys.executable, "tools/serve_human_qc_workbench.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    help_text = completed.stdout
    for token in (
        "Warn 人工复核",
        "--batch-root",
        "--quality-archive",
        "--reviewer",
        "--sam3-model",
        "8897",
    ):
        assert token in help_text
    assert "获取编辑锁" not in help_text
