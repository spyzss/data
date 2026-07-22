from pathlib import Path
import subprocess
import sys


DOCS = (
    Path("docs/PRD-asset-qc-pipeline.md"),
    Path("docs/PRD-qc-gated-json.md"),
    Path("docs/PRD-qc-unified-config.md"),
    Path("docs/asset-qc-json-format.md"),
    Path("docs/reviewer-guide.md"),
    Path("ACCEPTANCE.md"),
    Path("WORKFLOW_INTERFACE.md"),
    Path("docs/qc-dataflow-migration.md"),
)


def test_reviewer_docs_name_v2_profiles_and_single_source() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    for token in (
        "asset_qc_report.v2",
        "qc_acceptance_v2.1.0",
        "acceptance",
        "supplier_evaluation",
        "quality_archive/*.json",
        "sidecar 只作证据",
    ):
        assert token in combined


def test_docs_do_not_advertise_pass_sample_review() -> None:
    text = Path("ACCEPTANCE.md").read_text(encoding="utf-8")
    assert "正常 Pass 样本抽检" not in text
    assert "仅累计 warn 进入人工质检" in text


def test_warn_review_docs_publish_the_warn_first_operator_contract() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    for token in (
        "自动/SAM3 → Warn 人工复核 → 语义",
        "manual_review.issue_reviews",
        "manual_review.reviews[]",
        "Publisher audit",
        "not_required",
        "all_reviewed",
        "early_fail",
        "未查看的 Warn 保持原状",
        "最早的待复核 Warn",
        "Fail 只在“完成复核”时提交",
        "[start_frame, end_frame_exclusive)",
        "end_frame_exclusive - 1",
        "legacy freeze",
        "整条视频",
        "时间轴色块",
        "popover",
        "可拖动的播放针",
        "跳转到起始帧",
        "正文不显示检测分数或阈值字段",
        "色块仅显示颜色",
        "不堆叠色块",
        "仅在问题帧区间实时 overlay",
        "pending/failed 只锁定对应 Warn",
        "0.25×、0.5×、1×、1.5×、2×、3×",
        "人工原因可预选、多选和取消",
        "Other 为必填",
        "8897",
        "8898",
        "--sam3-model",
    ):
        assert token in combined

    for obsolete in (
        "自动 QC → 语义校准 → Warn 复核",
        "semantic_consistency (external)\n-> no candidate_issue_ids -> manual_review",
        "获取编辑锁",
    ):
        assert obsolete not in combined
    reviewer = Path("docs/reviewer-guide.md").read_text(encoding="utf-8")
    assert "短片" not in reviewer


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
