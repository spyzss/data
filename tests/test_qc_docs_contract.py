from pathlib import Path


DOCS = (
    Path("docs/PRD-qc-gated-json.md"),
    Path("docs/PRD-qc-unified-config.md"),
    Path("docs/asset-qc-json-format.md"),
    Path("ACCEPTANCE.md"),
    Path("WORKFLOW_INTERFACE.md"),
    Path("docs/qc-dataflow-migration.md"),
)


def test_reviewer_docs_name_v2_profiles_and_single_source() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    for token in (
        "asset_qc_report.v2",
        "qc_acceptance_v2.0.0",
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
