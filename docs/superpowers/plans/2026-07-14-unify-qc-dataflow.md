---
change: unify-qc-dataflow
design-doc: docs/superpowers/specs/2026-07-14-unify-qc-dataflow-design.md
base-ref: 44ee01f5221a00343e48046bfb477889a82a68cc
---

# 统一 QC 数据流实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把仓库中已合入的 Precheck、视频质量、SAM3 与正式批次报表统一到版本化 Config 驱动、单资产 QC JSON 写回、双 execution profile 控制的可恢复数据流。

**Architecture:** 保留现有检测算法和证据 sidecar，在 `qc_pipeline/adapters/` 将其结果转换为统一 `ModuleResult`；由 `qc_common/report_mutation.py` 独占单资产报告的模块归并、revision 和原子提交；由资产级 orchestrator 依配置顺序运行并根据 profile 决定 fail 截断或继续。正式人工队列和批次输出只投影 `quality_archive/*.json`，CSV、XLSX、Markdown 与 Parquet 都是可重建派生物。

**Tech Stack:** Python 3.11、dataclasses/typing、PyYAML、jsonschema Draft 2020-12、pytest、pandas/pyarrow、openpyxl、h5py、OpenCV。

## Global Constraints

- 本计划只实现 OpenSpec change `unify-qc-dataflow`；人工语义时间轴和 warn Pass/Fail 工作台属于 `add-human-semantic-warn-review`。
- 发布 `qc_acceptance_config_schema.v2`、`qc_acceptance_v2.0.0` 与 `asset_qc_report.v2`；`configs/qc_acceptance/qc_acceptance_v1.1.0.yaml` 的 SHA-256 必须保持 `0ef58453f381651711ec84490975a1af5e274a179647f6ea2cc83e42404ea41d`。
- 活跃入口 `configs/qc_acceptance.yaml` 必须与 `configs/qc_acceptance/qc_acceptance_v2.0.0.yaml` 字节一致。
- `overall_decision` 在流程未完成时只能为 `null`，完成后只能为 `pass` 或 `fail`；运行错误不能转换为质量结论。
- `acceptance` 的自动 hard fail 必须截断并跳过语义与人工阶段；`supplier_evaluation` 保留相同机器 fail 但继续执行。
- 自动 warn 必须累计到稳定去重的候选集合；所有自动检查 Pass 且候选为空时，不生成 warn 人工队列。
- `duplicate_check`、`content_validity`、`effective_duration` 在 v2.0.0 中必须 `enabled: false` 且 `disabled_reason: no_registered_implementation`，不能写成 Pass。
- `semantic_consistency`、`manual_review` 必须声明为 `execution_kind: external`；自动 orchestrator 到达外部阶段时写 `awaiting_external` 并暂停。
- v2 报告对人工语义扩展必须保持结构开放且原样保留，不得把一次语义修订限制为“单片段独立编辑”；后续工作台可用一个 revision 原子记录共享边界调整所影响的相邻两段。
- external-stage 语义 payload 的帧边界必须可原样承载严格递增半开区间 `[b_i, b_{i+1})`；统一数据流不得把内部边界误写成闭区间 end，也不得在投影时丢失 `UI end = b_{i+1} - 1` 的 off-by-one 合同。
- 同一资产模块串行写回且每次校验 expected revision；不同资产可由批次调度并行，不能共享可变报告对象。
- evidence 路径必须相对批次根目录，禁止 `..` 逃逸；大体量逐帧、mask、视频和 overlay 只保留 sidecar 引用。
- 正式批次决策与统计只遍历 `quality_archive/*.json`；sidecar 仅用于算法回归、证据展示和迁移对账。
- 每项任务遵循 TDD：先添加精确失败测试，确认失败原因，再写最小实现、运行相关回归并单独提交。

---

## 文件结构与职责

| 文件 | 职责 |
|---|---|
| `schemas/qc_acceptance_config.v2.schema.json` | v2 Config 的 profile、模块注册、外部/禁用状态和规则结构 |
| `configs/qc_acceptance/qc_acceptance_v2.0.0.yaml` | 不可变规则、顺序、阈值与 profile 快照 |
| `configs/qc_acceptance.yaml` | 与 v2.0.0 快照字节一致的活跃入口 |
| `schemas/asset_qc_report.v2.schema.json` | 通用模块 flow、runtime error、evidence、pipeline state 与二元结论 |
| `qc_common/contracts.py` | `ModuleResult`、`Issue`、`EvidenceRef`、稳定 ID 与枚举 |
| `qc_common/report.py` | 版本感知读取、Schema 校验、expected revision 与原子替换 |
| `qc_common/report_migration.py` | v1 到 v2 的纯函数迁移 |
| `qc_common/report_mutation.py` | 模块所有权归并、候选重建、状态写回与 revision 事务 |
| `qc_common/module_registry.py` | 自动 runner 注册与 enabled/unavailable 校验 |
| `qc_pipeline/context.py` | 不可变 `AssetContext`、源文件与批次路径边界 |
| `qc_pipeline/orchestrator.py` | 配置顺序、恢复、双 profile、external/disabled/error 状态机 |
| `qc_pipeline/adapters/precheck.py` | 五个 Precheck PRD 模块的确定性映射 |
| `qc_pipeline/adapters/video_quality.py` | batch 与 manifest range 视频结果的同一合同映射 |
| `qc_pipeline/adapters/sam3_containment.py` | window summary、overlay 与 containment evidence 映射 |
| `qc_reporting/projection.py` | 报告目录到 asset/issue/execution 三类规范化行 |
| `qc_reporting/aggregate.py` | profile 分组的资产/issue/人工/覆盖率统计 |
| `qc_reporting/cache.py` | 带 report revision/hash 清单的可删除缓存 |
| `tools/run_qc_pipeline.py` | 单资产/批次统一编排 CLI |
| `tools/build_qc_json_projection.py` | QC JSON 到 CSV/Parquet/Markdown/XLSX 的正式入口 |

## OpenSpec 任务覆盖索引

| OpenSpec | 本计划任务 |
|---|---|
| 1.1 | 1、2 |
| 1.2 | 3 |
| 1.3 | 4 |
| 2.1–2.6 | 5–10 |
| 3.1–3.4 | 11–13 |
| 4.1–4.4 | 14–17 |
| 5.1–5.4 | 18–21 |

### Task 1: 发布统一 Config v2 与严格加载规则

**Files:**
- Create: `schemas/qc_acceptance_config.v2.schema.json`
- Create: `configs/qc_acceptance/qc_acceptance_v2.0.0.yaml`
- Modify: `configs/qc_acceptance.yaml`
- Modify: `qc_common/schema.py:24-29`
- Modify: `qc_common/config.py:18-94`
- Create: `tests/test_qc_config_v2.py`
- Modify: `tests/test_qc_config.py:6-28`

**Interfaces:**
- Consumes: `load_qc_acceptance_config(path: Path | None = None)` 的现有调用方式。
- Produces: `LoadedQcConfig.pipeline_modules: tuple[str, ...]`、`LoadedQcConfig.default_profile: str`、`LoadedQcConfig.execution_profile(name: str) -> dict[str, str]`、`LoadedQcConfig.module_config(name: str) -> dict[str, Any]`、`LoadedQcConfig.assert_same_reference(reference: Mapping[str, str]) -> None`。
- Produces: `validate_qc_config(data: dict[str, Any])` 按 `schema_version` 选择 v1/v2 Schema，未知版本抛 `ValueError`。

- [x] **Step 1: 添加会失败的 v2 Config 合同测试**

```python
# tests/test_qc_config_v2.py
from pathlib import Path
import copy
import hashlib
import yaml
import pytest

from qc_common.config import load_qc_acceptance_config

V1_SHA256 = "0ef58453f381651711ec84490975a1af5e274a179647f6ea2cc83e42404ea41d"

def test_default_config_is_v2_snapshot_with_two_profiles() -> None:
    loaded = load_qc_acceptance_config()
    assert loaded.schema_version == "qc_acceptance_config_schema.v2"
    assert loaded.config_version == "qc_acceptance_v2.0.0"
    assert loaded.default_profile == "acceptance"
    assert loaded.execution_profile("acceptance")["fail_action"] == "stop"
    assert loaded.execution_profile("supplier_evaluation")["fail_action"] == "record_and_continue"
    assert loaded.module_config("semantic_consistency")["execution_kind"] == "external"
    assert loaded.module_config("duplicate_check") == {
        **loaded.module_config("duplicate_check"),
        "enabled": False,
        "disabled_reason": "no_registered_implementation",
    }

def test_active_config_matches_immutable_v2_snapshot() -> None:
    assert Path("configs/qc_acceptance.yaml").read_bytes() == Path(
        "configs/qc_acceptance/qc_acceptance_v2.0.0.yaml"
    ).read_bytes()
    assert hashlib.sha256(Path("configs/qc_acceptance/qc_acceptance_v1.1.0.yaml").read_bytes()).hexdigest() == V1_SHA256

def test_v2_rejects_enabled_module_without_implementation(tmp_path: Path) -> None:
    raw = copy.deepcopy(load_qc_acceptance_config().raw)
    raw["modules"]["duplicate_check"].update({"enabled": True})
    raw["modules"]["duplicate_check"].pop("implementation", None)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="enabled module duplicate_check"):
        load_qc_acceptance_config(path)
```

- [x] **Step 2: 运行测试并确认因 v2 文件/属性缺失而失败**

Run: `.venv/bin/python -m pytest tests/test_qc_config_v2.py tests/test_qc_config.py -q`

Expected: FAIL，首个失败包含 `qc_acceptance_config_schema.v1 != qc_acceptance_config_schema.v2` 或缺少 `default_profile`。

- [x] **Step 3: 发布 Config v2 并扩展 loader**

Config v2 顶层必须包含以下精确结构；现有各模块阈值和 rule 映射原值迁入相应 `parameters`/`rules`，不得改变数值：

```yaml
schema_version: qc_acceptance_config_schema.v2
config_version: qc_acceptance_v2.0.0
config_name: acceptance_gate
execution_profiles:
  acceptance: {fail_action: stop, runtime_error_action: stop_incomplete}
  supplier_evaluation: {fail_action: record_and_continue, runtime_error_action: stop_incomplete}
pipeline:
  default_profile: acceptance
  modules: [hdf5_text_info, quality_hand, keypoint_presence, keypoint_morphology, keypoint_temporal, video_quality, sam3_containment, semantic_consistency, manual_review, duplicate_check, content_validity, effective_duration]
modules:
  hdf5_text_info: {enabled: true, implementation: precheck.hdf5_text_info, parameters: {}, rules: {}}
  quality_hand: {enabled: true, implementation: precheck.quality_hand, parameters: {}, rules: {}}
  keypoint_presence: {enabled: true, implementation: precheck.keypoint_presence, parameters: {}, rules: {}}
  keypoint_morphology: {enabled: true, implementation: precheck.keypoint_morphology, parameters: {}, rules: {}}
  keypoint_temporal: {enabled: true, implementation: precheck.keypoint_temporal, parameters: {}, rules: {}}
  video_quality: {enabled: true, implementation: video_quality.unified, parameters: {}, rules: {}}
  sam3_containment: {enabled: true, implementation: sam3_containment.manifest, parameters: {}, rules: {}}
  semantic_consistency: {enabled: true, execution_kind: external, parameters: {}, rules: {}}
  manual_review: {enabled: true, execution_kind: external, parameters: {}, rules: {}}
  duplicate_check: {enabled: false, disabled_reason: no_registered_implementation, parameters: {}, rules: {}}
  content_validity: {enabled: false, disabled_reason: no_registered_implementation, parameters: {}, rules: {}}
  effective_duration: {enabled: false, disabled_reason: no_registered_implementation, parameters: {}, rules: {}}
```

`qc_common/config.py` 增加的核心实现：

```python
@property
def pipeline_modules(self) -> tuple[str, ...]:
    return tuple(str(name) for name in self.raw["pipeline"]["modules"])

@property
def default_profile(self) -> str:
    return str(self.raw["pipeline"]["default_profile"])

def execution_profile(self, name: str) -> dict[str, str]:
    try:
        return copy.deepcopy(self.raw["execution_profiles"][name])
    except KeyError as exc:
        raise ValueError(f"unknown execution profile: {name}") from exc

def module_config(self, name: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(self.raw["modules"][name])
    except KeyError as exc:
        raise ValueError(f"unknown pipeline module: {name}") from exc

def assert_same_reference(self, reference: Mapping[str, str]) -> None:
    expected = self.json_reference()
    for key in ("schema_version", "config_version", "config_hash"):
        if reference.get(key) != expected[key]:
            raise ValueError(f"QC config drift at {key}: {reference.get(key)} != {expected[key]}")
```

Loader 的语义校验必须逐项执行：pipeline 模块存在、rule ID 非空且全局唯一、profile 动作属于 Schema 枚举、enabled 模块必须二选一拥有 `implementation` 或 `execution_kind: external`、disabled 模块必须有 `disabled_reason`。默认入口加载时还要比较活跃文件与 `configs/qc_acceptance/<config_version>.yaml` 的字节。

- [x] **Step 4: 验证 Config 与现有视频阈值回归**

Run: `.venv/bin/python -m pytest tests/test_qc_config.py tests/test_qc_config_v2.py tests/test_acceptance_video_quality.py::test_default_video_quality_config_comes_from_unified_config tests/test_acceptance_video_quality.py::test_unified_video_threshold_override_changes_runtime_config -q`

Expected: PASS；v1 快照 hash 不变、活跃入口等于 v2 快照、视频阈值回归通过。

- [x] **Step 5: 提交**

```bash
git add schemas/qc_acceptance_config.v2.schema.json configs/qc_acceptance.yaml configs/qc_acceptance/qc_acceptance_v2.0.0.yaml qc_common/schema.py qc_common/config.py tests/test_qc_config.py tests/test_qc_config_v2.py
git commit -m "feat(qc): publish unified config v2"
```

### Task 2: 发布 Asset QC Report v2 Schema 与只读迁移器

**Files:**
- Create: `schemas/asset_qc_report.v2.schema.json`
- Create: `qc_common/report_migration.py`
- Modify: `qc_common/schema.py:32-37`
- Modify: `qc_common/report.py:16-35`
- Create: `tests/qc_report_fixtures.py`
- Create: `tests/test_asset_qc_schema_v2.py`
- Modify: `tests/test_asset_qc_schema.py`

**Interfaces:**
- Consumes: `load_asset_qc_report(path: Path) -> dict[str, Any] | None`。
- Produces: `validate_asset_qc_report(report: dict[str, Any]) -> None` 按报告自身版本选 Schema。
- Produces: `migrate_v1_to_v2(report: Mapping[str, Any], *, config_reference: Mapping[str, str], profile: str = "acceptance") -> dict[str, Any]`，纯函数、不写盘、不修改输入。
- Produces: `load_asset_qc_report(path, *, migrate_to_v2=False, config_reference=None, profile="acceptance")`；仅显式请求时投影 v1。

- [x] **Step 1: 添加 v2 状态约束和 v1 迁移失败测试**

```python
# tests/test_asset_qc_schema_v2.py
import copy
import pytest
from qc_common.report_migration import migrate_v1_to_v2
from qc_common.schema import validate_asset_qc_report
from tests.qc_report_fixtures import make_v1_video_report, make_v2_report

@pytest.mark.parametrize("status", ["pending", "running", "awaiting_external", "error"])
def test_unfinished_v2_report_requires_null_decision(status: str) -> None:
    report = make_v2_report(status=status, overall_decision=None)
    validate_asset_qc_report(report)
    report["overall_decision"] = "pass"
    with pytest.raises(ValueError, match="overall_decision"):
        validate_asset_qc_report(report)

def test_error_cannot_be_quality_fail() -> None:
    report = make_v2_report(status="error", overall_decision="fail")
    with pytest.raises(ValueError, match="overall_decision"):
        validate_asset_qc_report(report)

def test_migrate_v1_preserves_video_unknown_fields_and_revision() -> None:
    old = make_v1_video_report()
    old["extension_from_colleague"] = {"keep": True}
    frozen = copy.deepcopy(old)
    migrated = migrate_v1_to_v2(old, config_reference=old["qc_config"])
    assert old == frozen
    assert migrated["schema_version"] == "asset_qc_report.v2"
    assert migrated["report_revision"] == old["report_revision"]
    assert migrated["video_quality"] == old["video_quality"]
    assert migrated["extension_from_colleague"] == {"keep": True}
```

- [x] **Step 2: 运行测试并确认 v2 Schema/迁移模块不存在**

Run: `.venv/bin/python -m pytest tests/test_asset_qc_schema_v2.py tests/test_asset_qc_schema.py -q`

Expected: collection FAIL，包含 `No module named 'qc_common.report_migration'`。

- [x] **Step 3: 实现通用 Schema 和纯函数迁移**

v2 Schema 必须要求：`schema_version`、`asset_id`、`report_revision`、`qc_config`、`execution`、`pipeline_state`、`overall_decision`、`source_files`、`issues`、`runtime_errors`、`manual_review`；允许已登记 module block 与未知扩展字段共存。`manual_review` 及未来人工语义扩展不得用 Schema 固化为“一次只修改一个片段”，并必须允许后续 change 在一次 revision 中原子保存受共享边界影响的相邻两段 before/after。关键条件使用 `allOf/if/then`：

```json
{
  "properties": {
    "schema_version": {"const": "asset_qc_report.v2"},
    "overall_decision": {"enum": ["pass", "fail", null]},
    "pipeline_state": {
      "required": ["status", "last_completed_module", "next_module", "stop_reason"],
      "properties": {"status": {"enum": ["pending", "running", "awaiting_external", "stopped", "completed", "error"]}}
    }
  },
  "allOf": [
    {"if": {"properties": {"pipeline_state": {"properties": {"status": {"enum": ["pending", "running", "awaiting_external", "error"]}}}}}, "then": {"properties": {"overall_decision": {"type": "null"}}}},
    {"if": {"properties": {"pipeline_state": {"properties": {"status": {"const": "completed"}}}}, "then": {"properties": {"overall_decision": {"enum": ["pass", "fail"]}}},
    {"if": {"properties": {"pipeline_state": {"properties": {"status": {"const": "error"}}}}, "then": {"properties": {"overall_decision": {"type": "null"}}}
  ]
}
```

迁移器以深拷贝为基础，补齐以下精确字段并保留旧 module/unknown fields：

```python
def migrate_v1_to_v2(report, *, config_reference, profile="acceptance"):
    if report.get("schema_version") == "asset_qc_report.v2":
        return copy.deepcopy(dict(report))
    if report.get("schema_version") != "asset_qc_report.v1":
        raise ValueError(f"unsupported asset QC schema: {report.get('schema_version')}")
    migrated = copy.deepcopy(dict(report))
    migrated["schema_version"] = "asset_qc_report.v2"
    migrated["qc_config"] = dict(config_reference)
    migrated.setdefault("execution", {"profile": profile, "started_at": None, "updated_at": None})
    migrated.setdefault("source_files", {})
    migrated.setdefault("issues", [])
    migrated.setdefault("runtime_errors", [])
    manual = migrated.setdefault("manual_review", {})
    manual.setdefault("state", "not_evaluated")
    manual.setdefault("candidate_issue_ids", [])
    manual.setdefault("failures_for_batch_stats_issue_ids", [])
    migrated["pipeline_state"].setdefault("stop_reason", None)
    return migrated
```

- [x] **Step 4: 验证 v1/v2 双读和原 v1 回归**

Run: `.venv/bin/python -m pytest tests/test_asset_qc_schema.py tests/test_asset_qc_schema_v2.py tests/test_acceptance_video_quality.py::test_video_qc_preserves_existing_module_blocks_and_increments_revision -q`

Expected: PASS；读取 v1 不改盘，迁移结果通过 v2 Schema，原视频测试仍可读 v1 fixture。

- [x] **Step 5: 提交**

```bash
git add schemas/asset_qc_report.v2.schema.json qc_common/schema.py qc_common/report.py qc_common/report_migration.py tests/qc_report_fixtures.py tests/test_asset_qc_schema.py tests/test_asset_qc_schema_v2.py
git commit -m "feat(qc): add asset report v2 contract"
```

### Task 3: 定义统一 ModuleResult、Issue、EvidenceRef 与稳定 ID

**Files:**
- Create: `qc_common/contracts.py`
- Modify: `qc_common/__init__.py`
- Create: `tests/test_qc_contracts.py`

**Interfaces:**
- Produces: `Verdict = Literal["pass", "warn", "fail", "skipped"]`。
- Produces: frozen dataclasses `Issue`、`EvidenceRef`、`ModuleResult`，均提供 `to_dict() -> dict[str, Any]`。
- Produces: `build_issue_id(*, asset_id, module, rule_id, source_relative_path, coordinate_system, start_frame, end_frame, hand_side, evidence_kind) -> str`。
- Produces: `relative_evidence_path(path: Path, batch_root: Path) -> str`，路径逃逸抛 `ValueError`。

- [x] **Step 1: 添加稳定性、敏感性和路径边界测试**

```python
# tests/test_qc_contracts.py
from pathlib import Path
import pytest
from qc_common.contracts import build_issue_id, relative_evidence_path

BASE = dict(asset_id="asset-1", module="keypoint_temporal", rule_id="keypoint_temporal.strong_temporal_failure", source_relative_path="video/a.mp4", coordinate_system="source_inclusive", start_frame=10, end_frame=20, hand_side="left", evidence_kind="clip")

def test_issue_id_is_repeatable_and_reason_independent() -> None:
    first = build_issue_id(**BASE)
    second = build_issue_id(**dict(reversed(list(BASE.items()))))
    assert first == second
    assert first.startswith("keypoint_temporal:strong_temporal_failure:")
    assert len(first.rsplit(":", 1)[1]) == 20

def test_issue_id_changes_with_frame_or_hand() -> None:
    assert build_issue_id(**BASE) != build_issue_id(**{**BASE, "end_frame": 21})
    assert build_issue_id(**BASE) != build_issue_id(**{**BASE, "hand_side": "right"})

def test_evidence_path_cannot_escape_batch(tmp_path: Path) -> None:
    inside = tmp_path / "evidence" / "a.png"
    inside.parent.mkdir(); inside.write_bytes(b"x")
    assert relative_evidence_path(inside, tmp_path) == "evidence/a.png"
    with pytest.raises(ValueError, match="outside batch root"):
        relative_evidence_path(tmp_path.parent / "secret.png", tmp_path)
```

- [x] **Step 2: 运行测试并确认合同模块缺失**

Run: `.venv/bin/python -m pytest tests/test_qc_contracts.py -q`

Expected: collection FAIL，包含 `No module named 'qc_common.contracts'`。

- [x] **Step 3: 实现 frozen 合同与规范化哈希**

```python
Verdict = Literal["pass", "warn", "fail", "skipped"]

@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    kind: str
    path: str
    coordinate_system: str
    start_frame: int | None = None
    end_frame: int | None = None
    hand_side: str | None = None
    checksum: str | None = None
    mime_type: str | None = None
    generator_version: str | None = None

@dataclass(frozen=True)
class Issue:
    issue_id: str
    code: str
    severity: Literal["warn", "fail"]
    module: str
    issue_type: str
    metric: str
    observed_value: Any
    operator: str
    boundary_value: Any
    rule_id: str
    needs_manual_review: bool
    context: Mapping[str, Any] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()

@dataclass(frozen=True)
class ModuleResult:
    module: str
    verdict: Verdict
    evaluation: Mapping[str, Any]
    metrics: Mapping[str, Any]
    issues: tuple[Issue, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    runtime: Mapping[str, Any] = field(default_factory=dict)
```

`build_issue_id` 使用 `json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` 后 SHA-256，rule name 取 `rule_id.rsplit(".", 1)[-1]`，仅截取 20 个 hex。`to_dict()` 必须深度转为 JSON-safe 原生类型并保持 tuple 顺序；不得将 reason、时间戳或格式化浮点字符串加入 identity。

- [x] **Step 4: 验证合同序列化和路径规则**

Run: `.venv/bin/python -m pytest tests/test_qc_contracts.py -q`

Expected: PASS，稳定 ID 重跑一致且坐标/手侧变化会改变 ID。

- [x] **Step 5: 提交**

```bash
git add qc_common/contracts.py qc_common/__init__.py tests/test_qc_contracts.py
git commit -m "feat(qc): define module result contracts"
```

### Task 4: 实现模块所有权感知的 revision 报告事务

**Files:**
- Create: `qc_pipeline/context.py`
- Create: `qc_common/report_mutation.py`
- Modify: `qc_common/report.py:25-51`
- Create: `tests/test_report_mutation.py`

**Interfaces:**
- Consumes: Task 1 `LoadedQcConfig`、Task 2 `migrate_v1_to_v2`、Task 3 `ModuleResult`。
- Produces: frozen `AssetContext(asset_id: str, batch_root: Path, report_path: Path, source_files: Mapping[str, Any], source_range: tuple[int, int] | None = None, metadata: Mapping[str, Any] = field(default_factory=dict))`；构造时验证 report/evidence 根目录边界。
- Produces: `initialize_v2_report(context: AssetContext, config: LoadedQcConfig, profile: str, now: str) -> dict[str, Any]`。
- Produces: `apply_module_result(path: Path, *, context: AssetContext, config: LoadedQcConfig, profile: str, result: ModuleResult, expected_revision: int, next_module: str | None, now: str) -> dict[str, Any]`。
- Produces: `write_pipeline_transition(path, *, expected_revision, module, state, next_module, stop_reason, overall_decision, now) -> dict[str, Any]`。
- Produces exceptions: `ModuleOrderError`、`ConfigDriftError`，并复用 `StaleReportRevisionError`。

- [x] **Step 1: 添加所有权、重跑去重、revision 和原子失败测试**

```python
# tests/test_report_mutation.py
import copy, json
from pathlib import Path
import pytest
from qc_common.contracts import Issue, ModuleResult
from qc_common.report import StaleReportRevisionError
from qc_common.report_mutation import apply_module_result
from tests.qc_report_fixtures import make_asset_context, loaded_test_config

def test_module_rerun_replaces_only_owned_block_and_rebuilds_candidates(tmp_path: Path) -> None:
    context = make_asset_context(tmp_path, "a")
    config = loaded_test_config()
    issue = Issue("keypoint_temporal:jump:11111111111111111111", "jump", "warn", "keypoint_temporal", "temporal_jump", "joint_displacement_m_max", 0.2, ">", 0.05, "keypoint_temporal.strong_temporal_failure", True, {"start_frame": 4, "end_frame": 8})
    first = apply_module_result(context.report_path, context=context, config=config, profile="acceptance", result=ModuleResult("keypoint_temporal", "warn", {}, {"run": 1}, (issue,)), expected_revision=0, next_module="video_quality", now="2026-07-14T00:00:00Z")
    first["extension"] = {"keep": True}
    context.report_path.write_text(json.dumps(first), encoding="utf-8")
    second = apply_module_result(context.report_path, context=context, config=config, profile="acceptance", result=ModuleResult("keypoint_temporal", "pass", {}, {"run": 2}), expected_revision=1, next_module="video_quality", now="2026-07-14T00:01:00Z")
    assert second["extension"] == {"keep": True}
    assert second["keypoint_temporal"]["metrics"] == {"run": 2}
    assert second["issues"] == []
    assert second["manual_review"]["candidate_issue_ids"] == []

def test_stale_revision_never_changes_file(tmp_path: Path) -> None:
    context = make_asset_context(tmp_path, "a")
    config = loaded_test_config()
    apply_module_result(context.report_path, context=context, config=config, profile="acceptance", result=ModuleResult("hdf5_text_info", "pass", {}, {}), expected_revision=0, next_module="quality_hand", now="2026-07-14T00:00:00Z")
    before = context.report_path.read_bytes()
    with pytest.raises(StaleReportRevisionError):
        apply_module_result(context.report_path, context=context, config=config, profile="acceptance", result=ModuleResult("hdf5_text_info", "pass", {}, {}), expected_revision=0, next_module="quality_hand", now="2026-07-14T00:00:01Z")
    assert context.report_path.read_bytes() == before
```

- [x] **Step 2: 运行测试并确认事务模块不存在**

Run: `.venv/bin/python -m pytest tests/test_report_mutation.py -q`

Expected: collection FAIL，包含 `No module named 'qc_common.report_mutation'`。

- [x] **Step 3: 实现固定九步事务算法**

实现顺序必须与设计一致：load/init → v1 纯迁移 → revision/asset/config/next_module 校验 → 删除本模块旧 block/issues/evidence → 写新 block → 全量重建 candidate/fail IDs → 计算 profile 对应 exit gate → 仅终态计算 decision → revision+1/schema validate/atomic replace。

模块 block 的精确公共结构：

```python
module_block = {
    "flow": {
        "entry_gate": {"state": "ready", "eligible": True, "blocked_by_module": None, "required_inputs": [], "missing_inputs": [], "upstream_continue": True},
        "result_gate": {"verdict": result.verdict, "has_fail": result.verdict == "fail", "has_warn": result.verdict == "warn"},
        "exit_gate": {"state": exit_state, "continue_to_next_module": continue_to_next, "next_module": next_module},
    },
    "evaluation": copy.deepcopy(dict(result.evaluation)),
    "metrics": copy.deepcopy(dict(result.metrics)),
    "evidence": [item.to_dict() for item in result.evidence],
    "runtime": copy.deepcopy(dict(result.runtime)),
}
```

候选重建必须从 `report["issues"]` 计算，而非追加：

```python
report["manual_review"]["candidate_issue_ids"] = sorted({i["issue_id"] for i in report["issues"] if i["severity"] == "warn" and i["needs_manual_review"]})
report["manual_review"]["failures_for_batch_stats_issue_ids"] = sorted({i["issue_id"] for i in report["issues"] if i["severity"] == "fail"})
```

`qc_common/report.py` 在 `os.replace` 后打开父目录并 `os.fsync`；Schema 失败、Config drift、模块顺序错误、stale revision 均发生在替换之前。

- [x] **Step 4: 运行事务、Schema 与现有原子写回回归**

Run: `.venv/bin/python -m pytest tests/test_report_mutation.py tests/test_asset_qc_schema_v2.py tests/test_acceptance_video_quality.py::test_video_qc_preserves_existing_module_blocks_and_increments_revision -q`

Expected: PASS；重跑不累积 issue，未知字段保留，失败写入不改变原文件。

- [x] **Step 5: 提交**

```bash
git add qc_pipeline/context.py qc_common/report.py qc_common/report_mutation.py tests/test_report_mutation.py
git commit -m "feat(qc): add revision aware report mutation"
```

### Task 5: 适配 text integrity 与 quality_hand

**Files:**
- Create: `qc_pipeline/__init__.py`
- Create: `qc_pipeline/adapters/__init__.py`
- Create: `qc_pipeline/adapters/precheck.py`
- Modify: `tools/run_manifest_precheck.py:298-330`
- Create: `tests/test_precheck_qc_adapter.py`

**Interfaces:**
- Consumes: `adapt_precheck_results` 的 `CheckResult` 与 source-coordinate candidate rows；规则来自 `LoadedQcConfig.module_rules()`。
- Produces: `adapt_hdf5_text_info(*, asset_id: str, source_relative_path: str, results: Sequence[CheckResult], config: LoadedQcConfig) -> ModuleResult`。
- Produces: `adapt_quality_hand(*, asset_id: str, source_relative_path: str, results: Sequence[CheckResult], config: LoadedQcConfig) -> ModuleResult`。
- Produces: `precheck_config_from_unified(config: LoadedQcConfig, *, module_names: Sequence[str], output_dir: Path) -> PrecheckConfig`，把 v2 parameters 逐字段注入现有检查类。

- [x] **Step 1: 添加 pass/fail/warn golden 映射测试**

```python
# tests/test_precheck_qc_adapter.py
from qc_common.types import CheckResult
from qc_pipeline.adapters.precheck import adapt_hdf5_text_info, adapt_quality_hand
from tests.qc_report_fixtures import loaded_test_config

def test_text_integrity_missing_field_maps_to_hard_fail() -> None:
    result = adapt_hdf5_text_info(asset_id="a", source_relative_path="hdf5/a.h5", results=[CheckResult("text_integrity", 0, -1, {"missing_field_count": 1.0, "field_present_task": 0.0}, True, '{"missing_fields":["task"]}')], config=loaded_test_config())
    assert result.module == "hdf5_text_info"
    assert result.verdict == "fail"
    assert result.issues[0].rule_id == "hdf5_text.missing_required_field"
    assert result.issues[0].needs_manual_review is False

def test_quality_hand_single_side_low_maps_to_warn() -> None:
    rows = [CheckResult("quality_score", 0, 3, {"frame_score": 0.0, "quality_left": 0.0, "quality_right": 1.0}, None, "per-frame"), CheckResult("quality_score", 0, -1, {"pass_ratio": 0.5, "num_frames": 1.0}, False, "summary")]
    result = adapt_quality_hand(asset_id="a", source_relative_path="hdf5/a.h5", results=rows, config=loaded_test_config())
    assert result.verdict == "warn"
    assert result.issues[0].context["hand_side"] == "left"
    assert result.issues[0].context["start_frame"] == 3
```

- [x] **Step 2: 运行测试并确认 adapter 缺失**

Run: `.venv/bin/python -m pytest tests/test_precheck_qc_adapter.py -q`

Expected: collection FAIL，包含 `No module named 'qc_pipeline'`。

- [x] **Step 3: 实现两个确定性 adapter**

```python
def _worst(*verdicts: Verdict) -> Verdict:
    return max(verdicts, key={"skipped": -1, "pass": 0, "warn": 1, "fail": 2}.__getitem__)

def adapt_hdf5_text_info(*, asset_id, source_relative_path, results, config):
    summary = _summary(results, "text_integrity")
    missing = int(summary.metrics.get("missing_field_count", 0))
    invalid_json = "not valid JSON" in summary.reason or "no text_label" in summary.reason
    verdict = "fail" if missing or invalid_json else "pass"
    issues = () if verdict == "pass" else (_issue_from_row(asset_id=asset_id, module="hdf5_text_info", rule_id="hdf5_text.missing_required_field" if missing else "hdf5_text.missing_text_field", row=summary, source_relative_path=source_relative_path, severity="fail", needs_manual_review=False),)
    return ModuleResult("hdf5_text_info", verdict, {"decision": verdict, "reason": summary.reason}, dict(summary.metrics), issues)
```

`adapt_quality_hand` 必须逐帧识别 left/right 的 0 值：单手异常生成 warn、同帧双手异常生成 fail；invalid shape/value 按 v2 Config rule 生成 fail；`quality_hand is None` 映射为 `skipped` 且 `evaluation.reason=source_signal_not_provided`，不能映射成 Pass。

`tools/run_manifest_precheck._configured_precheck()` 必须停止读取本地默认阈值作为正式来源，改为调用 `precheck_config_from_unified()`；原 `--config-path` 只保留为遗留算法回归模式，并与正式 `--qc-config` 互斥。测试逐字段断言 quality、presence、morphology、temporal 的 runtime config 等于 v2 Config parameters。

- [x] **Step 4: 运行 adapter 与原检查回归**

Run: `.venv/bin/python -m pytest tests/test_precheck_qc_adapter.py tests/test_qc_modules_smoke.py -q`

Expected: PASS；原 CheckResult 行为不变，adapter 只解释输出。

- [x] **Step 5: 提交**

```bash
git add qc_pipeline/__init__.py qc_pipeline/adapters/__init__.py qc_pipeline/adapters/precheck.py tools/run_manifest_precheck.py tests/test_precheck_qc_adapter.py
git commit -m "feat(qc): adapt text and hand quality modules"
```

### Task 6: 适配 keypoint presence 与稳定帧区间

**Files:**
- Modify: `qc_pipeline/adapters/precheck.py`
- Modify: `tests/test_precheck_qc_adapter.py`

**Interfaces:**
- Consumes: `keypoint_missing` 和 `skeleton_quality_score` 行；source frame 已由 `tools/run_manifest_precheck._result_records()` 映射。
- Produces: `adapt_keypoint_presence(*, asset_id, source_relative_path, results, config) -> ModuleResult`。
- Produces: `_contiguous_ranges(frames: Sequence[int]) -> tuple[tuple[int, int], ...]`，闭区间合并。

- [x] **Step 1: 添加 NaN/Inf、连续缺失和稳定 ID 测试**

```python
def test_presence_merges_contiguous_invalid_frames_into_one_issue() -> None:
    rows = [CheckResult("skeleton_quality_score", 0, frame, {"keypoint_presence_invalid": 1.0, "valid_keypoint_count_left": 7.0, "valid_keypoint_count_right": 21.0}, True, "presence invalid") for frame in (10, 11, 12)]
    result = adapt_keypoint_presence(asset_id="a", source_relative_path="hdf5/a.h5", results=rows, config=loaded_test_config())
    assert result.verdict == "fail"
    assert len(result.issues) == 1
    assert result.issues[0].context == {"coordinate_system": "source_inclusive", "start_frame": 10, "end_frame": 12, "hand_side": "left"}
    assert adapt_keypoint_presence(asset_id="a", source_relative_path="hdf5/a.h5", results=rows, config=loaded_test_config()).issues[0].issue_id == result.issues[0].issue_id
```

- [x] **Step 2: 运行单测并确认函数缺失**

Run: `.venv/bin/python -m pytest tests/test_precheck_qc_adapter.py::test_presence_merges_contiguous_invalid_frames_into_one_issue -q`

Expected: FAIL，包含 `cannot import name 'adapt_keypoint_presence'`。

- [x] **Step 3: 实现 presence 规则映射与区间压缩**

```python
def _contiguous_ranges(frames):
    ranges: list[list[int]] = []
    for frame in sorted(set(int(value) for value in frames)):
        if not ranges or frame > ranges[-1][1] + 1:
            ranges.append([frame, frame])
        else:
            ranges[-1][1] = frame
    return tuple((start, end) for start, end in ranges)

def adapt_keypoint_presence(*, asset_id, source_relative_path, results, config):
    rows = [row for row in results if row.check in {"keypoint_missing", "skeleton_quality_score"} and row.frame_idx >= 0]
    invalid = [row for row in rows if bool(row.metrics.get("keypoint_presence_invalid")) or row.flag is True]
    # 每个 hand_side 分别按 source frame 合并区间；valid count < fail threshold 或 NaN/Inf => fail，
    # missing ratio 介于 warn/fail 边界 => warn；每个区间只产生一个稳定 issue。
```

`evaluation` 保存 `checked_frame_count`、`invalid_frame_count`、`invalid_frame_ratio`；`metrics` 保存左右手最小有效点数和连续区间。缺少整个 keypoint 字段使用 `keypoint_presence.missing_keypoint_field` fail；单纯没有供应商 `quality_hand` 不得被当作 keypoint 缺失。

- [x] **Step 4: 运行 presence 与 manifest 坐标回归**

Run: `.venv/bin/python -m pytest tests/test_precheck_qc_adapter.py tests/test_manifest_precheck_runner.py::test_manifest_precheck_outputs_source_frame_mapping_and_candidate_windows -q`

Expected: PASS；issue 使用 source-inclusive 坐标，区间不重复偏移。

- [x] **Step 5: 提交**

```bash
git add qc_pipeline/adapters/precheck.py tests/test_precheck_qc_adapter.py
git commit -m "feat(qc): adapt keypoint presence results"
```

### Task 7: 适配 keypoint morphology、metrics 与 evidence

**Files:**
- Modify: `qc_pipeline/adapters/precheck.py`
- Modify: `tests/test_precheck_qc_adapter.py`
- Modify: `tests/test_keypoint_morphology.py`

**Interfaces:**
- Consumes: `keypoint_morphology` frame/summary rows，summary 的 `morphology_verdict` 为 `pass|review|fail|not_applicable`。
- Produces: `adapt_keypoint_morphology(*, asset_id, source_relative_path, results, config) -> ModuleResult`。

- [x] **Step 1: 添加 review→warn、fail 和 not_applicable 测试**

```python
def test_morphology_review_maps_to_warn_with_frame_evidence() -> None:
    rows = [CheckResult("keypoint_morphology", 0, 5, {"morphology_verdict": "review", "which_thresholds_exceeded": ["left:bone_length_ratio_spread_review"], "left_bone_length_ratio_spread": 4.0}, None, "review"), CheckResult("keypoint_morphology", 0, -1, {"morphology_verdict": "review", "num_frames": 1}, None, "summary")]
    result = adapt_keypoint_morphology(asset_id="a", source_relative_path="hdf5/a.h5", results=rows, config=loaded_test_config())
    assert result.verdict == "warn"
    assert result.issues[0].rule_id == "keypoint_morphology.bone_length_ratio_spread"
    assert result.issues[0].context["start_frame"] == 5

def test_morphology_not_applicable_is_skipped_not_pass() -> None:
    rows = [CheckResult("keypoint_morphology", 0, -1, {"morphology_verdict": "not_applicable", "num_frames": 2}, None, "skipped_due_to_existence_invalid")]
    assert adapt_keypoint_morphology(asset_id="a", source_relative_path="hdf5/a.h5", results=rows, config=loaded_test_config()).verdict == "skipped"
```

- [x] **Step 2: 运行单测并确认 morphology adapter 缺失**

Run: `.venv/bin/python -m pytest tests/test_precheck_qc_adapter.py -k morphology -q`

Expected: FAIL，包含 `cannot import name 'adapt_keypoint_morphology'`。

- [x] **Step 3: 实现规则名解析和 evidence**

```python
MORPHOLOGY_REASON_TO_RULE = {
    "palm_scale_too_small": "keypoint_morphology.palm_scale_too_small",
    "bone_length_ratio_spread": "keypoint_morphology.bone_length_ratio_spread",
    "max_normalized_bone_length": "keypoint_morphology.max_normalized_bone_length",
    "zero_length_bone_count": "keypoint_morphology.zero_length_bone_count",
    "duplicate_joint_pair_count": "keypoint_morphology.duplicate_joint_pair_count",
    "collapsed_finger_count": "keypoint_morphology.collapsed_finger_count",
    "joint_angle_min_deg": "keypoint_morphology.joint_angle_min_deg",
    "joint_angle_violation_fraction": "keypoint_morphology.joint_angle_violation_fraction",
}
```

解析 `left:<metric>_review`、`right:<metric>_fail` 时剥离 side 与 verdict 后缀；同 rule/side 的连续 frame 合并。review 生成 warn 且 `needs_manual_review=True`，fail 生成 hard fail；`EvidenceRef.kind="frame_metrics"`、path 指向相对化 `check_results.json`，并记录 source-inclusive start/end/hand。

- [x] **Step 4: 运行 morphology adapter 与算法回归**

Run: `.venv/bin/python -m pytest tests/test_precheck_qc_adapter.py -k morphology tests/test_keypoint_morphology.py -q`

Expected: PASS；现有形态指标与阈值数值不变。

- [x] **Step 5: 提交**

```bash
git add qc_pipeline/adapters/precheck.py tests/test_precheck_qc_adapter.py tests/test_keypoint_morphology.py
git commit -m "feat(qc): adapt keypoint morphology results"
```

### Task 8: 适配 keypoint temporal 与人工候选 issue

**Files:**
- Modify: `qc_pipeline/adapters/precheck.py`
- Modify: `tests/test_precheck_qc_adapter.py`
- Modify: `tests/test_manifest_precheck_runner.py:85-142`

**Interfaces:**
- Consumes: `keypoint_temporal`、`skeleton_quality_score` 行及 `_map_candidate_window_to_source()` 产生的 candidate rows。
- Produces: `adapt_keypoint_temporal(*, asset_id, source_relative_path, results, candidate_windows, config) -> ModuleResult`。

- [x] **Step 1: 添加候选窗口与强时序 fail 的 golden 测试**

```python
def test_temporal_candidate_is_one_warn_issue_with_source_range() -> None:
    result = adapt_keypoint_temporal(
        asset_id="a", source_relative_path="hdf5/a.h5", results=[],
        candidate_windows=[{"asset_id": "a", "start_frame": 30, "end_frame": 42, "coordinate_space": "source", "hand_side": "both", "trigger_metrics": {"joint_displacement_m_max": 0.08}}],
        config=loaded_test_config(),
    )
    assert result.verdict == "warn"
    assert result.issues[0].context["start_frame"] == 30
    assert result.issues[0].context["end_frame"] == 42
    assert result.issues[0].needs_manual_review is True

def test_strong_temporal_failure_remains_hard_fail() -> None:
    rows = [CheckResult("skeleton_quality_score", 0, 9, {"skeleton_verdict": "suspect", "which_thresholds_exceeded": ["joint_acceleration_m_s2_max", "joint_displacement_m_max", "rotation_delta_max"]}, True, "threshold exceeded")]
    assert adapt_keypoint_temporal(asset_id="a", source_relative_path="hdf5/a.h5", results=rows, candidate_windows=[], config=loaded_test_config()).verdict == "fail"
```

- [x] **Step 2: 运行测试并确认函数缺失**

Run: `.venv/bin/python -m pytest tests/test_precheck_qc_adapter.py -k temporal -q`

Expected: FAIL，包含 `cannot import name 'adapt_keypoint_temporal'`。

- [x] **Step 3: 实现 temporal 映射**

```python
TEMPORAL_RULES = {
    "candidate": "keypoint_temporal.composite_frame_verdict",
    "skeleton_review": "keypoint_temporal.skeleton_quality_score",
    "projection": "keypoint_temporal.projection_review",
    "strong": "keypoint_temporal.strong_temporal_failure",
}
```

三个及以上核心 temporal metric 同帧超界映射 `strong` hard fail；projection/side-view/candidate window 映射 warn。每个 candidate window 创建一个稳定 issue，identity 使用 source path、source-inclusive start/end、hand side 与 `evidence_kind="candidate_window"`；`metrics` 保存 peak trigger metrics、候选数量和异常帧 union 数。`composite_frame_verdict` 只作为 evidence，不创建 module block。

- [x] **Step 4: 运行 temporal、坐标和候选回归**

Run: `.venv/bin/python -m pytest tests/test_precheck_qc_adapter.py -k temporal tests/test_manifest_precheck_runner.py::test_manifest_precheck_outputs_source_frame_mapping_and_candidate_windows tests/test_qc_modules_smoke.py -q`

Expected: PASS；候选只生成 `keypoint_temporal` issue，source 坐标只转换一次。

- [x] **Step 5: 提交**

```bash
git add qc_pipeline/adapters/precheck.py tests/test_precheck_qc_adapter.py tests/test_manifest_precheck_runner.py
git commit -m "feat(qc): adapt temporal candidate windows"
```

### Task 9: 统一 batch 与 manifest range 视频写回

**Files:**
- Create: `qc_pipeline/adapters/video_quality.py`
- Modify: `acceptance_pull/video_quality.py:2240-2397`
- Modify: `tools/run_manifest_video_quality.py:165-369`
- Create: `tests/test_video_quality_qc_adapter.py`
- Modify: `tests/test_acceptance_video_quality.py:885-1068`
- Modify: `tests/test_manifest_video_quality_runner.py:39-233`

**Interfaces:**
- Consumes: `VideoQualityResult` 与 `asset_qc_result_to_json()` 的现有 module payload。
- Produces: `adapt_video_quality_result(*, result: VideoQualityResult, config: LoadedQcConfig, batch_root: Path, source_range: tuple[int, int] | None = None) -> ModuleResult`。
- Produces: `write_video_quality_result(*, context: AssetContext, result: VideoQualityResult, config: LoadedQcConfig, profile: str, expected_revision: int, next_module: str) -> dict[str, Any]`。
- `write_video_quality_result` 只能接收已经按配置推进到 `pipeline_state.next_module == video_quality` 的报告；standalone runner 不得用 entry-module 后门绕过前置自动模块。没有前置报告时可以继续生成算法 sidecar，但必须返回结构化 prerequisite 状态且不得创建误导性的主报告。

- [x] **Step 1: 添加 batch/range 同合同与 revision 写回测试**

```python
def test_batch_and_range_video_use_same_module_shape(video_result, loaded_v2_config, tmp_path) -> None:
    batch = adapt_video_quality_result(result=video_result, config=loaded_v2_config, batch_root=tmp_path)
    ranged = adapt_video_quality_result(result=video_result, config=loaded_v2_config, batch_root=tmp_path, source_range=(20, 40))
    assert set(batch.to_dict()) == set(ranged.to_dict())
    assert batch.module == ranged.module == "video_quality"
    assert ranged.issues[0].context["coordinate_system"] == "source_video_inclusive"

def test_manifest_video_writes_pre_advanced_quality_archive(tmp_path: Path) -> None:
    advance_report_to_video_quality(tmp_path, asset_id="logical-a")
    summary = run_manifest_video_quality(manifest, tmp_path / "run", batch_root=tmp_path, profile="acceptance")
    report = json.loads((tmp_path / "quality_archive" / "logical-a.json").read_text())
    assert summary["completed_clip_count"] == 1
    assert report["schema_version"] == "asset_qc_report.v2"
    assert report["video_quality"]["flow"]["result_gate"]["verdict"] in {"pass", "warn", "fail"}

def test_fresh_manifest_video_never_bypasses_predecessors(tmp_path: Path) -> None:
    summary = run_manifest_video_quality(manifest, tmp_path / "run", batch_root=tmp_path, profile="acceptance")
    assert summary["pipeline_prerequisite_count"] == 1
    assert not (tmp_path / "quality_archive" / "logical-a.json").exists()
```

- [x] **Step 2: 运行测试并确认 manifest 未写主报告**

Run: `.venv/bin/python -m pytest tests/test_video_quality_qc_adapter.py tests/test_manifest_video_quality_runner.py -q`

Expected: FAIL，统一 adapter/写回接口或 prerequisite 状态尚不存在；测试不得通过放宽 fresh-report 模块顺序来变绿。

- [x] **Step 3: 将两条路径收敛到 adapter + report mutation**

```python
def write_video_quality_result(*, context, result, config, profile, expected_revision, next_module):
    module_result = adapt_video_quality_result(
        result=result,
        config=config,
        batch_root=context.batch_root,
        source_range=context.source_range,
    )
    return apply_module_result(
        context.report_path, context=context, config=config, profile=profile,
        result=module_result, expected_revision=expected_revision,
        next_module=next_module, now=utc_now(),
    )
```

删除 `_merge_video_quality_report()` 的正式写回职责；保留 `asset_qc_result_to_json()` 作为 v1 兼容/算法测试辅助。新 batch writer 和 manifest writer 在报告已经合法推进到 video 模块时必须调用上述接口；fresh standalone 运行只生成 sidecar 并记录结构化 prerequisite，等待 Task 11 orchestrator 提供前置状态。Manifest CLI 新增 `--batch-root`、`--profile`、`--config`，skip 判断在存在 QC JSON 时检查其中的 video module 与 source range，而不是仅看 sidecar 是否存在。

- [x] **Step 4: 运行视频算法、两类 runner 与报告事务回归**

Run: `.venv/bin/python -m pytest tests/test_video_quality_qc_adapter.py tests/test_acceptance_video_quality.py tests/test_manifest_video_quality_runner.py tests/test_report_mutation.py -q`

Expected: PASS；算法指标不变，合法推进的两类 runner 都通过 v2 mutation 写同一 module block；fresh standalone 不绕过模块顺序且不生成主报告。

- [x] **Step 5: 提交**

```bash
git add qc_pipeline/adapters/video_quality.py acceptance_pull/video_quality.py tools/run_manifest_video_quality.py tests/test_video_quality_qc_adapter.py tests/test_acceptance_video_quality.py tests/test_manifest_video_quality_runner.py
git commit -m "feat(qc): unify video quality report writes"
```

### Task 10: 适配 SAM3 window summary、overlay 与 evidence

**Files:**
- Create: `qc_pipeline/adapters/sam3_containment.py`
- Modify: `tools/run_manifest_sam3_containment.py:475-906`
- Modify: `configs/qc_acceptance/qc_acceptance_v2.0.0.yaml`
- Modify: `configs/qc_acceptance.yaml`
- Create: `tests/test_sam3_qc_adapter.py`
- Modify: `tests/test_manifest_sam3_containment_runner.py:184-345`
- Modify: `tests/test_qc_config_v2.py`

**Interfaces:**
- Produces: `adapt_sam3_containment(*, asset_id, batch_root, window_summaries, evidence_rows, config) -> ModuleResult`。
- Produces: `write_sam3_asset_result(*, context, window_summaries, evidence_rows, config, profile, expected_revision, next_module) -> dict[str, Any]`。

- [x] **Step 1: 添加 window verdict、相对 evidence 与缺失 sidecar 测试**

```python
def test_sam3_adapter_maps_fail_and_overlay_reference(tmp_path: Path) -> None:
    overlay = tmp_path / "sam3" / "combined_overlays" / "a_10.png"
    overlay.parent.mkdir(parents=True); overlay.write_bytes(b"png")
    result = adapt_sam3_containment(
        asset_id="a", batch_root=tmp_path,
        window_summaries=[{"asset_id": "a", "window_start_frame": 10, "window_end_frame": 20, "hand_side": "left", "containment_verdict": "strong_containment_mismatch", "inside_ratio": 0.1}],
        evidence_rows=[{"asset_id": "a", "window_start_frame": 10, "window_end_frame": 20, "hand_side": "both", "evidence_type": "combined_overlay", "source_path": str(overlay)}],
        config=loaded_test_config(),
    )
    assert result.verdict == "fail"
    assert result.evidence[0].path == "sam3/combined_overlays/a_10.png"
    assert result.issues[0].evidence_ids == (result.evidence[0].evidence_id,)

def test_missing_overlay_is_integrity_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evidence file does not exist"):
        adapt_sam3_containment(asset_id="a", batch_root=tmp_path, window_summaries=[], evidence_rows=[{"asset_id": "a", "evidence_type": "combined_overlay", "source_path": str(tmp_path / "missing.png")}], config=loaded_test_config())
```

- [x] **Step 2: 运行测试并确认 adapter 缺失**

Run: `.venv/bin/python -m pytest tests/test_sam3_qc_adapter.py -q`

Expected: collection FAIL，包含 `No module named 'qc_pipeline.adapters.sam3_containment'`。

- [x] **Step 3: 实现 SAM3 映射并按资产写回**

```python
SAM3_VERDICT = {
    "pass": ("pass", None),
    "acceptable": ("pass", None),
    "side_view_manual_review": ("warn", "sam3_containment.side_view_manual_review"),
    "projection_review": ("warn", "sam3_containment.projection_review"),
    "strong_containment_mismatch": ("fail", "sam3_containment.strong_containment_mismatch"),
}
```

每个 window/hand 生成最多一个 issue，context 保存 source-inclusive window；evidence ID 由 asset/window/hand/kind/path 的稳定 JSON 哈希得到。Runner 完成 sidecar 后按 asset 分组调用 writer；任一窗口异常导致该资产运行错误，不提交伪造 module success。阈值由统一 Config 注入 `FRAME_THRESHOLDS`/`WINDOW_THRESHOLDS`，测试固定现有有效数值。

实际 producer 的结构化字段是 `window_containment_verdict`，adapter 必须对 `containment_fail`、`acceptable_flagged`、`projection_review`、`side_view_manual_review`、`rotation_manual_review`、`mixed_review` 和 `review` 做显式 canonical 映射，禁止从文件名或 reason 推断。若 v2.0.0 快照遗漏 legacy frame/window 参数或数值与当前算法常量不一致，必须在合并发布前校正 v2 快照和字节一致的 active config，并用回归逐字段证明注入值等于现有有效算法值；不得用 hardcoded fallback 掩盖 Config 漂移，历史 v1 快照仍不可改。

- [x] **Step 4: 运行 SAM3 adapter、overlay 与原算法回归**

Run: `.venv/bin/python -m pytest tests/test_sam3_qc_adapter.py tests/test_manifest_sam3_containment_runner.py tests/test_sam3_keypoint_containment.py -q`

Expected: PASS；overlay 仍生成，主报告只保存相对引用和汇总结论。

- [x] **Step 5: 提交**

```bash
git add qc_pipeline/adapters/sam3_containment.py tools/run_manifest_sam3_containment.py tests/test_sam3_qc_adapter.py tests/test_manifest_sam3_containment_runner.py
git commit -m "feat(qc): adapt sam3 containment evidence"
```

### Task 11: 建立 registry、AssetContext 与可恢复资产 orchestrator

**Files:**
- Create: `qc_common/module_registry.py`
- Modify: `qc_pipeline/context.py`
- Create: `qc_pipeline/orchestrator.py`
- Create: `tools/run_qc_pipeline.py`
- Create: `tests/test_qc_orchestrator.py`

**Interfaces:**
- Produces: frozen `AssetContext(asset_id: str, batch_root: Path, report_path: Path, source_files: Mapping[str, Any], source_range: tuple[int, int] | None, metadata: Mapping[str, Any])`。
- Produces: `ModuleRunner = Callable[[AssetContext, LoadedQcConfig], ModuleResult]`。
- Produces: `ModuleRegistry.register(name: str, runner: ModuleRunner) -> None`、`resolve(name) -> ModuleRunner`、`has(name) -> bool`。
- Produces: `build_default_registry(context: AssetContext, config: LoadedQcConfig, *, segmenter_factory: Callable | None = None) -> ModuleRegistry`，注册五个 precheck、统一 video 与 SAM3 runner；每个资产 worker 构建独立 registry。
- Produces: `run_asset(context: AssetContext, *, config: LoadedQcConfig, profile: str, registry: ModuleRegistry, now: Callable[[], str] = utc_now) -> RunOutcome`。

- [x] **Step 1: 添加顺序、恢复、external pause 和 Config drift 测试**

```python
def test_orchestrator_resumes_from_next_module(tmp_path: Path) -> None:
    calls: list[str] = []
    registry = stub_registry(calls, {"hdf5_text_info": "pass", "quality_hand": "pass"})
    first = run_asset(context(tmp_path), config=config_with_modules(["hdf5_text_info", "quality_hand", "semantic_consistency"]), profile="acceptance", registry=registry)
    assert calls == ["hdf5_text_info", "quality_hand"]
    assert first.report["pipeline_state"]["status"] == "awaiting_external"
    assert first.report["pipeline_state"]["next_module"] == "semantic_consistency"
    calls.clear()
    second = run_asset(context(tmp_path), config=first.config, profile="acceptance", registry=registry)
    assert calls == []
    assert second.report["report_revision"] == first.report["report_revision"]

def test_asset_context_rejects_report_outside_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="report_path must be inside batch_root"):
        AssetContext("a", tmp_path / "batch", tmp_path / "outside.json", {}, None, {})
```

- [x] **Step 2: 运行测试并确认 orchestrator/registry 缺失**

Run: `.venv/bin/python -m pytest tests/test_qc_orchestrator.py -q`

Expected: collection FAIL，包含 `No module named 'qc_pipeline.orchestrator'`。

- [x] **Step 3: 实现 registry 与恢复循环**

```python
def run_asset(context, *, config, profile, registry, now=utc_now):
    config.execution_profile(profile)
    report = load_or_initialize(context, config, profile, now())
    config.assert_same_reference(report["qc_config"])
    start = config.pipeline_modules.index(report["pipeline_state"]["next_module"])
    executed: list[str] = []
    for module_name in config.pipeline_modules[start:]:
        module_config = config.module_config(module_name)
        if not module_config["enabled"]:
            report = record_disabled_transition(...)
            continue
        if module_config.get("execution_kind") == "external":
            report = record_awaiting_external(...)
            return RunOutcome(report, tuple(executed), "awaiting_external")
        runner = registry.resolve(module_name)
        result = runner(context, config)
        report = apply_module_result(...)
        executed.append(module_name)
        if report["pipeline_state"]["status"] in {"stopped", "error"}:
            break
    return RunOutcome(report, tuple(executed), report["pipeline_state"]["status"])
```

CLI 必须接受 `--batch-root`、`--manifest`、`--profile {acceptance,supplier_evaluation}`、`--config`、`--max-workers`、`--resume/--no-resume`。批次层只用 `ThreadPoolExecutor` 调度不同 asset_id；先拒绝 manifest 中重复 asset_id，再为每个 worker 创建独立 context/runner，不共享报告 dict。

`build_default_registry()` 的注册必须是实际 detector 调用，不是 sidecar reader：precheck runner 由 `precheck_config_from_unified()` 构造并将 CheckResult 交给 Tasks 5–8 adapter；video runner 调用 `analyze_video`/`analyze_video_frame_range` 后交给 Task 9 adapter；SAM3 runner 调用 manifest containment producer 后交给 Task 10 adapter。runner 只返回 `ModuleResult`，不得自行决定 next module 或 overall decision。

- [x] **Step 4: 运行 orchestrator、Config 与 mutation 测试**

Run: `.venv/bin/python -m pytest tests/test_qc_orchestrator.py tests/test_qc_config_v2.py tests/test_report_mutation.py -q`

Expected: PASS；恢复点来自 `next_module`，external 阶段不会被自动执行。

- [x] **Step 5: 提交**

```bash
git add qc_common/module_registry.py qc_pipeline/context.py qc_pipeline/orchestrator.py tools/run_qc_pipeline.py tests/test_qc_orchestrator.py
git commit -m "feat(qc): add resumable asset orchestrator"
```

### Task 12: 实现 acceptance 截断与 supplier evaluation 继续策略

**Files:**
- Modify: `qc_pipeline/orchestrator.py`
- Modify: `qc_common/report_mutation.py`
- Modify: `tests/test_qc_orchestrator.py`

**Interfaces:**
- Consumes: Task 11 `run_asset()`。
- Produces: 相同 `ModuleResult.verdict="fail"` 在 acceptance 生成 `exit_gate.state="stop_qc"`，在 supplier evaluation 生成 `exit_gate.state="continue"` 与 `continued_after_fail=True`。
- Produces: `mark_remaining_skipped_due_to_fail(report, modules, *, failed_module) -> dict[str, Any]`。

- [x] **Step 1: 添加同输入双 profile 状态轨迹测试**

```python
def test_profiles_keep_machine_fail_but_change_flow(tmp_path: Path) -> None:
    modules = ["hdf5_text_info", "video_quality", "sam3_containment", "semantic_consistency"]
    acceptance = run_stub_pipeline(tmp_path / "a", profile="acceptance", modules=modules, verdicts={"hdf5_text_info": "pass", "video_quality": "fail", "sam3_containment": "pass"})
    supplier = run_stub_pipeline(tmp_path / "s", profile="supplier_evaluation", modules=modules, verdicts={"hdf5_text_info": "pass", "video_quality": "fail", "sam3_containment": "pass"})
    assert acceptance["video_quality"]["flow"]["result_gate"]["verdict"] == "fail"
    assert supplier["video_quality"]["flow"]["result_gate"]["verdict"] == "fail"
    assert acceptance["pipeline_state"]["status"] == "stopped"
    assert acceptance["overall_decision"] == "fail"
    assert acceptance["manual_review"]["state"] == "skipped_due_to_fail"
    assert "sam3_containment" not in acceptance
    assert supplier["video_quality"]["flow"]["exit_gate"]["state"] == "continue"
    assert supplier["sam3_containment"]["flow"]["result_gate"]["verdict"] == "pass"
    assert supplier["pipeline_state"]["status"] == "awaiting_external"
```

- [x] **Step 2: 运行双 profile 测试并确认 supplier 被错误截断**

Run: `.venv/bin/python -m pytest tests/test_qc_orchestrator.py -k profiles -q`

Expected: FAIL，supplier report 在 video fail 后没有执行 SAM3。

- [x] **Step 3: 将机器 verdict 与 exit action 分离**

```python
profile_config = config.execution_profile(profile)
stop = result.verdict == "fail" and profile_config["fail_action"] == "stop"
exit_state = "stop_qc" if stop else "continue"
continued_after_fail = result.verdict == "fail" and not stop
```

Acceptance 截断时 `pipeline_state={status: stopped, last_completed_module: failed_module, next_module: None, stop_reason: hard_fail}`、`overall_decision=fail`、`manual_review.state=skipped_due_to_fail`，并在 `execution.module_states` 给所有后续模块记录 `skipped_due_to_fail`。Supplier 模式保留 fail issue/candidate 统计引用，在每个相关 module runtime 写 `continued_after_fail: true`，到 external 阶段才暂停，最终依赖人工阶段完成后仍必须因自动 fail 得到 fail。

- [x] **Step 4: 验证双 profile 与稳定 issue 一致**

Run: `.venv/bin/python -m pytest tests/test_qc_orchestrator.py tests/test_report_mutation.py -q`

Expected: PASS；两个 profile 的机器 issue ID、result verdict 相同，仅 exit/action/coverage 不同。

- [x] **Step 5: 提交**

```bash
git add qc_pipeline/orchestrator.py qc_common/report_mutation.py tests/test_qc_orchestrator.py
git commit -m "feat(qc): apply dual execution profiles"
```

### Task 13: 区分 disabled、skipped、unavailable 与 runtime error

**Files:**
- Modify: `qc_common/contracts.py`
- Modify: `qc_common/module_registry.py`
- Modify: `qc_common/report_mutation.py`
- Modify: `qc_pipeline/orchestrator.py`
- Modify: `tests/test_qc_orchestrator.py`

**Interfaces:**
- Produces: `ModuleUnavailableError(module: str)`。
- Produces: `record_runtime_error(path, *, module, error_type, message, expected_revision, context, config, profile, now) -> dict[str, Any]`。
- Report `execution.module_states[module].state` 枚举：`completed|disabled|skipped|not_implemented|runtime_error|awaiting_external|skipped_due_to_fail`。

- [x] **Step 1: 添加四类状态与 runtime error 非质量结论测试**

```python
def test_enabled_unregistered_module_is_error_not_pass(tmp_path: Path) -> None:
    report = run_with_unregistered_enabled_module(tmp_path, "duplicate_check")
    assert report["pipeline_state"]["status"] == "error"
    assert report["overall_decision"] is None
    assert report["runtime_errors"][0]["error_type"] == "module_unavailable"
    assert report["execution"]["module_states"]["duplicate_check"]["state"] == "not_implemented"

def test_disabled_module_is_recorded_without_module_pass_block(tmp_path: Path) -> None:
    report = run_with_disabled_module(tmp_path, "effective_duration")
    assert report["execution"]["module_states"]["effective_duration"]["state"] == "disabled"
    assert "effective_duration" not in report
```

- [x] **Step 2: 运行测试并确认状态混淆**

Run: `.venv/bin/python -m pytest tests/test_qc_orchestrator.py -k "unregistered or disabled or runtime" -q`

Expected: FAIL，缺少结构化 `module_states`/`runtime_errors`。

- [x] **Step 3: 实现结构化运行错误路径**

```python
runtime_error = {
    "module": module,
    "error_type": error_type,
    "message": message,
    "occurred_at": now,
    "retryable": error_type in {"stale_revision", "process_error", "evidence_integrity_error"},
}
report["runtime_errors"].append(runtime_error)
report["pipeline_state"] = {"status": "error", "last_completed_module": previous, "next_module": module, "stop_reason": error_type}
report["overall_decision"] = None
```

Disabled 模块只写 `module_states`；上游 skip 写 `skipped`；enabled 但 registry 缺失写 `not_implemented` + runtime error；runner exception、输入缺失、evidence/sidecar 写失败写 `runtime_error`。单资产异常由 batch worker 捕获并返回 error outcome，不取消其他 asset future。

- [x] **Step 4: 运行状态机、Schema 和批次隔离测试**

Run: `.venv/bin/python -m pytest tests/test_qc_orchestrator.py tests/test_asset_qc_schema_v2.py -q`

Expected: PASS；所有 error report 的 decision 为 null，disabled/unavailable 不产生 module pass block。

- [x] **Step 5: 提交**

```bash
git add qc_common/contracts.py qc_common/module_registry.py qc_common/report_mutation.py qc_pipeline/orchestrator.py tests/test_qc_orchestrator.py
git commit -m "feat(qc): separate runtime and quality states"
```

### Task 14: 从 QC JSON 生成 warn 人工队列输入

**Files:**
- Create: `qc_reporting/__init__.py`
- Create: `qc_reporting/projection.py`
- Modify: `tools/build_manual_review_queue.py:121-223`
- Create: `tests/test_qc_json_review_queue.py`
- Modify: `tests/test_manual_review_queue.py`

**Interfaces:**
- Produces: `iter_asset_reports(quality_archive: Path) -> Iterator[dict[str, Any]]`，逐份 Schema 校验。
- Produces: `project_warn_review_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]`。
- Formal CLI: `python -m tools.build_manual_review_queue --quality-archive <dir> --output-dir <dir>`；旧 sidecar 参数保留为显式 `--legacy-*` 对账路径，不得成为默认。

- [x] **Step 1: 添加仅 warn 入队、all-pass/auto-fail 不入队测试**

```python
def test_review_queue_comes_only_from_candidate_issue_ids(tmp_path: Path) -> None:
    write_report(tmp_path, asset="warn", issues=[warn_issue("w1", 10, 20)], candidates=["w1"], state="queued")
    write_report(tmp_path, asset="pass", issues=[], candidates=[], state="not_required")
    write_report(tmp_path, asset="fail", issues=[fail_issue("f1")], candidates=[], state="skipped_due_to_fail")
    rows = list(project_quality_archive_review_rows(tmp_path / "quality_archive"))
    assert [(row["asset_id"], row["issue_id"]) for row in rows] == [("warn", "w1")]
    assert rows[0]["window_start_frame"] == 10
    assert rows[0]["window_end_frame"] == 20
```

- [x] **Step 2: 运行测试并确认投影模块缺失**

Run: `.venv/bin/python -m pytest tests/test_qc_json_review_queue.py -q`

Expected: collection FAIL，包含 `No module named 'qc_reporting'`。

- [x] **Step 3: 实现候选 ID join 与正式 CLI**

```python
def project_warn_review_rows(report):
    issues = {item["issue_id"]: item for item in report["issues"]}
    rows = []
    for issue_id in report["manual_review"]["candidate_issue_ids"]:
        issue = issues[issue_id]
        if issue["severity"] != "warn" or not issue["needs_manual_review"]:
            raise ValueError(f"invalid manual candidate: {issue_id}")
        evidence = {item["evidence_id"]: item for item in report.get(issue["module"], {}).get("evidence", [])}
        rows.append(review_row_from_issue(report, issue, evidence))
    return rows
```

行必须包含 stable `review_id=issue_id`、supplier/asset、module/rule/reason、source-inclusive start/end、hand、machine verdict、metrics JSON、evidence/overlay 相对路径；禁止 pass-sample 抽样。候选引用不存在、重复或指向 fail 时整个资产投影失败并报 JSON path。

- [x] **Step 4: 运行新旧队列回归**

Run: `.venv/bin/python -m pytest tests/test_qc_json_review_queue.py tests/test_manual_review_queue.py -q`

Expected: PASS；新入口只读 QC JSON，旧 sidecar helper 仍可用于迁移对账测试。

- [x] **Step 5: 提交**

```bash
git add qc_reporting/__init__.py qc_reporting/projection.py tools/build_manual_review_queue.py tests/test_qc_json_review_queue.py tests/test_manual_review_queue.py
git commit -m "feat(qc): project warn queue from asset reports"
```

### Task 15: 实现 QC JSON 三表投影和 profile 分组统计

**Files:**
- Modify: `qc_reporting/projection.py`
- Create: `qc_reporting/aggregate.py`
- Create: `tests/test_qc_reporting_projection.py`
- Create: `tests/test_qc_reporting_aggregate.py`

**Interfaces:**
- Produces: frozen `BatchProjection(asset_rows: tuple[dict, ...], issue_rows: tuple[dict, ...], execution_rows: tuple[dict, ...], source_manifest: tuple[dict, ...])`。
- Produces: `project_quality_archive(path: Path) -> BatchProjection`。
- Produces: `aggregate_projection(projection: BatchProjection) -> dict[str, Any]`，顶层含 `overall` 与 `by_profile`。

- [x] **Step 1: 添加资产/issue 去重和 profile 隔离统计测试**

```python
def test_aggregation_counts_assets_and_issues_separately(tmp_path: Path) -> None:
    write_completed_report(tmp_path, "a", "acceptance", "fail", issues=[fail_issue("f1"), fail_issue("f2")])
    write_completed_report(tmp_path, "b", "supplier_evaluation", "pass", issues=[])
    projection = project_quality_archive(tmp_path / "quality_archive")
    stats = aggregate_projection(projection)
    assert stats["overall"]["asset_count"] == 2
    assert stats["overall"]["automatic_hard_fail_asset_count"] == 1
    assert stats["overall"]["automatic_hard_fail_issue_count"] == 2
    assert stats["overall"]["final_fail_asset_count"] == 1
    assert stats["by_profile"]["acceptance"]["asset_count"] == 1
    assert stats["by_profile"]["supplier_evaluation"]["module_coverage"]["sam3_containment"] == 1.0
```

- [x] **Step 2: 运行测试并确认聚合接口缺失**

Run: `.venv/bin/python -m pytest tests/test_qc_reporting_projection.py tests/test_qc_reporting_aggregate.py -q`

Expected: FAIL，缺少 `BatchProjection` 或 `aggregate_projection`。

- [x] **Step 3: 实现规范化投影和精确指标**

```python
def aggregate_projection(projection):
    return {
        "overall": _aggregate_group(projection.asset_rows, projection.issue_rows, projection.execution_rows),
        "by_profile": {
            profile: _aggregate_group(
                tuple(row for row in projection.asset_rows if row["profile"] == profile),
                tuple(row for row in projection.issue_rows if row["profile"] == profile),
                tuple(row for row in projection.execution_rows if row["profile"] == profile),
            )
            for profile in sorted({row["profile"] for row in projection.asset_rows})
        },
    }
```

`asset_rows` 一资产一行，含 profile/status/decision/report_revision/config hash/模块覆盖率；`issue_rows` 一 issue 一行，含 machine severity、human/effective verdict（字段不存在时 null）、rule/module/window；`execution_rows` 一 module state 一行，含 duration、continued_after_fail、runtime error。统计必须输出 total/completed/incomplete、自动 fail 资产与 issue、machine warn 资产与 issue、人工消解/确认（当前可为 0）、最终 pass/fail、pass rate、每模块 coverage 与 stop position。

- [x] **Step 4: 运行投影、Schema 和双 profile 测试**

Run: `.venv/bin/python -m pytest tests/test_qc_reporting_projection.py tests/test_qc_reporting_aggregate.py tests/test_asset_qc_schema_v2.py -q`

Expected: PASS；同资产多个 issue 不重复资产计数，两种 profile 不混合 coverage。

- [x] **Step 5: 提交**

```bash
git add qc_reporting/projection.py qc_reporting/aggregate.py tests/test_qc_reporting_projection.py tests/test_qc_reporting_aggregate.py
git commit -m "feat(qc): aggregate asset report projections"
```

### Task 16: 将正式 ledger 与 weekly report 入口迁移到统一投影

**Files:**
- Create: `tools/build_qc_json_projection.py`
- Modify: `tools/build_batch_qc_ledger.py:64-167`
- Modify: `tools/build_acceptance_ledger.py:77-102`
- Modify: `tools/build_weekly_supplier_acceptance_report.py`
- Modify: `tools/build_xjgt_acceptance_report.py:159-322`
- Create: `tests/test_qc_reporting_entrypoints.py`
- Modify: `tests/test_batch_qc_ledger.py`
- Modify: `tests/test_acceptance_ledger.py`
- Modify: `tests/test_weekly_supplier_acceptance_report.py`
- Modify: `tests/test_xjgt_acceptance_report.py`

**Interfaces:**
- Formal CLI: `python -m tools.build_qc_json_projection --quality-archive DIR --output-dir DIR --formats csv parquet xlsx markdown`。
- Existing formal tools 新增并默认要求 `--quality-archive`；sidecar 参数移动到 `--legacy-reconciliation-*`，只能输出差异，不能改正式 verdict。
- Produces: `write_projection_outputs(projection, statistics, output_dir, formats) -> dict[str, Path]`。

- [x] **Step 1: 添加 sidecar 冲突时 QC JSON 胜出的入口测试**

```python
def test_formal_ledger_ignores_conflicting_legacy_sidecar(tmp_path: Path) -> None:
    archive = write_completed_report(tmp_path, "a", "acceptance", "pass", issues=[])
    sidecar = tmp_path / "candidate_windows.json"
    sidecar.write_text('[{"asset_id":"a","auto_verdict":"fail"}]', encoding="utf-8")
    result = run_projection_cli(archive.parent, tmp_path / "out", legacy_candidate_windows=sidecar)
    ledger = pd.read_csv(result["asset_csv"])
    assert ledger.loc[0, "overall_decision"] == "pass"
    reconciliation = pd.read_csv(result["reconciliation_csv"])
    assert reconciliation.loc[0, "difference_type"] == "legacy_conflicts_with_qc_json"
```

- [x] **Step 2: 运行入口测试并确认现有工具仍解释 sidecar**

Run: `.venv/bin/python -m pytest tests/test_qc_reporting_entrypoints.py -q`

Expected: FAIL，现有 `build_batch_qc_ledger` 将 sidecar fail 当作正式结论。

- [x] **Step 3: 让所有正式输出消费同一 projection**

```python
projection = project_quality_archive(args.quality_archive)
statistics = aggregate_projection(projection)
paths = write_projection_outputs(
    projection, statistics, args.output_dir,
    formats=tuple(args.formats),
)
if args.legacy_reconciliation_candidate_windows:
    paths["reconciliation_csv"] = write_reconciliation_only(...)
```

CSV/Parquet 写 asset/issue/execution 三表；XLSX 固定 sheet 为 `Summary`、`Assets`、`Issues`、`Execution`、`Data_Dictionary`；Markdown 从同一 statistics 渲染。`build_acceptance_ledger` 与 weekly/XJGT 的正式结论列改为 projection 字段；遗留供应商特有 evidence 页可保留，但必须标注 `derived_evidence_only` 且不得回算 acceptance status。

- [x] **Step 4: 运行四类报表和遗留对账回归**

Run: `.venv/bin/python -m pytest tests/test_qc_reporting_entrypoints.py tests/test_batch_qc_ledger.py tests/test_acceptance_ledger.py tests/test_xjgt_acceptance_report.py tests/test_weekly_supplier_acceptance_report.py -q`

Expected: PASS；正式 verdict 都来自 QC JSON，旧输入只影响 reconciliation/evidence 页。

- [x] **Step 5: 提交**

```bash
git add tools/build_qc_json_projection.py tools/build_batch_qc_ledger.py tools/build_acceptance_ledger.py tools/build_weekly_supplier_acceptance_report.py tools/build_xjgt_acceptance_report.py tests/test_qc_reporting_entrypoints.py tests/test_batch_qc_ledger.py tests/test_acceptance_ledger.py tests/test_weekly_supplier_acceptance_report.py tests/test_xjgt_acceptance_report.py
git commit -m "feat(qc): migrate reports to asset json projection"
```

### Task 17: 增加可删除重建且不参与事实判定的缓存

**Files:**
- Create: `qc_reporting/cache.py`
- Modify: `tools/build_qc_json_projection.py`
- Create: `tests/test_qc_reporting_cache.py`

**Interfaces:**
- Produces: `build_source_manifest(quality_archive: Path) -> tuple[dict[str, Any], ...]`，每项含 relative path、asset_id、revision、sha256。
- Produces: `write_projection_cache(projection, cache_dir: Path) -> None`。
- Produces: `load_projection_cache(cache_dir, expected_manifest) -> BatchProjection | None`；任何不一致返回 None。

- [ ] **Step 1: 添加删除重建和 stale cache 无效测试**

```python
def test_cache_can_be_deleted_and_rebuilt_identically(tmp_path: Path) -> None:
    archive = make_archive(tmp_path)
    first = project_with_cache(archive, tmp_path / "cache")
    shutil.rmtree(tmp_path / "cache")
    second = project_with_cache(archive, tmp_path / "cache")
    assert first == second

def test_revision_change_invalidates_cache(tmp_path: Path) -> None:
    archive = make_archive(tmp_path)
    project_with_cache(archive, tmp_path / "cache")
    bump_report_revision(archive / "a.json")
    assert load_projection_cache(tmp_path / "cache", build_source_manifest(archive)) is None
```

- [ ] **Step 2: 运行测试并确认缓存模块不存在**

Run: `.venv/bin/python -m pytest tests/test_qc_reporting_cache.py -q`

Expected: collection FAIL，包含 `No module named 'qc_reporting.cache'`。

- [ ] **Step 3: 实现 manifest 驱动缓存**

```python
CACHE_FILES = {"assets": "assets.parquet", "issues": "issues.parquet", "execution": "execution.parquet"}

def load_projection_cache(cache_dir, expected_manifest):
    manifest_path = cache_dir / "source_reports.json"
    if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != list(expected_manifest):
        return None
    if not all((cache_dir / name).is_file() for name in CACHE_FILES.values()):
        return None
    return BatchProjection(...)
```

写缓存先写 `.tmp` 后逐文件 replace，最后写 manifest；CLI `--cache-dir` 只用于加速，cache missing/corrupt/stale 时自动从 QC JSON 重建。统计函数不得接受 cache 文件路径，只接受 `BatchProjection`。

- [ ] **Step 4: 验证缓存、投影和 CLI**

Run: `.venv/bin/python -m pytest tests/test_qc_reporting_cache.py tests/test_qc_reporting_projection.py tests/test_qc_reporting_entrypoints.py -q`

Expected: PASS；删除、损坏或 revision 变化都能回源重建且统计一致。

- [ ] **Step 5: 提交**

```bash
git add qc_reporting/cache.py tools/build_qc_json_projection.py tests/test_qc_reporting_cache.py
git commit -m "feat(qc): add rebuildable projection cache"
```

### Task 18: 同步 PRD、Schema 格式文档与 reviewer 指南

**Files:**
- Modify: `docs/PRD-qc-gated-json.md`
- Modify: `docs/PRD-qc-unified-config.md`
- Modify: `docs/asset-qc-json-format.md`
- Modify: `ACCEPTANCE.md`
- Modify: `WORKFLOW_INTERFACE.md`
- Create: `docs/qc-dataflow-migration.md`
- Create: `tests/test_qc_docs_contract.py`

**Interfaces:**
- 文档必须使用与代码相同的版本、profile、pipeline status、module state、CLI 与目录名。
- 迁移文档必须给出 v1 只读、首次 v2 写回、回滚到只读 sidecar 工具和禁止回退 master verdict 的操作步骤。

- [ ] **Step 1: 添加文档契约测试**

```python
def test_reviewer_docs_name_v2_profiles_and_single_source() -> None:
    combined = "\n".join(Path(path).read_text(encoding="utf-8") for path in DOCS)
    for token in ("asset_qc_report.v2", "qc_acceptance_v2.0.0", "acceptance", "supplier_evaluation", "quality_archive/*.json", "sidecar 只作证据"):
        assert token in combined

def test_docs_do_not_advertise_pass_sample_review() -> None:
    text = Path("ACCEPTANCE.md").read_text(encoding="utf-8")
    assert "正常 Pass 样本抽检" not in text
    assert "仅累计 warn 进入人工质检" in text
```

- [ ] **Step 2: 运行测试并确认旧文档缺少 v2/双 profile**

Run: `.venv/bin/python -m pytest tests/test_qc_docs_contract.py -q`

Expected: FAIL，缺少 v2 版本或 `supplier_evaluation`。

- [ ] **Step 3: 按已实现接口同步六份文档**

每份文档必须写明以下精确流程：

```text
自动 QC Gate
  acceptance: hard fail -> stopped/fail；不进入语义和人工质检
  supplier_evaluation: hard fail -> 记录并继续
-> semantic_consistency external
-> candidate_issue_ids 为空：manual_review=not_required
-> candidate_issue_ids 非空：manual_review=queued
-> 最终 overall_decision=pass|fail
-> 批次输出只投影 quality_archive/*.json
```

同时列出 Config 发布规则、v1 hash、evidence 相对路径、revision 冲突、runtime error 与质量 fail 的区别、新 CLI 完整示例和 sidecar 对账限制。

- [ ] **Step 4: 运行文档契约与 OpenSpec 校验**

Run: `.venv/bin/python -m pytest tests/test_qc_docs_contract.py -q && openspec validate unify-qc-dataflow --strict`

Expected: PASS；OpenSpec 显示 change valid。

- [ ] **Step 5: 提交**

```bash
git add docs/PRD-qc-gated-json.md docs/PRD-qc-unified-config.md docs/asset-qc-json-format.md ACCEPTANCE.md WORKFLOW_INTERFACE.md docs/qc-dataflow-migration.md tests/test_qc_docs_contract.py
git commit -m "docs(qc): document unified v2 dataflow"
```

### Task 19: 验证 pass、warn、hard fail、runtime error 的完整 revision 轨迹

**Files:**
- Create: `tests/fixtures/qc_pipeline/manifest.jsonl`
- Create: `tests/fixtures/qc_pipeline/expected_revision_trace.json`
- Create: `tests/test_qc_pipeline_revision_trace.py`

**Interfaces:**
- Consumes: `run_asset()`、stub registry 和真实 report mutation；不依赖模型权重。
- Produces: 固定 trace 行：`asset_id,module,revision,pipeline_status,result_verdict,exit_state,overall_decision,next_module`。

- [ ] **Step 1: 添加四类 fixture 的精确轨迹测试**

```python
def test_four_fixture_revision_trace_matches_golden(tmp_path: Path) -> None:
    actual = run_trace_fixtures(tmp_path, Path("tests/fixtures/qc_pipeline/manifest.jsonl"))
    expected = json.loads(Path("tests/fixtures/qc_pipeline/expected_revision_trace.json").read_text())
    assert actual == expected
    by_asset = group_by_asset(actual)
    assert by_asset["hard-fail"][-1]["pipeline_status"] == "stopped"
    assert by_asset["hard-fail"][-1]["overall_decision"] == "fail"
    assert by_asset["runtime-error"][-1]["pipeline_status"] == "error"
    assert by_asset["runtime-error"][-1]["overall_decision"] is None
```

- [ ] **Step 2: 运行测试并确认 golden/trace helper 缺失**

Run: `.venv/bin/python -m pytest tests/test_qc_pipeline_revision_trace.py -q`

Expected: FAIL，fixture 文件或 trace helper 不存在。

- [ ] **Step 3: 写入四资产 fixture 和真实 revision 捕获器**

```json
{"asset_id":"pass","verdicts":{"hdf5_text_info":"pass","quality_hand":"pass"}}
{"asset_id":"warn","verdicts":{"hdf5_text_info":"pass","quality_hand":"warn"}}
{"asset_id":"hard-fail","verdicts":{"hdf5_text_info":"fail"}}
{"asset_id":"runtime-error","errors":{"quality_hand":"input_missing"}}
```

每次 writer 返回后捕获实际报告，不手工推算 revision；assert revision 从 1 单调递增、warn candidate 全量重建、hard fail 不创建 external task、error 不产生 decision。

- [ ] **Step 4: 运行 revision trace 与事务回归**

Run: `.venv/bin/python -m pytest tests/test_qc_pipeline_revision_trace.py tests/test_report_mutation.py tests/test_qc_orchestrator.py -q`

Expected: PASS，golden 与实际状态轨迹逐字段一致。

- [ ] **Step 5: 提交**

```bash
git add tests/fixtures/qc_pipeline/manifest.jsonl tests/fixtures/qc_pipeline/expected_revision_trace.json tests/test_qc_pipeline_revision_trace.py
git commit -m "test(qc): cover complete revision trajectories"
```

### Task 20: 对同一批输入验证两种 profile 的端到端差异

**Files:**
- Create: `tests/test_qc_pipeline_profiles_e2e.py`
- Modify: `tests/fixtures/qc_pipeline/manifest.jsonl`

**Interfaces:**
- Consumes: `tools.run_qc_pipeline.run_batch(...)` 与 `project_quality_archive()`。
- 验证同一资产、同一 Config、同一 runner 输出，在两个 profile 下机器 issue 相同而模块覆盖/stop point 不同。

- [ ] **Step 1: 添加双批次 E2E 测试**

```python
def test_same_batch_has_profile_specific_flow_and_same_machine_findings(tmp_path: Path) -> None:
    acceptance = run_fixture_batch(tmp_path / "acceptance", profile="acceptance")
    supplier = run_fixture_batch(tmp_path / "supplier", profile="supplier_evaluation")
    a_report = read_report(acceptance, "hard-fail")
    s_report = read_report(supplier, "hard-fail")
    assert normalize_machine_issues(a_report) == normalize_machine_issues(s_report)
    assert a_report["pipeline_state"]["status"] == "stopped"
    assert "sam3_containment" not in a_report
    assert s_report["sam3_containment"]["flow"]["result_gate"]["verdict"] == "pass"
    a_stats = aggregate_projection(project_quality_archive(acceptance / "quality_archive"))
    s_stats = aggregate_projection(project_quality_archive(supplier / "quality_archive"))
    assert a_stats["overall"]["module_coverage"]["sam3_containment"] < s_stats["overall"]["module_coverage"]["sam3_containment"]
```

- [ ] **Step 2: 运行 E2E 并确认 batch 入口或 coverage 不完整**

Run: `.venv/bin/python -m pytest tests/test_qc_pipeline_profiles_e2e.py -q`

Expected: FAIL，缺少可调用 batch helper 或 supplier fail 后未继续。

- [ ] **Step 3: 补齐批次调用的可测试入口**

```python
def run_batch(contexts, *, config, profile, registry, max_workers):
    if len({context.asset_id for context in contexts}) != len(contexts):
        raise ValueError("duplicate asset_id in batch")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_asset, context, config=config, profile=profile, registry=registry): context.asset_id for context in contexts}
        return {asset_id: _isolate_future(future) for future, asset_id in futures.items()}
```

确保一个 asset error 不取消其他 future；每个资产 report revision 独立递增。测试 fixture 在 semantic external 暂停，因此 supplier report 尚不形成最终 pass，但已存在自动 fail 的资产在依赖人工 change 完成后 reducer 必须保持 fail。

- [ ] **Step 4: 运行 E2E、并发和投影测试**

Run: `.venv/bin/python -m pytest tests/test_qc_pipeline_profiles_e2e.py tests/test_qc_orchestrator.py tests/test_qc_reporting_aggregate.py -q`

Expected: PASS；同机器 findings、不同流转覆盖，分 profile 统计正确。

- [ ] **Step 5: 提交**

```bash
git add tools/run_qc_pipeline.py tests/fixtures/qc_pipeline/manifest.jsonl tests/test_qc_pipeline_profiles_e2e.py
git commit -m "test(qc): verify execution profile differences"
```

### Task 21: 全量回归、迁移对账与回滚演练

**Files:**
- Create: `tests/test_qc_migration_reconciliation.py`
- Modify: `docs/qc-dataflow-migration.md`
- Modify: `openspec/changes/unify-qc-dataflow/tasks.md`

**Interfaces:**
- Consumes: v1 fixture、遗留 sidecars、v2 projection 与全部正式 CLI。
- Produces: `reconcile_legacy_outputs(*, quality_archive, legacy_inputs) -> list[dict[str, Any]]`，只输出差异，不修改主报告。

- [ ] **Step 1: 添加 v1→v2、legacy 对账和只读回滚测试**

```python
def test_migration_reconciliation_never_rewrites_source_or_uses_legacy_verdict(tmp_path: Path) -> None:
    v1_path, legacy = write_migration_fixture(tmp_path)
    before = v1_path.read_bytes()
    projected = project_quality_archive(v1_path.parent)
    differences = reconcile_legacy_outputs(quality_archive=v1_path.parent, legacy_inputs=[legacy])
    assert v1_path.read_bytes() == before
    assert projected.asset_rows[0]["overall_decision"] is None
    assert differences[0]["difference_type"] == "legacy_conflicts_with_qc_json"
    assert differences[0]["authoritative_source"] == "asset_qc_json"
```

- [ ] **Step 2: 运行迁移测试和全量测试，记录首轮失败**

Run: `.venv/bin/python -m pytest tests/test_qc_migration_reconciliation.py -q && .venv/bin/python -m pytest -q`

Expected: 首条命令在 reconciliation helper 缺失处 FAIL；实现后第二条必须全量 PASS。

- [ ] **Step 3: 实现只读对账并完成回滚说明**

```python
def reconcile_legacy_outputs(*, quality_archive, legacy_inputs):
    authoritative = {row["asset_id"]: row for row in project_quality_archive(quality_archive).asset_rows}
    return sorted(
        compare_legacy_row(row, authoritative.get(normalize_asset_id(row.get("asset_id"))))
        for path in legacy_inputs
        for row in read_legacy_records(path)
    )
```

迁移文档写出三条可执行路径：正常升级（保留 sidecar、首次 module 写回迁为 v2）、只读验证（仅 projection/reconciliation）、回滚（停止 v2 writer，恢复旧 runner 只产 sidecar，但不得用旧报表覆盖已存在 v2 master JSON）。记录配置快照/hash、备份质量目录的操作和恢复验证命令。

- [ ] **Step 4: 执行完整验证并勾选 21 项 OpenSpec 任务**

Run:

```bash
.venv/bin/python -m pytest -q
openspec validate unify-qc-dataflow --strict
git diff --check
rg -n "precheck_clip_aggregates|candidate_windows|sam3_window_summary|video_quality_results|manual_review_labels" tools/build_batch_qc_ledger.py tools/build_acceptance_ledger.py tools/build_weekly_supplier_acceptance_report.py tools/build_xjgt_acceptance_report.py
```

Expected: pytest 全量 PASS；OpenSpec valid；`git diff --check` 无输出；最后的 `rg` 命中只允许出现在命名为 `legacy_reconciliation` 的参数、帮助文本或对账函数中。逐项将 `openspec/changes/unify-qc-dataflow/tasks.md` 的 21 个 checkbox 标为完成。

- [ ] **Step 5: 提交最终验证与迁移说明**

```bash
git add tests/test_qc_migration_reconciliation.py docs/qc-dataflow-migration.md openspec/changes/unify-qc-dataflow/tasks.md
git commit -m "test(qc): complete unified dataflow verification"
```

## 最终验收命令

```bash
.venv/bin/python -m pytest -q
openspec validate unify-qc-dataflow --strict
git diff --check
git status --short
```

期望：全量测试通过，OpenSpec 严格校验通过，无 whitespace error；工作区只包含本 change 的预期文件。不要归档 change，归档应在代码评审和用户验收后单独执行。
