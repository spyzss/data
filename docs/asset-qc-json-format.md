# 单资产 QC JSON 格式

## 1. 定位

每条数据从拉取完成开始，只维护一份主质检档案：

```text
<batch>/quality_archive/<asset_id>.json
```

这份 JSON 随数据走完整个 QC 流程。模块只更新自己拥有的 block 和 issue，
不得创建另一份模块专属主报告。CSV、HTML、sidecar、overlay 和 ledger 都只是
证据或批次派生产物。

当前正式 schema：

```text
schemas/asset_qc_report.v2.schema.json
```

旧 v1 报告必须先迁移到 v2；人工语义校准和 Warn 复核只写 v2 报告。

## 2. 顶层结构

视频质检完成后的典型结构如下。示例省略了部分 metrics：

```json
{
  "schema_version": "asset_qc_report.v2",
  "qc_config": {
    "schema_version": "qc_acceptance_config_schema.v2",
    "config_version": "qc_acceptance_v2.0.0",
    "config_name": "acceptance_gate",
    "config_path": "configs/qc_acceptance.yaml",
    "config_hash": "sha256:<64 lowercase hex characters>"
  },
  "asset_id": "file-008",
  "report_revision": 3,
  "execution": {
    "profile": "acceptance",
    "started_at": "2026-07-15T00:00:00Z",
    "updated_at": "2026-07-15T00:05:00Z"
  },
  "pipeline_state": {
    "status": "running",
    "last_completed_module": "video_quality",
    "next_module": "sam3_containment"
  },
  "overall_decision": null,
  "issues": [
    {
      "issue_id": "video_quality:fps_below_pass:001",
      "code": "fps_below_pass",
      "severity": "warn",
      "module": "video_quality",
      "issue_type": "low_fps",
      "metric": "video_basic.fps",
      "observed_value": 22.5,
      "operator": "<",
      "boundary_value": 24.0,
      "rule_id": "video_quality.fps_below_pass",
      "needs_manual_review": true,
      "context": {}
    }
  ],
  "manual_review": {
    "required": null,
    "state": "not_evaluated",
    "candidate_issue_ids": [
      "video_quality:fps_below_pass:001"
    ],
    "failures_for_batch_stats_issue_ids": []
  },
  "source_files": {
    "video": {
      "path": "video/file-008.mp4",
      "filename": "file-008.mp4",
      "extension": ".mp4"
    },
    "hdf5": {
      "path": "hdf5/file-008.hdf5",
      "exists": true
    }
  },
  "video_quality": {
    "stage": "video_prefilter",
    "module_version": "video_prefilter_v0.3.2",
    "flow": {
      "entry_gate": {
        "state": "ready",
        "eligible": true,
        "blocked_by_module": null,
        "required_inputs": ["source_files.video.path"],
        "missing_inputs": [],
        "upstream_continue": true
      },
      "result_gate": {
        "verdict": "warn",
        "has_fail": false,
        "has_warn": true
      },
      "exit_gate": {
        "state": "continue",
        "continue_to_next_module": true,
        "next_module": "sam3_containment"
      }
    },
    "evaluation": {
      "decision": "warn",
      "reasons": [],
      "warn_reasons": ["fps_below_pass"],
      "issue_ids": ["video_quality:fps_below_pass:001"],
      "should_run_mask_qc": true
    },
    "metadata": {},
    "sampling": {},
    "metrics": {},
    "errors": []
  },
  "reference_quality": {
    "mode": "none",
    "reference_video_path": null,
    "vmaf": null,
    "note": "当前无标准对照视频，未计算 VMAF。"
  }
}
```

## 3. 顶层字段

| 字段 | 规则 |
|---|---|
| `schema_version` | 正式人工流程固定为 `asset_qc_report.v2`；v1 只作为迁移输入。 |
| `qc_config` | 本次 pipeline 初始化时锁定的统一配置引用。 |
| `asset_id` | 资产唯一 ID，也是 JSON 文件名。 |
| `report_revision` | 每次成功写回加 1，用于防止旧结果覆盖新结果。 |
| `pipeline_state` | 当前流程位置，不代表单个模块质量。 |
| `overall_decision` | 只有流程停止或全部完成时才形成最终结论。 |
| `issues` | 所有模块共享的 warn/fail 事实表。 |
| `manual_review` | 人工路由输入、状态和结果。 |
| `<module_name>` | 模块自己的 gate、指标和证据。 |

### 3.1 `qc_config`

`qc_config` 在该资产建档时写入一次，并在整条 pipeline 中保持不变：

```json
{
  "schema_version": "qc_acceptance_config_schema.v2",
  "config_version": "qc_acceptance_v2.0.0",
  "config_name": "acceptance_gate",
  "config_path": "configs/qc_acceptance.yaml",
  "config_hash": "sha256:..."
}
```

规则：

- `config_hash` 必须根据实际读取的 YAML 字节计算。
- 模块不得在自己的 block 复制 thresholds 或 `config_ref`。
- issue 不重复写 `config_version`；顶层版本是唯一权威来源。
- 同一次 asset pipeline 不能中途更换 config。发现 hash/version 不一致必须拒绝写入。

### 3.2 `pipeline_state` 与 `overall_decision`

| `pipeline_state.status` | 含义 | `overall_decision` |
|---|---|---|
| `pending` | 已建档，尚未开始或等待当前 gate。 | 必须为 `null` |
| `running` | 自动 QC 仍在继续。 | 必须为 `null` |
| `stopped` | 某个模块 hard fail，后续 QC 已停止。 | 必须为 `fail` |
| `completed` | 所有应运行模块及适用的人工阶段完成。 | `pass` 或 `fail` |

`pending` 不是质量等级，也不等于 warn。`warn` 是模块对具体问题的判定；
流程未结束时只保存在 module verdict 和 `issues`，不提前写进
`overall_decision`。人工阶段尚未完成或出现运行错误时也必须为 `null`。

## 4. Issue 结构

每个触发的指标单独生成一个 issue 对象。一个视频有多个 warn 时，`issues`
中就有多个对象；不是把多个 metric/value 塞进同一个字段。

```json
{
  "issue_id": "video_quality:fps_below_pass:001",
  "code": "fps_below_pass",
  "severity": "warn",
  "module": "video_quality",
  "issue_type": "low_fps",
  "metric": "video_basic.fps",
  "observed_value": 22.5,
  "operator": "<",
  "boundary_value": 24.0,
  "rule_id": "video_quality.fps_below_pass",
  "needs_manual_review": true,
  "context": {}
}
```

| 字段 | 规则 |
|---|---|
| `issue_id` | 在该 asset 内稳定且唯一，供模块、人工和批次统计引用。 |
| `code` | 稳定原因码，不写自然语言句子。 |
| `severity` | 只能是 `warn` 或 `fail`。pass 不生成 issue。 |
| `module` | 产生问题的模块名。 |
| `issue_type` | 跨指标归类，如 `freeze`、`low_fps`、`exposure`。 |
| `metric` | 指标路径；无单一指标时可为 `null`。 |
| `observed_value` | 实际观测值，可为数值、布尔、字符串或结构。 |
| `operator` | 触发比较符，如 `<`、`>`、`==`；不适用时为 `null`。 |
| `boundary_value` | 触发边界；不适用时可为 `null`。 |
| `rule_id` | 回查统一 config 的稳定规则 ID。 |
| `needs_manual_review` | 该问题是否应成为人工候选。 |
| `context` | 区间、检测可靠性、文件位置等附加证据。 |

不再使用：

```text
reason_details
warn_reason_details
value
comparison
issue.config_version
module.thresholds
```

模块内 `reasons` / `warn_reasons` 可以保留简短 code 数组以兼容和快速展示，
但完整事实只在顶层 `issues` 保存一次。模块用 `issue_ids` 引用它们。

## 5. Gate 结构

每个自动 QC 模块都应写：

```json
{
  "flow": {
    "entry_gate": {},
    "result_gate": {},
    "exit_gate": {}
  }
}
```

通用规则：

- `pass`：`continue_to_next_module=true`。
- `warn`：生成 issue，追加到人工候选，继续下一模块。
- `fail` 且当前 profile 要求停止：`state=stop_qc`、
  `continue_to_next_module=false`、`next_module=null`；顶层
  `pipeline_state.status=stopped` 且 `pipeline_state.next_module=null`。
- 下游只读取上游 `exit_gate` 或顶层 `pipeline_state`，不解析自然语言原因。
- 上游已 fail 时，后续高成本模块不得运行。

## 6. 人工串行阶段

正式顺序是自动 QC → `semantic_calibration` → `manual_review`。两个人工阶段
复用同一份 QC JSON、同一 `report_revision` 与单资产 lease；CSV、progress JSON
和浏览器 localStorage 都不是正式事实源。

### 6.1 `semantic_calibration`

语义时间轴内部采用共享边界 `b_0...b_n` 和半开区间
`[start_frame, end_frame_exclusive)`。供应商 HDF5 使用闭区间，因此界面显示的
结束帧恒为 `end_frame_exclusive - 1`。首尾边界固定，只显示内部边界手柄；整段
色块不能拖动，也不支持拆分、合并、删除或整段平移。

拖动内部边界 `b_i` 会在同一 pending transaction 中同时修改左段结束帧和右段
起始帧。transaction 必须保存两段的 before/after 和两个 segment ID，但确认一次
只增加一次 `timeline_edit_count`。取消恢复两段且不计数。文字修改独立增加
`subtask_text_edit_count`。

```json
{
  "state": "in_progress",
  "source_dataset_path": "/label/subtask_label",
  "base_hdf5_sha256": "sha256:...",
  "final_hdf5_sha256": null,
  "timeline_edit_count": 1,
  "subtask_text_edit_count": 0,
  "pending_edit": null,
  "audit": []
}
```

存在 pending edit 时，其他边界、文字、任务切换、完成样本和下一资产都锁定，
必须先逐次确认或取消。完成语义阶段时，服务端在 HDF5 同目录生成临时副本，验证
只有 `/label/subtask_label` 的 canonical 内容变化并 `fsync`，随后原子替换；不
生成持久 `.bak`。

### 6.2 `manual_review`

自动模块只负责累计候选：

```json
{
  "required": null,
  "state": "not_evaluated",
  "candidate_issue_ids": ["video_quality:fps_below_pass:001"],
  "failures_for_batch_stats_issue_ids": []
}
```

到达人工路由模块后，由人工策略统一决定：

- `required=false`：无候选或按抽样策略无需人工。
- `required=true`：进入 `queued` / `in_progress` / `completed`。
- 自动 QC 已 hard fail：`required=false`、`state=skipped_due_to_fail`，问题直接供
  批次统计和返工使用。

人工结果必须引用 `issue_id`，并写结构化结论、reviewer、时间和备注；不得覆盖
机器观测、阈值、evidence 或机器 verdict：

- 人工 Pass：该机器 Warn 被消解，effective verdict 为 pass；
- 人工 Fail：确认该 Warn 为 fail，effective verdict 为 fail；
- 自动 hard fail：始终保持 fail，人工 Pass 不能覆盖。

每个 selected issue 都有 Pass/Fail 后才能完成样本。刷新页面从服务端 QC JSON
恢复状态；写入必须带 lease token 和 expected revision。lease 无效返回 423；
revision 过期返回 409，复核员应刷新后重新确认，禁止覆盖新 revision。

## 7. Video QC Block

当前视频模块写入：

- `flow`：入口、结果、出口 gate。
- `evaluation`：模块 decision、原因码和 issue 引用。
- `metadata`：帧数、FPS、时长、尺寸。
- `sampling`：配置抽样上限、实际抽样和解码数量。
- `metrics.video_basic`：基础可用性和尺寸。
- `metrics.timeline_metrics`：PTS、丢帧估算和时间间隔。
- `metrics.decode_metrics`：抽样解码完整性。
- `metrics.exposure_metrics`：黑帧、过暗、过曝。
- `metrics.sharpness_global`：全帧清晰度代理。
- `metrics.freeze_metrics`：低运动、候选/确认冻结和区间。
- `metrics.defect_metrics`：瑕疵总时长比例。
- `metrics.hdf5_alignment`：视频/HDF5 帧数对齐。
- `errors`：运行错误。

不包含手部 ROI 清晰度。HDF5 文本、关键点质量、mask 和语义结果属于各自模块，
视频 writer 不拥有也不重写这些 block。

### 7.1 Freeze 与掉帧证据

冻结区间至少记录：

```json
{
  "start_frame": 120,
  "end_frame": 158,
  "frame_count": 39,
  "start_time_sec": 4.0,
  "end_time_sec": 5.267,
  "duration_sec": 1.267,
  "duration_ms": 1267.0,
  "mean_frame_diff": 0.3,
  "mean_hist_diff": 0.002,
  "mean_ssim": 0.998,
  "mean_phash_hamming": 1.0,
  "motion_conflict": false,
  "motion_conflict_signals": [],
  "critical_window": false,
  "critical_keywords": []
}
```

时间轴至少记录 `drop_detection_source`、`drop_detection_reliable`、
`estimated_missing_frames` 和 `drop_frame_ratio`。这样后续裁切或人工复核能区分
可靠 PTS 证据与 OpenCV fallback。

## 8. 更新与并发规则

模块写回必须：

1. 读取当前 JSON 和 `report_revision`。
2. 校验 `asset_id` 与顶层 `qc_config` 未变化。
3. 只替换本模块拥有的 block 和本模块 issue。
4. 保留未知字段和其他模块 block。
5. 将 revision 加 1。
6. 通过 JSON Schema 校验。
7. 先写临时文件并 `fsync`，再原子替换目标文件。

revision 与预期不一致时必须报 stale-write 错误，不能静默覆盖。

## 9. 批次派生输出

以下内容可以从 `quality_archive/*.json` 生成，但都不是单资产主档案：

- 批次 CSV / XLSX；
- review queue 和 review index；
- issue 频率、供应商对比和有效时长汇总；
- sidecar、overlay、mask 证据；
- ledger events。

`sidecar` 是大体积模块明细的旁路文件；`ledger events` 是流程事件日志；CSV 是
表格视图。它们可被 JSON 用相对路径引用，但不能代替 `<asset_id>.json`。

正式聚合按 profile 输出下列指标，且同一资产只使用最大 `report_revision`：

| 字段 | 定义 |
|---|---|
| `auto_fail_assets` / `auto_fail_issues` | 机器 hard-fail 的去重资产数 / issue 数。 |
| `machine_warn_issues` | 顶层 `issues` 中机器 severity=warn 的总数，人工结果不会改写它。 |
| `human_checked_warn_issues` | 已写 Pass 或 Fail 的 warn 数。 |
| `human_resolved_warn_issues` | 人工 Pass、已消解的 warn 数。 |
| `human_confirmed_fail_issues` | 人工 Fail、确认失败的 warn 数。 |
| `timeline_edit_count` | 已确认共享边界事务数；按事务计数而不是按两个受影响段计数。 |
| `subtask_text_edit_count` | 已确认文字修改事务数。 |
| `final_pass_assets` / `final_fail_assets` | 最终二元结论的资产数。未完成和错误资产不进入通过率分母。 |

旧 manual CSV/progress JSON 只能通过一次性导入工具迁移到当前 QC JSON，且导入
前默认 dry-run；它们不能直接参与正式聚合。

`qc_reporting.export.write_aggregate_outputs` 将 overall 和每个 profile 的上述正式
指标先规范化为同一组长表行，再分别写为：

- `human_qc_aggregate.csv`；
- `human_qc_aggregate.parquet`；
- `human_qc_aggregate.xlsx` 的 `Metrics` sheet；
- `human_qc_aggregate.md`。

四种格式的列统一为 `scope/profile/metric/value_json`，指标名和 JSON 标量值必须
逐项一致。任何 profile 缺少正式指标时导出直接失败，不允许某一种格式静默缺列。
