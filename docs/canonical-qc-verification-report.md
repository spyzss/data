# Canonical QC → Curated LeRobot v3 验证报告

> **范围说明（2026-07-17）：** 本报告是当前 fixed Core 实现的历史验证证据，不是
> Canonical Data 目标架构完成证明。新的定位见
> `docs/superpowers/specs/2026-07-17-canonical-data-publisher-positioning-design.md`。
> 下文 `CanonicalQcEpisode` 是兼容实现类型。原 fixed-Core 结论保留；2026-07-17
> P0 addendum 已为 supplier extensions、batch metadata 和非零 revision 增加代码证据。

## 1. 验证范围

- 分支：`codex/human-qc-semantic-review`
- 变更：`canonical-qc-ingest-publisher`
- 输入：标准 HDF5 或标准 LeRobot episode
- 中间合同：`CanonicalQcEpisode` / `canonical_qc_episode.v1`
- 质量报告：每资产 `quality_archive/<asset_id>.json`，schema 为
  `asset_qc_report.v2`
- 输出：原子发布的 Curated LeRobot v3 release

第 1–7 节保留首版历史验证；第 8 节记录本次 Data Canonical/Publisher 修正。未实现
能力继续列为 Deferred，不用文档措辞替代代码和测试证据。

## 2. 数据流结论

```text
批次 manifest 的显式 identity
  -> Source Gate（配置、路径、Adapter、Canonical 合同）
  -> 自动 QC pipeline
  -> semantic_consistency（external）
  -> semantic_calibration（external）
  -> warn manual review（仅有 warn 时）
  -> final asset_qc_report.v2
  -> LeRobotV3Publisher 前置校验
  -> staging 写入和独立回读
  -> 原子 release commit / CURRENT.json
```

自动 QC 确定性 fail 不进入语义校准；自动阶段全部 pass 且没有 warn 时，在人工语义
阶段完成后不进入 warn 人工质检。供应商测评 profile 继续运行已启用模块，但不改变
Source Gate、报告 schema 或 Publisher 的 fail-closed 合同。

## 3. 合同对账

| 能力 | 状态 | 验证结论 |
|---|---|---|
| 显式 `asset_id/batch_id/supplier_id` | Implemented + Tested | CLI 强制提供；不从路径或损坏内容猜测。 |
| HDF5 Adapter | Implemented + Tested | 固定字段、shape/dtype、时间轴、语义、外置 MP4 和可选 hand evidence。 |
| LeRobot Adapter | Implemented + Tested | 固定 episode selector、Parquet、metadata、语义和外置 MP4。 |
| Canonical 合同 | Implemented + Tested | 数组只读、严格递增 `timestamp_ns`、半开区间 `[start,end)`、valid/NaN 和 calibration 合同。 |
| Source Gate QC JSON | Implemented + Tested | pass/fail/runtime_error 三态；确定性 Adapter/Core 失败也生成资产 JSON；JSON-only projection/aggregate 可统计。 |
| Source Gate 状态关联 | Implemented + Tested | 内部 Gate、顶层 pipeline/decision 和 Publisher 前置相互约束，矛盾 mutation fail closed。 |
| Source Gate retry/recovery | Implemented + Tested | 同诊断重试字节幂等；恢复以 CAS 增 revision，清当前错误并保留 history。 |
| Canonical 配置注册 | Implemented + Tested | active `v1.1.0` 与 immutable snapshot 字节一致；`v1.1.1` 为严格时间容差；pre-v1.1 不可执行。 |
| QC Bridge | Implemented + Tested | 保持 21 点顺序、数值、validity、视频和语义，输出现有 pipeline context。 |
| `quality_hand` Evidence | Implemented + Tested | 可选；unknown 不 fail；与机器观测分开统计，分歧可生成 warn。 |
| Canonical Data supplier extension inventory | Implemented + Tested | 标准 HDF5 非 Core dataset、标准 LeRobot 已登记规则定长列进入 immutable inventory；unsupported 类型 fail closed。 |
| Batch metadata / dataset attributes | Implemented + Tested | `canonical_batch_metadata.v1`、identity/content hash、CLI overlay 和 Publisher metadata 已绑定；跨批次检索索引另行实现。 |
| QC report batch projection | Implemented + Tested | asset row/source manifest 保留 `batch_id`；源文件删除后仍可按 JSON 汇总 Source Gate fail。 |
| Publisher 前置 | Implemented + Tested | 绑定 identity、report revision、Canonical revision/fingerprint、完整最终 Gate 与人工状态。 |
| Publisher 已登记字段 preservation | Implemented + Tested | frame extension 写 LeRobot feature；episode/batch extension 写 ndarray sidecar；data fingerprint 绑定 release identity。 |
| 非零人工语义修订发布 | Implemented + Tested | `canonical_revision_artifact.v1` 支持 task/subtask text 和共享边界，校验 CAS/fingerprint/revision/edit count 并绑定 manifest hash。 |
| LeRobot v3 writer | Implemented + Tested | 统一 writer 生成逐帧 Parquet、metadata、语义、视频、manifest 和 checksums。 |
| 独立官方 reader 回读 | Implemented + Tested | 校验 row/index/timestamp、arrays、subtask、视频 PTS 与 checksum。 |
| 原子发布和故障注入 | Implemented + Tested | staging/validate/commit 任一失败都不改变已有 release 或 `CURRENT.json`。 |
| 多相机/头腕/六目 profile | Deferred | 首版只支持 `camera_id=main` 单目合同。 |
| 镜头畸变模型 | Deferred | 首版只支持 `distortion_model=none`。 |
| 3D/2D 重投影质量规则 | Deferred | 当前只校验标定合同，未生成重投影质量结论。 |
| duplicate/content/effective duration 实现 | Deferred | 保留 pipeline 接口，尚无本 change 的算法实现证据。 |
| 自动语义模型替代 | Deferred | external stage 已解耦，但模型实现不在本 change。 |

## 4. Source Gate 失败报告合同

只有在显式 identity、批次内安全路径和可验证配置已建立后，系统才有可信的资产报告
目标。此后：

- `pass`：`result_gate=pass`、`exit_gate=continue`，Source Gate module state 为
  `completed`，再进入自动 QC；
- `fail`：要求 diagnostic，`result_gate=fail`、`exit_gate=stop_qc`，顶层固定为
  `stopped/fail`，fail issue 进入批次失败统计；
- `runtime_error`：要求 retryable diagnostic、禁止 result gate，exit 为
  `stop_incomplete`，顶层固定为 `error/null`，允许恢复；
- Publisher 若看见 `source_gate`，只接受完整的 completed/pass/continue 组合。

CLI 参数缺失、不安全路径或不可执行的 pre-v1.1 配置发生在可信报告目标建立之前，
只输出机器可读 CLI 错误，不伪造资产 QC JSON。

## 5. 时间与语义合同

- 内部统一 `timestamp_ns:int64[T]` 严格递增；不依赖转换后浮点秒作为主对齐键。
- Subtask 使用严格递增共享边界和半开区间 `[b_i,b_{i+1})`。
- 若 UI 展示闭区间，界面结束帧 `410` 对应内部右边界 `411`。
- Publisher 比较 semantic fingerprint、source fingerprint、Canonical revision 和
  QC report revision，防止时间轴或文本与最终报告错配。
- edit count 为 0 时禁止提供 artifact；非零修订必须提供匹配的
  `canonical_revision_artifact.v1`，不能回退发布旧文本。

## 6. 验证证据

最终冻结差异前的定向回归：

```text
.venv/bin/pytest -q \
  tests/test_canonical_qc_cli_e2e.py \
  tests/test_lerobot_v3_publish_prerequisites.py \
  tests/test_qc_reporting_projection.py \
  tests/test_asset_qc_schema_v2.py

121 passed in 144.32s
```

该组覆盖双输入 CLI、Source Gate JSON-only 统计、schema mutation、Publisher 前置、
semantic fingerprint、官方 reader E2E 与 projection。

首次 frozen review 找到并修复了 Source Gate 下游 runtime resume、runtime 到确定性
fail 的 CAS 转移、非法 locator 的可报告化、三态 retryable/module-state 关联，以及
Publisher identity/config 绑定缺口。修复后两位全新 reviewer 独立复审：代码审查
`14 passed`、spec 审查 `35 passed`，均明确无 Blocker/Important。

## 7. 最终门禁

| 门禁 | 结果 |
|---|---|
| 两个全新独立 reviewer，无 Blocker/Important | PASS：code 14 passed；spec 35 passed；均 CLEAN |
| `.venv/bin/python -m pytest -q` | PASS：1071 passed，1 skipped，322.10s |
| compileall | PASS：`canonical_qc lerobot_v3_publisher qc_pipeline tools`，exit 0 |
| `git diff --check` | PASS，exit 0 |
| schema/config 专项 | PASS：active alias 与 v1.1.0 snapshot 字节一致；v1.1.0/v1.1.1 可加载；pre-v1.1 明确不可执行 |
| HDF5/LeRobot semantic fingerprint 等价专项 | PASS：包含在 63 项专项中 |
| Publisher fault injection / atomicity 专项 | PASS：包含在 63 项专项中 |

专项命令合计 `63 passed in 48.35s`。配置快照 SHA-256：

- `canonical_qc_v1.0.0`: `cbb721a57b98735e901baefa08bbb0017e320069455bd13fe6e88d4405f77e54`
- `canonical_qc_v1.0.1`: `81c4d6f8a2d1eda8f02af76469538aadb3cccac79dd3d28a40568a61ac2a7321`
- `canonical_qc_v1.1.0`: `6620df570029761f034a0f1c283f6283cdc6caf45f55caf08fe451f509acc6ba`
- `canonical_qc_v1.1.1`: `2b30e457e25f6d96e844b8f374b24c3d047eb126f133f74a0d3a07603cce19ef`

最终结论：PASS。Task 11 的实现、边界说明、独立复审和 fresh 全量门禁均完成。

## 8. 2026-07-17 P0 Data Canonical / Publisher addendum

- `CanonicalDataEpisode` 作为兼容别名，新增 immutable `BatchMetadata`、
  `SupplierExtensions` 和完整 `data_fingerprint`。
- HDF5/LeRobot Adapter 枚举未被 Core/Evidence 消费的已支持字段；Publisher 与独立
  validator 对 feature、sidecar、metadata、stats、manifest 做 round-trip 对账。
- `canonical_revision_artifact.v1` 纯函数应用允许的语义 patch，不修改 Raw；CLI 新增
  `--batch-metadata` 和 `--revision-artifact`。
- 星际硅途真实 HDF5 样本以只读 inventory 验证：140 个 dataset（138 frame、2
  episode），前后 SHA-256 均为
  `ae72e18c82af92370eea257e7e7423c5d72e95beef3ebf052843e8f5346df5dc`。
- 京东真实 LeRobot 样本只读验证发现 `info.features` 声明 `float32`、物理 Parquet 为
  `double` 的 schema drift；Adapter 按设计 fail closed，样本 SHA-256 前后均为
  `8411a4331797a40e9bec8eba1a6b943b34927bff299626fa32d1a466afc720bd`。
- Fresh full-suite gate：`.venv/bin/python -m pytest -q` 为 `1091 passed, 1 skipped in
  83.09s`；`compileall` 与 `git diff --check` 均为 exit 0。

本 addendum 不宣称 MCAP/NPZ、多相机专用媒体或 object/vlen/ragged extension 已支持；
这些格式需要各供应商 Adapter/profile，不能删除字段后绕过发布门禁。
