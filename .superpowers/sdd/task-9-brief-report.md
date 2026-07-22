# Task 9C production overlay report

## 状态

DONE_WITH_CONCERNS

Task 9C 已在 `codex/human-qc-impl` 完成并提交。提交：
`ac83b72e0f0b1c1353624509d1115e6251048d3e`。

未修改 generic `overlay_worker.py`、Warn HTTP 路由、浏览器 JS、OpenSpec tasks、
`.comet/` 或计划文件。工作树中原有的计划/台账改动保持未提交、未触碰。

## TDD RED 证据

先只创建/修改测试，然后运行：

```bash
.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_renderer.py \
  tests/test_human_qc_launcher.py \
  tests/test_sam3_overlay_worker.py \
  tests/test_review_evidence.py
```

结果：`12 failed, 37 passed`。预期失败为：生产 renderer 模块不存在；launcher
未重建 `manifest_metadata`，没有显式模型/缓存/worker bound 参数，没有共享
runtime/worker/provider 接线，也没有 finally shutdown。

随后为“多区间 job 任一失败不得对外发布先前 ready 段”增加独立回归测试，先运行：

```bash
.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_renderer.py::test_production_provider_never_exposes_ready_segments_from_a_failed_multi_interval_job
```

结果：`1 failed`，实际状态集合为 `{"failed", "ready"}`，符合预期 RED。

## GREEN 证据

最终必测组合：

```bash
.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_renderer.py \
  tests/test_human_qc_launcher.py \
  tests/test_sam3_overlay_worker.py \
  tests/test_review_evidence.py
```

结果：`50 passed in 7.91s`。

SAM3 runner 回归（runner 未修改）：

```bash
.venv/bin/python -m pytest -q tests/test_qc_pipeline_sam3_runner.py
```

结果：`42 passed in 9.41s`。

邻接 Warn workbench/HTTP 回归：

```bash
.venv/bin/python -m pytest -q \
  tests/test_human_qc_workbench.py tests/test_human_qc_http_server.py
```

结果：`30 passed in 5.83s`。

其余验证：

```bash
.venv/bin/python -m py_compile \
  human_qc/sam3_overlay_renderer.py \
  tools/serve_human_qc_workbench.py \
  tests/test_sam3_overlay_renderer.py \
  tests/test_human_qc_launcher.py
git diff --check
.venv/bin/python tools/serve_human_qc_workbench.py --help >/dev/null
```

结果均为 exit 0。

一次完整组合运行中，既有 generic worker 的并发 cache quota 测试出现时序失败；
该测试随后连续独立运行 5 次全部通过，且最终完整组合通过。本任务未修改 worker。

## 提交文件

- `human_qc/sam3_overlay_renderer.py`（新增）
- `tools/serve_human_qc_workbench.py`
- `tests/test_sam3_overlay_renderer.py`（新增）
- `tests/test_human_qc_launcher.py`

## 设计说明

- `Sam3OverlayRenderer` 只消费 `StrictFrameProvider` 给出的显式
  source-frame -> physical-video-frame 映射，逐帧调用一个进程共享的
  `Sam3RuntimeProvider`，因此沿用其 inference lock。
- renderer 对 `[start, end)` 中每个 source frame 恰好处理一次；完整成功后才由
  generic worker fsync/replace。decode、SAM3、encoder 失败会删除临时输出。
- allowlist code 覆盖 model、source、mapping、input、decode、inference、encoder
  和通用 render 失败，不泄露路径、命令或 traceback。
- 空/错误 SAM3 mask 视为 inference failure，不发布空 mask overlay。
- `build_overlay_request()` 的 cache identity 同时包含可信 source SHA、区间、模型
  identity、runtime config、显式 mapping/keypoint recipe identity、renderer 输入与
  renderer version。
- `ProductionWorkerOverlayProvider` 是 `WorkerOverlayProvider` 子类；它只提交后台
  worker，不执行同步推理，并把 setup failure 立即投影为稳定 failed view。
- 多段 job 整体失败时，生产 provider 对外把全部段投影为 failed，避免暴露内部
  已完成但不可整体发布的 ready 段；generic worker 未被修改。
- launcher 用 `context_metadata_from_report()` 重建 metadata，构造一个共享
  `MediaCatalog`、一个 `Sam3RuntimeProvider`、一个 `BoundedOverlayWorker` 与一个
  production `WorkerOverlayProvider`，并把 provider 注入 `WarnWorkbenchService`。
- cache 为 `<asset batch root>/<relative cache dir>/<asset-id hash>`，通过 resolve +
  `relative_to(batch_root)` 拒绝逃逸。
- 未配置模型时使用即时 `overlay_model_unavailable` 终态 provider，不留下永久
  pending。`main()` 的 finally 同时关闭 server 与 overlay runtime/worker。

## 遗留顾虑

- 当前已存在的 supplier 报告未被假定拥有严格可重建 mapping。本实现只接受报告
  中显式的 `sam3_overlay_input.v1` recipe；缺失 recipe 的资产立即返回
  `overlay_mapping_unavailable`，不会猜测直接帧索引。
- 本任务按约束没有扩大到 runner recipe 持久化。后续若某个 supplier 要真正启用
  production overlay，需要由该 supplier runner 持久化经过验证的逐帧 mapping、
  keypoint input identity 与 source SHA；在此之前 fail-closed 是有意行为。
- 没有真实本地 SAM3 权重，因此本轮使用 fake locked segmenter/encoder 验证连续
  调用、失败原子性和缓存身份；实际模型加载与 GPU/codec 环境仍需部署验收。

---

# Task 9C review round 1 fix report

## 状态与提交

DONE_WITH_CONCERNS

Review round 1 的 7 项问题已修复并提交：
`aca6981`（`fix(sam3): harden production overlay wiring`）。

本轮只提交以下实现与聚焦测试；工作树中原有的计划文件与 `.comet` 台账修改仍未
暂存、未提交：

- `human_qc/sam3_overlay_renderer.py`
- `tools/serve_human_qc_workbench.py`
- `qc_pipeline/runners/sam3_containment.py`
- `tests/test_sam3_overlay_renderer.py`
- `tests/test_human_qc_launcher.py`
- `tests/test_qc_pipeline_sam3_runner.py`
- `tests/test_canonical_qc_runner_bridge.py`

## TDD RED / GREEN 证据

### 1. runner recipe 与真实重建链路

RED：

```bash
.venv/bin/python -m pytest -q \
  tests/test_qc_pipeline_sam3_runner.py::test_jdt_runner_persists_versioned_rehydratable_overlay_recipe \
  tests/test_qc_pipeline_sam3_runner.py::test_runner_report_rehydrates_through_launcher_and_production_worker_reaches_ready
```

结果：`2 failed`，分别为缺失 `overlay_input_recipe` 和生产 provider 无法重建映射。

GREEN：同一命令结果 `2 passed`。JDT runner 现在持久化可信视频 SHA、逐 source
frame 的物理视频帧映射，以及带 SHA/列名/row mapping 的 Parquet 2D 引用；真实
runner -> report JSON -> launcher -> `MediaCatalog` -> production worker 链路达到
`ready`。

DR/QY RED：

```bash
.venv/bin/python -m pytest -q \
  tests/test_qc_pipeline_sam3_runner.py::test_validated_dr_candidates_use_hdf5_calibration_without_jdt_parquet \
  tests/test_qc_pipeline_sam3_runner.py::test_qy_runner_persists_only_proven_continuous_overlay_inputs
```

结果：`2 failed`，均为缺失 recipe。

DR/QY GREEN：在同一选择中加入
`test_dr_invalid_depth_nan_and_invalid_hand_are_recorded`，结果 `3 passed`。DR 只在
声明视频、clip offset 和投影后有限有效点均可证明时生成 inline recipe；QY 只在
每个必需手侧存在且同一 source frame 映射到同一 video frame 时生成，否则不附加
recipe，保持 fail-closed。

冻结 report context RED：

```bash
.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_renderer.py::test_inline_recipe_rehydrates_from_frozen_report_context
```

结果：`1 failed`，`_FrozenMapping` 不能直接 JSON 序列化。修复后将该测试与 JDT
真实 E2E 一起运行，结果 `2 passed`。

Canonical offset RED：

```bash
.venv/bin/python -m pytest -q \
  tests/test_canonical_qc_runner_bridge.py::test_shifted_canonical_sam3_reads_physical_frame_and_reports_logical_frame
```

最初结果：`1 failed`，缺失 recipe。实现后第一次运行只暴露测试期望漏写
`sha256:` 前缀；修正期望后同一命令结果 `1 passed`，证明 logical frame `0`
严格映射到 physical frame `3`。Canonical 同样仅在视频来源、物理帧区间、候选
边界和有效有限 2D 点都可证明时附加 recipe。

### 2. batch-global cache quota 与安全默认值

RED：

```bash
.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_renderer.py::test_production_cache_quota_is_shared_across_assets_and_keeps_asset_identity \
  tests/test_human_qc_launcher.py::test_launcher_has_a_non_none_safe_default_batch_cache_limit
```

结果：`2 failed`，原实现按资产拆分 cache root，默认 quota 为 `None`。

GREEN：同一命令结果 `2 passed`。cache 改为 batch 共用
`<batch_root>/.human_qc/overlay-cache`，默认上限 2 GiB；fingerprint 显式包含
`asset_id`，跨资产共享目录时仍保持资产身份隔离。

### 3. 视频 source/seek/boundary 完整性

RED：参数化 seek false、seek 后位置错误、越界，以及同 asset source identity
变化的聚焦选择共 `4 failed`。GREEN：加入真实 JDT E2E 后共 `5 passed`。

随后新增同尺寸源文件替换测试：

```bash
.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_renderer.py::test_explicit_recipe_provider_rejects_source_file_substitution_before_decode
```

RED 为 `1 failed`；实现视频内容 SHA 在 decoder open 前复验后，与 seek/source
identity 选择合跑结果 `5 passed`。provider cache key 改为
`(asset_id, source.etag)`；读取前验证实际 source SHA、frame count、seek 返回值和
seek/read 前后位置。

### 4. model identity

```bash
.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_renderer.py::test_model_hash_covers_all_directory_content_even_when_size_and_mtime_are_preserved
```

RED：`1 failed`，嵌套 shard 内容在 size/mtime 不变时未改变 hash。GREEN：`1
passed`。现递归读取模型目录内所有 regular file 的完整内容并构造稳定 manifest
hash，不再依赖 size/mtime 作为内容身份。

### 5. MP4 产品验证

RED：真实 OpenCV MP4 round-trip 和 junk encoder product 两个测试均失败（缺少
probe metadata，坏文件被接受）。GREEN：两个新测试连同 renderer 关键回归选择
共 `7 passed`。encoder close 后使用 `canonical_qc.video_probe.probe_video` 验证
精确 frame count、宽高、fps 和 `mpeg4` codec；probe 或字段不匹配时删除产品并
返回 `overlay_encoder_failed`。

### 6. runtime 生命周期所有权

```bash
.venv/bin/python -m pytest -q \
  tests/test_human_qc_launcher.py::test_compatibility_service_retains_and_exposes_its_shutdown_owner \
  tests/test_human_qc_launcher.py::test_runtime_preload_failure_shuts_down_overlay_resources
```

RED：`2 failed`。GREEN：加入既有 `test_main_closes_overlay_worker_in_finally` 后
结果 `3 passed`。兼容 service 保留 runtime owner/shutdown；service 构建或 preload
异常会关闭已创建 runtime；`main()` 构建失败路径也由 finally 收口。

### 7. malformed frame 稳定错误码

```bash
.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_renderer.py::test_malformed_frame_array_maps_to_stable_decode_failure
```

RED：`1 failed`，ragged array 泄漏原始 `ValueError`。GREEN：`1 passed`，现在稳定
映射为 `overlay_decode_failed`。

## 最终验证

提交前聚焦组合：

```bash
.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_renderer.py \
  tests/test_human_qc_launcher.py \
  tests/test_sam3_overlay_worker.py \
  tests/test_review_evidence.py \
  tests/test_qc_pipeline_sam3_runner.py \
  tests/test_canonical_qc_runner_bridge.py
```

结果：`120 passed in 3.99s`。

真实链路/媒体四项：

```bash
.venv/bin/python -m pytest -q \
  tests/test_qc_pipeline_sam3_runner.py::test_runner_report_rehydrates_through_launcher_and_production_worker_reaches_ready \
  tests/test_sam3_overlay_renderer.py::test_production_cache_quota_is_shared_across_assets_and_keeps_asset_identity \
  tests/test_sam3_overlay_renderer.py::test_default_encoder_output_is_probed_as_exact_decodable_mp4 \
  tests/test_sam3_overlay_renderer.py::test_renderer_rejects_bad_mp4_product_after_encoder_close
```

结果：`4 passed in 1.01s`。

邻接 workbench/HTTP：

```bash
.venv/bin/python -m pytest -q \
  tests/test_human_qc_workbench.py tests/test_human_qc_http_server.py
```

结果：`30 passed in 5.77s`。

全量：

```bash
.venv/bin/python -m pytest -q
```

结果：`1830 passed, 1 skipped in 109.81s`。

静态验证：`git diff --check` 与三个修改实现文件的 `py_compile` 均 exit 0。

## 遗留顾虑

- 本地没有真实 SAM3 权重/GPU；production wiring、真实 MP4 编解码/probe 和 worker
  lifecycle 已验证，但真实模型加载与 GPU 推理仍需部署环境验收。
- 模型身份现在是递归全内容 SHA，正确性优先，但大型分片模型首次启动的 hash I/O
  成本需要在部署侧观测。
- 无法严格证明映射/2D 输入的 supplier 或历史 report 继续即时
  `overlay_mapping_unavailable`，不会猜测或降级为直接帧索引。

---

# Task 9C final production-bounds repair

## Root cause / data-flow 结论（修复前）

1. **failed job storage / quota**：`BoundedOverlayWorker._render()` 在多段 job
   中每段成功后立即把 temporary MP4 replace 为 final path；后段失败时
   `_failed_view(..., segments=produced)` 仍保留前段 `ready/path`。
   `_publish_rendered()` 只对 `candidate.status == "ready"` 执行 `_evict_for()`，
   `_evict_for()` 又只统计 `manifest.status == "ready"` 的目录。因此 failed
   job 中的 MP4 和 manifest 都绕过 batch quota；production provider 只在
   DTO 投影时把 path 隐去，没有清理真实文件。

2. **ready URL pin lifecycle**：`ProductionWorkerOverlayProvider ->
   WorkerOverlayProvider -> WarnWorkbenchService._overlay_segment_dto() ->
   MediaCatalog.allow_overlay()` 只登记 path，整条链没有调用 worker
   `pin()/unpin()`。因此 task DTO 已返回 ready URL 后，另一资产发布
   可以在浏览器首次请求前淘汰该文件；反过来，如果只做永久
   pin，cache 又永远无法淘汰。需要把有限 pin lease 与 MediaCatalog
   的可请求期绑定，GET/media access 续租，release/shutdown 解租，超期
   后 fail closed。

3. **container validation**：`Sam3OverlayRenderer.render_interval()` 只比较
   frame count/size/fps 和 `codec == "mpeg4"`；`canonical_qc.video_probe.probe_video()`
   未请求/暴露 ffprobe `format_name`。因此只要是 mpeg4 video stream，
   AVI 等非 MP4 container 即使改名为 `.mp4` 也会被当成成品。

4. **recipe strictness / historical report**：launcher `load_contexts()` 对报告
   schema、asset identity、SAM3 flow/result/exit gate 和 artifact 完成性都不做 recipe
   门禁，只要 `sam3_containment.runtime` 是 Mapping 就复制
   `overlay_input_recipe`。renderer v1 同时接受 inline `keypoints_2d` 和
   reference，存在双 payload/identity 歧义，cache identity 不一定绑定实际
   消费内容。

5. **runner behavioral isolation / report size**：JDT/DR/QY runner 在主 SAM3
   producer 运行前构建 overlay recipe；DR/QY builder 遍历所有候选帧，任一
   未采样帧投影/关键点异常都会改变原 sampled SAM3 的行为。
   DR/QY/Canonical 还把每帧关键点和逐帧 mapping 全部内联到
   module runtime，使 `asset_qc_report.v2` 随候选窗口无界增长。

6. **GET work/resource/TOCTOU**：`get_asset_task()` 调用
   `MediaCatalog.source()`，首次 GET 同步全文件 SHA-256 + ffprobe；production
   request factory 又在 GET 中首次构建 provider，v1 provider constructor 同步
   hash/read Parquet，并在 `(asset_id, source.etag)` dict 中无上限保留
   DataFrame/VideoCapture。decoder `_open()` 是“按路径 hash -> 按路径重新
   VideoCapture”，两步之间替换文件可使实际渲染源脱离 cache
   fingerprint。需在启动阶段冻结有界 source metadata，在 bounded worker
   中延迟物化 recipe，并使用同一已验证 OS file identity/fd 解码。

## 修复结果

1. failed job 现在在任一 segment 失败时删除全部 partial/final MP4，把所有
   segment 规范化为无 path 的 failed，并把 failed manifest 与 ready job 一起纳入
   batch byte quota；failed manifest 优先淘汰且不消耗 ready-job 数量配额。
2. production provider 在 ready DTO 发布前创建有限 worker pin lease，
   MediaCatalog 在实际媒体请求时续租，在 review lease release、asset replacement、
   runtime shutdown、allowlist expiry 或 backing file 消失时解租。跨资产淘汰不会
   删除仍可请求的 ready URL，lease 也不会永久阻止淘汰。
3. canonical ffprobe contract 新增 container format，renderer 只接受实际 MP4
   container 中的 mpeg4 stream；AVI 即使改名 `.mp4` 也 fail closed。
4. launcher 只从完整通过 `asset_qc_report.v2` schema 校验、asset/fingerprint
   一致、SAM3 flow 已完成且 artifact 为 computed/reused 的 runtime 接受严格
   `sam3_overlay_input.v2`。v1、双 inline/reference payload、stale fingerprint、
   不连续 mapping 和历史未完成 report 均不进入 production context。
5. JDT/DR/QY/Canonical recipe 改为 sampled SAM3 结束后的 best-effort
   post-processing；任何 overlay-only 异常都不改变既有 PASS/WARN/FAIL。
   report 只保留合并半开区间、线性 mapping ranges 和 SHA/size 绑定的 Parquet/JSON
   sidecar reference，不再内联逐帧 mapping/keypoints。
6. MediaCatalog 在启动期一次性冻结 source hash/probe，request path 只做 OS identity
   stat；frame provider 不再按 identity 永久缓存，Parquet/JSON/decoder 只在 bounded
   worker 中物化并在每个 interval 后关闭。sidecar 从同一已验证 fd 内容读取；视频
   SHA 和 OS identity 从同一打开 fd 验证，decoder 使用该 fd，关闭 TOCTOU 路径重开
   窗口。失败的 Parquet snapshot 也显式关闭。

## TDD 证据

各 finding 的最小 RED（修复前）与对应 GREEN：

```bash
.venv/bin/pytest -q \
  tests/test_sam3_overlay_worker.py::test_failed_multi_segment_jobs_delete_media_and_remain_inside_batch_quota
# RED: 1 failed（failed job 仍保留 ready segment path）
# GREEN: 1 passed

.venv/bin/pytest -q \
  tests/test_sam3_overlay_renderer.py::test_renderer_rejects_mpeg4_stream_inside_non_mp4_container \
  tests/test_sam3_overlay_renderer.py::test_renderer_closes_frame_provider_after_each_bounded_interval
# RED: 2 failed（AVI/mpeg4 被接受；provider close_calls == 0）
# GREEN: 2 passed

.venv/bin/pytest -q \
  tests/test_human_qc_launcher.py::test_launcher_accepts_recipe_only_from_current_completed_v2_sam3_report \
  tests/test_human_qc_launcher.py::test_launcher_rejects_historical_stale_ambiguous_or_gapped_overlay_recipe
# RED: 7 failed（六类无效/历史 recipe 均被接收）
# GREEN: 7 passed

.venv/bin/pytest -q \
  tests/test_qc_pipeline_sam3_runner.py::test_overlay_recipe_postprocessing_cannot_change_sampled_sam3_verdict \
  tests/test_qc_pipeline_sam3_runner.py::test_long_jdt_overlay_recipe_is_compact_and_does_not_inline_per_frame_data
# RED: 2 failed（recipe 异常在 QC 前传播；v1 逐帧 payload）
# GREEN: 3 passed（PASS/FAIL 参数化 + compact recipe）

.venv/bin/pytest -q \
  tests/test_human_qc_media.py::test_startup_frozen_source_makes_request_path_metadata_only_and_replacement_fails_closed \
  tests/test_human_qc_media.py::test_overlay_allowlist_lease_renews_releases_and_expires
# RED: 2 failed（无 startup freeze；allow_overlay 无 lease contract）
# GREEN: 2 passed

.venv/bin/pytest -q \
  tests/test_sam3_overlay_renderer.py::test_ready_overlay_url_is_pinned_until_catalog_lease_expiry_across_asset_eviction \
  tests/test_sam3_overlay_renderer.py::test_v2_sidecar_provider_decodes_from_verified_open_file_identity_not_reopened_path \
  tests/test_sam3_overlay_renderer.py::test_request_time_frame_provider_creation_does_not_read_parquet
# RED: 3 failed（无 catalog pin；无 expected fd identity；v2/deferred Parquet 未实现）
# GREEN: 3 passed
```

聚焦 + 邻近：

```bash
.venv/bin/pytest -q \
  tests/test_sam3_overlay_renderer.py tests/test_human_qc_launcher.py \
  tests/test_sam3_overlay_worker.py tests/test_review_evidence.py \
  tests/test_qc_pipeline_sam3_runner.py tests/test_canonical_qc_runner_bridge.py \
  tests/test_human_qc_media.py
# 155 passed in 4.88s

.venv/bin/pytest -q \
  tests/test_human_qc_workbench.py tests/test_human_qc_http_server.py \
  tests/test_canonical_video_probe.py
# 110 passed in 6.57s
```

最终全量与 diff：

```bash
.venv/bin/pytest -q
# 1848 passed, 1 skipped in 110.41s

git diff --check
# exit 0
```

## 范围与遗留风险

- 未修改 Task 10 route/browser JS；未修改 OpenSpec、`.comet` 或主计划文件。
- 提交主题：`fix(sam3): bound overlay production lifecycle`（最终 SHA 见任务回报）。
- 本机仍无真实 SAM3 权重/GPU，真实模型加载和 GPU 推理需部署环境验收；本任务已覆盖
  real MP4/ffprobe、真实 JDT report -> launcher -> bounded worker ready 链路。

---

# Task 9C final review Important repair

## 状态与范围

DONE

本轮只修复最终审查遗留的两项 Important：overlay cache 的 terminal quota/空目录
边界，以及 ISO BMFF `format_name` 无法证明实际 MP4 container 的问题。未修改
runner、recipe、launcher、HTTP、browser JS、OpenSpec、主计划或 `.comet`。

基线：`0adc80c157550aae75ac0fe39cde84a73a89115f`。本轮提交主题：
`fix(sam3): close overlay quota and container gaps`；包含本报告的最终提交 SHA 以任务
回报中的 `git rev-parse HEAD` 结果为准（Git commit 无法在自身已跟踪内容中嵌入其
最终 SHA）。

## Root cause

1. `_evict_for()` 同时扫描 ready/failed terminal job，但每淘汰一个 victim 都无条件
   递减 ready `target_count`。当较旧 failed manifest 先被淘汰时，代码错误地认为已
   释放 ready slot，`max_ready_jobs=1` 可以最终留下两个 ready manifest。
2. ready candidate 因 byte quota 失败转成 `overlay_cache_full` 后，旧分支直接
   `_persist()` failed manifest，没有再次走 failed-byte quota。对无法容纳最小 failed
   manifest 的极小 quota，删除 current job 后，finally 中 owner cleanup 又通过会
   `mkdir` 的 `_key_lock()` 重建 digest 目录，留下仅 lock 的不可扫描空目录；不同失败
   key 会持续累积。
3. ffprobe 对正常 MP4、QuickTime MOV 和 3GP 都返回
   `format_name=mov,mp4,m4a,3gp,3g2,mj2`。真实 ffmpeg 样本的可区分字段分别为
   `major_brand=isom`、`qt  `、`3gp4`；原 renderer 只检查 `format_name` 中含
   `mp4`，因此 MOV/3GP 改名 `.mp4` 后仍被误收。

## 最小修复

- ready quota 只在真实 ready victim 被淘汰时递减；failed victim 仍可先释放 byte，
  但不释放 ready-count。
- render 的所有 failed 终态（含 ready -> cache-full）在持久化前统一通过
  `_evict_failed_for()`。若最小 failed manifest 也放不下，保留无 path 的进程内安全
  failed view，不落盘超额内容。
- owner release 使用不创建 lock 的模式；current job 已被 quota cleanup 删除时不再
  重建 digest 目录。
- `ProbedVideo`/ffprobe contract 新增 `container_major_brand`。renderer 在既有
  format/codec/frame/size/fps 检查之外，只接受明确的 MP4 video major-brand allowlist；
  `qt`、`3gp*`、`M4A` 等均 fail closed，并继续映射稳定
  `overlay_encoder_failed`。

## TDD RED / GREEN 证据

先只修改测试并运行：

```bash
.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_worker.py::test_worker_reports_cache_full_without_publishing_a_ready_manifest \
  tests/test_sam3_overlay_worker.py::test_failed_job_eviction_does_not_release_a_ready_job_slot \
  tests/test_sam3_overlay_worker.py::test_distinct_failed_keys_do_not_accumulate_empty_job_directories_under_tiny_quota \
  tests/test_canonical_video_probe.py::test_probe_video_reads_metadata_rational_fps_and_normalized_pts \
  tests/test_sam3_overlay_renderer.py::test_renderer_requires_an_mp4_compatible_major_brand_from_real_ffmpeg
```

RED：`7 failed`。精确失败分别为：tiny quota 后 digest 目录仍存在；混合
failed/ready 下 ready manifest 为 2；不同 failed key 留下空 digest 目录；probe
contract 缺少 major brand；真实 QuickTime/3GP 未拒绝；正常 MP4 metadata 缺少
major brand。

最小实现后，同一选择 GREEN：`7 passed in 2.85s`。随后加入真实 ffmpeg
`M4A ` major-brand 视频 case，与 MOV/3GP/正常 MP4 四项合跑：

```bash
.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_renderer.py::test_renderer_requires_an_mp4_compatible_major_brand_from_real_ffmpeg
# 4 passed in 0.64s
```

## 最终验证

Task 9C 聚焦组合（原报告 155 项，加本轮新增参数化回归后为 161 项）：

```bash
.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_renderer.py tests/test_human_qc_launcher.py \
  tests/test_sam3_overlay_worker.py tests/test_review_evidence.py \
  tests/test_qc_pipeline_sam3_runner.py tests/test_canonical_qc_runner_bridge.py \
  tests/test_human_qc_media.py
# 161 passed in 5.15s
```

邻接测试：

```bash
.venv/bin/python -m pytest -q \
  tests/test_human_qc_workbench.py tests/test_human_qc_http_server.py \
  tests/test_canonical_video_probe.py
# 110 passed in 6.52s
```

全量：

```bash
.venv/bin/python -m pytest -q
# 1854 passed, 1 skipped in 120.71s
```

`py_compile`（4 个业务文件和 3 个聚焦测试文件）及 `git diff --check` 均 exit 0。

---

# Task 9C terminal-failure quota convergence follow-up

## 状态与范围

DONE

本轮继续只修复最终缓存验收遗漏：公开 interrupted recovery、`_run` exception
fallback、cleanup failure 三条 terminal-failed 路径的 quota 收敛，以及 heartbeat 与
quota cleanup 交错后重建空 digest 目录的问题。仅修改
`human_qc/overlay_worker.py`、`tests/test_sam3_overlay_worker.py` 和本报告；未触碰
runner、recipe、launcher、HTTP、browser、OpenSpec、主计划或 `.comet`。

基线：`9e7b7cd330487df5d0b81042b4612d3847ecc2df`。本轮独立提交 SHA 以最终任务
回报中的 `git rev-parse HEAD` 为准。

## Root cause 与锁约束

1. `_recover_interrupted_locked()`、`_run` 的 catch fallback 和
   `_persist_cleanup_failure()` 各自直接 `_persist(failed)`，绕过上一轮只在 render
   publication 分支加入的 failed-byte quota。`max_cache_bytes=1` 时，公开
   `worker.get()` 恢复 pending/generating 会落盘超额 failed manifest；fallback 与
   cleanup failure 也会留下 terminal digest 目录。
2. heartbeat 使用默认 `create=True` 的 `_key_lock()`，owner 续租又调用会创建父目录
   的 `_atomic_json()`。quota 已删除 current job 后、heartbeat 停止前的窗口内，续租
   线程可以重建 `.generation.lock`-only digest 目录。
3. 修复后的唯一 failed 落盘点为 `_publish_failed_locked()`；其调用方必须已按
   root publish lock -> current key lock 的顺序持锁。外层 `_publish_failed()` 统一
   获取这一顺序。公开 interrupted recovery 使用 root -> key -> nonblocking render
   fence；schedule 内部 recovery 只构造 view、不落 terminal manifest。heartbeat
   只尝试 nonblocking existing-key lock，不取 root，并使用 existing-parent-only 原子
   写，因此既不形成 key -> root 反序，也不能复活已删除目录。

## TDD RED / GREEN

先只新增四个确定性测试并运行：

```bash
.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_worker.py::test_public_interrupted_recovery_respects_tiny_quota_without_accumulating_job_dirs \
  tests/test_sam3_overlay_worker.py::test_run_exception_fallback_failed_view_respects_tiny_quota \
  tests/test_sam3_overlay_worker.py::test_cleanup_failure_failed_view_respects_tiny_quota \
  tests/test_sam3_overlay_worker.py::test_heartbeat_after_quota_cleanup_does_not_recreate_an_empty_job_directory
```

RED：`4 failed in 0.46s`。四项均稳定失败于 digest 目录仍存在：

- 公开 `worker.get()` 对预置 pending/generating 两个不同 key 返回安全
  `overlay_interrupted`，但持久化超额 failed manifest 并累积目录；
- 强制 publication exception 后，`_run` fallback 绕过 quota；
- 强制 owner cleanup error 后，cleanup-failed manifest 绕过 quota；
- 短 owner lease `0.03s` + Event 协调，确保 quota 删除后至少执行一次 heartbeat，
  稳定重建 lock-only 目录。

最小实现后同一命令 GREEN：`4 passed in 0.34s`。为验证交错测试不是偶然通过，
同一四项连续运行 5 轮，五轮均为 `4 passed`（单轮 0.18-0.24s）。

## 最终验证

```bash
.venv/bin/python -m pytest -q tests/test_sam3_overlay_worker.py
# 23 passed in 2.16s

.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_renderer.py tests/test_human_qc_launcher.py \
  tests/test_sam3_overlay_worker.py tests/test_review_evidence.py \
  tests/test_qc_pipeline_sam3_runner.py tests/test_canonical_qc_runner_bridge.py \
  tests/test_human_qc_media.py
# 165 passed in 5.22s

.venv/bin/python -m pytest -q \
  tests/test_human_qc_workbench.py tests/test_human_qc_http_server.py \
  tests/test_canonical_video_probe.py
# 110 passed in 5.88s

.venv/bin/python -m pytest -q
# 1858 passed, 1 skipped in 113.38s
```

`py_compile human_qc/overlay_worker.py tests/test_sam3_overlay_worker.py` 与
`git diff --check` 均 exit 0。

---

# Task 9C final-footprint and owner-handoff follow-up

## 状态与范围

DONE

本轮仅修复 failed publication 的最终真实目录 footprint，以及 cleanup failure 与
successor owner 交接的隔离。只修改 `human_qc/overlay_worker.py`、
`tests/test_sam3_overlay_worker.py` 和本报告；未触碰 runner、recipe、launcher、
HTTP、browser、OpenSpec、主计划或 `.comet`。

基线：`0f2983765dba815bda5420534eb5e3308b33a047`。本轮独立提交 SHA 以最终任务
回报中的 `git rev-parse HEAD` 为准。

## Root cause

1. `_publish_failed_locked()` 只把新 failed manifest 的序列化长度传给
   `_evict_failed_for()`，而该扫描有意排除 current job。旧 manifest 会被替换，但
   cleanup release 失败后 owner JSON 会长期残留；因此 quota 判断低估最终目录。
   reviewer 边界在本机精确复现为 `max_cache_bytes=647`、最终目录 `650 bytes`。
2. `_run` exception fallback 已携带原 owner token，但 cleanup-failed publication
   仍以 `owner_token=None` 调用。旧 worker release 报错、token 失效且 successor
   worker 接管后，旧 cleanup 可以在 key lock 内删除 successor 的新 owner、manifest
   和 job，令 successor 最终 `overlay_interrupted`。

## 最小修复与所有权语义

- failed publication 在删除 media 后，以最终新 manifest 字节替换旧
  `manifest.json`，再累加 current job 目录其余所有 regular file 的实际字节；owner
  JSON、generation lock、render fence 均被枚举（0-byte lock/fence 也不遗漏）。任何
  `iterdir/is_file/stat` 错误都视为 footprint unknown，fail closed 为 memory-only 并
  清理 current job，绝不低估。
- `_publish_failed()` 的 `owner_token` 改为必填；`_run` fallback 与 cleanup failure
  均传原 token。调用方在 root -> key 锁内重新验证 token；mismatch 时只返回已规范化
  的内存 failed view，不 parse/persist/discard/safe-remove，因此完全不触碰 successor
  durable state。直接调用 `_publish_failed_locked()` 的 recovery/render 分支仍在同一
  key lock 内先证明无 owner 或证明当前 token 所有权。

## TDD RED / GREEN

先只新增两个测试：

```bash
.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_worker.py::test_cleanup_failure_quota_counts_the_remaining_owner_file_in_final_footprint \
  tests/test_sam3_overlay_worker.py::test_stale_cleanup_failure_cannot_delete_a_successor_workers_owned_job
```

RED：`2 failed in 0.34s`。

- footprint case 的最终目录为 `650 bytes > 647`；
- 双 worker case 用 Barrier 保证 successor 的 BlockingRenderer 已启动、即新 owner 已
  接管后才放行旧 cleanup。旧 cleanup 观测 token 为 `None`，随后删除 successor job，
  successor 实际 `failed` 而非 `ready`。旧 worker 的 cleanup 阶段使用 reviewer
  `409` byte boundary，以隔离 handoff 语义。

最小实现后同一命令 GREEN：`2 passed in 0.33s`。同一 footprint + Barrier 组合连续
运行 10 轮，十轮全部通过（单轮 0.16-0.31s）。

## 最终验证

```bash
.venv/bin/python -m pytest -q tests/test_sam3_overlay_worker.py
# 25 passed in 1.69s

.venv/bin/python -m pytest -q \
  tests/test_sam3_overlay_renderer.py tests/test_human_qc_launcher.py \
  tests/test_sam3_overlay_worker.py tests/test_review_evidence.py \
  tests/test_qc_pipeline_sam3_runner.py tests/test_canonical_qc_runner_bridge.py \
  tests/test_human_qc_media.py
# 167 passed in 5.06s

.venv/bin/python -m pytest -q \
  tests/test_human_qc_workbench.py tests/test_human_qc_http_server.py \
  tests/test_canonical_video_probe.py
# 110 passed in 5.97s

.venv/bin/python -m pytest -q
# 1860 passed, 1 skipped in 113.57s
```

`py_compile human_qc/overlay_worker.py tests/test_sam3_overlay_worker.py` 与
`git diff --check` 均 exit 0。
