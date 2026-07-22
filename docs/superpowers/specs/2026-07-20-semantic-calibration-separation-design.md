# 语义校准独立化与人工质检前置设计

日期：2026-07-20

实现状态（2026-07-22）：已落地独立 `semantic_calibration` 包、独立 HTTP server、
`/api/semantic/...`、独立静态页面和 `tools/serve_semantic_calibration.py`；正式操作
说明见 `docs/semantic-calibration-workbench.md`。

## 目标

将人工语义校准从现有 `human_qc` 共用工作台中完整拆出，形成独立前端、独立后端、独立启动入口和独立 API 命名空间。现有语义校准能力必须原样保留；人工质检前后端不再承载、导入或渲染语义校准代码。

流水线顺序调整为：

`自动 QC → 人工质检 → 语义校准 → 后续模块/完成`

人工质检确认 Fail 后，资产立即终止后续流程，不得创建或打开语义校准任务。人工质检确认 Pass 后，资产才进入语义校准。没有人工候选、因而被既有规则标记为 `not_required` 的资产保持现有跳过行为，并可直接进入语义校准，避免无 Warn 资产永久阻塞。

## 已确认的范围

### 必须保留

- `asset_qc_report.v2` 作为跨模块正式状态源。
- 语义校准的完整视频、字幕、subtask 时间轴和文字编辑体验。
- 时间轴只允许拖动内部左右边界，不允许拖动整个区间。
- 相邻 subtask 共享边界：拖动一个边界必须在同一事务中同时修改左段结束帧和右段起始帧。
- 单一 pending edit、逐次确认/取消、revision 检查和 reviewer lease。
- HDF5 临时副本校验、fsync、原子替换、失败时保持原文件不变。
- 修改次数与 before/after 审计。
- 当前人工质检 JSON 合同、候选全量选入策略和后端判定语义，供后续人工质检重做时继续使用。
- acceptance 与 supplier evaluation 已有的自动 hard-fail 差异；人工 Pass 不得覆盖已有机器 hard fail。

### 本次移除或停用

- 语义校准与 Warn 人工质检共用一个 `WorkbenchService`、HTTP server 和静态页面的设计。
- 语义页面中的 Warn adapter、Warn 样式、问题证据、Pass/Fail 和人工质检文案。
- 人工质检服务对语义页面、语义 adapter、语义编辑 API 和 HDF5 写回能力的托管。
- 语义完成后再创建人工质检任务的旧流转。
- 当前共用页面作为正式入口的地位。人工质检页面后续单独重做，本次不设计其新界面。

## 架构边界

### 语义校准模块

新增顶层 `semantic_calibration` 包，并由其独占以下职责：

- 统一 subtask source adapter 和共享边界时间轴。
- boundary/text pending transaction。
- reviewer lease 与 report revision 校验。
- 语义任务投影、任务队列和 eligibility 检查。
- HDF5 临时写入、完整校验和原子替换。
- 独立 HTTP server 与静态资源。

正式启动入口为 `tools/serve_semantic_calibration.py`。前端资源放入独立的语义静态目录，只加载语义应用和语义样式。

语义模块可以依赖稳定的 `qc_common` 报告、配置和原子写入合同，但不得依赖 `human_qc` 包、Warn service、Evidence service 或人工质检静态资源。

### 人工质检模块

`human_qc` 继续拥有当前 Warn 候选、人工判定、SAM3 evidence 和 QC JSON 写回能力，但不得导入 `semantic_calibration`。现有人工质检前端不作为本次交付目标；当前共用前端入口停止承担语义校准展示。

人工质检与语义校准唯一允许的耦合面是持久化的 `asset_qc_report.v2` 状态合同。两边不得通过 Python service 对象、前端 adapter、DOM 或同一 HTTP facade 直接通信。

### 共享基础设施

确实与领域无关的原子 JSON 写入、report revision、路径限制等能力继续放在 `qc_common`。reviewer lease 若两边都需要，迁入中立公共模块；不得为了复用而让语义校准反向导入 `human_qc`。

## 流转与状态合同

### 人工质检前置门禁

语义任务 eligibility 由服务端根据最新 QC JSON 计算，浏览器参数不得绕过。规则如下：

| 人工质检状态 | 人工结论 | 语义行为 | 流水线行为 |
| --- | --- | --- | --- |
| `queued` / `in_progress` / `not_evaluated` | 未形成 | 不创建、不列出、不可直接打开 | 停留在人工质检 |
| `completed` | 所有选中 issue 均为 Pass | 创建或恢复语义待办 | `next_module=semantic_consistency` |
| `completed` | 任一选中 issue 为 Fail | 标记 `semantic_calibration.state=skipped_due_to_fail` | `status=stopped`，`next_module=null` |
| `not_required` | 无人工候选 | 创建或恢复语义待办 | `next_module=semantic_consistency` |
| `skipped_due_to_fail` / `error` | 不可通过 | 不创建、不列出、不可直接打开 | 保持终止或错误状态 |

人工 Pass 使用已有 `selected_issue_ids` 与 `issue_reviews` 推导：所有已选 issue 均已判定且 verdict 全部为 `pass`。当前 `all_candidates` 策略继续在进入人工质检时将 `candidate_issue_ids` 快照到 `selected_issue_ids`，但文档明确它是可替换的 selection seam，后续可以改为抽样、风险或预算策略。

不新增一个可被客户端伪造的“允许进入语义”布尔字段。资格由人工质检的持久化事实和流水线游标共同决定。

### 原子状态迁移

人工质检最后一项保存时，服务端在一次 revision-aware 报告写入中完成状态迁移：

- 全部 Pass：人工质检标记完成，流水线游标推进到 `semantic_consistency`，语义状态初始化为可开始。
- 任一 Fail：人工质检标记完成，`overall_decision` 保持或归约为 Fail，流水线停止，语义状态标记为 `skipped_due_to_fail`。

若进程在报告落盘后、语义服务载入前退出，语义服务重启时按 QC JSON 幂等恢复任务，不依赖跨服务内存消息。

语义完成后只负责完成自身阶段并推进后续模块或结束流水线；不得再选择人工候选、创建人工质检任务或调用人工质检 service。

## 独立语义 API

语义 server 只暴露 `/api/semantic/...`：

- `GET /api/semantic/assets`：返回符合门禁的语义资产和状态。
- `GET /api/semantic/assets/{asset_id}/task`：返回单资产语义任务。
- `POST /api/semantic/assets/{asset_id}/lease/acquire`
- `POST /api/semantic/assets/{asset_id}/lease/renew`
- `POST /api/semantic/assets/{asset_id}/lease/release`
- `POST /api/semantic/assets/{asset_id}/boundary/pending`
- `POST /api/semantic/assets/{asset_id}/text/pending`
- `POST /api/semantic/assets/{asset_id}/pending/confirm`
- `POST /api/semantic/assets/{asset_id}/pending/cancel`
- `POST /api/semantic/assets/{asset_id}/complete`

语义 server 不提供 Warn task、人工 Pass/Fail、issue evidence 或通用 `/api/assets/{id}/task` 路由。请求一个未通过前置门禁的资产时，返回稳定的业务错误码和可读原因；不能返回语义任务 DTO，也不能允许写操作。

## 独立语义页面

直接打开语义服务根地址时，页面必须立即进入以下三种稳定状态之一，不能停留在“未加载资产”的空壳：

1. `?asset_id=...` 指定了可用资产：加载该任务。
2. 未指定资产：自动加载第一条待处理语义任务。
3. 没有待办：显示明确的空队列完成态。

页面保留资产选择器，以便在符合门禁的语义资产之间切换。指定的资产尚未完成人工质检或已被人工 Fail 终止时，显示对应门禁原因，但不把它混入可编辑队列。

页面只包含：

- 资产与样本进度。
- 视频和当前帧。
- subtask 共享边界时间轴。
- 当前 subtask 文本编辑。
- pending edit 的双段 before/after、确认和取消。
- 完成语义校准操作与保存状态。

页面不得包含 Warn 原因、机器指标、SAM3 overlay、人工 Pass/Fail 或“人工质检任务”标签。reviewer 未填写时给出面向操作者的提示；获取 lease 的交互文案使用“开始校准/占用任务”，不裸露“编辑锁”技术术语。

## 时间轴与写入不变量

- 内部统一使用严格递增半开边界序列 `b_0 ... b_n`，第 `i` 段为 `[b_i, b_{i+1})`。
- 只渲染内部边界 `b_1 ... b_{n-1}` 的拖动手柄；首段左边界和末段右边界固定。
- 拖动 `b_i` 同时改变左段结束和右段开始，形成一个不可拆分的 pending transaction。
- pending transaction 保存 `boundary_id`、两个受影响 subtask ID 和两段 before/after。
- 确认一次只增加一次 `timeline_edit_count`；取消不增加。
- 前后端都拒绝空档、重叠、逆序、零长度或越过外侧相邻边界。
- 源 HDF5 使用闭区间时，source adapter 负责无歧义转换。
- 完成时写入同目录临时 HDF5，重新打开验证全部目标 dataset、JSON 和边界，再 fsync 并 `os.replace`；失败不得更新完成状态或破坏原文件。

## 并发、错误与恢复

- 所有语义写请求携带 lease token 和 expected report revision。
- lease 缺失或过期返回 423 类业务错误，引导重新占用任务。
- revision 冲突返回 409，保留服务端数据并要求重新加载，不自动覆盖。
- 存在 pending edit 时禁止修改其他 segment、切换资产或完成任务。
- 页面刷新从服务端最新报告恢复，不使用 localStorage 作为正式状态源。
- `finalizing` 状态启动时继续采用现有恢复策略：根据事务记录和 HDF5 hash 完成或回滚报告状态。
- API 不向浏览器暴露绝对路径、命令行或 Python 异常原文。

## 迁移策略

1. 先用回归测试锁定现有语义时间轴、编辑事务、HDF5 写入和页面交互。
2. 在中立包中建立人工质检 → 语义校准的状态迁移函数，并调整 pipeline module 顺序。
3. 将语义领域代码迁入 `semantic_calibration`，更新内部引用和测试；`human_qc` 不保留对语义实现的运行时导入。
4. 建立独立 Semantic service facade、HTTP server、静态页面和启动命令。
5. 将人工质检完成逻辑改为 Pass 推进语义、Fail 终止；删除语义完成后推进人工质检的逻辑。
6. 停用共用工作台作为正式入口，并同步操作文档、数据合同和 OpenSpec。

整个迁移在同一功能分支完成，不合并到 `main`。迁移过程中保持 QC JSON 兼容读取；旧顺序下尚未完成的报告通过显式迁移函数重排游标，不能依靠 UI 猜测或静默覆盖。已完成报告保持只读历史，不倒转已发生的人工/语义审计事件。

## 测试与验收

### 后端与合同

- 人工质检全 Pass 后原子推进到语义待办。
- 任一人工 Fail 后流水线终止，语义状态为 `skipped_due_to_fail`。
- 人工未完成的资产不出现在语义列表，直接请求也被拒绝。
- `not_required` 资产保持现有跳过行为并进入语义。
- supplier evaluation 中人工 Pass 不覆盖机器 hard fail。
- 语义完成不再创建、选择或修改人工质检任务。
- `semantic_calibration` 和 `human_qc` 之间不存在双向或反向领域导入。
- 两个 server 的 API 路由集合互斥；语义 server 的 Warn 路由返回 404。

### 语义回归

- 只能拖内部左右边界，不能移动整个 segment。
- 中间边界拖动同时更新相邻两段且只计一次编辑。
- 非法边界在前后端均被拒绝。
- 文字编辑、pending 确认/取消、刷新恢复、lease 与 revision 冲突行为保持一致。
- HDF5 写入成功、校验失败、原子替换失败和 `finalizing` 恢复测试全部通过。

### 页面与运行验收

- 根地址自动打开第一条符合门禁的语义任务。
- `?asset_id=` 深链正确加载或显示明确门禁原因。
- 页面不存在 Warn、SAM3、Evidence、Pass/Fail 或人工质检组件和文案。
- 桌面与窄屏下视频、时间轴、subtask 编辑器和 pending 确认区布局正常。
- 使用真实测试 HDF5 完成一次边界修改、取消、再次修改、确认及最终原子写入。
- 全量 Python、Node 和 OpenSpec strict 验证通过。

## 非目标

- 不在本次重做人工质检 UI。
- 不改变 SAM3 检测输出或 evidence 生成方式。
- 不增加语义质量 Pass/Fail；语义校准只记录 completed、skipped 或 error 和修改审计。
- 不改变 HDF5 的业务字段含义。
- 不合并到 `main`，也不删除人工质检 JSON 历史。
