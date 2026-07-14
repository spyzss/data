## 1. 统一合同与 Schema

- [x] 1.1 为 execution profile、通用 module flow、runtime error、evidence 和二元 final decision 增加配置及 JSON Schema 测试
- [x] 1.2 在 `qc_common` 定义统一 `ModuleResult`、`Issue`、`EvidenceRef` 和稳定 issue ID 接口
- [x] 1.3 实现模块所有权感知的 revision-aware 报告 mutation，并验证重跑去重与未知字段保留

## 2. 已合入自动模块适配

- [x] 2.1 将 text integrity 与 quality_hand 结果适配为 `hdf5_text_info` 和 `quality_hand` module block
- [x] 2.2 将 keypoint missing/skeleton quality 适配为 `keypoint_presence` module block 和稳定帧区间 issue
- [x] 2.3 将 keypoint morphology 适配为 `keypoint_morphology` module block、metrics 和 evidence
- [x] 2.4 将 keypoint temporal/candidate windows 适配为 `keypoint_temporal` module block 和人工候选 issue
- [x] 2.5 统一 batch 视频与 manifest range 视频路径，使两者通过同一 `video_quality` 报告 mutation 写回
- [x] 2.6 将 SAM3 window summary、overlay 和 containment 结果适配为 `sam3_containment` module block 和 evidence 引用

## 3. 统一编排与执行策略

- [x] 3.1 实现按版本化配置和 registry 运行的资产级 orchestrator，并支持从 `next_module` 恢复
- [x] 3.2 实现 acceptance profile 的 hard-fail 截断、后续阶段跳过和最终 fail
- [x] 3.3 实现 supplier_evaluation profile 的 fail 记录后继续、完整执行轨迹和最终 fail 保留
- [x] 3.4 对 disabled、skipped、not_implemented 和 runtime error 建立互不混淆的状态与测试

## 4. 人工路由输入与批次投影

- [x] 4.1 从 QC JSON issues/evidence 生成 warn 人工队列输入，停止直接拼接多套 sidecar 结论
- [x] 4.2 实现仅遍历 `quality_archive/*.json` 的批次投影和按 profile 分组统计
- [x] 4.3 将现有 batch ledger/weekly report 正式入口迁移到统一投影，并保留遗留结果对账测试
- [ ] 4.4 增加可删除重建的 Parquet/CSV 缓存并验证缓存不参与事实判定

## 5. 文档与端到端验证

- [ ] 5.1 同步 PRD、统一配置、Schema 文档和 reviewer 指南中的双 profile 与唯一事实源规则
- [ ] 5.2 使用 pass、warn、hard fail、runtime error 四类 fixture 验证单资产完整 revision 轨迹
- [ ] 5.3 使用同一批输入端到端验证 acceptance 截断与 supplier_evaluation 全流程的差异
- [ ] 5.4 运行全量测试并记录旧 sidecar 到 QC JSON 的迁移与回滚说明
