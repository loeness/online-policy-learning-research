# H036 接纳诊断：主控接受与实现范围

日期：2026-10-23（Asia/Singapore）。主控接受[NEXT_DESIGN](NEXT_DESIGN_20261023.md)的诊断方向及有限实现范围。主控已阅读实验方预审，并委派只读静态复核；未发现方法归因上的实质阻断。该接受授权本地实现、必要合成检查和冻结，**不是实际恢复拟合、策略评分或SSH授权**。

## 接受的最小方法

- 同一lambda=1、相同训练目标和搜索轨迹，分别记录原新训练KL可接纳的F64与所有有限点中完整目标最低的U64，二者同初值、同顺序、同精确平局规则；记录不反馈优化器。
- V保持原两级gate；UG只取消closure筛选、保留float32转后复查；UR从同一U64取消转后回退，仅作诊断。UG−V与UR−UG分别解释，UR−V不归因于单一规则。
- 全16条逻辑配对；11条用原F/U完整float64哈希相同及原无float32回退作复用证明，准确引用旧D评分；恢复集合仅26204/26206/26209/26212/26213，最多5次原路径恢复、15个新增完整评分包。不补lambda=0，不按holdout选点或筛样本。
- 恢复先核对原有closure数量/顺序、候选hash、完整训练标量及F身份；任一不符整批REPLAY_MISMATCH。全部候选锁定后才新增留出评分；不扩为16次恢复或自动重试。
- 保持模型、教师、buffer、wildcard、原codec、89 B actor与262144 B完整持久上限。新影子状态及日志仅在诊断工件，不进入learner；原delivery_v1/源码/E163证据保持原样。

## 算法实现的文件所有权与冻结条件

新增源码仅限：`src/online_rl/retention_acceptance_replay.py`、`src/online_rl/retention_acceptance_diagnostic.py`、`configs/h036_acceptance_diagnostic_v1.json`、`tests/test_retention_acceptance_diagnostic.py`。新方法/交接与冻结位于H036的`METHOD_ACCEPTANCE_V1.md`、`HANDOFF_ACCEPTANCE_V1.md`、`delivery_acceptance_v1/`；必要设计澄清可追加NEXT_DESIGN，不覆盖历史设计。算法方不修改E/H索引、E164预算、实验runner或实验方审查文件。

冻结前必须落实两个具体契约：

1. 每个复用项准确绑定原C/D的prepared_id、三文件/foundation身份、binding、v1 source identity、candidate/deployment/metrics哈希和run002 provenance。48份标量引用不能代替输入与部署完整绑定。helper的dtype/shape/内容hash与候选原字节SHA分开。
2. 原评分接口adapter明确使用`format=h036_fit_v1`及其实际要求的prepared_id、lambda、有限float32 3×3 candidate、diagnostics.candidate_sha256；独立记录诊断选择来源，不伪造原fit的optimizer统计。原评分器不改。

只做设计第7节必要合成检查，证明新training-only loop的F与原训练路径逐点一致、U记账无干预、两gate分开、holdout不能影响选择、hash复用/别名及锁定不可篡改。原已通过且未变的23项和1876项审计不再全部重跑。真实恢复、模型反序列化及新评分均留给独立验证阶段，不用于开发“试跑”。

交付必须可直接编排：方法、API、精确配置/入口、输入及源码身份、必要检查结果、调用/资源计数语义、全部关键停止点。只提交设计不完成实现责任。

## E164 与资源状态

主控将本验证预留为[E164](../../../experiments/exp164/README.md)，仍属H036诊断细化，不新增生产算法H。已有标量重放为2026-10-23设计支持的结果后分析，原记录保存；不能追称为未见结果的预注册确认。

规划单次60 CPU秒/120墙钟秒、0交互、0GPU、单worker/单数值线程、64 MiB新增工件/cache/temp；只是待审规划值，不是当前额度。E163两次reservation不复用。算法交付冻结后，实验方完成独立一致性/公平性与静态资源估算；主控据具体可审阅版本完成适用资源授权及正式准入，才可在新E范围执行。

当前不扩大结构、不搜索lambda、不启动StageA或新种子。实验方若提出实质可行性阻断，算法方在冻结前修复；若涉及方法/范围变化回交主控，不在runner中代定。
