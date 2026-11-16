# E043 — EvidenceRouter 短流 pilot 回放

**主 H：** [H009](../../research/records/H009.md) · **角色：** 保存流路由 pilot · **运行状态：** COMPLETED。

**回顾性说明：** 本 README 于 2026-10-20 导入；pilot 是开发诊断，短段结果没有用于正式效果估计。

## 设计与结果

seed 331 的既有 identification pilot 行为流作为共同输入，比较新 EvidenceRouter 与原 robust-bank W32。无新环境交互；评估只有一条 768-step 短流。短段 ARI 从原 .442 降至约 .397，显示新规则在少证据时不是所有指标都改进；不由此推断正式长流无效。主动探索未做比较。

## 来源与限制

- 输入流：[../../outputs/knowledge_identification_pilot/stream_seed_331.npz](../../outputs/knowledge_identification_pilot/stream_seed_331.npz)；producer、metrics、源码 snapshot：[../../outputs/knowledge_v2_router_pilot](../../outputs/knowledge_v2_router_pilot)。
- 规程和算法说明：[../../docs/KNOWLEDGE_V2_VALIDATION.md](../../docs/KNOWLEDGE_V2_VALIDATION.md)、[../../src/online_rl/evidence_router.py](../../src/online_rl/evidence_router.py)、[../../scripts/evaluate_evidence_router.py](../../scripts/evaluate_evidence_router.py)。
- 新增环境交互 0；offline evaluations 1。summary 提供 input manifest hash 与 source hash；完整 execution SHA 未核实。
- 单个短 pilot 无独立 figure；结果表在后续 V2 报告及 E044 summary。该 artifact 无训练 command，复现说明不冒充历史命令。

## 归属与分析版本导航

主生命周期：[H009](../../research/records/H009.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

本 E 未识别到独立派生报告；原始运行摘要、检查输出及缺口见上文与 run provenance。
