# E047 — V2 在线优化 pilot

**主 H：** [H009](../../research/records/H009.md) · **角色：** pilot · **运行状态：** COMPLETED；不进入 formal estimate。

**回顾性说明：** 本 README 于 2026-10-20 导入；本条记录不重新分析 pilot 或调参。

## 设计与观测

seed 787，三个条件（EvidenceRouter int8、旧 gate、新路由非阻塞整合）各 8,192 步，共 24,576 环境交互；每周期 3 个 fit rollout +1 holdout，checkpoint 边界为 W0 + saved checkpoint ≤256 KiB。主指标是 reward/step、success、容量与 wall/CPU。所有 run-level 结果保存在 summary 和条件目录；此单 seed 试验不形成正式方法排名。

## 来源与限制

- Protocol：[../../docs/KNOWLEDGE_V2_VALIDATION.md](../../docs/KNOWLEDGE_V2_VALIDATION.md)；producer、summary、outputs 和 source snapshots：[../../outputs/knowledge_v2_pilot](../../outputs/knowledge_v2_pilot)。
- 输入：[../../data/foundation.pt](../../data/foundation.pt)、[../../configs/c1.json](../../configs/c1.json)；runner 与组件参考：[../../scripts/run_knowledge_v2.py](../../scripts/run_knowledge_v2.py)、[../../scripts/knowledge_v2_components.py](../../scripts/knowledge_v2_components.py)。
- 完整 execution commit 未核实；源码快照随输出保留。历史原命令未恢复，protocol command 若引用只能标 reproduction。
- 实耗 24,576 步，CPU 与 wall 在 root summary。无独立 figure；formal comparison 的派生图见 E048。

## 归属与分析版本导航

主生命周期：[H009](../../research/records/H009.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

本 E 未识别到独立派生报告；原始运行摘要、检查输出及缺口见上文与 run provenance。
