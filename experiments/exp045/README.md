# E045 — V2 因果归因 pilot

**主 H：** [H009](../../research/records/H009.md) · **角色：** pilot · **运行状态：** COMPLETED；不进入 formal effect estimate。

**回顾性说明：** 本 README 于 2026-10-20 导入；pilot 用于排查流程和配置，不按其结果重新选择正式 seed。

## 设计与观测

seed 787，6 条件（learned int8、preserve-trial、oracle int8、learned dense、oracle dense、oracle gate）各 8,192 步；PPO 周期 3 个 fit rollout +1 holdout，条件共享外部 schedule。比较在线 reward/step、success、路由权限、修正字节。总计 49,152 环境步。summary 保留所有条件和 run 级指标；oracle 条件收到隐藏 context，仅为诊断。此单 seed pilot 不支持归因结论，正式比较在 E046。

## 输入、代码与 evidence

- Protocol：[../../docs/KNOWLEDGE_V2_VALIDATION.md](../../docs/KNOWLEDGE_V2_VALIDATION.md)；producer、summary、run outputs 和 per-run source snapshots：[../../outputs/knowledge_causal_pilot](../../outputs/knowledge_causal_pilot)。
- 输入为 [../../data/foundation.pt](../../data/foundation.pt)、[../../configs/c1.json](../../configs/c1.json)；实现参考 [../../scripts/run_knowledge_causal_ablation.py](../../scripts/run_knowledge_causal_ablation.py)。
- 完整执行 commit 未从 pilot metadata 核实。输出中的 manifest 保存执行源码；协议复现命令不表示历史调用。
- 运行 evidence 可见于 seed_787 的 6 条件目录；无专属 figure，NOT_APPLICABLE：pilot 只作因果 harness 检查。

## 归属与分析版本导航

主生命周期：[H009](../../research/records/H009.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

本 E 未识别到独立派生报告；原始运行摘要、检查输出及缺口见上文与 run provenance。
