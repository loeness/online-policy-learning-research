# E041 — 旧版 loop 终点重访评估

**主 H：** [H008](../../research/records/H008.md) · **角色：** 历史 persistence 评估 · **运行状态：** COMPLETED，但已判为不适合持久性推断。

**回顾性说明：** 本 README 于 2026-10-20 导入。它保留错误评估，不抹除其交互成本，也不作为参数持久性结论。

## 设计、失效与实耗

旧版从 loop formal checkpoint 清除短历史后做终点重访。原始 summary 计入 105,844 额外环境步。代码复核发现两项问题：路由槽重新分配时未清除被驱逐情境对应的策略修正；随机 action 流只在情境级设种子，并跨 episode 连续消耗，episode 长度不同后不再配对。由此旧结果不适合估计修正参数的保留效果。没有可用的独立有效主指标或合格结论；旧训练流未受影响。

## 来源与审计

- producer artifact 嵌在 [../../outputs/knowledge_loop_formal](../../outputs/knowledge_loop_formal) 的 21 个 seed × arm 记录与 summary 中；该路径也服务 E040 的训练 cohort。
- 错误分析与修正规程：[../../docs/KNOWLEDGE_REVISIT_CORRECTION.md](../../docs/KNOWLEDGE_REVISIT_CORRECTION.md)；综合报告保留旧结果排除说明：[../../outputs/knowledge_organization_report/结果与研究判断.md](../../outputs/knowledge_organization_report/结果与研究判断.md)。
- 方法参考 [../../scripts/run_knowledge_loop.py](../../scripts/run_knowledge_loop.py)；原独立命令与完整 execution SHA 未恢复。不要复用 E040 的训练结论作为此评估的有效结果。
- 旧 figure 不单列：该无效重访不用于持久性图表。新修正评估独立列为 E042。

## 归属与分析版本导航

主生命周期：[H008](../../research/records/H008.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
