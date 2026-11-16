# E050 — V2 终点配对 persistence 评估

**主 H：** [H009](../../research/records/H009.md) · **角色：** 正式终点重访 · **运行状态：** COMPLETED。

**回顾性说明：** 本 README 于 2026-10-20 导入。该重访沿用已记录的逐 episode 配对修正方案，不做训练或超参选择。

## 设计与结果

输入为 V2 formal 的 15 个 seed × method 终点（3 seed ×5 method）。清空短历史，保留策略修正 vs 清除修正；路由统计可更新，槽被新原型占用时清除过期修正；不进行梯度更新。六情境各 8 个完整 episode，两条件逐 episode 使用匹配 reset/action RNG。主指标为保留相对清除的 paired return 与成功率。

共 1,440 episode、80,319 environment interactions、0 policy updates。非阻塞共享整合保留成功率 31.94%、清除后 13.89%、配对回报 +.985；新路由独立 int8 也达到 30.56% / +.888；single-head +.759；旧路由修复 int8 +.040。结果显示多种方法保留了训练参数作用，但共享整合相对同路由独立 int8 的增量较小，不能归因于共享单独机制。

## 输入、成本与 provenance

- 输入终点：[../../outputs/knowledge_v2_formal](../../outputs/knowledge_v2_formal)；producer、per-seed/method revisit outputs、summary 和 source snapshot：[../../outputs/knowledge_v2_revisit](../../outputs/knowledge_v2_revisit)。
- 设计：[../../docs/KNOWLEDGE_V2_REVISIT.md](../../docs/KNOWLEDGE_V2_REVISIT.md)；报告：[../../outputs/knowledge_v2_report/验证与优化结论.md](../../outputs/knowledge_v2_report/验证与优化结论.md)、analysis.json、audit.json。
- runner：[../../scripts/revisit_knowledge_v2.py](../../scripts/revisit_knowledge_v2.py)、[../../scripts/restore_knowledge_v2.py](../../scripts/restore_knowledge_v2.py)；原始执行 commit 未核实。checkpoint restore 需遵循 variant-aware factory。
- 实耗 80,319 步、1,440 episodes、零梯度；每方法及 seed 的结果在 summary。无单独重访 figure；综合线上图不替代配对重访账本。

## 归属与分析版本导航

主生命周期：[H009](../../research/records/H009.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
