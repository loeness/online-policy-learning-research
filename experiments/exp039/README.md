# E039 — 未知边界 online loop pilot

**主 H：** [H008](../../research/records/H008.md) · **角色：** pilot / Stage A · **运行状态：** COMPLETED；不计正式估计。

**回顾性说明：** 本 README 于 2026-10-20 导入；此处不补做或重算 pilot。

## 设计与观测

seed 997，4 条件（frozen、int8、evidence gate、adaptive_assignment）各 2,048 个交互步；真实 C1 在线流，不同条件共享外部设置。主指标为固定步数的奖励/步与成功率。pilot 共 8,192 training interactions。单 seed 的 adaptive_assignment 达 100% 成功率、奖励/步 +.24661；其余三条件成功率约 8.8%–11.4%，奖励/步约 −.0204 至 −.0211。它是强结构先验的一次 pilot 结果，不作正式方法结论。

## 来源与限制

- 协议与 runner：[../../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md](../../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md)、[../../scripts/run_knowledge_loop.py](../../scripts/run_knowledge_loop.py)；producer 与每臂数据：[../../outputs/knowledge_loop_pilot](../../outputs/knowledge_loop_pilot)。
- 训练命令和完整执行 commit 未在本次核实；协议复现说明不代表历史 invocation。
- 正式结果图和综合分析在 [../../outputs/knowledge_organization_report/online_return.png](../../outputs/knowledge_organization_report/online_return.png) / .pdf 与结果报告。pilot 无单独 figure，使用综合图时不得与正式 seed 混合。
- pilot 实际交互 8,192 步；无剩余预算授权。

## 归属与分析版本导航

主生命周期：[H008](../../research/records/H008.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
