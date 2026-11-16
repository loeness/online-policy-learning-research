# E046 — V2 六条件因果归因正式组

**主 H：** [H009](../../research/records/H009.md) · **角色：** 正式新 seed 因果诊断 · **运行状态：** COMPLETED。

**回顾性说明：** 本 README 于 2026-10-20 导入。原协议规定了六条件和诊断边界；oracle 不满足未知情境约束。

## 设计与结果

Seeds 541、557、569，6 条件 ×32,768 步，共 589,824 环境步。相同 W0、PPO 和隐藏 schedule；对比原路由/原分配、仅修复 rollback、oracle route int8、旧路由 dense、oracle dense、oracle route + 旧 gate。主指标在线奖励/步向上；success 与 202 B int8 /664 B dense 修正字节为辅助。

正式均值：原路由 int8 −.016267；preserve-trial 修复 −.014477；oracle int8 −.004182；旧路由 dense −.016333；oracle dense −.006097；oracle gate −.011049。修复后的提升集中在 seed 557；oracle int8 三个 seed 均优于修复路由，dense 没有胜过同路由/诊断 int8，oracle gate 仍低于 oracle int8。结论提示辨识/绑定和 gate 是瓶颈，不能把 oracle 收益算为可部署方法增益。

## 来源、成本与限制

- producer、formal_analysis、summary、所有 seed × condition outputs 和源码 snapshot：[../../outputs/knowledge_causal_formal](../../outputs/knowledge_causal_formal)。
- 输入：[../../data/foundation.pt](../../data/foundation.pt)、[../../configs/c1.json](../../configs/c1.json)；协议：[../../docs/KNOWLEDGE_V2_VALIDATION.md](../../docs/KNOWLEDGE_V2_VALIDATION.md)；runner：[../../scripts/run_knowledge_causal_ablation.py](../../scripts/run_knowledge_causal_ablation.py)。
- 派生分析、figure 与独立审计：[../../outputs/knowledge_v2_report/analysis.json](../../outputs/knowledge_v2_report/analysis.json)、causal_v2.png / .pdf、audit.json；图由正式 causal 分析生成，原始绘图命令未恢复。
- 实际正式交互 589,824 步。完整执行 Git SHA 未核实；因果 harness 的 V2 protocol source list 有执行后补充，原 execution manifest 保留，见 H009。
- Oracle 路由来自外部 hidden schedule，非未知任务算法结果；dense 与 int8 预算不同。

## 归属与分析版本导航

主生命周期：[H009](../../research/records/H009.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
