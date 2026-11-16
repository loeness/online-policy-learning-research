# E049 — behavior-reference gate 分数诊断

**主 H：** [H009](../../research/records/H009.md) · **角色：** 单独评分诊断 · **运行状态：** COMPLETED。

**回顾性说明：** 本 README 于 2026-10-20 导入；本诊断不属于新 gate performance result。

## 设计与结果

seed 787、8,192 步，在原 gate 调用位置同时计算旧候选/ incumbent score 与以实际 behavior probability 为分母的诊断 score；每次仍返回旧分数，保持旧算法动作不变。主诊断指标是候选/回退 probe 的 score sign disagreement，而非回报改进。10 次 probe 中 4 次符号不同；动作与 reward 和原 gate pilot 的 8,192 步逐项一致。两个分数都不是安全改进保证。

## 成本与来源

- Producer、summary、steps、checkpoint 与 source snapshot：[../../outputs/knowledge_v2_score_diagnostic](../../outputs/knowledge_v2_score_diagnostic)；输入 reference pilot 由输出元数据标识为 seed 787 的旧 gate 路径。
- 文档与代码：[../../docs/KNOWLEDGE_V2_IMPLEMENTATION_NOTES.md](../../docs/KNOWLEDGE_V2_IMPLEMENTATION_NOTES.md)、[../../scripts/diagnose_knowledge_gate_score.py](../../scripts/diagnose_knowledge_gate_score.py)、[../../docs/KNOWLEDGE_V2_VALIDATION.md](../../docs/KNOWLEDGE_V2_VALIDATION.md)。
- 成本 8,192 environment steps；CPU 9.5 秒、wall 9.588 秒。完整 execution SHA 未核实，source SHA 在 summary。
- 无单独图；score table/list保存在 summary。历史实现行为未修改，未得到纠正 gate 的效果证据。

## 归属与分析版本导航

主生命周期：[H009](../../research/records/H009.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
