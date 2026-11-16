# E044 — EvidenceRouter 正式共同流回放

**主 H：** [H009](../../research/records/H009.md) · **角色：** 正式离线路由机制诊断 · **运行状态：** COMPLETED。

**回顾性说明：** 本 README 于 2026-10-20 导入；这是原有实际行为流上的 retrospective algorithm comparison，不是独立泛化样本。

## 设计与结果

对 seeds 331、347、359 的 identification formal 流做离线回放，比较新 router 与 robust-bank W32；相同流输入、无环境交互。主要指标为 route coverage、含 unresolved 的 ARI、复用/分配/驱逐；这些算法离线计算不能重计环境步。新 router coverage 93.6% vs 87.5%，ARI 约 .901 vs .650，generation purity 约 99.4% vs 93.2%，重访 generation reuse 约 93.1% vs 61.7%，每流分配/驱逐 6/0 vs 27/21。结果支持改善这批流上的分裂与复用，不能证明噪声动力学或新分布泛化。

## 输入、成本与来源

- 输入真实流：../../outputs/knowledge_identification_formal（seed 331 / 347 / 359 的 stream 文件）；历史 producer 与指标：[../../outputs/knowledge_v2_router_formal](../../outputs/knowledge_v2_router_formal)。
- 协议/算法：[../../docs/KNOWLEDGE_V2_VALIDATION.md](../../docs/KNOWLEDGE_V2_VALIDATION.md)、[../../src/online_rl/evidence_router.py](../../src/online_rl/evidence_router.py)、[../../scripts/evaluate_evidence_router.py](../../scripts/evaluate_evidence_router.py)；source snapshot 在输出根。
- 新环境交互 0，离线回放 3 条流；输出的 input manifest hash 标识输入。完整 execution SHA 未核实。
- V2 report：[../../outputs/knowledge_v2_report/验证与优化结论.md](../../outputs/knowledge_v2_report/验证与优化结论.md)。无单独绘图；指标见 metrics.json / summary.json。未测 active exploration。

## 归属与分析版本导航

主生命周期：[H009](../../research/records/H009.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
