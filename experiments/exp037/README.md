# E037 — 从零辨识 pilot

**主 H：** [H008](../../research/records/H008.md) · **角色：** pilot · **运行状态：** COMPLETED；不用于正式机制结论。

**回顾性说明：** 本 README 于 2026-10-20 导入；不是预注册。pilot 的短流按协议排除于正式统计。

## 设计与结果

seed 331 从零生成一条 768 步环境流，比较有支持度原型、朴素原型和仅近期窗口三类方法，多个窗口 8/32/64 在相同流上离线回放。路由覆盖、ARI（包含未解析状态）、转移预测与分裂/驱逐为诊断指标；不测试独立主动探索。一次生成实际环境交互 768 步，9 次离线路由计算不另计交互。短 pilot 保留，不能由它单独支持最终辨识判断。

## 来源与证据

- 协议：[../../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md](../../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md)；历史 producer、stream、metrics 与 source snapshot：[../../outputs/knowledge_identification_pilot](../../outputs/knowledge_identification_pilot)。
- summary 中记录原命令 [scripts/run_task_free_identification.py](../../scripts/run_task_free_identification.py) --phase pilot；完整执行 SHA 未核实。输入 stream 是 pilot 内生成并保留的 [outputs/knowledge_identification_pilot/stream_seed_331.npz](../../outputs/knowledge_identification_pilot/stream_seed_331.npz)。
- 结果综述：[../../outputs/knowledge_organization_report/结果与研究判断.md](../../outputs/knowledge_organization_report/结果与研究判断.md)；无单独图（共享图见综述），绘图命令不适用本 pilot。
- 成本 768 环境步；source hash、protocol hash、环境版本在 summary。无新的剩余 compute 授权。

## 归属与分析版本导航

主生命周期：[H008](../../research/records/H008.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
