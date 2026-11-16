# E034 — 七种原始 codec 的静态正式比较

**主 H：** [H008](../../research/records/H008.md) · **角色：** 正式静态比较 · **运行状态：** COMPLETED；含一项中止重复尝试，见 [run002 provenance](runs/run002/provenance.json)。

**回顾性说明：** 本 README 于 2026-10-20 导入，不是执行前计划。原协议在正式静态队列前写明控制条件和保留门槛。

## 设计与结果

3 个已训练教师 checkpoint（source seed 181、193、211）、6 个目标、7 种原始 codec（dense float32、逐行 sparse Top-K、FP16、int8、global mean sparse、centroid2 sparse、truncated SVD）和 5 档预算；每个可行配置评估 32 个完整 episode。以真实二进制总字节、成功率与 episode return 比较；成功率损失须 ≤5 个百分点且回报损失 ≤0.25 才算通过 dense 保留门槛。评估是 oracle 隔离，不是部署路由。

实际 277,327 交互步，CPU 120.719 秒、墙钟 123.009 秒。320 B 内独立 int8 为 202 B、成功率 100%、平均回报 3.382；双原型共享残差 316 B、100%、3.335，未同时胜过同预算最佳独立编码。SVD 280 B 为 85.24%、2.594。24 个配置不适用预算，129 组配置与 21,312 episode 行保留。结论不支持共享残差有独有优势。

## 输入、审计与 provenance

- 教师输入：[outputs/c1_supplement/representation_alternatives/hierarchical_output_mixer/formal/hierarchical_output_mixer/none/low/s181/history/agent.pt](../../outputs/c1_supplement/representation_alternatives/hierarchical_output_mixer/formal/hierarchical_output_mixer/none/low/s181/history/agent.pt)；[outputs/c1_supplement/representation_alternatives/hierarchical_output_mixer/formal/hierarchical_output_mixer/none/low/s193/history/agent.pt](../../outputs/c1_supplement/representation_alternatives/hierarchical_output_mixer/formal/hierarchical_output_mixer/none/low/s193/history/agent.pt)；[outputs/c1_supplement/representation_alternatives/hierarchical_output_mixer/formal/hierarchical_output_mixer/none/low/s211/history/agent.pt](../../outputs/c1_supplement/representation_alternatives/hierarchical_output_mixer/formal/hierarchical_output_mixer/none/low/s211/history/agent.pt)。对应 hash 在静态 summary 中。基础输入 [data/foundation.pt](../../data/foundation.pt)。协议：[../../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md](../../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md)。
- 历史 producer 与原始 episode ledger：[../../outputs/knowledge_static_formal](../../outputs/knowledge_static_formal)；解读和图在 [../../outputs/knowledge_organization_report/结果与研究判断.md](../../outputs/knowledge_organization_report/结果与研究判断.md) 与 static_capacity.png / .pdf。
- 该历史组没有运行时 source snapshot、runner hash 或 codec hash；明确限制见 [outputs/knowledge_static_formal/serialization_audit/source_provenance.json](../../outputs/knowledge_static_formal/serialization_audit/source_provenance.json)。完整执行 Git SHA 未知，不从当前源码反推。
- 脚本 [../../scripts/run_static_consolidation.py](../../scripts/run_static_consolidation.py) 是实现参考。protocol 给出的重现命令不是原执行命令；原命令未可靠恢复。
- run002 保存中止的重复尝试，85,781 步是候选部分的下界，完整 run 总交互未知；不纳入本 E 的正式统计或结果。

## 归属与分析版本导航

主生命周期：[H008](../../research/records/H008.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
