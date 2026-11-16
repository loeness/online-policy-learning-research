# E033 — 静态整合 pilot

**主 H：** [H008](../../research/records/H008.md) · **角色：** pilot · **运行状态：** COMPLETED · **科学地位：** 排除于正式效果统计。

**回顾性说明：** 本 README 于 2026-10-20 导入；不是预注册。原三步协议把 pilot 用于小规模软件检查。

## 设计与观测

对既有 s181 教师终点的有效 6×27 修正矩阵做静态 codec 评估；4 个 episode / target / candidate，oracle 仅在评估时隔离情境。优化对象为真实编码字节、平均成功率和回报相对 dense 的保留；不训练策略，不测 optimizer 或在线延续。实际交互 5,793 步，CPU 3.438 秒、墙钟 3.765 秒。pilot 输出和数据保留；该小样本不支持预算或编码效果结论，正式结果见 E034。

## 来源与证据

- 协议：[../../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md](../../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md)；历史 producer：[../../outputs/knowledge_static_pilot](../../outputs/knowledge_static_pilot)。
- 源教师 checkpoint 路径在正式静态 summary 的 source_records 中列出；本 pilot 输入版本/hashes见 [../../outputs/knowledge_static_pilot/summary.json](../../outputs/knowledge_static_pilot/summary.json)。该路径由历史 summary 指明，不因本导入复制。
- 代码快照/执行 commit 未在 pilot summary 中确认。可参考脚本 [../../scripts/run_static_consolidation.py](../../scripts/run_static_consolidation.py)；协议中命令只能作为 reproduction，不能冒充历史命令。
- 仅保留 episode ledger 与 summary；无专属图，NOT_APPLICABLE：pilot 只作软件/流程检查。

## 归属与分析版本导航

主生命周期：[H008](../../research/records/H008.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

本 E 未识别到独立派生报告；原始运行摘要、检查输出及缺口见上文与 run provenance。
