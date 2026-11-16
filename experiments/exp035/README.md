# E035 — 独立 global Top-K 静态公平控制

**主 H：** [H008](../../research/records/H008.md) · **角色：** 静态控制 / 独立新比较 · **运行状态：** COMPLETED。

**回顾性说明：** 本 README 于 2026-10-20 导入。该控制是在发现逐行 Top-K 与共享残差预算分配不对称后新增；原七方法组保留不变。

## 设计、结果与成本

复用 s181、s193、s211 同一教师和同一源/评估流；对有效 6×27 矩阵整体选取全局最大幅值项，不含共享参数。比较相同字节档位、同评估 episode 下的物理编码和行为保留。实际 61,961 环境步，CPU 24.359 秒、墙钟 24.634 秒。结果纳入总体静态解释：加入该控制后，不能把共享残差表现归因于跨行预算分配优势；整体结论仍是共享方法未胜过同预算最佳独立编码。

## 来源与链接

- 协议补充及方法：[../../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md](../../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md)、[../../scripts/run_static_consolidation.py](../../scripts/run_static_consolidation.py)。
- 输入教师：与 E034 相同的三个 checkpoint，路径及 hash 见 [../../outputs/knowledge_static_global_control/summary.json](../../outputs/knowledge_static_global_control/summary.json)。
- producer、episode ledger、binary audit：[../../outputs/knowledge_static_global_control](../../outputs/knowledge_static_global_control)；source snapshot 位于其 source_snapshot 子目录。派生比较和图：[../../outputs/knowledge_organization_report/analysis.json](../../outputs/knowledge_organization_report/analysis.json)、static_capacity.png / .pdf。
- 汇总记录冻结协议提交为缩写 66ce6ad，不是完整 SHA；original_execution_commit 保持未知。代码 snapshot/hash 已保留。
- 历史实耗 61,961 步；无未来预算授权。绘图命令未在本记录核实。

## 归属与分析版本导航

主生命周期：[H008](../../research/records/H008.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
