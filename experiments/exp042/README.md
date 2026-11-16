# E042 — 修正配对的 loop 终点重访

**主 H：** [H008](../../research/records/H008.md) · **角色：** 修正后 persistence 正式评估 · **运行状态：** COMPLETED。

**回顾性说明：** 本 README 于 2026-10-20 导入；KNOWLEDGE_REVISIT_CORRECTION.md 冻结了修正语义，当前文档不构成新的预注册。

## 设计与结果

输入为 loop formal 各 seed/方法的同一训练终点。seed 401、419、433；6 个含修正方法 ×6 个情境 ×8 个完整 episode ×保留/清除两条件。清空短历史；每个情境新分配槽时只清除不再属于新原型的修正，不进行梯度更新；每个情境 × episode 匹配 reset 与独立 action RNG seed。

共 1,728 episode、102,991 环境步、零 policy gradient update。保留参数相对清除的平均配对回报：single-head +1.022（保留成功率 22.92%）；sparse +.100；int8 +.182；SVD +.058；evidence gate +.127；冻结策略为 0。single-head 三个 seed 的配对增益均正；复杂组织方法仅小幅为正，未超过简单 single-head。

## 输入、来源与成本

- 历史 producer 和每 seed/arm 结果：[../../outputs/knowledge_loop_revisit_corrected](../../outputs/knowledge_loop_revisit_corrected)；输入训练终点：[../../outputs/knowledge_loop_formal](../../outputs/knowledge_loop_formal)。
- 协议：[../../docs/KNOWLEDGE_REVISIT_CORRECTION.md](../../docs/KNOWLEDGE_REVISIT_CORRECTION.md)；runner：[../../scripts/run_knowledge_revisit_corrected.py](../../scripts/run_knowledge_revisit_corrected.py)；source snapshot 位于 corrected 输出根。
- 结果/图源：[../../outputs/knowledge_organization_report/结果与研究判断.md](../../outputs/knowledge_organization_report/结果与研究判断.md)、analysis.json、online_return.png / .pdf。该图兼含 formal 在线流，重访配对数应引用 corrected summary 而非图重建。
- 实耗 102,991 步，CPU 101.922 秒、墙钟 104.452 秒。输入/源码 hash 在 summary；完整执行 Git SHA 未核实。
- 不做新的参数调优，原始 loop outputs 不覆盖。

## 归属与分析版本导航

主生命周期：[H008](../../research/records/H008.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
