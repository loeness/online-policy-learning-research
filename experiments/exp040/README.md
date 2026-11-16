# E040 — 未知边界 online loop 正式组

**主 H：** [H008](../../research/records/H008.md) · **角色：** 正式闭环比较 · **运行状态：** COMPLETED。

**回顾性说明：** 本 README 于 2026-10-20 导入。原协议包含 frozen、single-head、sparse、int8、SVD、evidence gate 和 adaptive_assignment 七臂。

## 设计与结果

每个 seed（401、419、433）7 臂从相同 W0、零修正和空原型库开始，各运行 32,768 步隐藏边界流；每 1,024 步的 768 步用于 PPO、256 步用于 holdout。主要指标是全部在线奖励 / 固定交互步，辅助为在线 episode 成功率、真实 actor 字节、完整状态成本和墙钟。

共 688,128 training interactions。神经方法中 single-head 最佳：奖励/步 −.01242、成功率 13.73%、平均墙钟 44.5 秒；evidence gate 为 −.01722 / 11.93%；int8 为 −.01662 / 12.03%。adaptive_assignment 为 +.24325 / 100%、平均墙钟 31.9 秒，但用源先验和可见结构；它是本环境强统计控制，不等价于 task-free 神经组织。该结果反对当前复杂组织方案已有在线优势。修正包上限 320 B 不等于整智能体容量；完整恢复文件约 130–155 KB。

## 成本、来源与边界

- producer、21 seed × arm summary 和 raw outputs：[../../outputs/knowledge_loop_formal](../../outputs/knowledge_loop_formal)；代码快照在各运行子目录 source_snapshot。
- 输入：[../../data/foundation.pt](../../data/foundation.pt)、[../../configs/c1.json](../../configs/c1.json)；协议和代码参考：[../../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md](../../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md)、[../../scripts/run_knowledge_loop.py](../../scripts/run_knowledge_loop.py)。
- 正式环境交互 688,128 步。报告中的按方法墙钟均值为实测，不是硬件基准。
- 结果/图：[../../outputs/knowledge_organization_report/结果与研究判断.md](../../outputs/knowledge_organization_report/结果与研究判断.md)、analysis.json、online_return.png / .pdf。图源为正式分析；命令记录未核实。
- 历史输出记录 source snapshots；完整 execution SHA 未从该 cohort 中核实。原始训练程序与配置不在本次修改范围。
- 该输出中还嵌有旧重访字段：旧重访作为单独无效推断问题登记于 E041，不与本正式训练回报合并。

## 归属与分析版本导航

主生命周期：[H008](../../research/records/H008.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
