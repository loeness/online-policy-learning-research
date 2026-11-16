# E065：S2 六局部槽容量压力 pilot

**H：** [H012](../../research/records/H012.md) · **性质：** seed 839 开发 pilot。**状态：** 原运行已完成；2026-10-20 回顾性整理。

## 设计与决策

四种缓存条件在十二个已知情境上运行：独立、较快独立、持续共享和初期共享后冻结。六个局部槽在 24 段固定循环日程中使用 LRU；pilot 每 bout 1,024 训练步。目标是验证淘汰确实清除局部参数并保留足够运行证据，以判断是否进入 formal cohort。

## 结果、成本与限制

四臂合计 98,304 训练步、56,327 评估步，共 154,631 次交互。Seed 839 的 shared arm summary 记录 24 次 cache miss、18 次 eviction、24 次 int8 commit，actor 230 B，单臂 online success 31.30%。这是开发种子的 pilot 指标，不是正式比较或显著性证据。Pilot report audit 保留在原目录。

## 产物与复现

- 协议：[SHARED_WRITE_CAPACITY_PROTOCOL.md](../../docs/SHARED_WRITE_CAPACITY_PROTOCOL.md)；配置：[shared_write_capacity.json](../../configs/shared_write_capacity.json)；代码入口：[run_shared_write_capacity.py](../../scripts/run_shared_write_capacity.py)。
- [Pilot summary](../../outputs/knowledge_shared_write_capacity_pilot/summary.json)、[audit](../../outputs/knowledge_shared_write_capacity_pilot_report/audit.json)、[figure](../../outputs/knowledge_shared_write_capacity_pilot_report/capacity_results.png)。
- 一个 cohort-level producer invocation 的根为 [outputs/knowledge_shared_write_capacity_pilot](../../outputs/knowledge_shared_write_capacity_pilot)；[runs/run001/provenance.json](runs/run001/provenance.json) 列出 4 个 seed-condition 输出单元，不表示 4 个已知独立 CLI invocation。原始 command 和 execution commit 未记录；文档中的 runner 只作复现参考。原始输出不复制。

**成本限制：** 历史批准计算/时间预算未找到记录。Pilot 仅覆盖固定循环和给定家族/ID。

## 归属与分析版本导航

主生命周期：[H012](../../research/records/H012.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
