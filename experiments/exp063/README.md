# E063：S1b probe-based grouping pilot

**H：** [H011](../../research/records/H011.md) · **性质：** 单 seed 开发 pilot，检查 probe 分组与运行流程。**状态：** cohort producer 已完成；后处理先失败后从既有产物修复；2026-10-20 回顾性整理。

## 设计与决策

检验每个已知 route 收集 512 步 uniform-action probe 后，数值转移 profile 是否可用于分组并让共享参数影响留出目标。seed 827、四条件、四个 seen bouts 和两个 held-out bouts，各 2,048 个训练步。Grouper 不收 family、context name 或 reward；route 和切换时刻仍给定。

## 结果、成本与限制

四臂合计 49,152 训练步、12,288 probe 步、24,343 评估步，共 85,783 次交互。Root summary 中 auto-active/oracle-active online success 均为 17.22%，auto-frozen 15.79%，independent 12.75%。这是单 seed 的短程筛查。一个 cohort-level producer invocation 产生 4 个 seed-condition 输出单元。训练完成后，汇总因 active/frozen 前缀比较包含不同 condition 名称而失败；`POSTPROCESSING_FAILURE.json` 记录训练产物未改变。修复仅重建汇总，额外交互为 0。此后处理失败作为单独的 postprocessing attempt 记录，不把 producer cohort 改记为失败，也不构成训练重跑。

## 产物与复现

- 协议：[SHARED_WRITE_AUTO_PROTOCOL.md](../../docs/SHARED_WRITE_AUTO_PROTOCOL.md)；配置：[shared_write_auto.json](../../configs/shared_write_auto.json)；代码入口：[run_shared_write_auto.py](../../scripts/run_shared_write_auto.py) 与 [grouping](../../src/online_rl/shared_write_grouping.py)。
- [Pilot summary](../../outputs/knowledge_shared_write_auto_pilot/summary.json)、[post-processing failure receipt](../../outputs/knowledge_shared_write_auto_pilot/POSTPROCESSING_FAILURE.json)、[repaired pilot audit](../../outputs/knowledge_shared_write_auto_pilot_report/audit.json)、[pilot figures](../../outputs/knowledge_shared_write_auto_pilot_report/)。
- 一个 cohort-level producer invocation 的根为 [outputs/knowledge_shared_write_auto_pilot](../../outputs/knowledge_shared_write_auto_pilot)；[runs/run001/provenance.json](runs/run001/provenance.json) 列出 4 个 seed-condition 输出单元和独立的 postprocessing failure/reconstruction phase metadata。没有新增 producer invocation 或环境交互。历史命令和执行 Git commit 未记录；protocol/runner 是复现参考。训练 raw data 和 frozen snapshots 留在 outputs 原位置。

**成本限制：** 历史获批预算未找到记录。探测成本真实计入；单 seed 不能支持泛化判断。

## 归属与分析版本导航

主生命周期：[H011](../../research/records/H011.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
