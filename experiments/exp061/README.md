# E061：S1a 真实共享写入 pilot

**H：** [H010](../../research/records/H010.md) · **性质：** 历史 pilot，单 seed 开发筛查，不是新的预注册。**状态：** 原运行已完成；2026-10-20 回顾性整理。

## 设计与决策

问题是继续更新共享修正是否会改变 held-out 目标策略，相比冻结共享和独立局部学习。四条件从同一 W0 出发，已知 route 与 family；seed 811，四个 seen bouts 和两个 held-out bouts 各 2,048 个训练步。主指标为 A2/B2 cold return 与 online success，方向越高越好。这个单 seed 只用于早期机制筛查，不能作确认。

## 结果、成本与限制

共 49,152 训练步、22,426 评估步、71,578 次交互。汇总记录的 online success：oracle-active 16.35%、oracle-frozen 13.24%、独立 15.20%、较快独立 13.17%；各单臂数值见 root summary。种子单一且 pilot 训练长度较短，不能据此判断稳定收益。结果 audit 保留在原目录。

## 产物与复现

- 历史协议：[SHARED_WRITE_PROTOCOL.md](../../docs/SHARED_WRITE_PROTOCOL.md)；冻结配置：[shared_write.json](../../configs/shared_write.json)；代码入口：[run_shared_write.py](../../scripts/run_shared_write.py)。
- 产物：[pilot summary](../../outputs/knowledge_shared_write_pilot/summary.json)、[audit](../../outputs/knowledge_shared_write_pilot_report/audit.json)、[figures](../../outputs/knowledge_shared_write_pilot_report/)。
- 一个 cohort-level producer invocation 的根为 [outputs/knowledge_shared_write_pilot](../../outputs/knowledge_shared_write_pilot)；[runs/run001/provenance.json](runs/run001/provenance.json) 列出其中 4 个 seed-condition 输出单元及 per-component source manifest。它们不是 4 个已知独立 CLI invocation。实际历史命令与执行 Git commit 未记录；协议和 runner 是复现参考，不是实际命令证明。没有复制 raw logs 或 source snapshots。

**成本限制：** 原批准的计算/时间预算未找到记录。检查点是阶段状态，不提供任意时刻轨迹续训。

## 归属与分析版本导航

主生命周期：[H010](../../research/records/H010.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
