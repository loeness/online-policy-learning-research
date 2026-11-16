# E064：S1b family-free probe grouping formal cohort

**H：** [H011](../../research/records/H011.md) · **性质：** 三个新部署种子的受控分组比较。**状态：** 原运行已完成；2026-10-20 回顾性整理。

## 设计与决策

在已知六个 route 与 context slot 下，比较独立、oracle family grouping、仅由 numeric transition profile 决定的 auto-active、以及 auto-frozen。每个 context 首次出现先做 512 步均匀动作 probe；主指标为家族 pair grouping 与 A2/B2 留出策略表现。Formal seeds：677、691、709。Probe 不做梯度更新，所有条件支付相同预算。

## 结果、成本与限制

12 个 seed-condition 输出单元（3 seeds × 4 arms）共 491,520 训练步、36,864 probe 步、110,611 评估步，总计 638,995 次交互，属于一个 formal cohort producer invocation。三个种子均把六个情境分到与 audit-only family labels 一致的两组，auto-active 与 oracle-active 的策略结果相同。Online success：auto/oracle 42.57%、independent 29.99%、auto-frozen 35.14%。Seed 677 的留出成功率下降，不能声称每次共享更新都改善策略。正式 audit 51 项通过。

这是已知路由与边界下的 probe-mediated transfer；A2/B2 在 boot 前已花费 probe steps。它没有实现未知 context/route discovery，也没有测策略收益的安全 gate。

## 产物与复现

- 协议：[SHARED_WRITE_AUTO_PROTOCOL.md](../../docs/SHARED_WRITE_AUTO_PROTOCOL.md)；配置：[shared_write_auto.json](../../configs/shared_write_auto.json)；代码入口：[runner](../../scripts/run_shared_write_auto.py)、[grouper](../../src/online_rl/shared_write_grouping.py)、[shared policy](../../src/online_rl/shared_write_policy.py)。
- [Formal summary](../../outputs/knowledge_shared_write_auto_formal/summary.json)、[结果报告](../../outputs/knowledge_shared_write_auto_report/阶段S1b结果.md)、[audit](../../outputs/knowledge_shared_write_auto_report/audit.json)、[cold-transfer figure](../../outputs/knowledge_shared_write_auto_report/probe_mediated_cold_transfer.png)。
- 一个 cohort-level producer invocation 的根为 [outputs/knowledge_shared_write_auto_formal](../../outputs/knowledge_shared_write_auto_formal)；[runs/run001/provenance.json](runs/run001/provenance.json) 列出 12 个 seed-condition 输出单元、source manifest/snapshot 与 output paths，不表示 12 个已知独立 CLI invocation。实际 shell command 和执行 commit 未记录；协议、runner 与 per-component snapshots 是复现参考，不能断言它们就是原执行命令/commit。原始轨迹未复制。

**成本限制：** 历史批准预算未知；正式结果只适用于该单一 W0、生成器、probe/action 先验和给定 route。

## 归属与分析版本导航

主生命周期：[H011](../../research/records/H011.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
