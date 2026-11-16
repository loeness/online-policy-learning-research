# E066：S2 十二情境、六局部槽 formal capacity cohort

**H：** [H012](../../research/records/H012.md) · **性质：** 三个新种子的受控缓存容量比较。**状态：** 原运行已完成；2026-10-20 回顾性整理。

## 设计与决策

十二情境固定循环，六局部槽 LRU；淘汰时清除局部参数，比较 independent-cache、较快独立、持续 shared-cache 与 frozen-shared-cache。已知 context ID 和 family membership 由外部管理器提供。Formal seeds 727、743、757；每条件 24 bouts × 4,096 训练步。Primary evidence 是重访但局部重新学习前的回报、全程 online reward/success 和已淘汰情境的终点评估。

## 结果、成本与限制

12 个 seed-condition 输出单元（3 seeds × 4 arms）共 1,179,648 训练步、262,692 评估步，总计 1,442,340 次交互，属于一个 formal cohort producer invocation。三种子均值 online success 为 shared 81.28%、independent 28.73%、较快 independent 32.09%、frozen shared 53.68%；重访前且未局部重学的 success 为 shared 60.42%，两独立条件各 17.71%，frozen 24.65%。所有 24 段均 miss、共 18 次 eviction。驻留情境终点 success 98.61%，已淘汰情境 56.25%，所以共享参数未完全保存局部特化。正式 audit 148 项通过。

Shared actor 实际 230 B，independent 204 B；相同 230 B cap 不是相同实际存储。LRU 与人为循环顺序会造成高淘汰压力；这不证明在自然复现流或总字节受限下仍有同样收益。三个种子、一个 W0 和小环境限制外推。

## 产物与复现

- 协议：[SHARED_WRITE_CAPACITY_PROTOCOL.md](../../docs/SHARED_WRITE_CAPACITY_PROTOCOL.md)；配置：[shared_write_capacity.json](../../configs/shared_write_capacity.json)；代码入口：[runner](../../scripts/run_shared_write_capacity.py)、[capacity env](../../src/online_rl/shared_write_capacity_env.py)。
- [Formal summary](../../outputs/knowledge_shared_write_capacity_formal/summary.json)、[阶段 S2 report](../../outputs/knowledge_shared_write_capacity_report/阶段S2结果.md)、[audit](../../outputs/knowledge_shared_write_capacity_report/audit.json)、[capacity figure](../../outputs/knowledge_shared_write_capacity_report/capacity_results.png)、[independent review](../../docs/SHARED_WRITE_CAPACITY_REVIEW.md)。
- 一个 cohort-level producer invocation 的根为 [outputs/knowledge_shared_write_capacity_formal](../../outputs/knowledge_shared_write_capacity_formal)；[runs/run001/provenance.json](runs/run001/provenance.json) 列出 12 个 seed-condition 输出单元、per-component config 与 source manifest/snapshot，不表示 12 个已知独立 CLI invocation。原实际 shell command 与执行 Git commit 未记录；protocol 与 runner 只是复现参考。raw traces 不复制。

**成本限制：** 历史批准预算未知。actor/package cap 不能当作完整归档容量；周期 checkpoint 不包含环境和未完成 rollout 状态，不能精确续跑轨迹。

## 归属与分析版本导航

主生命周期：[H012](../../research/records/H012.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
