# E069：S3 same-dynamics reward-conflict formal cohort

**H：** [H013](../../research/records/H013.md) · **性质：** 三个正式种子的冲突目标诊断。**状态：** 原运行已完成；2026-10-20 回顾性整理。

## 设计与决策

Plus/minus 两 route 共享 A0 动力学、观测和终止条件，minus 奖励为相反符号。Probe profile 不含 reward 或 route/mode label。比较独立、持续共享及 plus 后冻结共享；formal seeds 773、787、797；每 mode 512 probe steps、8,192 training steps。Primary metric 是 plus 学完后 minus 的 signed cold return；之后分别报告 minus adaptation 与 plus retention。

## 结果、成本与限制

9 个 seed-condition 输出单元（3 seeds × 3 arms）合计 147,456 training、9,216 probe、51,176 evaluation、207,848 总交互，属于一个 formal cohort producer invocation。共享更新把未训练 minus 的 cold return 平均降低 0.5477，逐 seed 为 −0.6375、−0.41875、−0.586875；独立局部头变化为 0。Frozen 与 active 在 plus 后有相同冷迁移。其后学习 minus 时 active 的 plus-return retention 变化平均 −0.0444（−0.0244、−0.1163、+0.0075），方向混合；没有严重崩溃证据。正目标最终 success 只有 27.08%，所以不能把结果描述成已建立的强安全或灾难性遗忘结论。120/120 formal audit checks 通过。

此实验说明在这组反例中，同动力学 profile 不足以判断共享更新是否有益；它不证明所有动力学 grouping 或完整情境识别都失败。

## 产物与复现

- 协议：[SHARED_WRITE_CONFLICT_PROTOCOL.md](../../docs/SHARED_WRITE_CONFLICT_PROTOCOL.md)；配置：[shared_write_conflict.json](../../configs/shared_write_conflict.json)；代码入口：[runner](../../scripts/run_shared_write_conflict.py)、[conflict env](../../src/online_rl/shared_write_conflict_env.py)、[grouping](../../src/online_rl/shared_write_grouping.py)。
- [Formal summary](../../outputs/knowledge_shared_write_conflict_formal/summary.json)、[S3 result](../../outputs/knowledge_shared_write_conflict_report/阶段S3结果.md)、[report JSON/audit](../../outputs/knowledge_shared_write_conflict_report/report.json)、[figure](../../outputs/knowledge_shared_write_conflict_report/signed_reward_and_plus_retention.png)。
- 一个 cohort-level producer invocation 的根为 [outputs/knowledge_shared_write_conflict_formal](../../outputs/knowledge_shared_write_conflict_formal)；[runs/run001/provenance.json](runs/run001/provenance.json) 列出 9 个 seed-condition 输出单元、per-component config、source manifest/snapshot 与 summary 路径，不表示 9 个已知独立 CLI invocation。原命令与执行 commit 未记录；linked protocol/runner 只作复现参考。原始轨迹不复制。

**成本限制：** 历史批准计算/时间预算未找到记录。单一 W0、三种子、人工固定两情境和弱学习的正目标限定了解释范围。

## 归属与分析版本导航

主生命周期：[H013](../../research/records/H013.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
