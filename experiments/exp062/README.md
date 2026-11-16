# E062：S1a 真实共享写入 formal cohort

**H：** [H010](../../research/records/H010.md) · **性质：** 5 个正式部署种子的受控机制比较。**状态：** 原运行已完成；2026-10-20 回顾性整理。

## 设计与决策

比较 oracle-active 持续学习共享修正、oracle-frozen 初期共享后冻结、独立局部修正和较快独立学习。五个新 seed 为 601、619、631、647、659；固定 W0、给定 route/family、六情境序列。训练 40,960 步/臂，三 fit rollout 加一个 holdout rollout/1,024 训练步。主指标是 A2/B2 cold return 及 boot→seen 变化；同时记录在线成功率与旧情境保持。

## 结果、成本与限制

20 个 seed-condition 输出单元（5 seeds × 4 arms）共 819,200 训练步、1,010,652 含评估交互，属于一个 formal cohort producer invocation。持续共享 cold return 均值 −0.3368，冻结共享 −0.8621；active 在 5/5 seeds 上高于 frozen。Online success 为 active 47.70%、frozen 34.22%、独立 29.73%、较快独立 33.80%。一些种子的目标成功率下降，因此不是单调改进保证。Actor 实际字节为 shared 738 B、independent 666 B；两者只共用 1,024 B 上限，不能主张容量节省。正式 audit 50 项检查通过。

## 产物与复现

- 冻结协议：[SHARED_WRITE_PROTOCOL.md](../../docs/SHARED_WRITE_PROTOCOL.md)；配置：[shared_write.json](../../configs/shared_write.json)；代码入口：[run_shared_write.py](../../scripts/run_shared_write.py)、[policy](../../src/online_rl/shared_write_policy.py)、[storage](../../src/online_rl/shared_write_storage.py)。
- [Formal summary](../../outputs/knowledge_shared_write_formal/summary.json)、[结果报告](../../outputs/knowledge_shared_write_report/阶段S1a结果.md)、[audit](../../outputs/knowledge_shared_write_report/audit.json)、[cold-transfer figure](../../outputs/knowledge_shared_write_report/cold_transfer.png)。
- 一个 cohort-level producer invocation 的根为 [outputs/knowledge_shared_write_formal](../../outputs/knowledge_shared_write_formal)；[runs/run001/provenance.json](runs/run001/provenance.json) 列出 20 个 seed-condition 输出单元及各自的 config、source manifest/snapshot 与 summary 路径，不表示 20 个已知独立 CLI invocation。实际历史命令与执行 Git commit 未记录；source manifest 和 snapshot 只提供逐文件版本证据，协议及 runner 仅为复现参考。原始轨迹不复制。

**成本限制：** 历史批准的计算/时间预算未找到记录。任务 family 和 routes 已知，单一 W0 与小型确定性生成器限制外推。

## 归属与分析版本导航

主生命周期：[H010](../../research/records/H010.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
