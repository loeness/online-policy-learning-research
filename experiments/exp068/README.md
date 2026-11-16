# E068：S3 same-dynamics reward-conflict pilot

**H：** [H013](../../research/records/H013.md) · **性质：** 单 seed 短 pilot。**状态：** 原运行已完成；2026-10-20 回顾性整理。

## 设计与决策

在共享相同动作动力学的 plus/minus route 上，令 minus 奖励反号。比较独立局部头、持续共享与 plus 后冻结共享；每个 mode 均先做 512 步 probe。Pilot seed 853、每 mode 2,048 training steps。观察共享是否对负目标 cold return 产生可见影响，并记录旧目标保持。

## 结果、成本与限制

Pilot 共 12,288 training、3,072 probe、9,183 evaluation、24,543 总交互。该短 seed 中，shared/frozen 的 minus cold transfer 为 0，旧 plus 回报变化也为 0；这不排除正式种子中的负迁移。Pilot 结果保留，但只属于流程/短预算观察。42/42 audit checks 通过。

## 产物与复现

- 协议：[SHARED_WRITE_CONFLICT_PROTOCOL.md](../../docs/SHARED_WRITE_CONFLICT_PROTOCOL.md) 与 [SHARED_WRITE_PROTOCOL.md](../../docs/SHARED_WRITE_PROTOCOL.md)；配置：[shared_write_conflict.json](../../configs/shared_write_conflict.json)；代码入口：[run_shared_write_conflict.py](../../scripts/run_shared_write_conflict.py)。
- [Pilot summary](../../outputs/knowledge_shared_write_conflict_pilot/summary.json)、[pilot report](../../outputs/knowledge_shared_write_conflict_pilot_report/report.md)、[audit data](../../outputs/knowledge_shared_write_conflict_pilot_report/report.json)。
- 一个 cohort-level producer invocation 的根为 [outputs/knowledge_shared_write_conflict_pilot](../../outputs/knowledge_shared_write_conflict_pilot)；[runs/run001/provenance.json](runs/run001/provenance.json) 列出 3 个 seed-condition 输出单元，不表示 3 个已知独立 CLI invocation。原始命令和执行 Git commit 未记录；保留 per-component source manifest/snapshot 链接，不复制 raw data。

**成本限制：** 历史批准预算未找到记录；单 seed 短程零差不能作稳定性结论。

## 归属与分析版本导航

主生命周期：[H013](../../research/records/H013.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
