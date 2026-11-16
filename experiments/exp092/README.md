# E092 — Known-boundary continual stream pilot attempts v1–v4

- **主 H：** H015（../../research/records/H015.md）
- **类别 / operational status：** pilot / four preserved attempts；COMPLETED after three failed/stopped attempts
- **归档：** retrospectively imported 2026-10-20；本记录不表示本次执行或预注册。

## 设计与结果

Known local-index phase; one pilot seed 997, short bout schedule to exercise cache allocation and shared grouping. v1-v3 are retained failures/stops; v4 is the completed pilot. They are distinct producer attempts under one pilot design and are not formal seeds.

- **Seeds / arms：** 997; seven stream conditions in final v4.
- **指标与方向：** pilot correctness/coverage; online reward and cost are descriptive only.
- **配置：** [configs/write_stream.json](../../configs/write_stream.json)；[outputs/write_stream_known_pilot_v4/frozen_config.json](../../outputs/write_stream_known_pilot_v4/frozen_config.json)
- **历史批准 compute/time budget：** UNKNOWN。

v1 failed before environment interaction (0). v2 stopped after 40 training + 128 probe = 168; v3 after 3,368 training + 512 probe = 3,880. v4 completed 28,317 total interactions. Aggregate physical cost across preserved attempts is 32,365 interactions. Only v4 reaches the final pilot report; no pilot result validates formal performance.

**实际成本：** 32,365 interactions across all four attempts; v4 itself 24,576 training + 2,304 probe + 1,437 evaluation = 28,317.

## 证据与复现信息

- **原始输出路径（未复制；含原始 run artifacts）:** [outputs/write_stream_known_pilot](../../outputs/write_stream_known_pilot)；[outputs/write_stream_known_pilot_v2](../../outputs/write_stream_known_pilot_v2)；[outputs/write_stream_known_pilot_v3](../../outputs/write_stream_known_pilot_v3)；[outputs/write_stream_known_pilot_v4](../../outputs/write_stream_known_pilot_v4)
- **报告 / derived analysis:** [outputs/write_stream_known_pilot_report/report.md](../../outputs/write_stream_known_pilot_report/report.md)；[outputs/write_stream_known_pilot_report/analysis.json](../../outputs/write_stream_known_pilot_report/analysis.json)
- **审计 / failures / decision records:** [outputs/write_stream_known_pilot/FAILED_COHORT.json](../../outputs/write_stream_known_pilot/FAILED_COHORT.json)；[outputs/write_stream_known_pilot_v2/FAILED_COHORT.json](../../outputs/write_stream_known_pilot_v2/FAILED_COHORT.json)；[outputs/write_stream_known_pilot_v3/FAILED_COHORT.json](../../outputs/write_stream_known_pilot_v3/FAILED_COHORT.json)；[outputs/write_current_reports/index.json](../../outputs/write_current_reports/index.json)
- **图表:** [outputs/write_stream_known_pilot_report/write_stream_metrics.png](../../outputs/write_stream_known_pilot_report/write_stream_metrics.png)；[outputs/write_stream_known_pilot_report/write_stream_metrics.pdf](../../outputs/write_stream_known_pilot_report/write_stream_metrics.pdf)
- **执行日志:** [outputs/write_execution_logs/attempt_history](../../outputs/write_execution_logs/attempt_history)
- **源快照 manifest：** [outputs/write_stream_known_pilot_v4/seed_997/trial_gate_relaxed/source_snapshot/manifest.json](../../outputs/write_stream_known_pilot_v4/seed_997/trial_gate_relaxed/source_snapshot/manifest.json)
- **输入数据路径：** [data/pretrain_source.npz](../../data/pretrain_source.npz) (SHA-256 493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2)；[data/foundation.pt](../../data/foundation.pt) (SHA-256 e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222)；[outputs/write_decision_selection/selection.json](../../outputs/write_decision_selection/selection.json) (SHA-256 17f739ee1c6d6278a7231fd8320ca4b2910ed422a972d3a3e74324a4eae58fac)
- **历史命令：** 完整历史 argv 未找到；下方重跑说明与历史命令分开。
- **重跑说明：** 历史 argv 未完整恢复。按 [docs/WRITE_STREAM_PROTOCOL.md](../../docs/WRITE_STREAM_PROTOCOL.md) 与冻结 pilot config，在新目录做 pilot；不要覆盖 v1-v4。
- **版本缺口：** 未找到实际执行的完整 Git commit SHA。Linked manifests 的逐文件 hashes 留在旧路径；本次不生成 source snapshot。
- **Per-invocation provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)；[runs/run002/provenance.json](runs/run002/provenance.json)；[runs/run003/provenance.json](runs/run003/provenance.json)；[runs/run004/provenance.json](runs/run004/provenance.json)


## 归属与分析版本导航

主生命周期：[H015](../../research/records/H015.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
