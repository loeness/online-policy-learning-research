# E094 — Unknown-boundary continual stream pilot

- **主 H：** H015（../../research/records/H015.md）
- **类别 / operational status：** pilot；COMPLETED
- **归档：** retrospectively imported 2026-10-20；本记录不表示本次执行或预注册。

## 设计与结果

Unknown route phase; router sees visible observations/transitions only. One pilot seed 997, all selected stream conditions in the pilot config. Pilot is separate from formal cohort.

- **Seeds / arms：** 997。
- **指标与方向：** pilot flow/audit coverage; reward is descriptive, not selection or confirmation.
- **配置：** [configs/write_stream.json](../../configs/write_stream.json)；[outputs/write_stream_unknown_pilot/frozen_config.json](../../outputs/write_stream_unknown_pilot/frozen_config.json)
- **历史批准 compute/time budget：** UNKNOWN。

25,987 interactions completed: 24,576 training + 1,411 evaluation; zero paid probe interactions. Pilot did not create an independent performance claim.

**实际成本：** 25,987 interactions.

## 证据与复现信息

- **原始输出路径（未复制；含原始 run artifacts）:** [outputs/write_stream_unknown_pilot](../../outputs/write_stream_unknown_pilot)
- **报告 / derived analysis:** [outputs/write_stream_unknown_pilot_report/report.md](../../outputs/write_stream_unknown_pilot_report/report.md)；[outputs/write_stream_unknown_pilot_report/analysis.json](../../outputs/write_stream_unknown_pilot_report/analysis.json)
- **审计 / failures / decision records:** [outputs/write_current_reports/index.json](../../outputs/write_current_reports/index.json)
- **图表:** [outputs/write_stream_unknown_pilot_report/write_stream_metrics.png](../../outputs/write_stream_unknown_pilot_report/write_stream_metrics.png)；[outputs/write_stream_unknown_pilot_report/write_stream_metrics.pdf](../../outputs/write_stream_unknown_pilot_report/write_stream_metrics.pdf)
- **执行日志:** NOT_APPLICABLE（无独立 artifact）
- **源快照 manifest：** [outputs/write_stream_unknown_pilot/seed_997/trial_gate_relaxed/source_snapshot/manifest.json](../../outputs/write_stream_unknown_pilot/seed_997/trial_gate_relaxed/source_snapshot/manifest.json)
- **输入数据路径：** [data/pretrain_source.npz](../../data/pretrain_source.npz) (SHA-256 493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2)；[data/foundation.pt](../../data/foundation.pt) (SHA-256 e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222)；[outputs/write_decision_selection/selection.json](../../outputs/write_decision_selection/selection.json) (SHA-256 17f739ee1c6d6278a7231fd8320ca4b2910ed422a972d3a3e74324a4eae58fac)
- **历史命令：** 完整历史 argv 未找到；下方重跑说明与历史命令分开。
- **重跑说明：** Historical argv was not recovered. Reproduction recipe (not historical command): use [docs/WRITE_STREAM_PROTOCOL.md](../../docs/WRITE_STREAM_PROTOCOL.md) with phase unknown, the frozen pilot settings, and a fresh output root.
- **版本缺口：** 未找到实际执行的完整 Git commit SHA。Linked manifests 的逐文件 hashes 留在旧路径；本次不生成 source snapshot。
- **Per-invocation provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)


## 归属与分析版本导航

主生命周期：[H015](../../research/records/H015.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
