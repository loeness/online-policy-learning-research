# E093 — Known-boundary continual write stream main

- **主 H：** H015（../../research/records/H015.md）
- **类别 / operational status：** exploratory main stream；COMPLETED
- **归档：** retrospectively imported 2026-10-20；本记录不表示本次执行或预注册。

## 设计与结果

Continuous known-local-index phase over 12 reward-aware C1 tasks and reverse recurrence; three seeds, seven conditions; cache eviction and probes are charged. The controlled runner does not expose true family labels to the grouper.

- **Seeds / arms：** 1031、1033、1039；7 conditions, 21 runs.
- **指标与方向：** online reward/train step and reward/(training+probe interactions); 3-seed descriptive paired comparisons.
- **配置：** [configs/write_stream.json](../../configs/write_stream.json)；[configs/write_followup.json](../../configs/write_followup.json)；[outputs/write_stream_known/frozen_config.json](../../outputs/write_stream_known/frozen_config.json)；[outputs/write_decision_selection/selection.json](../../outputs/write_decision_selection/selection.json)
- **历史批准 compute/time budget：** UNKNOWN。

1,136,281 interactions. Full-stream reward/interaction means: trial_gate_relaxed 0.006329, reward_shared 0.006081. The later interaction-matched full-use independent7_lru control scored 0.007487 (E097); selected gate did not beat the stronger control. No held-out final policy claim.

**实际成本：** 1,032,192 training + 47,744 probe + 56,345 evaluation = 1,136,281 interactions.

## 证据与复现信息

- **原始输出路径（未复制；含原始 run artifacts）:** [outputs/write_stream_known](../../outputs/write_stream_known)
- **报告 / derived analysis:** [outputs/write_stream_known_report/report.md](../../outputs/write_stream_known_report/report.md)；[outputs/write_stream_known_report/analysis.json](../../outputs/write_stream_known_report/analysis.json)
- **审计 / failures / decision records:** [outputs/write_current_reports/index.json](../../outputs/write_current_reports/index.json)
- **图表:** [outputs/write_stream_known_report/write_stream_metrics.png](../../outputs/write_stream_known_report/write_stream_metrics.png)；[outputs/write_stream_known_report/write_stream_metrics.pdf](../../outputs/write_stream_known_report/write_stream_metrics.pdf)
- **执行日志:** NOT_APPLICABLE（无独立 artifact）
- **源快照 manifest：** [outputs/write_stream_known/seed_1039/trial_gate_relaxed/source_snapshot/manifest.json](../../outputs/write_stream_known/seed_1039/trial_gate_relaxed/source_snapshot/manifest.json)
- **输入数据路径：** [data/pretrain_source.npz](../../data/pretrain_source.npz) (SHA-256 493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2)；[data/foundation.pt](../../data/foundation.pt) (SHA-256 e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222)；[outputs/write_decision_selection/selection.json](../../outputs/write_decision_selection/selection.json) (SHA-256 17f739ee1c6d6278a7231fd8320ca4b2910ed422a972d3a3e74324a4eae58fac)
- **历史命令：** 完整历史 argv 未找到；下方重跑说明与历史命令分开。
- **重跑说明：** Historical argv was not recovered in the archived execution log set. Reproduction recipe (not historical command): follow [docs/WRITE_STREAM_PROTOCOL.md](../../docs/WRITE_STREAM_PROTOCOL.md) using the frozen known config, selection file, and a fresh output root.
- **版本缺口：** 未找到实际执行的完整 Git commit SHA。Linked manifests 的逐文件 hashes 留在旧路径；本次不生成 source snapshot。
- **Per-invocation provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)


## 归属与分析版本导航

主生命周期：[H015](../../research/records/H015.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
