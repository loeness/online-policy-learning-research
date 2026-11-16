# E095 — Unknown-boundary continual stream main，含 interruption/resume

- **主 H：** H015（../../research/records/H015.md）
- **类别 / operational status：** exploratory main stream with retained interrupted attempt；COMPLETED with an earlier interrupted partial cohort and transparent metadata repair
- **归档：** retrospectively imported 2026-10-20；本记录不表示本次执行或预注册。

## 设计与结果

Continuous unknown-boundary phase on 12 task contexts; three seeds × four conditions. Router sees no task name/family/preference/switch notification; mixed uniform action behavior and route events are logged. An externally interrupted partial cohort is retained; later resume only completed missing cells. Summary schema repair derives missing fields from verified frozen config.

- **Seeds / arms：** 1031、1033、1039；4 conditions, 12 completed runs。
- **指标与方向：** reward/(training+probe interactions), training reward, positive task success, router/profile diagnostics; evaluation is reported separately.
- **配置：** [configs/write_stream.json](../../configs/write_stream.json)；[configs/write_followup.json](../../configs/write_followup.json)；[outputs/write_stream_unknown/frozen_config.json](../../outputs/write_stream_unknown/frozen_config.json)；[outputs/write_stream_unknown/RESUME_NOTE.json](../../outputs/write_stream_unknown/RESUME_NOTE.json)；[outputs/write_stream_unknown/METADATA_REPAIR_NOTE.json](../../outputs/write_stream_unknown/METADATA_REPAIR_NOTE.json)
- **历史批准 compute/time budget：** UNKNOWN。

Completed cohort: 622,487 interactions; an earlier interrupted attempt separately recorded 36,442 training steps. These are distinct physical expenditures; do not assume the partial attempt is included in the final completed summary. Full-stream means: trial_gate_relaxed 0.005722, reward_shared 0.005546, budget-matched independent7_lru 0.008187 (E098). Gate positive training success was about 13.2% vs 21.6%; endpoint success about 18.1% vs 26.4%. Route labels are observer diagnostics, not ground truth.

**实际成本：** Completed cohort 589,824 training + 32,663 evaluation = 622,487; earlier interruption 36,442 training steps. At least 658,929 physical interactions including both attempts; no double-counting inside the completed cohort.

## 证据与复现信息

- **原始输出路径（未复制；含原始 run artifacts）:** [outputs/write_stream_unknown_interrupted](../../outputs/write_stream_unknown_interrupted)；[outputs/write_stream_unknown](../../outputs/write_stream_unknown)
- **报告 / derived analysis:** [outputs/write_stream_unknown_report/report.md](../../outputs/write_stream_unknown_report/report.md)；[outputs/write_stream_unknown_report/analysis.json](../../outputs/write_stream_unknown_report/analysis.json)；[outputs/write_stream_unknown_report_pre_schema_repair/report.md](../../outputs/write_stream_unknown_report_pre_schema_repair/report.md)；[outputs/write_stream_unknown_report_pre_schema_repair/analysis.json](../../outputs/write_stream_unknown_report_pre_schema_repair/analysis.json)
- **审计 / failures / decision records:** [outputs/write_stream_unknown_interrupted/FAILED_COHORT.json](../../outputs/write_stream_unknown_interrupted/FAILED_COHORT.json)；[outputs/write_stream_unknown/RESUME_NOTE.json](../../outputs/write_stream_unknown/RESUME_NOTE.json)；[outputs/write_stream_unknown/METADATA_REPAIR_NOTE.json](../../outputs/write_stream_unknown/METADATA_REPAIR_NOTE.json)；[outputs/write_current_reports/index.json](../../outputs/write_current_reports/index.json)
- **图表:** [outputs/write_stream_unknown_report/write_stream_metrics.png](../../outputs/write_stream_unknown_report/write_stream_metrics.png)；[outputs/write_stream_unknown_report/write_stream_metrics.pdf](../../outputs/write_stream_unknown_report/write_stream_metrics.pdf)；[outputs/write_stream_unknown_report_pre_schema_repair/write_stream_metrics.png](../../outputs/write_stream_unknown_report_pre_schema_repair/write_stream_metrics.png)；[outputs/write_stream_unknown_report_pre_schema_repair/write_stream_metrics.pdf](../../outputs/write_stream_unknown_report_pre_schema_repair/write_stream_metrics.pdf)
- **执行日志:** [outputs/write_execution_logs](../../outputs/write_execution_logs)
- **源快照 manifest：** [outputs/write_stream_unknown/seed_1039/trial_gate_relaxed/source_snapshot/manifest.json](../../outputs/write_stream_unknown/seed_1039/trial_gate_relaxed/source_snapshot/manifest.json)；[outputs/write_stream_unknown_interrupted/seed_1031/independent7_lru/source_snapshot/manifest.json](../../outputs/write_stream_unknown_interrupted/seed_1031/independent7_lru/source_snapshot/manifest.json)
- **输入数据路径：** [data/pretrain_source.npz](../../data/pretrain_source.npz) (SHA-256 493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2)；[data/foundation.pt](../../data/foundation.pt) (SHA-256 e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222)；[outputs/write_decision_selection/selection.json](../../outputs/write_decision_selection/selection.json) (SHA-256 17f739ee1c6d6278a7231fd8320ca4b2910ed422a972d3a3e74324a4eae58fac)
- **历史命令：** 完整历史 argv 未找到；下方重跑说明与历史命令分开。
- **重跑说明：** Historical command/runner arguments were not fully recovered. Reproduction recipe (not historical command): follow [docs/WRITE_STREAM_PROTOCOL.md](../../docs/WRITE_STREAM_PROTOCOL.md) for phase unknown and preserve the frozen config; start with a new output directory.
- **版本缺口：** 未找到实际执行的完整 Git commit SHA。Linked manifests 的逐文件 hashes 留在旧路径；本次不生成 source snapshot。
- **Per-invocation provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)；[runs/run002/provenance.json](runs/run002/provenance.json)


## 归属与分析版本导航

主生命周期：[H015](../../research/records/H015.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
