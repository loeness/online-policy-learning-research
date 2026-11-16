# E098 — Unknown phase full-use independent budget baseline

- **主 H：** H015（../../research/records/H015.md）
- **类别 / operational status：** budget-matched control；COMPLETED
- **归档：** retrospectively imported 2026-10-20；本记录不表示本次执行或预注册。

## 设计与结果

Unknown phase independent7_lru, 3 seeds × 24 bouts × 2,048 training steps; every fresh row used for PPO, zero probes. Total training interaction budget matches unknown shared stream; PPO update count and compute are not matched.

- **Seeds / arms：** 1031、1033、1039。
- **指标与方向：** reward/(training+probe interaction) 越高越好；evaluation separate.
- **配置：** [configs/write_followup.json](../../configs/write_followup.json)；[docs/WRITE_BUDGET_BASELINE.md](../../docs/WRITE_BUDGET_BASELINE.md)；[outputs/write_stream_unknown_budget_baseline/frozen_config.json](../../outputs/write_stream_unknown_budget_baseline/frozen_config.json)
- **历史批准 compute/time budget：** UNKNOWN。

155,524 interactions (147,456 training + 8,068 evaluation). Mean reward/interaction 0.008187 versus E095 trial_gate_relaxed 0.005722 and reward_shared 0.005546. Audit passed.

**实际成本：** 155,524 interactions.

## 证据与复现信息

- **原始输出路径（未复制；含原始 run artifacts）:** [outputs/write_stream_unknown_budget_baseline](../../outputs/write_stream_unknown_budget_baseline)
- **报告 / derived analysis:** [outputs/write_stream_unknown_budget_baseline_report/report.md](../../outputs/write_stream_unknown_budget_baseline_report/report.md)；[outputs/write_stream_unknown_budget_baseline_report/analysis.json](../../outputs/write_stream_unknown_budget_baseline_report/analysis.json)
- **审计 / failures / decision records:** [outputs/write_current_reports/index.json](../../outputs/write_current_reports/index.json)
- **图表:** [outputs/write_stream_unknown_budget_baseline_report/write_stream_metrics.png](../../outputs/write_stream_unknown_budget_baseline_report/write_stream_metrics.png)；[outputs/write_stream_unknown_budget_baseline_report/write_stream_metrics.pdf](../../outputs/write_stream_unknown_budget_baseline_report/write_stream_metrics.pdf)
- **执行日志:** [outputs/write_execution_logs/followups_status.json](../../outputs/write_execution_logs/followups_status.json)
- **源快照 manifest：** [outputs/write_stream_unknown_budget_baseline/seed_1039/independent7_lru/source_snapshot/manifest.json](../../outputs/write_stream_unknown_budget_baseline/seed_1039/independent7_lru/source_snapshot/manifest.json)
- **输入数据路径：** [data/pretrain_source.npz](../../data/pretrain_source.npz) (SHA-256 493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2)；[data/foundation.pt](../../data/foundation.pt) (SHA-256 e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222)；[outputs/write_decision_selection/selection.json](../../outputs/write_decision_selection/selection.json) (SHA-256 17f739ee1c6d6278a7231fd8320ca4b2910ed422a972d3a3e74324a4eae58fac)
- **历史命令：** 确认的 argv 在相应 runs/runNNN/provenance.json，并链接命令证据。
- **重跑说明：** 历史 argv 在 [outputs/write_execution_logs/followups_status.json](../../outputs/write_execution_logs/followups_status.json)；未来运行必须使用新输出目录。
- **版本缺口：** 未找到实际执行的完整 Git commit SHA。Linked manifests 的逐文件 hashes 留在旧路径；本次不生成 source snapshot。
- **Per-invocation provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)

## 归属与分析版本导航

主生命周期：[H015](../../research/records/H015.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
