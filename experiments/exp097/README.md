# E097 — Known phase full-use independent budget baseline main，含失败 clone attempt

- **主 H：** H015（../../research/records/H015.md）
- **类别 / operational status：** budget-matched formal control with failed retry；COMPLETED after retained failed attempt
- **归档：** retrospectively imported 2026-10-20；本记录不表示本次执行或预注册。

## 设计与结果

Known phase independent7_lru; 3 seeds, 24 bouts, 2,144 training steps/bout. Matches shared arms' training+probe interaction budget, but spends all steps on PPO and has different fit rows, updates and compute. First full run failed during endpoint clone; fixed rerun is separate run.

- **Seeds / arms：** 1031、1033、1039；independent7_lru only。
- **指标与方向：** reward/(training+probe interaction) 越高越好；CPU and PPO updates separately reported; evaluation not in training+probe denominator.
- **配置：** [configs/write_followup.json](../../configs/write_followup.json)；[docs/WRITE_BUDGET_BASELINE.md](../../docs/WRITE_BUDGET_BASELINE.md)；[outputs/write_stream_known_budget_baseline/frozen_config.json](../../outputs/write_stream_known_budget_baseline/frozen_config.json)
- **历史批准 compute/time budget：** UNKNOWN。

Completed baseline: 162,413 interactions (154,368 training + 8,045 evaluation). Mean known full-stream reward/interaction 0.007487, ahead of E093 trial_gate_relaxed 0.006329 and reward_shared 0.006081. Failed clone attempt separately consumed 154,368 training steps before endpoint evaluation deepcopy error; keep both attempts.

**实际成本：** Completed 162,413 + failed attempt 154,368 = 316,781 actual interactions; failed attempt had training cost only.

## 证据与复现信息

- **原始输出路径（未复制；含原始 run artifacts）:** [outputs/write_stream_known_budget_failed_clone](../../outputs/write_stream_known_budget_failed_clone)；[outputs/write_stream_known_budget_baseline](../../outputs/write_stream_known_budget_baseline)
- **报告 / derived analysis:** [outputs/write_stream_known_budget_baseline_report/report.md](../../outputs/write_stream_known_budget_baseline_report/report.md)；[outputs/write_stream_known_budget_baseline_report/analysis.json](../../outputs/write_stream_known_budget_baseline_report/analysis.json)
- **审计 / failures / decision records:** [outputs/write_stream_known_budget_failed_clone/FAILED_COHORT.json](../../outputs/write_stream_known_budget_failed_clone/FAILED_COHORT.json)；[outputs/write_current_reports/index.json](../../outputs/write_current_reports/index.json)
- **图表:** [outputs/write_stream_known_budget_baseline_report/write_stream_metrics.png](../../outputs/write_stream_known_budget_baseline_report/write_stream_metrics.png)；[outputs/write_stream_known_budget_baseline_report/write_stream_metrics.pdf](../../outputs/write_stream_known_budget_baseline_report/write_stream_metrics.pdf)
- **执行日志:** [outputs/write_execution_logs/failed_clone_attempt/known_budget_baseline.stderr.log](../../outputs/write_execution_logs/failed_clone_attempt/known_budget_baseline.stderr.log)；[outputs/write_execution_logs/followups_status.json](../../outputs/write_execution_logs/followups_status.json)
- **源快照 manifest：** [outputs/write_stream_known_budget_baseline/seed_1039/independent7_lru/source_snapshot/manifest.json](../../outputs/write_stream_known_budget_baseline/seed_1039/independent7_lru/source_snapshot/manifest.json)；[outputs/write_stream_known_budget_failed_clone/seed_1039/independent7_lru/source_snapshot/manifest.json](../../outputs/write_stream_known_budget_failed_clone/seed_1039/independent7_lru/source_snapshot/manifest.json)
- **输入数据路径：** [data/pretrain_source.npz](../../data/pretrain_source.npz) (SHA-256 493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2)；[data/foundation.pt](../../data/foundation.pt) (SHA-256 e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222)；[outputs/write_decision_selection/selection.json](../../outputs/write_decision_selection/selection.json) (SHA-256 17f739ee1c6d6278a7231fd8320ca4b2910ed422a972d3a3e74324a4eae58fac)
- **历史命令：** 确认的 argv 在相应 runs/runNNN/provenance.json，并链接命令证据。
- **重跑说明：** 成功 cohort 的历史 argv 见 [outputs/write_execution_logs/followups_status.json](../../outputs/write_execution_logs/followups_status.json)。将命令用于未来实验时应换新输出路径，且不得覆盖 failed clone。
- **版本缺口：** 未找到实际执行的完整 Git commit SHA。Linked manifests 的逐文件 hashes 留在旧路径；本次不生成 source snapshot。
- **Per-invocation provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)；[runs/run002/provenance.json](runs/run002/provenance.json)


## 归属与分析版本导航

主生命周期：[H015](../../research/records/H015.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
