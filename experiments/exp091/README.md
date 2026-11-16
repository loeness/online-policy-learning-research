# E091 — 有界 LRU 父模块回收 validation

- **主 H：** H014（../../research/records/H014.md）
- **类别 / 运行状态：** bounded capacity recycle validation；COMPLETED; formal/unknown follow-up SKIPPED per prespecified rule
- **归档性质：** retrospectively imported on 2026-10-20；不是本次运行，也不是预注册。

## 冻结设计与配置

Known phase，与 E090 使用相同两 seeds、同条件、同 cumulative profile；唯一预定差别为 parent_recycle=true。将 bounded recycle 与 fixed parents 做 paired validation，若 paired mean gain 非正则不进入 formal/unknown。

- **Seeds / arms：** 1049、1051；reward_shared 与 trial_gate_relaxed。
- **指标与方向：** 在线 reward / training+probe interaction 越高越好；prespecified paired recycle gain。
- **配置：** [configs/write_followup.json](../../configs/write_followup.json)；[outputs/write_parent_recycle_validation/frozen_config.json](../../outputs/write_parent_recycle_validation/frozen_config.json)；[docs/WRITE_RECYCLE_PROTOCOL.md](../../docs/WRITE_RECYCLE_PROTOCOL.md)
- **历史批准 compute/time budget：** UNKNOWN（未从现存记录核实）。

## 实际结果与成本

119,428 interactions。recycle − fixed parent paired mean gain: trial_gate_relaxed −0.000651，reward_shared −0.000748；formal 与 unknown follow-up 按预设规则跳过。旧报告曾标 audit failed；更晚的派生 report/index 均报告 0 audit failures，通过状态保留版本差异。

**实际成本：** 98,304 training + 9,216 probe + 11,908 evaluation = 119,428 interactions。

## 工件与复现 provenance

- **原始 run 输出（保留在旧路径，含其原有 steps / summary / checkpoints）:** [outputs/write_parent_recycle_validation](../../outputs/write_parent_recycle_validation)；[outputs/write_parent_selection](../../outputs/write_parent_selection)
- **报告 / 机器分析:** [outputs/write_parent_recycle_validation_report/report.md](../../outputs/write_parent_recycle_validation_report/report.md)；[outputs/write_current_reports/write_parent_recycle_validation/report.md](../../outputs/write_current_reports/write_parent_recycle_validation/report.md)
- **审计 / 失败 / 决策记录:** [outputs/write_parent_selection/selection.json](../../outputs/write_parent_selection/selection.json)；[outputs/write_parent_selection/formal_decision.json](../../outputs/write_parent_selection/formal_decision.json)；[outputs/write_parent_selection/SKIPPED.json](../../outputs/write_parent_selection/SKIPPED.json)；[outputs/write_current_reports/index.json](../../outputs/write_current_reports/index.json)
- **图表:** [outputs/write_parent_recycle_validation_report/write_stream_metrics.png](../../outputs/write_parent_recycle_validation_report/write_stream_metrics.png)；[outputs/write_current_reports/write_parent_recycle_validation/write_stream_metrics.png](../../outputs/write_current_reports/write_parent_recycle_validation/write_stream_metrics.png)
- **执行日志:** [outputs/write_execution_logs/failed_clone_attempt/followups_status.json](../../outputs/write_execution_logs/failed_clone_attempt/followups_status.json)
- **代码 source snapshot manifest：** [outputs/write_parent_recycle_validation/seed_1051/trial_gate_relaxed/source_snapshot/manifest.json](../../outputs/write_parent_recycle_validation/seed_1051/trial_gate_relaxed/source_snapshot/manifest.json)
- **输入数据路径：** [data/pretrain_source.npz](../../data/pretrain_source.npz) (SHA-256 493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2)；[data/foundation.pt](../../data/foundation.pt) (SHA-256 e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222)；[outputs/write_decision_selection/selection.json](../../outputs/write_decision_selection/selection.json) (SHA-256 17f739ee1c6d6278a7231fd8320ca4b2910ed422a972d3a3e74324a4eae58fac)
- **历史执行命令：** 可在各 run provenance.json 中找到，并注明命令证据：[outputs/write_execution_logs/failed_clone_attempt/followups_status.json](../../outputs/write_execution_logs/failed_clone_attempt/followups_status.json)
- **未来重跑说明：** 历史 argv 可在 [outputs/write_execution_logs/failed_clone_attempt/followups_status.json](../../outputs/write_execution_logs/failed_clone_attempt/followups_status.json) 中恢复；此历史 validation 不授权已跳过的 formal/unknown cohorts。
- **Git commit：** MISSING。逐文件 source manifest 可引用，但不含能核实的完整执行 commit SHA；归档上下文 commit 不作为运行 SHA。
- **Figures：** [outputs/write_parent_recycle_validation_report/write_stream_metrics.png](../../outputs/write_parent_recycle_validation_report/write_stream_metrics.png)；[outputs/write_current_reports/write_parent_recycle_validation/write_stream_metrics.png](../../outputs/write_current_reports/write_parent_recycle_validation/write_stream_metrics.png)
- **Run provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)

## 归属与分析版本导航

主生命周期：[H014](../../research/records/H014.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
- [analysis003](analyses/analysis003/README.md)
