# E090 — Cumulative profile validation，无父模块回收

- **主 H：** H014（../../research/records/H014.md）
- **类别 / 运行状态：** capacity evidence validation；COMPLETED
- **归档性质：** retrospectively imported on 2026-10-20；不是本次运行，也不是预注册。

## 冻结设计与配置

Known phase，2 seeds × 2 conditions，1,024 steps/bout；cumulative_profiles=true，parent_recycle=false。此 cohort 没有 cumulative_profiles=false 对照，故仅记录能力与当前 gate/baseline 表现。

- **Seeds / arms：** 1049、1051；reward_shared 与 trial_gate_relaxed。
- **指标与方向：** online reward / training+probe interaction 越高越好；cohort audit 与 group coverage。
- **配置：** [configs/write_followup.json](../../configs/write_followup.json)；[outputs/write_parent_cumulative_validation/frozen_config.json](../../outputs/write_parent_cumulative_validation/frozen_config.json)；[docs/WRITE_RECYCLE_PROTOCOL.md](../../docs/WRITE_RECYCLE_PROTOCOL.md)
- **历史批准 compute/time budget：** UNKNOWN（未从现存记录核实）。

## 实际结果与成本

119,364 interactions。trial_gate_relaxed 相对 reward_shared mean difference −0.0005。预设 parent-recycle 选取逻辑在另一个 E 下触发后续验证；本 E 本身不能隔离累计统计的因果作用。旧 report 标 audit failed；后续 [outputs/write_current_reports/index.json](../../outputs/write_current_reports/index.json) 和当前 report 对相同 run 记录 audit 通过。

**实际成本：** 98,304 training + 9,216 probe + 11,844 evaluation = 119,364 interactions。

## 工件与复现 provenance

- **原始 run 输出（保留在旧路径，含其原有 steps / summary / checkpoints）:** [outputs/write_parent_cumulative_validation](../../outputs/write_parent_cumulative_validation)
- **报告 / 机器分析:** [outputs/write_parent_cumulative_validation_report/report.md](../../outputs/write_parent_cumulative_validation_report/report.md)；[outputs/write_current_reports/write_parent_cumulative_validation/report.md](../../outputs/write_current_reports/write_parent_cumulative_validation/report.md)；[outputs/write_current_reports/index.json](../../outputs/write_current_reports/index.json)
- **审计 / 失败 / 决策记录:** [outputs/write_parent_selection/selection.json](../../outputs/write_parent_selection/selection.json)；[outputs/write_parent_selection/SKIPPED.json](../../outputs/write_parent_selection/SKIPPED.json)；[outputs/write_current_reports/index.json](../../outputs/write_current_reports/index.json)
- **图表:** [outputs/write_parent_cumulative_validation_report/write_stream_metrics.png](../../outputs/write_parent_cumulative_validation_report/write_stream_metrics.png)；[outputs/write_current_reports/write_parent_cumulative_validation/write_stream_metrics.png](../../outputs/write_current_reports/write_parent_cumulative_validation/write_stream_metrics.png)
- **执行日志:** [outputs/write_execution_logs/failed_clone_attempt/followups_status.json](../../outputs/write_execution_logs/failed_clone_attempt/followups_status.json)；[outputs/write_execution_logs/followups_status.json](../../outputs/write_execution_logs/followups_status.json)
- **代码 source snapshot manifest：** [outputs/write_parent_cumulative_validation/seed_1051/trial_gate_relaxed/source_snapshot/manifest.json](../../outputs/write_parent_cumulative_validation/seed_1051/trial_gate_relaxed/source_snapshot/manifest.json)
- **输入数据路径：** [data/pretrain_source.npz](../../data/pretrain_source.npz) (SHA-256 493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2)；[data/foundation.pt](../../data/foundation.pt) (SHA-256 e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222)；[outputs/write_decision_selection/selection.json](../../outputs/write_decision_selection/selection.json) (SHA-256 17f739ee1c6d6278a7231fd8320ca4b2910ed422a972d3a3e74324a4eae58fac)
- **历史执行命令：** 可在各 run provenance.json 中找到，并注明命令证据：[outputs/write_execution_logs/failed_clone_attempt/followups_status.json](../../outputs/write_execution_logs/failed_clone_attempt/followups_status.json)
- **未来重跑说明：** 历史实际 argv 见 outputs/write_execution_logs/failed_clone_attempt/followups_status.json（pilot coordinator）；完整 cohort命令可按同文件字段复现，使用新输出目录。
- **Git commit：** MISSING。逐文件 source manifest 可引用，但不含能核实的完整执行 commit SHA；归档上下文 commit 不作为运行 SHA。
- **Figures：** [outputs/write_parent_cumulative_validation_report/write_stream_metrics.png](../../outputs/write_parent_cumulative_validation_report/write_stream_metrics.png)；[outputs/write_current_reports/write_parent_cumulative_validation/write_stream_metrics.png](../../outputs/write_current_reports/write_parent_cumulative_validation/write_stream_metrics.png)
- **Run provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)


## 归属与分析版本导航

主生命周期：[H014](../../research/records/H014.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
