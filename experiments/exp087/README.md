# E087 — Detour shifted-generator write stream，W0-29

- **主 H：** H014（../../research/records/H014.md）
- **类别 / 运行状态：** shifted foundation comparison；COMPLETED
- **归档性质：** retrospectively imported on 2026-10-20；不是本次运行，也不是预注册。

## 冻结设计与配置

与 E086 相同 detour stream 和条件，独立预训练 foundation seed 29；不与 W0-17 pooling。

- **Seeds / arms：** foundation seed 29；deployment seeds 1061、1063。
- **指标与方向：** reward / training+probe interaction 越高越好；foundation 分开。
- **配置：** [configs/write_followup.json](../../configs/write_followup.json)；[configs/write_stream.json](../../configs/write_stream.json)；[outputs/write_detour_w0_29/frozen_config.json](../../outputs/write_detour_w0_29/frozen_config.json)
- **历史批准 compute/time budget：** UNKNOWN（未从现存记录核实）。

## 实际结果与成本

325,959 interactions，audit 通过。Gate mean 0.001185，reward_shared 0.000856；后续 full-use independent W0-29 E089 为 0.001890，独立基线领先。

**实际成本：** 294,912 training + 13,568 probe + 17,479 evaluation = 325,959 interactions。

## 工件与复现 provenance

- **原始 run 输出（保留在旧路径，含其原有 steps / summary / checkpoints）:** [outputs/write_detour_w0_29](../../outputs/write_detour_w0_29)
- **报告 / 机器分析:** [outputs/write_detour_w0_29_report/report.md](../../outputs/write_detour_w0_29_report/report.md)；[outputs/write_detour_w0_29_report/analysis.json](../../outputs/write_detour_w0_29_report/analysis.json)
- **审计 / 失败 / 决策记录:** [outputs/write_current_reports/index.json](../../outputs/write_current_reports/index.json)
- **图表:** [outputs/write_detour_w0_29_report/write_stream_metrics.png](../../outputs/write_detour_w0_29_report/write_stream_metrics.png)；[outputs/write_detour_w0_29_report/write_stream_metrics.pdf](../../outputs/write_detour_w0_29_report/write_stream_metrics.pdf)
- **执行日志:** [outputs/write_execution_logs/followups_status.json](../../outputs/write_execution_logs/followups_status.json)
- **代码 source snapshot manifest：** [outputs/write_detour_w0_29/seed_1061/trial_gate_relaxed/source_snapshot/manifest.json](../../outputs/write_detour_w0_29/seed_1061/trial_gate_relaxed/source_snapshot/manifest.json)
- **输入数据路径：** [data/pretrain_source.npz](../../data/pretrain_source.npz) (SHA-256 493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2)；[outputs/write_decision_foundations/seed_29/foundation.pt](../../outputs/write_decision_foundations/seed_29/foundation.pt) (SHA-256 f3d665f3541d8658a236031726f9f152b15ead6710469c62e5d0b6f1a0dd5074)；[outputs/write_decision_selection/selection.json](../../outputs/write_decision_selection/selection.json) (SHA-256 17f739ee1c6d6278a7231fd8320ca4b2910ed422a972d3a3e74324a4eae58fac)
- **历史执行命令：** 可在各 run provenance.json 中找到，并注明命令证据：[outputs/write_execution_logs/followups_status.json](../../outputs/write_execution_logs/followups_status.json)
- **未来重跑说明：** 历史命令见 [outputs/write_execution_logs/followups_status.json](../../outputs/write_execution_logs/followups_status.json)；复现时应使用同一 detour config、W0-29 foundation 与新输出目录。
- **Git commit：** MISSING。逐文件 source manifest 可引用，但不含能核实的完整执行 commit SHA；归档上下文 commit 不作为运行 SHA。
- **Figures：** [outputs/write_detour_w0_29_report/write_stream_metrics.png](../../outputs/write_detour_w0_29_report/write_stream_metrics.png)；[outputs/write_detour_w0_29_report/write_stream_metrics.pdf](../../outputs/write_detour_w0_29_report/write_stream_metrics.pdf)
- **Run provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)


## 归属与分析版本导航

主生命周期：[H014](../../research/records/H014.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
