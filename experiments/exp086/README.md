# E086 — Detour shifted-generator write stream，W0-17

- **主 H：** H014（../../research/records/H014.md）
- **类别 / 运行状态：** shifted foundation comparison；COMPLETED
- **归档性质：** retrospectively imported on 2026-10-20；不是本次运行，也不是预注册。

## 冻结设计与配置

Known phase 的 detour generator（wall/opening）上，以 W0-17 比较 trial_gate_relaxed、reward_shared 和 independent7_lru。两个部署 seeds 单独汇总。

- **Seeds / arms：** foundation seed 17；deployment seeds 1061、1063。
- **指标与方向：** reward / training+probe interaction 越高越好；training-only 和 final evaluation 单独列。
- **配置：** [configs/write_followup.json](../../configs/write_followup.json)；[configs/write_stream.json](../../configs/write_stream.json)；[outputs/write_detour_w0_17/frozen_config.json](../../outputs/write_detour_w0_17/frozen_config.json)
- **历史批准 compute/time budget：** UNKNOWN（未从现存记录核实）。

## 实际结果与成本

326,124 interactions，audit 通过。Gate reward/interaction 0.001504，reward_shared 0.000948；旧 holdout-using independent 对照约 0.001338，出现小幅描述性优势。后续 full-use budget control E088 为 0.001843，优势未保持。

**实际成本：** 294,912 training + 13,568 probe + 17,644 evaluation = 326,124 interactions。

## 工件与复现 provenance

- **原始 run 输出（保留在旧路径，含其原有 steps / summary / checkpoints）:** [outputs/write_detour_w0_17](../../outputs/write_detour_w0_17)
- **报告 / 机器分析:** [outputs/write_detour_w0_17_report/report.md](../../outputs/write_detour_w0_17_report/report.md)；[outputs/write_detour_w0_17_report/analysis.json](../../outputs/write_detour_w0_17_report/analysis.json)
- **审计 / 失败 / 决策记录:** [outputs/write_current_reports/index.json](../../outputs/write_current_reports/index.json)
- **图表:** [outputs/write_detour_w0_17_report/write_stream_metrics.png](../../outputs/write_detour_w0_17_report/write_stream_metrics.png)；[outputs/write_detour_w0_17_report/write_stream_metrics.pdf](../../outputs/write_detour_w0_17_report/write_stream_metrics.pdf)
- **执行日志:** [outputs/write_execution_logs/followups_status.json](../../outputs/write_execution_logs/followups_status.json)
- **代码 source snapshot manifest：** [outputs/write_detour_w0_17/seed_1061/trial_gate_relaxed/source_snapshot/manifest.json](../../outputs/write_detour_w0_17/seed_1061/trial_gate_relaxed/source_snapshot/manifest.json)
- **输入数据路径：** [data/pretrain_source.npz](../../data/pretrain_source.npz) (SHA-256 493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2)；[outputs/write_decision_foundations/seed_17/foundation.pt](../../outputs/write_decision_foundations/seed_17/foundation.pt) (SHA-256 730e0033786f3aa00a05b21fadf5ec3259917a7c6fbc651dc5aea7dcbb655979)；[outputs/write_decision_selection/selection.json](../../outputs/write_decision_selection/selection.json) (SHA-256 17f739ee1c6d6278a7231fd8320ca4b2910ed422a972d3a3e74324a4eae58fac)
- **历史执行命令：** 可在各 run provenance.json 中找到，并注明命令证据：[outputs/write_execution_logs/followups_status.json](../../outputs/write_execution_logs/followups_status.json)
- **未来重跑说明：** 历史命令见 [outputs/write_execution_logs/followups_status.json](../../outputs/write_execution_logs/followups_status.json)；复现时应使用同一 detour config、W0-17 foundation 与新输出目录。
- **Git commit：** MISSING。逐文件 source manifest 可引用，但不含能核实的完整执行 commit SHA；归档上下文 commit 不作为运行 SHA。
- **Figures：** [outputs/write_detour_w0_17_report/write_stream_metrics.png](../../outputs/write_detour_w0_17_report/write_stream_metrics.png)；[outputs/write_detour_w0_17_report/write_stream_metrics.pdf](../../outputs/write_detour_w0_17_report/write_stream_metrics.pdf)
- **Run provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)


## 归属与分析版本导航

主生命周期：[H014](../../research/records/H014.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
