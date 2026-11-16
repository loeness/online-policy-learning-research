# E089 — 同交互预算 full-use independent control，W0-29 detour

- **主 H：** H014（../../research/records/H014.md）
- **类别 / 运行状态：** budget-matched control；COMPLETED
- **归档性质：** retrospectively imported on 2026-10-20；不是本次运行，也不是预注册。

## 冻结设计与配置

与 E088 同预算规则，但固定 foundation seed 29；不得将 seed 1061/1063 当作 4 个 independent foundation seeds。

- **Seeds / arms：** foundation seed 29；deployment seeds 1061、1063。
- **指标与方向：** all-stream reward / training+probe interaction 越高越好；resource matching limits 见 config。
- **配置：** [configs/write_detour_budget_supplement.json](../../configs/write_detour_budget_supplement.json)；[outputs/write_detour_budget_w0_29/frozen_config.json](../../outputs/write_detour_budget_w0_29/frozen_config.json)
- **历史批准 compute/time budget：** UNKNOWN（未从现存记录核实）。

## 实际结果与成本

108,675 interactions（102,912 training + 5,763 evaluation；0 probe）。Independent7_lru mean reward/interaction 0.001890，高于 E087 gate 0.001185。Audit 通过。

**实际成本：** 108,675 interactions。

## 工件与复现 provenance

- **原始 run 输出（保留在旧路径，含其原有 steps / summary / checkpoints）:** [outputs/write_detour_budget_w0_29](../../outputs/write_detour_budget_w0_29)
- **报告 / 机器分析:** [outputs/write_detour_budget_w0_29_report/report.md](../../outputs/write_detour_budget_w0_29_report/report.md)；[outputs/write_detour_budget_w0_29_report/analysis.json](../../outputs/write_detour_budget_w0_29_report/analysis.json)
- **审计 / 失败 / 决策记录:** [outputs/write_detour_budget_logs/coordinator_status.json](../../outputs/write_detour_budget_logs/coordinator_status.json)；[outputs/write_current_reports/index.json](../../outputs/write_current_reports/index.json)
- **图表:** [outputs/write_detour_budget_w0_29_report/write_stream_metrics.png](../../outputs/write_detour_budget_w0_29_report/write_stream_metrics.png)；[outputs/write_detour_budget_w0_29_report/write_stream_metrics.pdf](../../outputs/write_detour_budget_w0_29_report/write_stream_metrics.pdf)
- **执行日志:** [outputs/write_detour_budget_logs](../../outputs/write_detour_budget_logs)
- **代码 source snapshot manifest：** [outputs/write_detour_budget_w0_29/seed_1061/independent7_lru/source_snapshot/manifest.json](../../outputs/write_detour_budget_w0_29/seed_1061/independent7_lru/source_snapshot/manifest.json)
- **输入数据路径：** [data/pretrain_source.npz](../../data/pretrain_source.npz) (SHA-256 493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2)；[outputs/write_decision_foundations/seed_29/foundation.pt](../../outputs/write_decision_foundations/seed_29/foundation.pt) (SHA-256 f3d665f3541d8658a236031726f9f152b15ead6710469c62e5d0b6f1a0dd5074)；[outputs/write_decision_selection/selection.json](../../outputs/write_decision_selection/selection.json) (SHA-256 17f739ee1c6d6278a7231fd8320ca4b2910ed422a972d3a3e74324a4eae58fac)
- **历史执行命令：** 完整 argv 未恢复；不得把下列重跑说明误认为历史命令。
- **未来重跑说明：** coordinator_status.json 记录 cohort 成功但不保存精确 argv；冻结参数与脚本入口在 [configs/write_detour_budget_supplement.json](../../configs/write_detour_budget_supplement.json)、[docs/WRITE_BUDGET_BASELINE.md](../../docs/WRITE_BUDGET_BASELINE.md)。
- **Git commit：** MISSING。逐文件 source manifest 可引用，但不含能核实的完整执行 commit SHA；归档上下文 commit 不作为运行 SHA。
- **Figures：** [outputs/write_detour_budget_w0_29_report/write_stream_metrics.png](../../outputs/write_detour_budget_w0_29_report/write_stream_metrics.png)；[outputs/write_detour_budget_w0_29_report/write_stream_metrics.pdf](../../outputs/write_detour_budget_w0_29_report/write_stream_metrics.pdf)
- **Run provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)


## 归属与分析版本导航

主生命周期：[H014](../../research/records/H014.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
