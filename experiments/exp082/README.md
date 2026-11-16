# E082 — 支持感知修复后的 write-decision validation

- **主 H：** H014（../../research/records/H014.md）
- **类别 / 运行状态：** validation / candidate selection；COMPLETED
- **归档性质：** retrospectively imported on 2026-10-20；不是本次运行，也不是预注册。

## 冻结设计与配置

复用 validation seeds 与 8 条件；修复 generation 冲突，并仅在双方可辨识的共同特征支持上比较 reward。revision cohort 不与失败 cohort 合并。validation 选择规则按 online reward / training step 选两个 proposed conditions，同时保留 independent/dynamics/reward baselines。

- **Seeds / arms：** 859、863；8 conditions。
- **指标与方向：** online reward / training step 越高越好；正向训练 goal rate 与 final evaluation 为次要指标；seed 为统计单位。
- **配置：** [configs/write_decision_supported.json](../../configs/write_decision_supported.json)；[outputs/write_decision_supported_validation/frozen_config.json](../../outputs/write_decision_supported_validation/frozen_config.json)
- **历史批准 compute/time budget：** UNKNOWN（未从现存记录核实）。

## 实际结果与成本

485,042 interactions 完成。选择 trial_gate_relaxed 与 projected_gate；验证均值分别为 0.006219 与 0.005731，简单 reward_shared 为 0.0063。gate 指标仍是 surrogate，不是真实收益或安全保证。2 seeds 仅作描述性选择。

**实际成本：** 327,680 training + 24,576 probe + 132,786 evaluation = 485,042 interactions。

## 工件与复现 provenance

- **原始 run 输出（保留在旧路径，含其原有 steps / summary / checkpoints）:** [outputs/write_decision_supported_validation](../../outputs/write_decision_supported_validation)
- **报告 / 机器分析:** [outputs/write_decision_validation_report/report.md](../../outputs/write_decision_validation_report/report.md)；[outputs/write_decision_validation_report/analysis.json](../../outputs/write_decision_validation_report/analysis.json)
- **审计 / 失败 / 决策记录:** [outputs/write_decision_archive_audit/software_checks.txt](../../outputs/write_decision_archive_audit/software_checks.txt)；[outputs/write_decision_selection/selection.json](../../outputs/write_decision_selection/selection.json)
- **图表:** [outputs/write_decision_validation_report/online_outcomes.png](../../outputs/write_decision_validation_report/online_outcomes.png)；[outputs/write_decision_validation_report/evaluation_goal_rates.png](../../outputs/write_decision_validation_report/evaluation_goal_rates.png)；[outputs/write_decision_validation_report/costs.png](../../outputs/write_decision_validation_report/costs.png)；[outputs/write_decision_validation_report/capacity.png](../../outputs/write_decision_validation_report/capacity.png)
- **执行日志:** NOT_APPLICABLE（存档中没有独立文件）。
- **代码 source snapshot manifest：** [outputs/write_decision_supported_validation/seed_863/trial_rollback/source_snapshot/manifest.json](../../outputs/write_decision_supported_validation/seed_863/trial_rollback/source_snapshot/manifest.json)
- **输入数据路径：** [data/pretrain_source.npz](../../data/pretrain_source.npz) (SHA-256 493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2)；[data/foundation.pt](../../data/foundation.pt) (SHA-256 e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222)
- **历史执行命令：** 完整 argv 未恢复；不得把下列重跑说明误认为历史命令。
- **未来重跑说明：** 历史完整 argv 未保存。重跑配方（非历史命令）：按 [docs/WRITE_DECISION_PROTOCOL.md](../../docs/WRITE_DECISION_PROTOCOL.md) 使用 [configs/write_decision_supported.json](../../configs/write_decision_supported.json) 与全新输出目录。
- **Git commit：** MISSING。逐文件 source manifest 可引用，但不含能核实的完整执行 commit SHA；归档上下文 commit 不作为运行 SHA。
- **Figures：** [outputs/write_decision_validation_report/online_outcomes.png](../../outputs/write_decision_validation_report/online_outcomes.png)；[outputs/write_decision_validation_report/evaluation_goal_rates.png](../../outputs/write_decision_validation_report/evaluation_goal_rates.png)；[outputs/write_decision_validation_report/costs.png](../../outputs/write_decision_validation_report/costs.png)；[outputs/write_decision_validation_report/capacity.png](../../outputs/write_decision_validation_report/capacity.png)
- **Run provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)


## 归属与分析版本导航

主生命周期：[H014](../../research/records/H014.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
