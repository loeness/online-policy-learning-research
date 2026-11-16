# E084 — trial rollback 的 post-selection causal control

- **主 H：** H014（../../research/records/H014.md）
- **类别 / 运行状态：** causal control；COMPLETED
- **归档性质：** retrospectively imported on 2026-10-20；不是本次运行，也不是预注册。

## 冻结设计与配置

独立补跑 trial_rollback，保持 trial A+C 学习表达并禁止持久 C writes，作为 gate benefit 是否超过本地在线学习的 control。该单臂 cohort 与之前 formal 是分开的；跨 cohort 同 seed 结果仅描述，不当作同次随机配对。

- **Seeds / arms：** 1009、1013、1019；单一条件 trial_rollback。
- **指标与方向：** online reward / training step 与 evaluation；越高越好。
- **配置：** [outputs/write_decision_causal_control/frozen_config.json](../../outputs/write_decision_causal_control/frozen_config.json)；[docs/WRITE_CONTROL_SUPPLEMENT.md](../../docs/WRITE_CONTROL_SUPPLEMENT.md)
- **历史批准 compute/time budget：** UNKNOWN（未从现存记录核实）。

## 实际结果与成本

152,067 interactions 完成。trial_rollback mean online reward / training step 为 0.0148；三个 seeds 上 trial_gate_relaxed 均高于它，但此跨 cohort 差异不证明 gate 优于 reward_shared（formal reward_shared 为 0.019149）。

**实际成本：** 122,880 training + 4,608 probe + 24,579 evaluation = 152,067 interactions。

## 工件与复现 provenance

- **原始 run 输出（保留在旧路径，含其原有 steps / summary / checkpoints）:** [outputs/write_decision_causal_control](../../outputs/write_decision_causal_control)
- **报告 / 机器分析:** [outputs/write_decision_causal_report/report.md](../../outputs/write_decision_causal_report/report.md)；[outputs/write_decision_causal_report/analysis.json](../../outputs/write_decision_causal_report/analysis.json)
- **审计 / 失败 / 决策记录:** [outputs/write_decision_causal_control/summary.json](../../outputs/write_decision_causal_control/summary.json)
- **图表:** [outputs/write_decision_causal_report/online_outcomes.png](../../outputs/write_decision_causal_report/online_outcomes.png)；[outputs/write_decision_causal_report/evaluation_goal_rates.png](../../outputs/write_decision_causal_report/evaluation_goal_rates.png)；[outputs/write_decision_causal_report/costs.png](../../outputs/write_decision_causal_report/costs.png)；[outputs/write_decision_causal_report/capacity.png](../../outputs/write_decision_causal_report/capacity.png)
- **执行日志:** NOT_APPLICABLE（存档中没有独立文件）。
- **代码 source snapshot manifest：** [outputs/write_decision_causal_control/seed_1019/trial_rollback/source_snapshot/manifest.json](../../outputs/write_decision_causal_control/seed_1019/trial_rollback/source_snapshot/manifest.json)
- **输入数据路径：** [data/pretrain_source.npz](../../data/pretrain_source.npz) (SHA-256 493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2)；[data/foundation.pt](../../data/foundation.pt) (SHA-256 e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222)
- **历史执行命令：** 完整 argv 未恢复；不得把下列重跑说明误认为历史命令。
- **未来重跑说明：** 历史完整 argv 未找到。重跑配方（非历史命令）：使用 [scripts/run_write_control.py](../../scripts/run_write_control.py)、冻结 cohort config 与新输出目录；旧 seed 仅作复现，不是新确认。
- **Git commit：** MISSING。逐文件 source manifest 可引用，但不含能核实的完整执行 commit SHA；归档上下文 commit 不作为运行 SHA。
- **Figures：** [outputs/write_decision_causal_report/online_outcomes.png](../../outputs/write_decision_causal_report/online_outcomes.png)；[outputs/write_decision_causal_report/evaluation_goal_rates.png](../../outputs/write_decision_causal_report/evaluation_goal_rates.png)；[outputs/write_decision_causal_report/costs.png](../../outputs/write_decision_causal_report/costs.png)；[outputs/write_decision_causal_report/capacity.png](../../outputs/write_decision_causal_report/capacity.png)
- **Run provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)


## 归属与分析版本导航

主生命周期：[H014](../../research/records/H014.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
