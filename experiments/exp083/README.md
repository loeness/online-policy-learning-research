# E083 — 新 seeds 上的 write-decision formal comparison

- **主 H：** H014（../../research/records/H014.md）
- **类别 / 运行状态：** formal exploratory comparison；COMPLETED
- **归档性质：** retrospectively imported on 2026-10-20；不是本次运行，也不是预注册。

## 冻结设计与配置

按 validation 选择，正式 cohort 仅比较 independent_fast、dynamics_shared、reward_shared、trial_gate_relaxed 与 projected_gate；不根据 formal 排名调规则。4096 training steps/bout，seed 是统计单位。

- **Seeds / arms：** 1009、1013、1019；5 conditions。
- **指标与方向：** online reward / training step、训练成功率、final evaluation；越高越好。另记录 gate decisions、实际容量与 CPU/wall。
- **配置：** [configs/write_decision_supported.json](../../configs/write_decision_supported.json)；[outputs/write_decision_formal/frozen_config.json](../../outputs/write_decision_formal/frozen_config.json)；[outputs/write_decision_selection/selection.json](../../outputs/write_decision_selection/selection.json)
- **历史批准 compute/time budget：** UNKNOWN（未从现存记录核实）。

## 实际结果与成本

759,378 interactions 完成。formal 均值：reward_shared 0.019149，trial_gate_relaxed 0.018754，projected_gate 0.015320。Relaxed gate 未胜过简单 reward-aware sharing。旧任务行为保持与 online proxy 不能证明 gate 的单次错误率；formal 文档副本有一处后选补充段落差异，PROVENANCE_NOTE.json 已披露。

**实际成本：** 614,400 training + 23,040 probe + 121,938 evaluation = 759,378 interactions。

## 工件与复现 provenance

- **原始 run 输出（保留在旧路径，含其原有 steps / summary / checkpoints）:** [outputs/write_decision_formal](../../outputs/write_decision_formal)
- **报告 / 机器分析:** [outputs/write_decision_formal_report/report.md](../../outputs/write_decision_formal_report/report.md)；[outputs/write_decision_formal_report/analysis.json](../../outputs/write_decision_formal_report/analysis.json)
- **审计 / 失败 / 决策记录:** [outputs/write_decision_formal/PROVENANCE_NOTE.json](../../outputs/write_decision_formal/PROVENANCE_NOTE.json)；[outputs/write_decision_archive_audit/prior_artifacts.json](../../outputs/write_decision_archive_audit/prior_artifacts.json)；[outputs/write_decision_archive_audit/software_checks.txt](../../outputs/write_decision_archive_audit/software_checks.txt)
- **图表:** [outputs/write_decision_formal_report/online_outcomes.png](../../outputs/write_decision_formal_report/online_outcomes.png)；[outputs/write_decision_formal_report/evaluation_goal_rates.png](../../outputs/write_decision_formal_report/evaluation_goal_rates.png)；[outputs/write_decision_formal_report/costs.png](../../outputs/write_decision_formal_report/costs.png)；[outputs/write_decision_formal_report/capacity.png](../../outputs/write_decision_formal_report/capacity.png)
- **执行日志:** NOT_APPLICABLE（存档中没有独立文件）。
- **代码 source snapshot manifest：** [outputs/write_decision_formal/seed_1019/trial_gate_relaxed/source_snapshot/manifest.json](../../outputs/write_decision_formal/seed_1019/trial_gate_relaxed/source_snapshot/manifest.json)
- **输入数据路径：** [data/pretrain_source.npz](../../data/pretrain_source.npz) (SHA-256 493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2)；[data/foundation.pt](../../data/foundation.pt) (SHA-256 e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222)；[outputs/write_decision_selection/selection.json](../../outputs/write_decision_selection/selection.json) (SHA-256 17f739ee1c6d6278a7231fd8320ca4b2910ed422a972d3a3e74324a4eae58fac)
- **历史执行命令：** 完整 argv 未恢复；不得把下列重跑说明误认为历史命令。
- **未来重跑说明：** 历史完整 argv 未恢复。重跑配方（非历史命令）：按 protocol 的 frozen selection 与 [configs/write_decision_supported.json](../../configs/write_decision_supported.json) 新建 formal output。
- **Git commit：** MISSING。逐文件 source manifest 可引用，但不含能核实的完整执行 commit SHA；归档上下文 commit 不作为运行 SHA。
- **Figures：** [outputs/write_decision_formal_report/online_outcomes.png](../../outputs/write_decision_formal_report/online_outcomes.png)；[outputs/write_decision_formal_report/evaluation_goal_rates.png](../../outputs/write_decision_formal_report/evaluation_goal_rates.png)；[outputs/write_decision_formal_report/costs.png](../../outputs/write_decision_formal_report/costs.png)；[outputs/write_decision_formal_report/capacity.png](../../outputs/write_decision_formal_report/capacity.png)
- **Run provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)


## 归属与分析版本导航

主生命周期：[H014](../../research/records/H014.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
