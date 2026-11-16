# E081 — Write-decision 原 validation cohort（失败且不用于选择）

- **主 H：** H014（../../research/records/H014.md）
- **类别 / 运行状态：** validation / failed implementation cohort；FAILED（16 个单元中 9 个完成；不纳入后续 selection）
- **归档性质：** retrospectively imported on 2026-10-20；不是本次运行，也不是预注册。

## 冻结设计与配置

对 8 个条件做 2 个 validation seeds × 2,048 steps/bout 的选择前比较。观察到真实旧槽 anchor 后，部分 gate 条件触发 generation 元数据冲突；同时缺失终止特征被误判成 reward mismatch。

- **Seeds / arms：** 859、863；8 conditions；16 attempted cells。
- **指标与方向：** online reward / training step；gate attempts/accepts；成本分 training/probe/evaluation。selection criterion 是 validation mean online reward / training step。
- **配置：** [configs/write_decision.json](../../configs/write_decision.json)；[outputs/write_decision_validation/frozen_config.json](../../outputs/write_decision_validation/frozen_config.json)
- **历史批准 compute/time budget：** UNKNOWN（未从现存记录核实）。

## 实际结果与成本

FAILED/不完整。9/16 cell 完成，未用于候选选择。修复前原始成本：267,264 training + 22,016 probe + 103,391 evaluation = 392,671 interactions。失败原因和 partial data 保留；该 cohort 与修复版分开。

**实际成本：** 392,671 interactions；CPU/wall cohort aggregate 未在该失败 marker 中给出，故记 UNKNOWN。

## 工件与复现 provenance

- **原始 run 输出（保留在旧路径，含其原有 steps / summary / checkpoints）:** [outputs/write_decision_validation](../../outputs/write_decision_validation)
- **报告 / 机器分析:** NOT_APPLICABLE（存档中没有独立文件）。
- **审计 / 失败 / 决策记录:** [outputs/write_decision_validation/FAILED_COHORT.json](../../outputs/write_decision_validation/FAILED_COHORT.json)；[outputs/write_decision_archive_audit/software_checks.txt](../../outputs/write_decision_archive_audit/software_checks.txt)
- **图表:** NOT_APPLICABLE（该类 observer/failed diagnostic 无独立图表，详见记录说明）。
- **执行日志:** [outputs/write_execution_logs](../../outputs/write_execution_logs)
- **代码 source snapshot manifest：** [outputs/write_decision_validation/seed_863/trial_rollback/source_snapshot/manifest.json](../../outputs/write_decision_validation/seed_863/trial_rollback/source_snapshot/manifest.json)
- **输入数据路径：** [data/pretrain_source.npz](../../data/pretrain_source.npz) (SHA-256 493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2)；[data/foundation.pt](../../data/foundation.pt) (SHA-256 e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222)
- **历史执行命令：** 完整 argv 未恢复；不得把下列重跑说明误认为历史命令。
- **未来重跑说明：** 历史 argv 未找到。重跑配方（非历史命令）：依据 [docs/WRITE_DECISION_PROTOCOL.md](../../docs/WRITE_DECISION_PROTOCOL.md) 与 [configs/write_decision.json](../../configs/write_decision.json) 使用新的空输出目录；此失败 cohort 不应用于选择。
- **Git commit：** 完整 SHA 未核实。原失败 marker 记录短版本 `583a25f7`，照实保留于 run provenance；不补造完整执行 SHA，归档上下文 commit 不作为运行 SHA。
- **Figures：** NOT_APPLICABLE / unavailable
- **Run provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)


## 归属与分析版本导航

主生命周期：[H014](../../research/records/H014.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
