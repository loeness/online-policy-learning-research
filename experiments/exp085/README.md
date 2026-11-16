# E085 — Formal accepted/rejected proposal 的 observer counterfactual audit

- **主 H：** H014（../../research/records/H014.md）
- **类别 / 运行状态：** paid observer diagnostic；COMPLETED
- **归档性质：** retrospectively imported on 2026-10-20；不是本次运行，也不是预注册。

## 冻结设计与配置

从每个 formal run 选择首尾非零 accepted/rejected proposals（去重，每 run 最多四个），对候选与 incumbent 在当前 cold parent 和旧 resident routes 上做配对完整 episode observer evaluation。Observer 结果不反馈 learner 或方法选择。

- **Seeds / arms：** formal seeds 1009、1013、1019；被审查 proposal 共 23 个。
- **指标与方向：** 8 对配对完整模拟 episode；accepted harm 阈值、cold benefit 阈值为描述性标签，不是安全率。
- **配置：** [configs/write_decision_supported.json](../../configs/write_decision_supported.json)；[docs/WRITE_DECISION_PROTOCOL.md](../../docs/WRITE_DECISION_PROTOCOL.md)
- **历史批准 compute/time budget：** UNKNOWN（未从现存记录核实）。

## 实际结果与成本

42,714 observer environment steps，0 learner/probe steps。见到一例 accepted 但旧任务变差、一例 rejected 但 candidate 有益。只审查少量被抽样 decisions，余项未定；不能估计 false accept/reject rate。

**实际成本：** 42,714 observer interactions；CPU 20.140625 s，wall 20.5199756 s。

## 工件与复现 provenance

- **原始 run 输出（保留在旧路径，含其原有 steps / summary / checkpoints）:** [outputs/write_decision_counterfactual](../../outputs/write_decision_counterfactual)
- **报告 / 机器分析:** NOT_APPLICABLE（存档中没有独立文件）。
- **审计 / 失败 / 决策记录:** [outputs/write_decision_counterfactual/summary.json](../../outputs/write_decision_counterfactual/summary.json)；[outputs/write_decision_archive_audit/evidence_performance.json](../../outputs/write_decision_archive_audit/evidence_performance.json)
- **图表:** NOT_APPLICABLE（该类 observer/failed diagnostic 无独立图表，详见记录说明）。
- **执行日志:** NOT_APPLICABLE（存档中没有独立文件）。
- **代码 source snapshot manifest：** MISSING；见每次 run provenance.json 中的说明。
- **输入数据路径：** [outputs/write_decision_formal](../../outputs/write_decision_formal)；[outputs/write_decision_formal_report/analysis.json](../../outputs/write_decision_formal_report/analysis.json)
- **历史执行命令：** 完整 argv 未恢复；不得把下列重跑说明误认为历史命令。
- **未来重跑说明：** 审计脚本可对被记录 formal artifacts 做只读重建；历史 argv/source snapshot 未找到。不得把后来结果反馈到原选择。
- **Git commit：** MISSING。逐文件 source manifest 可引用，但不含能核实的完整执行 commit SHA；归档上下文 commit 不作为运行 SHA。
- **Figures：** NOT_APPLICABLE / unavailable
- **Run provenance：** [runs/run001/provenance.json](runs/run001/provenance.json)


## 归属与分析版本导航

主生命周期：[H014](../../research/records/H014.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
