# E019：C1 compact basis 与强 context-head 替代 cohort

原始执行日期未单独记录；该 cohort 属于 C1 supplement。导入日期为 2026-10-20。

- **H：**[H006](../../research/records/H006.md)。**类型：**formal alternative representation + validation + source-retention diagnostic。**状态：**COMPLETED。
- **问题/对照：**新 seeds 上比较 frozen tree reference、learned source-basis tensor、frozen independent/hierarchical output mixer、shared full context-head 与 EWC context-head；确认 compact basis 能否保有 online performance，并记录源策略保留和容量。
- **设置：**18 formal / 15 validation rows；formal deployment seeds 181/193/211；validation 919/923。每 formal run 63,488 training + 17,408 evaluation interactions；18-run training 合计 1,142,784，formal eval 合计 313,344。另有 post-run matched W0 source probes 10,752 interactions；validation interaction 总数未从本报告分拆。
- **代码/数据/图：**producer [scripts/run_c1_supplement.py](../../scripts/run_c1_supplement.py)；[src/online_rl/c1_tensor_experiment.py](../../src/online_rl/c1_tensor_experiment.py)、`c1_tensor_controls.py`、`c1_context_head_experiment.py`、`c1_context_head_controls.py`；protocol/amendments [docs/C1_TENSOR_ALTERNATIVES.md](../../docs/C1_TENSOR_ALTERNATIVES.md)、`C1_CONTEXT_HEAD_ALTERNATIVE.md`、`C1_SOURCE_BASIS_DERIVATION.md`；数据 W0/source data；产物 [outputs/c1_supplement/context_head_alternatives/](../../outputs/c1_supplement/context_head_alternatives)、`representation_alternatives/`、`source_retention/`、`alternative_results.md`、`alternative_analysis.json`、`alternative_summary.csv`、`cohort_source_manifest.json`；图 `alternative_comparisons.png`、`model_comparisons.png`、`neural_comparisons.png` 及相应 PDF。
- **结果/限制：**hierarchical output mixer 与 learned tensor 各 3/3 六任务达标；frozen independent mixer 一个 seed 仅 87.0% 六任务均值；context head 平均 96.12%，EWC 80.61%。hierarchical mixer correction/Adam 2,972 B、complete state 约149 KB。后者具更大 state cap。Source-basis 适配与可见动作置换先验匹配，不能推广至源策略行空间以外。
- **复现参考：**冻结的 cohort 配置与 hash manifest 已保留；准确历史 CLI 未恢复。绘图源版本与数据由 [outputs/c1_supplement/](../../outputs/c1_supplement) report/analysis/manifest 指定；没有独立 plot command provenance 时视为缺口。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/c1_supplement/source_retention/](../../outputs/c1_supplement/source_retention)
- [outputs/c1_supplement/alternative_results.md](../../outputs/c1_supplement/alternative_results.md)
- [outputs/c1_supplement/context_head_alternatives/](../../outputs/c1_supplement/context_head_alternatives)
- [outputs/c1_supplement/representation_alternatives/](../../outputs/c1_supplement/representation_alternatives)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H006](../../research/records/H006.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
