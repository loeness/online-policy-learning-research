# E125 / analysis002 · 历史派生产物版本

回溯分类，未执行新分析。编号是导航序号，不冒称历史运行次数或时间顺序。

- 版本位置：[outputs/retention_prediction_diagnostics](../../../../outputs/retention_prediction_diagnostics)
- 结构化来源：[experiments/exp125/analyses/analysis002/provenance.json](provenance.json)
- 关联运行：[run001](../../runs/run001/provenance.json)
- 主实验：[E125](../../README.md)

具体输入/排除以原审计或哈希清单为准；关联运行不表示把失败、中断与完成运行合池。历史分析命令、执行 SHA、未绑定的绘图来源仍为 MISSING，当前文件哈希不补足这些缺口。

已引用的分析代码（历史执行版本未补造）：
- [scripts/diagnose_retention_prediction.py](../../../../scripts/diagnose_retention_prediction.py)

上游运行：E122/run001, E124/run001；映射依据 SCRIPT_DEFAULT_INPUTS。除明确的原输入哈希外，不把默认参数/事后关联称为历史实际 argv。

## 原始报告、审计与图表

- [outputs/retention_prediction_diagnostics/action_counts.csv](../../../../outputs/retention_prediction_diagnostics/action_counts.csv)
- [outputs/retention_prediction_diagnostics/analysis.json](../../../../outputs/retention_prediction_diagnostics/analysis.json)
- [outputs/retention_prediction_diagnostics/calibration_propensity.csv](../../../../outputs/retention_prediction_diagnostics/calibration_propensity.csv)
- [outputs/retention_prediction_diagnostics/comparison_coverage.csv](../../../../outputs/retention_prediction_diagnostics/comparison_coverage.csv)
- [outputs/retention_prediction_diagnostics/comparison_coverage_by_method.csv](../../../../outputs/retention_prediction_diagnostics/comparison_coverage_by_method.csv)
- [outputs/retention_prediction_diagnostics/input_hashes.json](../../../../outputs/retention_prediction_diagnostics/input_hashes.json)
- [outputs/retention_prediction_diagnostics/prequential_predictions.csv](../../../../outputs/retention_prediction_diagnostics/prequential_predictions.csv)
- [outputs/retention_prediction_diagnostics/relation_aggregates.csv](../../../../outputs/retention_prediction_diagnostics/relation_aggregates.csv)
- [outputs/retention_prediction_diagnostics/report.md](../../../../outputs/retention_prediction_diagnostics/report.md)
- [outputs/retention_prediction_diagnostics/residual_decomposition.csv](../../../../outputs/retention_prediction_diagnostics/residual_decomposition.csv)
- [outputs/retention_prediction_diagnostics/seed_bootstrap_differences.csv](../../../../outputs/retention_prediction_diagnostics/seed_bootstrap_differences.csv)
- [outputs/retention_prediction_diagnostics/stream_metrics.csv](../../../../outputs/retention_prediction_diagnostics/stream_metrics.csv)
- [outputs/retention_prediction_diagnostics/utility_mse_component_comparison.csv](../../../../outputs/retention_prediction_diagnostics/utility_mse_component_comparison.csv)

## 共享来源与执行身份

同一报告/图可由多个 E 引用，不转移生产归属、不增加执行次数或独立样本。此入口的关系为 `COLOCATED_WITH_THIS_E_OUTPUT`。 原生产根路径交叉定位：E125。
原件已记录的输入/代码哈希继续有效；本包装中的 null 表示没有额外重建完整执行 commit/argv，不是断言原件完全没有来源信息。
