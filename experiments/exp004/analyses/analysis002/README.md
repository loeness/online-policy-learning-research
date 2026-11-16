# E004 / analysis002 · 历史派生产物版本

回溯分类，未执行新分析。编号是导航序号，不冒称历史运行次数或时间顺序。

- 版本位置：[outputs/current_analysis](../../../../outputs/current_analysis)
- 结构化来源：[experiments/exp004/analyses/analysis002/provenance.json](provenance.json)
- 关联运行：[run001](../../runs/run001/provenance.json)
- 主实验：[E004](../../README.md)

具体输入/排除以原审计或哈希清单为准；关联运行不表示把失败、中断与完成运行合池。历史分析命令、执行 SHA、未绑定的绘图来源仍为 MISSING，当前文件哈希不补足这些缺口。

已引用的分析代码（历史执行版本未补造）：
- [outputs/current_analysis/analyze_existing.py](../../../../outputs/current_analysis/analyze_existing.py)
- [outputs/current_analysis/plot_diagnostics.py](../../../../outputs/current_analysis/plot_diagnostics.py)

上游运行：E003/run001, E005/run001, E009/run001, E010/run001；映射依据 NATIVE_INPUT_HASH_MANIFEST。除明确的原输入哈希外，不把默认参数/事后关联称为历史实际 argv。

## 原始报告、审计与图表

- [outputs/current_analysis/diagnostics.json](../../../../outputs/current_analysis/diagnostics.json)
- [outputs/current_analysis/mechanism_diagnostics.pdf](../../../../outputs/current_analysis/mechanism_diagnostics.pdf)
- [outputs/current_analysis/mechanism_diagnostics.png](../../../../outputs/current_analysis/mechanism_diagnostics.png)
- [outputs/current_analysis/provenance.json](../../../../outputs/current_analysis/provenance.json)
- [outputs/current_analysis/深度分析.md](../../../../outputs/current_analysis/深度分析.md)

## 共享来源与执行身份

同一报告/图可由多个 E 引用，不转移生产归属、不增加执行次数或独立样本。此入口的关系为 `COLOCATED_WITH_THIS_E_OUTPUT`。 原生产根路径交叉定位：E004。 原生输入身份记录：[outputs/current_analysis/provenance.json](../../../../outputs/current_analysis/provenance.json)。
原件已记录的输入/代码哈希继续有效；本包装中的 null 表示没有额外重建完整执行 commit/argv，不是断言原件完全没有来源信息。
