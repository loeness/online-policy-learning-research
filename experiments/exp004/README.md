# E004：已保存 A/B cohort 的机制事后分析

- **H：**[H001](../../research/records/H001.md)、[H002](../../research/records/H002.md)。**类型：**read-only diagnostic / analysis。**状态：**COMPLETED。
- **问题/对照：**用已保存 A/B run files 重算和绘制机制诊断；不产生新策略轨迹、不重新选择方法。
- **数据、种子、指标：**输入仅为 [outputs/validated/](../../outputs/validated)；种子 41/53/67；指标由 [outputs/current_analysis/analyze_existing.py](../../outputs/current_analysis/analyze_existing.py) 中的诊断定义。
- **成本：**0 新环境 interactions；CPU/wall time 未记录。
- **代码/图：**producer/analysis [outputs/current_analysis/analyze_existing.py](../../outputs/current_analysis/analyze_existing.py)、`plot_diagnostics.py`；provenance [outputs/current_analysis/provenance.json](../../outputs/current_analysis/provenance.json)；图 [outputs/current_analysis/mechanism_diagnostics.png](../../outputs/current_analysis/mechanism_diagnostics.png)、`.pdf`；JSON [outputs/current_analysis/diagnostics.json](../../outputs/current_analysis/diagnostics.json)。
- **结果/限制：**所有内容为派生诊断，原 cohort 未改变；未发现 execution SHA。
- **复现参考（非已执行命令）：**`python outputs/current_analysis/analyze_existing.py`，随后 `python outputs/current_analysis/plot_diagnostics.py`。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/current_analysis/](../../outputs/current_analysis)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H001](../../research/records/H001.md)。另关联：[H002](../../research/records/H002.md)、[H003](../../research/records/H003.md)、[H004](../../research/records/H004.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
- [analysis003](analyses/analysis003/README.md)
