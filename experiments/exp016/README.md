# E016：C1 无标签 transition 的在线动作模型基线

- **H：**[H006](../../research/records/H006.md)。**类型：**formal baseline + validation + diagnostic。**状态：**COMPLETED。
- **问题/对照：**比较 W0_no_model、window_counts、window_assignment 和 adaptive_assignment；模型只见真实 `(s,a,s')`、x bands、边界与动作双射先验，无 task ID 或切换提示。另做 clear-history 和 frozen-tree diagnostics。
- **设置：**三正式 seeds 131/149/167，各 63,488-step stream；validation 与 diagnostic runs 见 `step1_models` manifest/summary。指标包括真实在线 success/return、统计更新、state bytes、CPU 与清空历史后的表现。
- **成本：**model formal 761,856; model validation 126,976; clear-model diagnostic 55,296; frozen-tree diagnostic 36,864 interactions，均来自 [outputs/c1_supplement/结果解读.md](../../outputs/c1_supplement/结果解读.md) interaction ledger。
- **代码/数据/图：**producer [scripts/run_c1_models.py](../../scripts/run_c1_models.py)（报告中的程序入口；源码 [src/online_rl/c1_model_experiment.py](../../src/online_rl/c1_model_experiment.py)、`c1_strong_models.py`）；config [configs/c1_supplement.json](../../configs/c1_supplement.json)；数据 W0/source data；输出 [outputs/c1_supplement/step1_models/](../../outputs/c1_supplement/step1_models)、`model_diagnostic/`、`tree_diagnostic/`；图 [outputs/c1_supplement/model_comparisons.png](../../outputs/c1_supplement/model_comparisons.png)、`.pdf`。
- **结果/限制：**window/adaptive assignment 100% success、window counts 81.16%、无模型 W0 10.10%。统计状态每 event 更新，所以“零 neural gradients”不是“无在线学习”。方法获知任务相关几何先验，清空历史后模型表现下降。
- **复现参考：**使用 step1_models frozen configs 与历史 runner；准确 CLI 未从已保存源恢复。不得将文档化复现方式称作已执行命令。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/c1_supplement/step1_models/](../../outputs/c1_supplement/step1_models)
- [outputs/c1_model_execution.log](../../outputs/c1_model_execution.log)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H006](../../research/records/H006.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
- [analysis003](analyses/analysis003/README.md)
