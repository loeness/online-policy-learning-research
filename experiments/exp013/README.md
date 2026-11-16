# E013：修正路由后的 C1 temperature-1 learnability 检查

- **H：**[H005](../../research/records/H005.md)。**类型：**diagnostic validation / negative outcomes。**状态：**COMPLETED。
- **问题/对照：**校正 calibration index 后，评估 temperature 1 下 lr 0.003/0.02 各 16,384 steps、lr 0.01 延长至 32,768 steps；gate 观察 A0/A1，失败保留，不进入正式排名。
- **成本：**[outputs/c1/validation_ledger.json](../../outputs/c1/validation_ledger.json) 含每个 run 的 train/eval interactions；本记录未将所有失败与早停条目重加总，aggregate total 为 MISSING。
- **代码/数据/图：**producer `scripts/run_c1.py --mode precheck`；config [configs/c1.json](../../configs/c1.json)；输入 W0 和源数据；输出 [outputs/c1/validation_lr/](../../outputs/c1/validation_lr)、[outputs/c1/validation_temperature/](../../outputs/c1/validation_temperature)、`validation_ledger.json`。图：NOT_APPLICABLE（learnability gate 表和逐项 JSON 是此阶段证据）。
- **结果/限制：**多项 temperature-1 设置的 A0 未达目标，包括上述预算内失败；是这些设置下的真实学习负结果，不等价于结构假设失败。精确实际命令/执行 SHA 未记录。
- **复现参考：**`python scripts/run_c1.py --mode precheck`；仅为复现提示。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/c1/validation_lr/](../../outputs/c1/validation_lr)
- [outputs/c1/validation_ledger.json](../../outputs/c1/validation_ledger.json)
- [outputs/c1/validation_temperature/](../../outputs/c1/validation_temperature)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H005](../../research/records/H005.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

本 E 未识别到独立派生报告；原始运行摘要、检查输出及缺口见上文与 run provenance。
