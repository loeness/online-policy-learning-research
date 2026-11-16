# E014：C1 选定 temperature-2 六情境 gate

- **H：**[H005](../../research/records/H005.md)。**类型：**independent validation gate。**状态：**COMPLETED。
- **问题/对照：**固定 lr=0.01、temperature 2；新的独立预检中六个任务均达 80% complete-episode gate 后才冻结 formal comparison。
- **设置与成本：**6 contexts × 8,192 training + 1,024 evaluation steps = 55,296 interactions；该 gate 使用 validation paths，未混入 formal seeds 71/83/97。
- **代码/数据/图：**producer `scripts/run_c1.py --mode precheck`；[configs/c1.json](../../configs/c1.json)、[outputs/c1/precheck_selected/](../../outputs/c1/precheck_selected)、[outputs/c1/validation_ledger.json](../../outputs/c1/validation_ledger.json)；输入 [data/foundation.pt](../../data/foundation.pt)、[data/pretrain_source.npz](../../data/pretrain_source.npz)。图：NOT_APPLICABLE（gate 由六项结果 ledger 判定，没有单独 figure）。
- **结果/限制：**六项 100% evaluation success，支持共同表示/学习器达到预定 gate，不是 C1 方法比较或统计确证。validation ledger 和 offline representation diagnostic 不能由当前 precheck 命令单独完整重建。
- **复现参考：**`python scripts/run_c1.py --mode precheck`（复现新 gate，不代表可重生旧 ledger）。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/c1/precheck_selected/](../../outputs/c1/precheck_selected)
- [outputs/c1/validation_ledger.json](../../outputs/c1/validation_ledger.json)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H005](../../research/records/H005.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

本 E 未识别到独立派生报告；原始运行摘要、检查输出及缺口见上文与 run provenance。
