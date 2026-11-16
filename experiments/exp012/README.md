# E012：C1 初版 precheck 的校准索引失败

- **H：**[H005](../../research/records/H005.md)。**类型：**failed implementation diagnostic。**状态：**FAILED（运行器异常；不作为策略学习失败）。
- **问题/对照：**验证 C1 稠密残差在六个情境的共同学习 gate；初版 calibration index 转换错误使 prototype precheck 无效。
- **设置：**[configs/c1.json](../../configs/c1.json)、`c1_protocol.md`；初版 [outputs/c1/precheck/](../../outputs/c1/precheck)。`validation_ledger.json` 保留 40 条账目，包括一次异常前已发生而未写日志的真实 1 step；完整总成本没有在本记录单独重算。
- **数据/代码/图：**producer `scripts/run_c1.py --mode precheck`；输入 [data/pretrain_source.npz](../../data/pretrain_source.npz)、[data/foundation.pt](../../data/foundation.pt)；输出及 ledger [outputs/c1/precheck/](../../outputs/c1/precheck)、[outputs/c1/validation_ledger.json](../../outputs/c1/validation_ledger.json)。图：NOT_APPLICABLE（失败实现诊断，保留 machine ledger；不做性能绘图）。
- **结论/限制：**失败归为校准索引 implementation error，不是任务不可学习的证据；该 cohort 未用于正式训练或选参。execution SHA 和精确实际 CLI 未记录。
- **复现参考：**`python scripts/run_c1.py --mode precheck`；该命令当前创建选中 gate，不能复现完整历史 ledger。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/c1/precheck/](../../outputs/c1/precheck)
- [outputs/c1/validation_ledger.json](../../outputs/c1/validation_ledger.json)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H005](../../research/records/H005.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

本 E 未识别到独立派生报告；原始运行摘要、检查输出及缺口见上文与 run provenance。
