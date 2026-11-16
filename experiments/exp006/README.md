# E006：B+ 优化器状态隔离

- **H：**[H003](../../research/records/H003.md)。**类型：**control experiment。**状态：**COMPLETED。
- **问题/对照：**在保留 residual、清空 current history 的条件下比较保留 Adam 与重置 Adam，进行 CW→CCW 8,192-step learning continuation；无梯度条件用于区分纯 policy weights 的贡献。
- **设置与指标：**三个 source checkpoints（41/53/67）；训练 return、策略变化、实际 optimizer updates 与交互成本。
- **成本：**49,152 formal interactions；额外评价成本以各保存 run 为准，汇总未单列。
- **代码/数据/图：**[scripts/run_ab_controls.py](../../scripts/run_ab_controls.py)、[scripts/resume_ab_controls.py](../../scripts/resume_ab_controls.py)、[src/online_rl/controls.py](../../src/online_rl/controls.py)；[configs/ab_plus.json](../../configs/ab_plus.json)；输入 [outputs/validated/](../../outputs/validated)；产物 [outputs/ab_plus/optimizer_isolation/](../../outputs/ab_plus/optimizer_isolation)、[outputs/ab_plus/summary.json](../../outputs/ab_plus/summary.json)、[outputs/ab_plus/结果解读.md](../../outputs/ab_plus/结果解读.md)、audit；汇总图 [outputs/ab_plus/supplement_summary.png](../../outputs/ab_plus/supplement_summary.png)、`.pdf`。
- **事实/限制：**这是 Adam state retain/reset 的组合干预，不应解释为已证实的普遍优化器优劣；来源 terminal 不含完整 RNG。
- **复现参考：**producer 脚本见上（复现提示，不是历史实际命令）。执行 SHA 未记录。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/ab_plus/](../../outputs/ab_plus)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H003](../../research/records/H003.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
