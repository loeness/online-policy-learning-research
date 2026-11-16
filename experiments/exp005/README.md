# E005：B+ 无梯度参数 × 当前历史对照

- **H：**[H003](../../research/records/H003.md)。**类型：**control diagnostic。**状态：**COMPLETED。
- **问题/对照：**从三个 B_history_keep terminal 分别保留参数和历史（P+H+）、清空当前历史但保留参数（P+H0）、清参数并清历史（P0H0）；三组仍保留 learned routing prototypes。
- **设置：**seeds 41/53/67；normal、CW、CCW、CW-slip、CCW-slip；每分支 2,048 步、梯度关闭。主要指标为 return/success 与状态恢复一致性。
- **成本：**92,160 formal environment interactions（provenance 汇总）。总时间没有独立可核对值。
- **代码/数据/图：**producer [scripts/run_ab_controls.py](../../scripts/run_ab_controls.py)、[scripts/resume_ab_controls.py](../../scripts/resume_ab_controls.py)；[configs/ab_plus.json](../../configs/ab_plus.json)；输入 [outputs/validated/](../../outputs/validated) terminal bundles、[data/foundation.pt](../../data/foundation.pt)；产物 [outputs/ab_plus/no_gradient/](../../outputs/ab_plus/no_gradient)、[outputs/ab_plus/summary.json](../../outputs/ab_plus/summary.json)、[outputs/ab_plus/结果解读.md](../../outputs/ab_plus/结果解读.md)、`audit.json`、`source_audit.json`；图 [outputs/ab_plus/supplement_summary.png](../../outputs/ab_plus/supplement_summary.png)、`.pdf`。
- **结果/限制：**P+H0 在所测环境保有策略修正的行为；P0H0 不是删除全部 route knowledge。B+ 从 reconstructed bundles 使用新声明的 RNG，而非原训练精确续训。
- **复现参考（非已执行命令）：**[scripts/run_ab_controls.py](../../scripts/run_ab_controls.py) 和 [scripts/resume_ab_controls.py](../../scripts/resume_ab_controls.py)；确切历史参数未恢复。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/ab_plus/](../../outputs/ab_plus)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H003](../../research/records/H003.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
