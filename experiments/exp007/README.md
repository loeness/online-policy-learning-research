# E007：A/B 日程的解析 history-only baseline

- **H：**[H002](../../research/records/H002.md)、[H003](../../research/records/H003.md)。**类型：**strong baseline control。**状态：**COMPLETED。
- **问题/对照：**固定 W0 actor 加在线估计的全局四分之一旋转与逆映射，对照冻结 W0 和神经 A/B PPO；估计只读近期状态转移，不读取隐藏标签，不做梯度更新。
- **设置：**16-event window、至少 5 个 informative moves；A CW 16,384 steps，B 固定 32,768-step 日程；记录在线成功、回报、状态和 CPU。
- **成本：**provenance 记录 analytic + frozen 总计 294,912 formal interactions；不同 arms/runs 的分项保留在输出。
- **代码/数据/图：**producer [scripts/run_ab_controls.py](../../scripts/run_ab_controls.py)、[scripts/resume_ab_controls.py](../../scripts/resume_ab_controls.py)；[src/online_rl/analytic.py](../../src/online_rl/analytic.py)；config [configs/ab_plus.json](../../configs/ab_plus.json)；输入 [outputs/validated/](../../outputs/validated)；结果 [outputs/ab_plus/analytic/](../../outputs/ab_plus/analytic) 与 summary/report/audit；图 [outputs/ab_plus/supplement_summary.png](../../outputs/ab_plus/supplement_summary.png)、`.pdf`。
- **结果/限制：**解析 history-only 在所测 A/B 全局旋转任务上强于当前 PPO arms，无梯度不等于无在线学习（其统计状态持续更新）。不代表通用 POMDP 方法。
- **复现参考：**producer 脚本见上，确切实际 CLI 未记录。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/ab_plus/](../../outputs/ab_plus)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H002](../../research/records/H002.md)。另关联：[H003](../../research/records/H003.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
