# E002：A/B 初始 pilot

- **H：**[H001](../../research/records/H001.md)、[H002](../../research/records/H002.md)。**类型：**pilot。**状态：**COMPLETED（B 路由部分因实现缺陷而无效）。
- **问题/对照：**在共同 W0 上比较冻结、full PPO、global residual 与 history routing/keep/reset/oracle arms；检查 CW 学习、上下文变化和历史重访。
- **环境、种子、指标：**NavigationEnv；seeds 11/23/37；CW 与固定 B 交互日程；online return、context success、route diagnostics、状态和成本。每 run 16,384（A）或 32,768（B）training steps。共享 W0。
- **成本：**24 runs、638,976 training interactions；评价 steps 保留在 [outputs/pilot/](../../outputs/pilot) 报告和各 run summary 中，pilot 汇总成本未在 provenance 中单独给出。
- **代码/数据/图：**producer [scripts/run_experiments.py](../../scripts/run_experiments.py)，实现 [src/online_rl/experiment.py](../../src/online_rl/experiment.py)、`policy.py`、`routing.py`；[configs/pilot.json](../../configs/pilot.json)；输入 [data/pretrain_source.npz](../../data/pretrain_source.npz)、[data/foundation.pt](../../data/foundation.pt)；产物 [outputs/pilot/](../../outputs/pilot)、报告 [outputs/pilot/report.md](../../outputs/pilot/report.md)；图 [outputs/pilot/learning_summary.png](../../outputs/pilot/learning_summary.png)、`.pdf`。
- **结果/限制：**A global residual 在 3 seeds 的 CW 评估达 100%，但 source-after-shift success 为 0%。B 路由在混合切换窗口时可创建重复 prototype；错误路由实现已保留并明确标记，不能当作目标算法有效性证据。未单独快照初始 pilot 路由源码。
- **复现参考：**`scripts/run_experiments.py suite --config configs/pilot.json --output outputs/pilot`（文档给出的复现流程，不是已证实的历史实际 CLI）。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/pilot/](../../outputs/pilot)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H001](../../research/records/H001.md)。另关联：[H002](../../research/records/H002.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
