# E015：原始固定结构 C1 formal cohort

- **H：**[H005](../../research/records/H005.md)。**类型：**formal mechanism comparison。**状态：**COMPLETED。
- **问题/对照：**从共同 W0 比较 flat leaves、tree parent+local leaves 与 dictionary，在低/高容量下检验固定 family/x-band prior 的共享机制；A0/B0/A1/B1 已见、A2/B2 留出，后续重访及隐藏标签 history continuation。
- **设置：**seeds 71/83/97；18 runs。每 run 固定 63,488 training interactions；总 training 1,142,784，评价交互另计且未在此摘要加总。指标为 cold return、final task success、old-task retention/recovery、history routing、bytes 与成本。配置与预注册式 gate 见 `c1_protocol.md`、[configs/c1.json](../../configs/c1.json)。
- **代码/数据/图：**producer `scripts/run_c1.py --mode formal`；实现 [src/online_rl/c1_experiment.py](../../src/online_rl/c1_experiment.py)、`c1_policy.py`、`c1_routing.py`、`c1_envs.py`；输入 W0/source data；输出 [outputs/c1/formal/](../../outputs/c1/formal)、`summary.json`、`analysis.json`、`validation_ledger.json`、`representation_diagnostic.json`、`budget_preflight.json`、`provenance.json`；图 `cumulative_returns.png`、`cold_reward.png`、`history_routing.png`、`adaptive_bytes_vs_success.png`。
- **结果/限制：**低档 tree 最终均值 100%，高档 99.60%；全部 arms 90/108 task instances 达 80%。主阶段已知 task ID/oracle calibration；history router 用既有带标签校准 bank，history_eval 则 oracle routed。树/任务先验匹配且一个 W0，结果描述性。
- **复现参考：**`python scripts/run_c1.py --mode formal`（历史准确参数由 frozen config 管理，命令本身是复现描述）。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/c1/formal/](../../outputs/c1/formal)
- [outputs/c1/summary.json](../../outputs/c1/summary.json)
- [outputs/c1/analysis.json](../../outputs/c1/analysis.json)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H005](../../research/records/H005.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
