# E017：C1 tree 与 flat grouped 的精确等价诊断

- **H：**[H005](../../research/records/H005.md)、[H006](../../research/records/H006.md)。**类型：**paired diagnostic / equivalence control。**状态：**COMPLETED。
- **问题/对照：**检验固定父+叶 additive policy 重写成 flat grouped additive form 后，logits、梯度、权重、Adam state 与 PPO rollout 是否一致；分别在两个 capacity tiers 配对。
- **设置与成本：**source basis 与 policy 来自 C1 supplement cohort；两档 PPO parity interaction 合计 4,096。附属 grouped-dictionary formal 行属于 E018 的 39-run neural cohort，不在此重复计费。
- **代码/数据/图：**producer [scripts/run_c1_supplement.py](../../scripts/run_c1_supplement.py)；模块 [src/online_rl/c1_modular_controls.py](../../src/online_rl/c1_modular_controls.py)、[src/online_rl/c1_supplement_experiment.py](../../src/online_rl/c1_supplement_experiment.py)；证据 [outputs/c1_supplement/step2_grouping/equivalence/ppo_parity.json](../../outputs/c1_supplement/step2_grouping/equivalence/ppo_parity.json)、[docs/C1_GROUPED_EQUIVALENCE.md](../../docs/C1_GROUPED_EQUIVALENCE.md)。图：NOT_APPLICABLE（该精确 parity 诊断以 machine audit/paired PPO evidence 表示，未单独绘图）。输入 W0/source data 与 step2 checkpoints。
- **结果/限制：**配对 comparison 的最大 logits 差为 0，输出/梯度/状态一致。该结果否定该任务配置中树容器独有的性能优势，不否定共享加法因子本身。
- **复现参考：**历史 parity runner 参数未收录；检查 [docs/C1_GROUPED_EQUIVALENCE.md](../../docs/C1_GROUPED_EQUIVALENCE.md) 和保留 artifacts。不得构造执行命令。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/c1_supplement/step2_grouping/equivalence/](../../outputs/c1_supplement/step2_grouping/equivalence)
- [outputs/c1_neural_integration_checks.log](../../outputs/c1_neural_integration_checks.log)
- [outputs/c1_step2_execution.log](../../outputs/c1_step2_execution.log)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H005](../../research/records/H005.md)。另关联：[H006](../../research/records/H006.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

本 E 未识别到独立派生报告；原始运行摘要、检查输出及缺口见上文与 run provenance。
