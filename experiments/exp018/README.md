# E018：C1 主补充队列的调参与持续学习基线

- **H：**[H006](../../research/records/H006.md)。**类型：**formal neural comparison + validation。**状态：**COMPLETED。
- **问题/对照：**在独立 validation seed 冻结设置后，比较 flat/tree/dictionary/grouped dictionary 与 full actor、EWC、行为 replay、tree+replay；检查新任务学习、旧任务保留、hidden-router continuation 与容量/成本。
- **设置：**formal seeds 131/149/167、validation seed 911；同一 W0；fixed schedule 与可见结构先验。39 formal neural runs、24 validation runs；主要指标包括六任务完成率、old retention、cold gain、hidden stream success/return、bytes、更新与 CPU。
- **成本：**formal neural training 2,476,032 + evaluation 678,912 interactions；validation training 761,856 + evaluation 208,896。E017 配对等价的 4,096 已单独记账。
- **代码/数据/图：**producer [scripts/run_c1_supplement.py](../../scripts/run_c1_supplement.py)、[scripts/run_c1.py](../../scripts/run_c1.py)；[src/online_rl/c1_supplement_experiment.py](../../src/online_rl/c1_supplement_experiment.py)、`c1_continual_controls.py`、`c1_modular_controls.py`；configs/frozen settings 见 [configs/c1_supplement.json](../../configs/c1_supplement.json)、[outputs/c1_supplement/step3_tuning_and_continual/](../../outputs/c1_supplement/step3_tuning_and_continual)；input [data/foundation.pt](../../data/foundation.pt)、[data/pretrain_source.npz](../../data/pretrain_source.npz)；reports [outputs/c1_supplement/结果解读.md](../../outputs/c1_supplement/结果解读.md)、`analysis.json`、`optimizer_step_audit.json`；图 `model_comparisons.png`、`neural_comparisons.png`。
- **结果/限制：**低/高 tree 99.67%/100%; full 34.80%, EWC 18.27%, replay 26.84%; low tree+replay 94.37% 且立即旧成功率下降。hook 双计数已在 derived report 纠正，原 JSON/checkpoints 不改。Full methods 使用更大预算；CLEAR-inspired replay 不是 CLEAR 复现。
- **复现参考：**protocol/amendments 和 source manifest 供复现；准确 historical CLI 未恢复。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/c1_supplement/](../../outputs/c1_supplement)
- [outputs/c1_step3_execution.log](../../outputs/c1_step3_execution.log)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H006](../../research/records/H006.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
