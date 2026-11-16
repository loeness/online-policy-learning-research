# E009：主 Top-K residual 压缩筛选

- **H：**[H004](../../research/records/H004.md)。**类型：**pilot / compression screen。**状态：**COMPLETED，未选出合格候选。
- **问题/对照：**比较 original eight-slot、compact dense 与 Top-K 100/75/50/25/12.5%，检验实际状态节省、五种环境下行为损失及继续更新能力。
- **设置：**B+ source checkpoints 41/53/67；五个已知环境；每 context 部署 2,048 steps；固定支持继续学习 8,192 steps；预设状态、success、return 阈值见 [configs/compression.json](../../configs/compression.json)。
- **成本：**184,320 formal interactions；阈值未随结果调整。
- **代码/数据/图：**producer [scripts/run_compression_attempt.py](../../scripts/run_compression_attempt.py)；[src/online_rl/compression_experiment.py](../../src/online_rl/compression_experiment.py)、`compression.py`；config [configs/compression.json](../../configs/compression.json)；输入 [outputs/validated/](../../outputs/validated)、[outputs/ab_plus/deployment_bundles/](../../outputs/ab_plus/deployment_bundles)；结果 [outputs/compression_attempt/](../../outputs/compression_attempt)，含 bundles/validation/test/summary/report/provenance；图 [outputs/compression_attempt/compression_summary.png](../../outputs/compression_attempt/compression_summary.png)、`.pdf`。
- **结果/限制：**没有 fraction 满足所有门槛；100% index overhead 使 adaptive numeric state 比 compact dense 高 16.65%；较高压缩档在部分 context 超过损失标准。不是所有稀疏化方法的结论。
- **复现参考：**`python scripts/run_compression_attempt.py --config configs/compression.json`（复现参考，具体历史命令未记录）。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/compression_attempt/](../../outputs/compression_attempt)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H004](../../research/records/H004.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
