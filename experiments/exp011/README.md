# E011：九种 residual 压缩替代方法

- **H：**[H004](../../research/records/H004.md)。**类型：**formal exploratory comparison。**状态：**COMPLETED。
- **问题/对照：**对相同 B checkpoints 比较 compact dense、Top-K50、exact action contrast、centered/activation Top-K、columns32、rank-2 SVD、蒸馏 rank-2、BF16；统一冷 Adam、部署与 8,192-step online continuation。
- **设置：**seeds 41/53/67；五个环境；预先冻结 9 arms 与新 RNG offset；报告 adaptive state、完整 bytes、性能、更新、推理/总时间。相关外部灵感 LoRA/Wanda/distillation 见 protocol，非完整复现。
- **成本：**774,144 environment interactions；9 arms 全部完成；总线上 CPU/单样本计时不作为稳定加速证据。
- **代码/数据/图：**producer [scripts/run_compression_alternatives.py](../../scripts/run_compression_alternatives.py)；reporter [scripts/report_compression_alternatives.py](../../scripts/report_compression_alternatives.py)；config/protocol [configs/compression_alternatives.json](../../configs/compression_alternatives.json)、`compression_alternatives_protocol.md`；数据 [outputs/ab_plus/](../../outputs/ab_plus)；结果 [outputs/compression_alternatives/](../../outputs/compression_alternatives)、summary/analysis/spectral diagnostics/provenance；图 `compression_alternatives_pareto.png`、`compression_alternatives_training.png` 及 PDF，均在该目录。
- **事实/限制：**rank-2 与 BF16 在这组轨迹有较低直接损失；centered/activation Top-K 未改善其针对性对照，columns32损失较大；contrast 超 Top-K50 byte target。非 RAM 和非速度普遍结论。
- **复现参考：**`python scripts/run_compression_alternatives.py --config configs/compression_alternatives.json`；报表由 [scripts/report_compression_alternatives.py](../../scripts/report_compression_alternatives.py) 生成。命令用于复现，不是历史执行证据。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/compression_alternatives/](../../outputs/compression_alternatives)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H004](../../research/records/H004.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
