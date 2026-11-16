# E003：A/B 修正实现后的验证 cohort

- **H：**[H001](../../research/records/H001.md)、[H002](../../research/records/H002.md)。**类型：**独立验证 cohort（非预注册统计确证）。**状态：**COMPLETED。
- **问题/对照：**保留已选超参，修正 history router 稳定尾部匹配；复查在线 A 适应及 B 的历史路径重用。
- **环境、种子、指标：**同一 W0，seeds 41/53/67；A 16,384-step CW，B 32,768-step 固定任务日程；online return/success、重访恢复、source-after-shift、交互与容量。
- **成本：**24 runs；638,976 online training interactions，加 35,221 post-training evaluation interactions。总 online time 按各 run summary；aggregate provenance 有记录。
- **代码/数据/图：**producer [scripts/run_experiments.py](../../scripts/run_experiments.py)；[configs/selected.json](../../configs/selected.json)；输入 [data/pretrain_source.npz](../../data/pretrain_source.npz)、[data/foundation.pt](../../data/foundation.pt)；结果 [outputs/validated/](../../outputs/validated)、[outputs/validated/analysis.json](../../outputs/validated/analysis.json)、[outputs/validated/provenance.json](../../outputs/validated/provenance.json)；图 [outputs/validated/learning_summary.png](../../outputs/validated/learning_summary.png)、`.pdf`。来源 hashes 在 provenance。
- **结果/限制：**global residual 3/3 CW 成功率 100%，同样 3/3 source-after-shift 成功率 0%；history_keep 相较 history_reset 的重访恢复较好。所有 seeds 共用一个 W0；三对 seed 的 bootstrap 只作探索性描述。
- **复现参考：**`scripts/run_experiments.py suite --config configs/selected.json --output outputs/validated`（文档式复现，不是已证实历史命令）。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/validated/](../../outputs/validated)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H001](../../research/records/H001.md)。另关联：[H002](../../research/records/H002.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
