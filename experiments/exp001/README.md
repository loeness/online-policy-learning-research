# E001：A/B 配置验证

- **H：**[H001](../../research/records/H001.md)。**类型：**配置验证；回顾记录，非新实验。**状态：**COMPLETED。
- **问题/对照：**在 held-out validation seed 901 选择 A/B PPO learning rate 与 entropy coefficient；main seeds 11/23/37 不参与选参。支持信号为固定 selection score 的最佳 trial；较差 trial 保留。
- **环境、种子、指标：**同一 NavigationEnv/W0；三个配置试验，seed 901。指标为 online return、final CW success、online/evaluation interactions、elapsed time 与状态字节。
- **设置与成本：**configs 中保留原配置；选中 lr=0.01、entropy=0.06、route window=16。三项共 24,576 training + 11,430 evaluation interactions；记录 elapsed time 合计 14.1296064 s。无独立 compute budget 记录。
- **代码/数据/图：**producer [scripts/autoresearch.py](../../scripts/autoresearch.py)；配置 [configs/pilot.json](../../configs/pilot.json)、[configs/selected.json](../../configs/selected.json)；数据 [data/pretrain_source.npz](../../data/pretrain_source.npz)、[data/foundation.pt](../../data/foundation.pt)；选择和逐 trial 记录 [outputs/autoresearch_validation/selection.json](../../outputs/autoresearch_validation/selection.json)、[outputs/autoresearch_validation/results.csv](../../outputs/autoresearch_validation/results.csv)。图：NOT_APPLICABLE（配置筛选表和 CSV 已足够，未发现该 cohort 专属图）。
- **结果/限制：**trial 2 score 1.201188、CW success 1.0，故选定；属于有限 seed 的配置选择。执行 SHA、精确实际命令未记录，不能用复现命令冒充。
- **复现参考（非已执行命令）：**`.venv/Scripts/python.exe scripts/autoresearch.py`；需要原 Windows 环境和相同 W0/config。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/autoresearch_validation/](../../outputs/autoresearch_validation)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H001](../../research/records/H001.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

本 E 未识别到独立派生报告；原始运行摘要、检查输出及缺口见上文与 run provenance。
