# E008：匹配初始化的 A/B PPO 参考

- **H：**[H001](../../research/records/H001.md)、[H002](../../research/records/H002.md)、[H003](../../research/records/H003.md)。**类型：**matched-reference control。**状态：**COMPLETED。
- **问题/对照：**以相同 W0、新环境/动作 RNG 安排重跑 A global residual 与 B history_keep，并匹配各自冻结 W0 reference，避免直接把旧随机流结果当作配对对照。
- **设置与指标：**A 16,384 steps、B 32,768-step schedule；seed 41/53/67 的新初始化；return/success 与训练成本。
- **成本：**147,456 formal interactions，六个 run；aggregate report 与 provenance 保存。
- **代码/数据/图：**producer [scripts/run_matched_online_references.py](../../scripts/run_matched_online_references.py)；config [configs/ab_plus.json](../../configs/ab_plus.json)；输入 [data/foundation.pt](../../data/foundation.pt)、[data/pretrain_source.npz](../../data/pretrain_source.npz)；结果 [outputs/ab_plus/matched_online/](../../outputs/ab_plus/matched_online)、[outputs/ab_plus/summary.json](../../outputs/ab_plus/summary.json)、report；图 [outputs/ab_plus/supplement_summary.png](../../outputs/ab_plus/supplement_summary.png)、`.pdf`。
- **限制：**不同方法在线轨迹自然不同，共同随机安排不等于逐步相同观测；三个部署 seed 仍共用一份 W0。
- **复现参考：**`python scripts/run_matched_online_references.py`（仅复现参考；历史 CLI 未恢复）。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/ab_plus/](../../outputs/ab_plus)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H001](../../research/records/H001.md)。另关联：[H002](../../research/records/H002.md)、[H003](../../research/records/H003.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
