# E010：固定 50% support 的在线修复诊断

- **H：**[H004](../../research/records/H004.md)。**类型：**diagnostic / predeclared repair。**状态：**COMPLETED。
- **问题/对照：**固定 50% support，不重新搜索比例；相对 compact dense，在新的 CW→CCW stream 上额外训练 8,192 steps，比较修复前后五环境表现。
- **设置、输入：**source seeds 41/53/67，[configs/compression_repair.json](../../configs/compression_repair.json)；输入主 screen 的 50% bundles 与 dense compact reference；主指标为修复前后 success/return 和成本。
- **成本：**172,032 formal interactions。具体 wall/CPU 与最低有效修复步数未记录。
- **代码/数据/图：**producer [scripts/run_compression_repair.py](../../scripts/run_compression_repair.py)；数据/输出 [outputs/compression_attempt/repair/](../../outputs/compression_attempt/repair)（before/continuation/after/config/summary/report）；图：NOT_APPLICABLE（该诊断报告未记录独立图）。
- **结果/限制：**报告显示平均 post-repair 指标达到五个百分点容忍范围，需额外经验；不能追认主直接压缩 gate 通过。脚本硬编码 [outputs/compression_attempt](../../outputs/compression_attempt)，是复现限制。
- **复现参考：**`python scripts/run_compression_repair.py`，须提供同结构输入；此为说明性复现方式，不是历史执行记录。
- **Provenance：**[run001](runs/run001/provenance.json)。

## 整合后的原始产物入口

- [outputs/compression_attempt/repair/](../../outputs/compression_attempt/repair)

此 run 是历史 cohort 的导航包装；原始输出保持原位，未据子目录数量推定实际命令调用次数。

## 归属与分析版本导航

主生命周期：[H004](../../research/records/H004.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
