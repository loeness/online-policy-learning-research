# E038 — 从零无标签辨识正式组

**主 H：** [H008](../../research/records/H008.md) · **角色：** 正式机制比较 · **运行状态：** COMPLETED。

**回顾性说明：** 本 README 于 2026-10-20 导入，不是预注册。正式方案和判据记录在原协议中。

## 设计与结果

seed 331、347、359 各生成一条六情境随机顺序/长度的真实行为流（切换可在 episode 内）；路由器从空库运行，不接收 context ID、family 或切换时钟。比较 support-aware bank、naive novelty bank 与仅保留近期窗口的转移预测；原型最多 6，窗口 8/32/64。主辨识指标为解析覆盖和含 unresolved 的 ARI，另报告预测、复用、分配/驱逐、延迟与字节。

三条流合计 18,644 环境步；27 次算法回放复用同一实际流，不重复计环境交互。W32 support-aware bank 覆盖 87.5%、ARI .650、纯度 93.2%、每流平均创建/驱逐 27/21、generation 复用 61.7%。朴素库纯度 99.8% 但 ARI 近零、复用为 0；仅近期窗口预测准确率 90.1% 但无持久路由。支持度法优于朴素新颖性，却没有解决重复分裂。主动探索没有独立公平对照。

## 输入、成本与来源

- 协议：[../../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md](../../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md)；producer、summary、metrics、stream、配置和源码快照：[../../outputs/knowledge_identification_formal](../../outputs/knowledge_identification_formal)。
- summary 记录命令 [scripts/run_task_free_identification.py](../../scripts/run_task_free_identification.py) --phase formal；该命令来自历史 summary。输入是该目录生成的 stream_seed_331/347/359.npz，精确路径均在 root 中；底层基础数据版本 hash 见 summary。
- 实际 cost 18,644 步；CPU 94.813 秒、墙钟 97.490 秒（含流生成与离线评估）。
- 结果与 figure 源分析：[../../outputs/knowledge_organization_report/结果与研究判断.md](../../outputs/knowledge_organization_report/结果与研究判断.md)、analysis.json、identification.png / .pdf。图绘制命令未恢复。
- 源代码快照和源 hash 已保存；完整执行 Git SHA 未核实。历史环境版本详见 summary，不推断为当前环境。

## 归属与分析版本导航

主生命周期：[H008](../../research/records/H008.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
