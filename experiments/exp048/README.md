# E048 — V2 五条件在线闭环正式组

**主 H：** [H009](../../research/records/H009.md) · **角色：** 正式优化方法比较 · **运行状态：** COMPLETED。

**回顾性说明：** 本 README 于 2026-10-20 导入；保存的正式输出和来源快照为执行证据。

## 设计与结果

Seeds 541、557、569，5 条件 ×32,768 步，合计 491,520 环境交互。条件：single-head、修复旧路由 int8、新 EvidenceRouter int8、新路由旧 gate、新路由非阻塞整合。共同 W0 和隐藏 schedule；每个 1,024 步有 768 步 PPO 更新、256 步 holdout。主指标固定交互步的在线 reward/step，辅助成功率、真实编码、完整 checkpoint/package 和时间。

Single-head 均值 −.005001、23.36% 成功率，为神经方法最佳；EvidenceRouter int8 −.012829 /17.54%；nonblocking −.010876 /19.69%，相对同新路由 int8 只有 2/3 seed 的回报增量为正。96 次整合提交仅 4 次采用 148 B SVD，其余为 202 B int8；恢复包没有变小，平均约 236,469 B。nonblocking 不是对完整预算的显著压缩结果。

## 输入、成本与 evidence

- Producer、五臂 run outputs、summary 和 source snapshots：[../../outputs/knowledge_v2_formal](../../outputs/knowledge_v2_formal)。
- Protocol：[../../docs/KNOWLEDGE_V2_VALIDATION.md](../../docs/KNOWLEDGE_V2_VALIDATION.md)；算法说明：[../../docs/KNOWLEDGE_V2_STRUCTURE_AND_ALGORITHM.md](../../docs/KNOWLEDGE_V2_STRUCTURE_AND_ALGORITHM.md)；输入 [../../data/foundation.pt](../../data/foundation.pt)、[../../configs/c1.json](../../configs/c1.json)。
- 代码参考：[../../scripts/run_knowledge_v2.py](../../scripts/run_knowledge_v2.py)、[../../scripts/knowledge_v2_components.py](../../scripts/knowledge_v2_components.py)、[../../scripts/run_knowledge_loop.py](../../scripts/run_knowledge_loop.py)。
- 主报告、derived analysis、figure 与 audit：[../../outputs/knowledge_v2_report/验证与优化结论.md](../../outputs/knowledge_v2_report/验证与优化结论.md)、analysis.json、online_v2.png / .pdf、audit.json。图来自正式运行；绘图命令未恢复。
- 历史实耗 491,520 步，wall 701.540 秒，CPU 689.766 秒。code snapshot 已留存；完整执行 Git SHA 未核实。W0 + checkpoint 的边界不是全进程峰值内存保证。

## 归属与分析版本导航

主生命周期：[H009](../../research/records/H009.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
