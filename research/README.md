# 研究档案入口

| 阅读目的 | 入口 |
| --- | --- |
| 当前有哪些问题、证据、结论和下一步 | [RESEARCH_LOG.md](RESEARCH_LOG.md) |
| 每个想法的历史轮次与完整生命周期 | [records/](records/) |
| 所有 E 的协议、运行与报告入口 | [实验目录](../experiments/README.md) |
| 文献、软件引用与 H/E 借用关系 | [CITATIONS.md](CITATIONS.md) · [机器映射](citations.json) |
| 完整结构复核结果与已知缺口 | [STRUCTURE_AUDIT.md](STRUCTURE_AUDIT.md) |
| 查一个旧产物属于哪里 | [ARTIFACT_CATALOG.json](ARTIFACT_CATALOG.json) |
| 此次分类覆盖、编号、缺失来源和范围 | [MIGRATION_REPORT.md](MIGRATION_REPORT.md) |
| 长推导、设计与参考笔记入口 | [appendix/](appendix/) |
| 原来按时间编排的实验目录 | [旧实验索引](../docs/EXPERIMENT_INDEX.md) |

此处保存研究记录；实际算法和生产/审计/报告工具仍在 src/ 与 scripts/。
历史 outputs/ 及协议保持原路径，避免破坏源码快照和复现引用。未来 E 可以
直接使用 runs/runNNN/producer/ 接收新输出；本次没有把旧结果复制成“新运行”。

历史 run 来源记录可以对应保存的 cohort / attempt；108 份来源记录不等于已核实的 108 次原始 CLI 调用。新运行仍须一 invocation / attempt 一 run。分析导航区分历史版本与索引；未知精确输入或执行版本保持 MISSING。
