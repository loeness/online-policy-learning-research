# E036 — 同秩策略 KL 蒸馏控制

**主 H：** [H008](../../research/records/H008.md) · **角色：** 静态结果后的正式后续控制 · **运行状态：** COMPLETED。

**回顾性说明：** 本 README 于 2026-10-20 导入。协议明确将该同秩蒸馏列为看到低预算 SVD 损失后提出的后续机制，不并入最初冻结比较。

## 设计与结果

3 个教师（s181、s193、s211）×3 个预算（148、280、412 B），用同秩 SVD 因子初始化，以 256 个固定源观测 ×6 个教师路由拟合动作分布 KL；Adam lr=.01、256 次全批更新，无新增 fit 交互、无奖励查询、无测试回报挑迭代。评估 1,728 episode、32,106 环境步；评估后回报和成功率向上，实际因子字节向下为好。

148 B 成功率由 SVD 的 61.11% 升至 93.92%，但三个 seed 未同时通过原始回报保留门槛；280 B 与 412 B 在三教师上均通过。该实验支持“权重误差不足以衡量控制知识损失”的设计线索；策略蒸馏源自已有工作，且源观测覆盖和已知教师路由限制外推，不能主张独有创新。

## 输入、证据与缺口

- 协议：../../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md（补充 C）；producer、summary、episodes 与源码目录：[../../outputs/knowledge_distillation_control](../../outputs/knowledge_distillation_control)。
- 参考教师的评估 ledger 复用原 formal CSV，不重新计价；source dataset hash、reference CSV hash、protocol hash 记录于 summary。源数据的原始路径没有完整恢复，记为 MISSING；不新增复制。
- 实现参考：[../../scripts/run_knowledge_distillation.py](../../scripts/run_knowledge_distillation.py)；历史执行 full Git SHA 未核实。source 文件快照在输出根 source 目录。
- 解释、图及审计：[../../outputs/knowledge_organization_report/结果与研究判断.md](../../outputs/knowledge_organization_report/结果与研究判断.md)、analysis.json、static_capacity.png / .pdf。
- 实耗 32,106 evaluation interactions；fit interactions 0。无独立图生成命令 provenance，使用共享综述图链接并注明由 analysis 派生。

## 归属与分析版本导航

主生命周期：[H008](../../research/records/H008.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
