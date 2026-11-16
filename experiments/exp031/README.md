# E031 — P0 软件 smoke check

**主 H：** [H007](../../research/records/H007.md) · **角色：** pilot / 软件 smoke · **运行状态：** COMPLETED · **科学结果：** 不纳入 P0 效果结论。

**回顾性说明：** 本 README 于 2026-10-20 导入，不是执行前计划。历史 P0 协议把它明确排除于正式队列。

## 设计与结果

目的为缩小版检查 P0 数据流和配对评估输出。seed 271、source A0、256 个 PPO 训练步；六个 target 各 2 个 episode，最多 64 步，统一 route 1。主机制指标不适用；检查目标是生成训练、基线与候选数据，不声称任何迁移效果。实际共 1,772 步，墙钟 4.744 秒、进程 CPU 1.391 秒。summary 标明 smoke_test=true；该结果不用于方法判断。

## 输入、来源与证据

- 协议和输入：[../../docs/PRE_CORE_VALIDATION_PROTOCOL.md](../../docs/PRE_CORE_VALIDATION_PROTOCOL.md)、[../../configs/c1.json](../../configs/c1.json)、[../../data/foundation.pt](../../data/foundation.pt)。
- 历史 producer：[../../outputs/pre_core_p0_smoke](../../outputs/pre_core_p0_smoke)；汇总、轨迹、迁移表和 provenance 均留在原目录。报告：[../../outputs/pre_core_p0_smoke/README.md](../../outputs/pre_core_p0_smoke/README.md)。无科学图，NOT_APPLICABLE：这是软件 smoke。
- 精确 smoke runner 快照：[../../archive/source_versions/PRE_CORE_P0_smoke_runner.py](../../archive/source_versions/PRE_CORE_P0_smoke_runner.py)。原执行 HEAD 与协议提交均为 e8d715ae3d4a85173d0df3c80e90674780dc8a31；文件 SHA 和环境记于输出 provenance。
- 原始命令未在该 provenance 中核实；任何协议复现命令均只能标为 reproduction，不能称历史命令。
- 不适用正式 seed 批次、主指标、阶段预算或科学结论。历史消费 1,772 环境步；不形成未来 compute 授权。

## 归属与分析版本导航

主生命周期：[H007](../../research/records/H007.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
