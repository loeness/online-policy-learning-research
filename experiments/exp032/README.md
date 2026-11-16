# E032 — 正式 P0 局部更新与跨情境迁移诊断

**主 H：** [H007](../../research/records/H007.md) · **角色：** 正式探索性机制诊断 · **运行状态：** COMPLETED。

**回顾性说明：** 本 README 于 2026-10-20 导入，不是预注册。原协议 PRE_CORE_VALIDATION_PROTOCOL.md 在正式运行前冻结；协议承诺不等于 H/E 记录当时已经存在。

## 设计与结果

3 个部署 seed（271、283、307）×6 个 source，各独立使用同一 W0、零修正、新 optimizer，在 route 1 上进行 4,096 PPO 步；每个学得修正原样用于 6 个 target，每目标 32 个配对完整 episode、最多 64 步。主要指标为候选相对零修正基线的配对平均 episode return，越高越好；辅助为成功率、长度、迁移矩阵与描述性 cosine 相关。

18/18 个自身更新均改善，seed 均值增益 1.35、1.58、1.13。90 个非对角迁移格中 51 正、39 负；同家族均值 +0.51、+0.86、+0.49，跨家族 −0.09、−0.09、−0.16。梯度 cosine 与迁移的相关在三个 seed 为 .582、.686、.096；正 cosine 仍在 16、20、22 个正 cosine cell 中分别观察到 3、3、10 个负迁移。匹配 baseline 成功率 9.03%，更新后 34.38%，单次范围 12.5%–75%。这些是依赖单元的描述性结果，不是显著性或校准的伤害概率。

## 成本、输入与来源

- 实际交互 309,351 步：训练 73,728、基线评估 34,712、候选评估 200,911；CPU 126.52 秒、墙钟 129.90 秒。
- 输入：[../../data/foundation.pt](../../data/foundation.pt)、[../../configs/c1.json](../../configs/c1.json)；规程：[../../docs/PRE_CORE_VALIDATION_PROTOCOL.md](../../docs/PRE_CORE_VALIDATION_PROTOCOL.md)；方法代码：[../../scripts/run_pre_core_p0.py](../../scripts/run_pre_core_p0.py)。
- 历史 producer：[../../outputs/pre_core_p0](../../outputs/pre_core_p0)；报告、analysis、summary、audit、配对 CSV、迁移矩阵及分支 checkpoint 均保留原位。图为 transfer_diagnostic.png / .pdf；其源数据是该目录 analysis 与 transfer_matrix，历史绘图命令未核实。
- provenance 的 git_head_at_launch 为完整 SHA e4515be40e3d808743123631f062efc3fdfb8425；协议提交为 e8d715ae3d4a85173d0df3c80e90674780dc8a31。见 [../../outputs/pre_core_p0/provenance.json](../../outputs/pre_core_p0/provenance.json)。
- 规程总上限 331,776 步，历史实耗 309,351。无剩余预算记录；本导入不授权新运行。

## 归属与分析版本导航

主生命周期：[H007](../../research/records/H007.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
