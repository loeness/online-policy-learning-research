# E161 后的操作前信息、绑定与算法借鉴审查

日期：2026-10-21。类型：结果后只读源码/已存证据审查及方案分析。

用户指定的“消融实验（1）”“结论&贡献”聊天已通过 read_thread 读取；科学事实以本地 E161 冻结报告和执行 source snapshot 为准。本次未拟合预测器、重算 E161 统计、调用环境、连接远端或使用 E161 剩余预算。一个 prefix 的字段/shape 抽查不是对全部 16 个状态的独立审计。

## 结论

停止 E161 原样升级完整流的决定保持不变。已有结果支持小的候选质量增量，未支持值得升级的替换收益。当前最需要分离：候选压缩质量、旧父知识的可继承性、以及对未来需求的预测。

信息审查不是空白：E161 checkpoint 有旧组 bounded evidence，可提出有限函数约束；但 H024 的完整六维统计不在该 checkpoint，也不能读取 observer 的旧 source 矩阵冒充部署记忆。下一步优先冻结 H033 的特征可用性合同；若要做科学排序诊断，必须先另登记 E，现阶段不运行。

## 1. 既有结论的证据边界

[E161 报告](../../../experiments/exp161/analyses/analysis001/report.md) 主比较 behavioral−mean = +.001591365，预设单侧下界 +.000044580；behavioral−keep = +.001300812，下界 −.001131791。两点均小于 .002，联合推进未通过。三种日程下的 keep 取舍对 mean 与 behavioral 都存在，不能归功于拟合特有的操作选择能力。

这不证明真实效应为零或必然小于 .002。旧数据不能继续充当新算法的独立确认；本次审查没有修改 H026/E161 原状态、阈值、原始工件或 analysis001。

## 2. 操作前可用信息合同（源码事实）

以 [E161 冻结的 learner capture](../../../experiments/exp161/runs/run001/producer/source_snapshot/scripts/run_storage_decision.py:180) 为主，而非把其他 runner 的 state 名称混入本实验。

| 信息 | E161 决策前是否实际存在 | 可怎样使用 / 限制 |
| --- | --- | --- |
| W0、当前 local/parent 参数、parent_group、slot_group_tags | 在 core_state/learner 中 | 计算当前有效函数和绑定；不等于仍保存旧 source 私有参数 |
| slot_generation、fitted_steps、cache、recent_groups | 在 learner 中 | 可核对来源代次、当前 local 拟合量、有限近期组频率和 LRU 状态；不能推出未来日程 |
| 旧/新组 bounded evidence | 在 learner.evidence 中，每组最多 256 条 finalized PPO 行 | 含 obs、action、behavior_probs、log-prob、advantage、returns、episode/task/bout；只能约束这些历史观测及已存行为标签 |
| 两个新来源各 128 行及 owner/slot/generation | 已计费 source sidecar | 当前新候选的实际输入；不能从完整 raw log 补出未存观测 |
| H024 的六维 BoundedRetentionStats | 不在 E161 checkpoint | 相关模块存在不等于本实验已经保存 recovery EWMA、parent source-fit/age 等；若未来增添需定义更新时序与字节 |
| old source 完整 effective matrices | observer_prefix 中，明确 excluded from deployment capacity | 禁作免费 selector 或保护 teacher 输入；旧 local 已淘汰后不能重建其任意状态输出 |
| future_regime、未来 bout 序列、branch reward、完整模拟器内部状态、全量日志 | 外部评估资料 | 不作部署特征；只可在另登记的外部研究诊断中构造目标/验证信息泄漏 |
| seed/RNG/路径名 | 为恢复而存在 | 不默认作为估值特征，不能通过查表/重演评估生成器获得未来日程 |

证据结构见 [BoundedGroupEvidence](../../../experiments/exp161/runs/run001/producer/source_snapshot/src/online_rl/storage_decision.py:245)。其 FIFO 截断规则会使较早任务或较早策略的行丢失；动作概率是采样时标签，不自动等于当前 incumbent 或最终旧任务策略。

抽查 [s26201 prefix](../../../experiments/exp161/runs/run001/producer/cases/s26201_dynamics_late/prefix/checkpoint.pt)：两组各 256 行，obs=(256,13)、behavior_probs=(256,4)，checkpoint 为 176565 B；当前驻留任务是新组 A0+/A1+。这些字段验证的是存在性，不验证覆盖充分性、标签质量或预测力。

H024 统计的定义在 [retention_stream_value.py](../../../experiments/exp161/runs/run001/producer/source_snapshot/src/online_rl/retention_stream_value.py:158)；E161 runner 仅导入 apply_retention_action，不能因为同模块还有统计类就报告它已接入。

## 3. 确定的结构限制及尚不能归因的原因

1. **候选目标与保存价值不同。** 当前拟合器只最小化新来源的 KL 与向 mean 的正则，没有旧行为损失、未来效用或淘汰后恢复目标。这是目标范围事实，不能据此断言换价值损失就会提高回报。
2. **唯一 parent 加硬绑定形成实际取舍。** replace 将父绑定给新组；新旧 local 的补偿只保持驻留 effective matrix。LRU 清零 local 后，旧组不再匹配父绑定，不能继承原共享知识。单纯在 new-bound 候选里加旧 KL 无法恢复这个调用路径。[补偿/绑定代码](../../../experiments/exp161/runs/run001/producer/source_snapshot/src/online_rl/storage_decision.py:172)、[新分配继承代码](../../../experiments/exp161/runs/run001/producer/source_snapshot/scripts/run_storage_decision.py:371)。
3. **wildcard 也不是免费解。** 一个九系数、无情境条件的父函数，对重叠状态上互相冲突的目标只能折中；改成 wildcard 还会改变哪些组继承它。参数平均 wildcard 已有负例，不能把相同均值改名为新合并。
4. **表示/数据/codec 是候选解释。** source basis 被限制在 W0 输出对比空间；有限观测可能覆盖不足；replace 的安装/编码路径与 keep 不同。缺少对应干预，不能指定其中任何一项为 E161 未通过的原因。
5. **actor Adam 不是简单的新增混杂解释。** 共同 prefix 已完成 codec/actor optimizer 清理，各分支从这份状态起步；不能把 replace 再清一次就当成已知损失来源。仅让 keep 多过一次 codec 也不保证匹配替换的误差大小，因为被编码矩阵不同。

## 4. H033 应预测哪个量

设 X 是允许的操作前状态，F 是外部未来日程，Delta(X,F)=J(replace;X,F)−J(keep;X,F)。同一 X 被分配不同 F 时，不能要求算法区分其尚未发生的分支。

但这不禁止学习预先固定分布 p* 下的 `tau(X)=E[Delta(X,F)|X,F~p*]`。不同 prefix 的候选质量/旧知识状态可能使 tau 有变化。区别是“预测哪个未来会被评估器安排”与“在固定目标分布下估计操作期望增量”。E161 尚未验证后者有可利用信号。

未来小型诊断若被选中，应先在 H033 下另登记 E：

- 用可核对的少量特征，禁止未来、observer 矩阵、路径/seed 查表；特征构造/标准化只用训练 fold。
- 原 old/new-major 等权为一个明确目标，balanced 仍单列；不通过混入 balanced 改写 E161 主目标。
- 同一 seed 的全部 prefix/future/replica/arm 成组留出，避免 branch 随机拆分；仅 16 个 prefix，最多作结果后可行性筛查。
- 直接评价 held-out 操作增量、选择回报及相对固定 keep/replace/可用简单规则的增量。总体 outcome MSE 只作辅助；oracle 选择未来最佳分支不是可部署基线。
- 模型、fold、超参数、停止条件和分析预算先冻结；不因旧六维 ridge 失败就扩大模型。

[R-learner 原文第 2 节](https://arxiv.org/html/1712.04912v4) 提供分离共同结果与处理增量、cross-fitting 的参考；其 i.i.d./无混杂及速率结论不能直接迁移到持续改变 learner 的流。E161 已付费得到配对分支时，先直接研究配对差，无需为套用论文额外引入 propensity/DR。未来只有所选反馈时，才需单独定义随机 logger、支持与 off-policy 目标；一步 logger 价值不等于完整流价值。

[SPIBB 原文](https://proceedings.mlr.press/v97/laroche19a.html) 的不确定区域保留 baseline 原则可启发 abstain/keep；它不能解决未来未知，也不使旧 lower score 自动变成校准下界。这里保留的是 memory action，原文是在 batch MDP 中限制策略，安全保证不转移。

## 5. 借鉴哪些机制，先改哪一处

优先级 1 是完成上述信息合同；它已发现存在旧观测，同时排除了免费的旧 source 函数与 H024 统计。没有新增选择器效力结论。

若有可利用排序信号，H033 可考虑面向 delta 的小模型与保守 keep；它仍需独立验证，不能因 R-learner/DR 名称宣称解决。

若要减少替换的旧知识代价，登记为 [H036](../../records/H036.md)：保持九系数/codec，比较新组拟合与带 incumbent 旧行为项的拟合，同时把 new-bound 与 wildcard 分开。关键对照是 **同 wildcard、同缓冲和字节下，lambda=0 vs lambda>0**，另保留 keep 与 E161 原替换参考。只加旧 KL、却不让旧组调用候选，不能测试这个问题。

H036 的旧 teacher 限定为 local=0 的 incumbent 初始化 prior，用旧行前四维可见坐标直接计算 W0+旧 parent；不读取该行旧 route id 去调用当前驻留 slot。这样保护目标不会因 slot 已复用而暗中变成新任务策略。它与保存的历史 behavior_probs 及 observer 旧 source 函数是三个不同对象。

[P&C](https://proceedings.mlr.press/v80/schwarz18a/schwarz18a.pdf) 式 (2) 把新策略蒸馏与 online EWC 旧知识保护结合。本案只借用“整合时保护旧知识”的原则；本提案用既有有限观测作函数约束，与原算法不同，蒸馏和保护本身不构成新颖性。

[A-GEM](https://arxiv.org/html/1812.00420v2) 第 4 节的有界参考记忆/梯度约束适合考察连续更新干扰，但不能直接限制离散父重绑定的长期损失。仓库 H014 已有相关 projected gate，H006 已试 EWC/行为回放改编；不能把这些作为尚未试过的万能新方案。

暂缓备选 [H037](../../records/H037.md) 记录同组淘汰前 residual handoff。它会增加写入时点和算子，不与候选/绑定/编码同轮修改。多 parent/模块字典仍归 H031；若要尝试，必须在相同完整字节下重新分配容量，不能白加 parent 后归因为结构收益。

## 6. 下一决策与范围

本次只有源码/已有字段审查和提案登记；没有新 E、训练、排序诊断或远端运行。下一次只选择一个问题：H033 的受限结果后排序筛查，或 H036 的旧行为保护机制。H035 不解释当前已知身份实验的结果。若信号不足或保护无效，保留其边界/负结果，不用复杂拓扑或额外种子挽救 E161。
