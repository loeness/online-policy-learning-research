# 研究总索引

初始化：2026-10-20；历史分类整合：2026-10-20。

本索引已将现有研究按 H 猜想、E 实验和实际执行入口回溯归类。所有旧数据、
失败记录、协议和源码快照保留原位置，新目录通过链接和 provenance 引用它们。
补录日期不是原始提出日期或预注册日期；未记录的执行 commit/命令保持缺失。

覆盖边界：归类了旧归档清单的全部输出入口、设计/复盘文档和配置入口，以及
共享代码、数据、测试和引用。此次没有重新训练、重算全部统计或独立复核每条
科学结论；原始数据的含义和限制仍以有效协议、review 与审计记录为准。

## 从这里恢复研究

1. 最新机制状态：[H026 行为候选与较晚在线状态](records/H026.md)。
2. 紧邻证据链：H025 反馈窗口/误差来源 → H023 预测 → H024 完整流保存策略；
   预测改善没有建立完整组织算法的策略增量。
3. 动态结构与尚未执行工作见 H027–H035。完整 C2/C3、动态 DAG、事件触发维护
   等设计不能从相关组件实验推成已经完成。
4. [全部 E 实验入口](../experiments/README.md) ·
   [机器可读产物归属](ARTIFACT_CATALOG.json) ·
   [归档覆盖及缺口](MIGRATION_REPORT.md)。

引用入口：[文献与来源总索引](CITATIONS.md)；它区分算法借用、相关工作、软件工具、附件来源及仓库自身的引用方式。

跨阶段判断：[2026-10-20 研究结论、创新候选与后续验证分析](RESEARCH_REVIEW_2026-10-20.md)。该分析区分已支持机制、H026 条件性信号与待证明算法；不改变 H 状态或实验授权，没有新增环境交互。

2026-10-21 下一轮方案：沿 H026 的已执行证据开展一次有停止规则的独立确认，
预留 [E161](../experiments/exp161/README.md)。H035 保留为情境摘要的独立提案，
编号不代表执行先后。E161 已于 2026-10-21 获用户执行授权，已完成独立确认；E161 COMPLETED / INCONCLUSIVE，详见下方最新状态。上限 250 万交互、累计 1 CPU 小时、45 分钟 campaign 墙钟，含失败/重试；预算以 [E161 账本](../experiments/exp161/budget.json) 为准。

[完整结构复核](STRUCTURE_AUDIT.md) 记录此次补齐项和历史复现缺口；归档分类完成不等于每个实验拥有完整可复现来源。

## 猜想与实验

| ID / 生命周期 | 猜想 | 状态 | 实验 | 证据与来源 | 下一决策 |
| --- | --- | --- | --- | --- | --- |
| [H001](records/H001.md) | 预训练后的在线适应 A | TESTED | [E001](../experiments/exp001/README.md), [E002](../experiments/exp002/README.md), [E003](../experiments/exp003/README.md), [E004](../experiments/exp004/README.md), [E008](../experiments/exp008/README.md) | [补充材料](appendix/H001/README.md)；[run / analysis](records/H001.md#2026-10-20-结构复核导航) | 相关后续已归档：H002, H003；先读后续结果，避免重复原阶段已执行建议。 |
| [H002](records/H002.md) | 历史路由与策略路径重用 B | TESTED | [E002](../experiments/exp002/README.md), [E003](../experiments/exp003/README.md), [E004](../experiments/exp004/README.md), [E007](../experiments/exp007/README.md), [E008](../experiments/exp008/README.md) | [补充材料](appendix/H002/README.md)；[run / analysis](records/H002.md#2026-10-20-结构复核导航) | 相关后续已归档：H003, H005, H008, H009；先读后续结果，避免重复原阶段已执行建议。 |
| [H003](records/H003.md) | 持久参数、近期历史与优化器状态的作用 | TESTED | [E004](../experiments/exp004/README.md), [E005](../experiments/exp005/README.md), [E006](../experiments/exp006/README.md), [E007](../experiments/exp007/README.md), [E008](../experiments/exp008/README.md) | [补充材料](appendix/H003/README.md)；[run / analysis](records/H003.md#2026-10-20-结构复核导航) | 相关后续已归档：H004, H005；先读后续结果，避免重复原阶段已执行建议。 |
| [H004](records/H004.md) | 残差表示压缩、修复与替代表示 | INCONCLUSIVE | [E004](../experiments/exp004/README.md), [E009](../experiments/exp009/README.md), [E010](../experiments/exp010/README.md), [E011](../experiments/exp011/README.md) | [补充材料](appendix/H004/README.md)；[run / analysis](records/H004.md#2026-10-20-结构复核导航) | 相关后续已归档：H006, H007, H008；先读后续结果，避免重复原阶段已执行建议。 |
| [H005](records/H005.md) | 固定布局下的 C1 共享策略机制 | TESTED | [E012](../experiments/exp012/README.md), [E013](../experiments/exp013/README.md), [E014](../experiments/exp014/README.md), [E015](../experiments/exp015/README.md), [E017](../experiments/exp017/README.md) | [补充材料](appendix/H005/README.md)；[run / analysis](records/H005.md#2026-10-20-结构复核导航) | 相关后续已归档：H006, H007；先读后续结果，避免重复原阶段已执行建议。 |
| [H006](records/H006.md) | C1 补充强基线、替代紧凑表示与可辨识边界 | INCONCLUSIVE | [E016](../experiments/exp016/README.md), [E017](../experiments/exp017/README.md), [E018](../experiments/exp018/README.md), [E019](../experiments/exp019/README.md) | [补充材料](appendix/H006/README.md)；[run / analysis](records/H006.md#2026-10-20-结构复核导航) | 相关后续已归档：H007, H008；先读后续结果，避免重复原阶段已执行建议。 |
| [H007](records/H007.md) | 预核心阶段的局部学习与迁移信号 | INCONCLUSIVE | [E031](../experiments/exp031/README.md), [E032](../experiments/exp032/README.md) | [补充材料](appendix/H007/README.md)；[run / analysis](records/H007.md#2026-10-20-结构复核导航) | 相关后续已归档：H008, H009；先读后续结果，避免重复原阶段已执行建议。 |
| [H008](records/H008.md) | 有界在线知识组织的静态、辨识与闭环验证 | INCONCLUSIVE | [E033](../experiments/exp033/README.md), [E034](../experiments/exp034/README.md), [E035](../experiments/exp035/README.md), [E036](../experiments/exp036/README.md), [E037](../experiments/exp037/README.md), [E038](../experiments/exp038/README.md), [E039](../experiments/exp039/README.md), [E040](../experiments/exp040/README.md), [E041](../experiments/exp041/README.md), [E042](../experiments/exp042/README.md) | [补充材料](appendix/H008/README.md)；[run / analysis](records/H008.md#2026-10-20-结构复核导航) | 相关后续已归档：H009, H010；先读后续结果，避免重复原阶段已执行建议。 |
| [H009](records/H009.md) | V2 路由、局部更新保留与整合门控诊断 | INCONCLUSIVE | [E043](../experiments/exp043/README.md), [E044](../experiments/exp044/README.md), [E045](../experiments/exp045/README.md), [E046](../experiments/exp046/README.md), [E047](../experiments/exp047/README.md), [E048](../experiments/exp048/README.md), [E049](../experiments/exp049/README.md), [E050](../experiments/exp050/README.md) | [补充材料](appendix/H009/README.md)；[run / analysis](records/H009.md#2026-10-20-结构复核导航) | 相关后续已归档：H010, H011, H012, H013；先读后续结果，避免重复原阶段已执行建议。 |
| [H010](records/H010.md) | 真实在线交互是否通过共享参数传递知识 | TESTED | [E061](../experiments/exp061/README.md), [E062](../experiments/exp062/README.md) | [补充材料](appendix/H010/README.md)；[run / analysis](records/H010.md#2026-10-20-结构复核导航) | 相关后续已归档：H011, H012, H013；先读后续结果，避免重复原阶段已执行建议。 |
| [H011](records/H011.md) | 用在线探测证据进行共享家族分组 | TESTED | [E063](../experiments/exp063/README.md), [E064](../experiments/exp064/README.md) | [补充材料](appendix/H011/README.md)；[run / analysis](records/H011.md#2026-10-20-结构复核导航) | 相关后续已归档：H013；先读后续结果，避免重复原阶段已执行建议。 |
| [H012](records/H012.md) | 局部容量耗尽时共享参数能否保留可复用经验 | TESTED | [E065](../experiments/exp065/README.md), [E066](../experiments/exp066/README.md) | [补充材料](appendix/H012/README.md)；[run / analysis](records/H012.md#2026-10-20-结构复核导航) | 相关后续已归档：H014, H017, H019；先读后续结果，避免重复原阶段已执行建议。 |
| [H013](records/H013.md) | 分组支持、局部歧义与奖励冲突 | INCONCLUSIVE | [E067](../experiments/exp067/README.md), [E068](../experiments/exp068/README.md), [E069](../experiments/exp069/README.md) | [补充材料](appendix/H013/README.md)；[run / analysis](records/H013.md#2026-10-20-结构复核导航) | 相关后续已归档：H014, H025, H026；先读后续结果，避免重复原阶段已执行建议。 |
| [H014](records/H014.md) | 在线写入决策与有限父模块回收 | INCONCLUSIVE | [E081](../experiments/exp081/README.md), [E082](../experiments/exp082/README.md), [E083](../experiments/exp083/README.md), [E084](../experiments/exp084/README.md), [E085](../experiments/exp085/README.md), [E086](../experiments/exp086/README.md), [E087](../experiments/exp087/README.md), [E088](../experiments/exp088/README.md), [E089](../experiments/exp089/README.md), [E090](../experiments/exp090/README.md), [E091](../experiments/exp091/README.md) | [补充材料](appendix/H014/README.md)；[run / analysis](records/H014.md#2026-10-20-结构复核导航) | 相关后续已归档：H015, H016, H017, H018；先读后续结果，避免重复原阶段已执行建议。 |
| [H015](records/H015.md) | 已知与未知边界下的在线写入流 | INCONCLUSIVE | [E092](../experiments/exp092/README.md), [E093](../experiments/exp093/README.md), [E094](../experiments/exp094/README.md), [E095](../experiments/exp095/README.md), [E096](../experiments/exp096/README.md), [E097](../experiments/exp097/README.md), [E098](../experiments/exp098/README.md) | [补充材料](appendix/H015/README.md)；[run / analysis](records/H015.md#2026-10-20-结构复核导航) | 相关后续已归档：H017, H019, H024；先读后续结果，避免重复原阶段已执行建议。 |
| [H016](records/H016.md) | 删除在线局部参数后，公共修正是否保留知识 | INCONCLUSIVE | [E099](../experiments/exp099/README.md), [E100](../experiments/exp100/README.md) | [补充材料](appendix/H016/README.md)；[run / analysis](records/H016.md#2026-10-20-结构复核导航) | 相关后续已归档：H017, H019；先读后续结果，避免重复原阶段已执行建议。 |
| [H017](records/H017.md) | 自然淘汰后的在线共享写入与恢复 | INCONCLUSIVE | [E101](../experiments/exp101/README.md), [E102](../experiments/exp102/README.md), [E103](../experiments/exp103/README.md) | [补充材料](appendix/H017/README.md)；[run / analysis](records/H017.md#2026-10-20-结构复核导航) | 相关后续已归档：H018, H019；先读后续结果，避免重复原阶段已执行建议。 |
| [H018](records/H018.md) | 历史收益分数能否预测自然回访收益 | INCONCLUSIVE | [E104](../experiments/exp104/README.md), [E105](../experiments/exp105/README.md) | [补充材料](appendix/H018/README.md)；[run / analysis](records/H018.md#2026-10-20-结构复核导航) | 相关后续已归档：H019, H022, H025；先读后续结果，避免重复原阶段已执行建议。 |
| [H019](records/H019.md) | 稀缺父位置的保存操作与未来复现分布 | INCONCLUSIVE | [E111](../experiments/exp111/README.md), [E112](../experiments/exp112/README.md), [E113](../experiments/exp113/README.md) | [补充材料](appendix/H019/README.md)；[run / analysis](records/H019.md#2026-10-20-结构复核导航) | 相关后续已归档：H020, H021, H022, H023, H024；先读后续结果，避免重复原阶段已执行建议。 |
| [H020](records/H020.md) | 候选来源剂量与保存内容效应 | INCONCLUSIVE | [E114](../experiments/exp114/README.md), [E115](../experiments/exp115/README.md) | [补充材料](appendix/H020/README.md)；[run / analysis](records/H020.md#2026-10-20-结构复核导航) | 相关后续已归档：H025, H026；先读后续结果，避免重复原阶段已执行建议。 |
| [H021](records/H021.md) | 持久保存与未来继续学习的交互 | INCONCLUSIVE | [E116](../experiments/exp116/README.md), [E117](../experiments/exp117/README.md) | [补充材料](appendix/H021/README.md)；[run / analysis](records/H021.md#2026-10-20-结构复核导航) | 相关后续已归档：H025, H026；先读后续结果，避免重复原阶段已执行建议。 |
| [H022](records/H022.md) | 固定 lower 保存规则的独立确认 | INCONCLUSIVE | [E119](../experiments/exp119/README.md), [E120](../experiments/exp120/README.md) | [补充材料](appendix/H022/README.md)；[run / analysis](records/H022.md#2026-10-20-结构复核导航) | 相关后续已归档：H023, H024, H025；先读后续结果，避免重复原阶段已执行建议。 |
| [H023](records/H023.md) | 所选反馈中的历史特征预测价值 | TESTED | [E121](../experiments/exp121/README.md), [E122](../experiments/exp122/README.md) | [补充材料](appendix/H023/README.md)；[run / analysis](records/H023.md#2026-10-20-结构复核导航) | 相关后续已归档：H024, H025；先读后续结果，避免重复原阶段已执行建议。 |
| [H024](records/H024.md) | 重复在线写入的学习型保存器增量 | INCONCLUSIVE | [E123](../experiments/exp123/README.md), [E124](../experiments/exp124/README.md) | [补充材料](appendix/H024/README.md)；[run / analysis](records/H024.md#2026-10-20-结构复核导航) | 相关后续已归档：H025, H026；先读后续结果，避免重复原阶段已执行建议。 |
| [H025](records/H025.md) | 反馈来源分解与保存操作的观察窗口 | INCONCLUSIVE | [E125](../experiments/exp125/README.md), [E126](../experiments/exp126/README.md), [E127](../experiments/exp127/README.md), [E128](../experiments/exp128/README.md) | [补充材料](appendix/H025/README.md)；[run / analysis](records/H025.md#2026-10-20-结构复核导航) | 相关后续已归档：H026；先读后续结果，避免重复原阶段已执行建议。 |
| [H026](records/H026.md) | 有限观测行为候选与较晚在线状态 | INCONCLUSIVE | [E129](../experiments/exp129/README.md), [E130](../experiments/exp130/README.md), [E131](../experiments/exp131/README.md), [E132](../experiments/exp132/README.md), [E161](../experiments/exp161/README.md) | [补充材料](appendix/H026/README.md)；[run / analysis](records/H026.md#2026-10-20-结构复核导航) | E161 已完成：behavioral−mean +.001591，behavioral−keep +.001301；后者下界<0，且两点估计<.002。联合确认未过，停止本轮，不补 seed 或升级完整流。 |
| [H027](records/H027.md) | 动态知识拓扑与 C2 结构操作 | PROPOSED | —（未执行完整提案） | [补充材料](appendix/H027/README.md)；[run / analysis](records/H027.md#2026-10-20-结构复核导航) | 未实施部分先评估必要性；不自动启动 P2 或超预算验证。 |
| [H028](records/H028.md) | 真正满容量下的保留、替换与 C3 | PROPOSED | —（未执行完整提案） | [补充材料](appendix/H028/README.md)；[run / analysis](records/H028.md#2026-10-20-结构复核导航) | 未实施部分先评估必要性；不自动启动 P2 或超预算验证。 |
| [H029](records/H029.md) | 歧义下的有预算主动辨别与可靠回退 | PROPOSED | —（未执行完整提案） | [补充材料](appendix/H029/README.md)；[run / analysis](records/H029.md#2026-10-20-结构复核导航) | 未实施部分先评估必要性；不自动启动 P2 或超预算验证。 |
| [H030](records/H030.md) | 受影响状态的事件触发局部维护 | PROPOSED | [E118](../experiments/exp118/README.md) | [补充材料](appendix/H030/README.md)；[run / analysis](records/H030.md#2026-10-20-结构复核导航) | 未实施部分先评估必要性；不自动启动 P2 或超预算验证。 |
| [H031](records/H031.md) | 交叉因素下的树、模块字典与 DAG | PROPOSED | —（未执行完整提案） | [补充材料](appendix/H031/README.md)；[run / analysis](records/H031.md#2026-10-20-结构复核导航) | 未实施部分先评估必要性；不自动启动 P2 或超预算验证。 |
| [H032](records/H032.md) | 收益、实际字节与成本的统一结构选择 | PROPOSED | —（未执行完整提案） | [补充材料](appendix/H032/README.md)；[run / analysis](records/H032.md#2026-10-20-结构复核导航) | 未实施部分先评估必要性；不自动启动 P2 或超预算验证。 |
| [H033](records/H033.md) | 共享结果基线、动作对比与双重稳健反馈 | INCONCLUSIVE（最小诊断 NO_ADVANCEMENT；完整提案未验证） | [E162](../experiments/exp162/README.md)（已完成，本地离线诊断） | [统计报告](../experiments/exp162/analyses/analysis001/report.md)；[计量披露](../experiments/exp162/analyses/analysis002/README.md)；[方法 v1](appendix/H033/METHOD_V1.md) | G_min=+.000720<.002，折稳定性1/4<3/4；停止固定三特征 ridge，不补种子/调参/完整流；不外推为所有过去信息无用。 |
| [H034](records/H034.md) | 更新频率、结构存储与容量误差的理论问题 | PROPOSED | —（未执行完整提案） | [补充材料](appendix/H034/README.md)；[run / analysis](records/H034.md#2026-10-20-结构复核导航) | 未实施部分先评估必要性；不自动启动 P2 或超预算验证。 |
| [H035](records/H035.md) | 有界递归情境摘要与因果统计顺序 | PROPOSED | —（未执行完整提案） | [补充材料](appendix/H035/README.md)；[run / analysis](records/H035.md#2026-10-20-结构复核导航) | 未实施部分先评估必要性；不自动启动 P2 或超预算验证。 |
| [H036](records/H036.md) | 固定容量下带旧行为约束的共享候选整合 | INCONCLUSIVE（v1 NO_ADVANCEMENT；新E165实现中） | [E163](../experiments/exp163/README.md)、[E164](../experiments/exp164/README.md)均COMPLETE_VALID；[E165](../experiments/exp165/README.md) PLANNED | [E164报告](../experiments/exp164/analyses/a001/report.md)、[主控分析](appendix/H036/COORDINATOR_E164_SYNTHESIS_20261023.md)、[E165合同](appendix/H036/COORDINATOR_E165_CONTRACT_20261024.md) | 用户2026-10-24授权新轮并要求额度恢复后继续；E165全16有限C→U缩步，S1−S0/S1−D，0fit。先实现冻结与独立静态审查，全部checks/诊断/审计远端；规划120CPU/240累计wall/128MiB，尚无runtime许可。E163/E164停止决定与退役额度不变。 |
| [H037](records/H037.md) | 同组局部槽淘汰前的残差交接 | PROPOSED（暂缓） | —（未实施、未分配 E） | E161 结果后源码审查；关联 H014/H027/H036 | 记录备选；父绑定不匹配时不适用，当前不新增淘汰时写入机制。 |

## 编号与事实来源

此次并行整理预留 H001–H035、E001–E160；表中仅列真实创建的历史记录，未用
编号保持跳过。后续编号从 H036/E161 及以上核对分配。run 编号只在所属 E 内
递增，analysis 编号也只在所属 E 内递增。

2026-10-21 已预留 E161 给 H026 的确认方案；后续从 H036/E162 及以上再次核对分配。

2026-10-21 E161 结果后算法审查已预留 H036；后续从 H037/E162 及以上再次核对分配。

2026-10-21 同次审查记录暂缓备选 H037；后续从 H038/E162 及以上再次核对分配。

H 是完整问题生命周期；本轮计划是 H 中的日期小节；pilot、正式比较和不同
科学诊断使用各自 E。失败重试保留为 E 下不同 run，报告修正保留为不同 analysis。

历史交互/成本可能有别名、公共前缀、重复执行和未知中断开销，不能把所有 E
的摘要不加区分地相加为全项目成本。本次整理没有新增环境交互或 GPU 训练，
也没有扩大实验预算。原始提出者未核实时明确记为未知，不从聊天印象补造。

2026-10-21 执行状态：H026 → [E161](../experiments/exp161/README.md) 已在服务器 COMPLETED / INCONCLUSIVE（16 新 seed、288 分支、2244608 实际交互）；[预算账本](../experiments/exp161/budget.json) 记录句柄和上限。H026 保持 INCONCLUSIVE；独立审计 PASS，原始工件与失败审计完整保留。

2026-10-21 最终决策：E161 两项联合确认与 .002 点估计推进门槛均未过；[实际结果及边界](../experiments/exp161/README.md#2026-10-21-实际结果与停止决定)。按固定规模停止，无新增后续实验；H035 继续 PROPOSED。

2026-10-21 结果后解释：[H026 解释与问题筛选](records/H026.md#2026-10-21-结果解释与下一问题筛选结果后分析) 区分候选质量、替换代价和不可见未来；[H033 必要条件提议](records/H033.md#2026-10-21-e161-后的必要条件问题结果后提议) 先审查过去信息是否足以支持操作排序。仅登记问题细化，未重算、实现或运行新验证，不改变 E161 停止决定与预算边界。

2026-10-21 算法实现侧只读审查：[操作前信息、绑定与借鉴](appendix/H033/20261021_INFORMATION_AND_STRUCTURE_REVIEW.md) 确认旧组 bounded evidence 存在、H024 完整 stats 不在 E161；区分期望增量与预知外部未来。新登记 H036 旧行为保护候选、H037 暂缓交接备选；无新 E、训练或远端操作，E161 原停止决定保持不变。

2026-10-21 主控接受 H033 最小排序诊断 v1 的冻结交付并预留 [E162](../experiments/exp162/README.md)。用户明确要求交由消融实验（1）审查执行；仅本地离线复用，0 新环境交互，CPU 300 秒/墙钟 600 秒/单线程/256 MiB 新输出上限，见 [E162 账本](../experiments/exp162/budget.json)。推进以留出选择价值及折稳定性筛查，非 MSE/确认性显著性；具体标准先于结果固定。H036/H037/H035 不启动，E161 不续跑；后续编号从 H038/E163 及以上再次核对。

2026-10-21 E162 完成并由主控集成：[H033 结果与停止决定](records/H033.md#2026-10-21-e162-实际结果与主控集成)。相对 keep +.002033，但相对最强简单频率规则仅 +.000720，四折仅1/4通过；两项预设条件均未过，NO_ADVANCEMENT，H033 更新为 INCONCLUSIVE。新增交互0，科学流水线墙钟10.898秒、campaign454.681秒；完整 CPU 因启动器漏计为 UNKNOWN，保留原证据并退役全部300秒 reservation，不释放续跑预算。固定最小模型停止；无 H035/H036/完整流/新种子实验。后续编号仍从 H038/E163 及以上再次核对。

2026-10-22 主控接受 H036 同一父模块的旧/新双目标整合方向，登记 [E163 Stage0](../experiments/exp163/README.md) 与[受限实现合同](appendix/H036/COORDINATOR_CONTRACT_20261022.md)。仅预留本地0交互、CPU120秒/墙钟180秒/256 MiB规划边界；execution_authorized=false，待算法实现冻结交付与独立审查。至少12/16编码后旧KL降低25%且保留新KL收益50%为机制筛查，不作为奖励确认。H036仍PROPOSED；无新拟合/交互/远端，无StageA额度；E161/E162停止不变。后续编号从 H038/E164 及以上再次核对。

2026-10-22 后续用户指示：实现任务已获授权并发送；等待算法冻结后交消融实验（1）通过SSH执行E163，再由主控分析。执行位置改为robot的既有 `/data/robot/p001/e163`，必要代码/配置/固定输入仅按白名单；原数值预算和冻结/审查前置条件不变。E163当前NOT_RUN，StageA未授权；详见E163当前协议/账本与合同追加位置说明。

2026-10-22 H036 v1冻结接受：21项源码/配置、48项输入及参考身份核验通过；冻结副本23项合成检查通过，source identity 0685e20c095b4c49c12d04041e3857f19b584c062dbe3eea456af1561fa661ea。H036改IMPLEMENTED，E163待消融实验独立审查/SSH准入，尚未拟合、execution_authorized=false；软件接受不等于机制有效。

2026-10-22 E163 READY验收：科学与runner独立复审PASS，远端49文件/33依赖及27项传输白名单核验完成。主控授权E163_RUN001_20261022_APPROVED，绑定run_spec 6e7604856c28ce33218f732eb893c1cc56827cab7eb4e51abee21b994f4491e8；动态容量刷新及同campaign计量预检先于拟合，原数值限额不变。结果待消融返回主控集成，无其他阶段授权。

2026-10-22 E163 run001技术停止：首个CPU生命周期预检发现RUSAGE_CHILDREN/wait4不符，exec复用SSH shell PID造成起点/历史计数污染；科学worker未启、0fit/0交互，H036仍IMPLEMENTED而非机制阴性。原证据保留；验证侧仅准备repair002的最小计量/启动修复，第二次执行未授权，资源读数按最终账本披露，不自动释放或重置预算。

2026-10-22 E163失败结算与修复验收：独立审计ACCOUNTING_INVALID、完整CPU/墙钟UNKNOWN，原单次reservation退役。repair002已冻结、独立静态复审PASS且获主控验收；科学方法/输入/门槛不变，尚无修复传输或执行。追加单次CPU120秒/墙钟180秒/0交互预算待用户答复，不能把原额度或修复上限当作新授权。H036机制仍未评价，无StageA。

2026-10-22 用户明确追加一次重试预算；主控登记stage0_campaign002并发布[E163_RUN002_20261022_APPROVED](../experiments/exp163/preparation/coordinator_start002.json)，独立120CPU秒/180墙钟秒/0交互，冻结repair002在原SSH范围执行。保留run001全部证据、UNKNOWN成本及退役额度；动态远端准入及同campaign计量/停止预检先于固定32fit，无自动第三次或后续阶段授权。当前无新增科学结论。

2026-10-23 E163结果交回并由主控接受：[run002报告](../experiments/exp163/analyses/a002/report.md)与1876项独立审计零失败确认COMPLETE_VALID，32fit、0交互，8/16<预设12/16，NO_ADVANCEMENT。H036更新INCONCLUSIVE，固定v1停止；7个通过项D为非C0，另1个合法C0通过不归因于学到保护。失败分为3个C无正新收益、2个旧KL降幅不足和余下3个新收益保留不足，不能由均值推成普遍收益或函数类不兼容。run002观测CPU8.625535秒/墙钟10.011705487秒，含收尾预留上界13.625535/20.011705487；累计完整成本仍UNKNOWN，两次额度关闭且无复用。先定位已有拟合/接纳与编码诊断中的失败来源，再决定受限设计；没有新fit/StageA授权，后续编号仍从H038/E164核对。

2026-10-23 用户要求共享下一目标并按设计→主控确认→实现冻结→独立验证推进。已发布[双对话交接合同](appendix/H036/COORDINATOR_NEXT_CONTRACT_20261023.md)，当前属于H036搜索/接纳/留出问题的设计细化。算法实现拥有方法/结构与实现交付，消融实验并行审查，主控接受后才登记新E及数值预算。无新E运行或E163额度复用。

2026-10-23 已核对并预留E164，主控接受同轨迹接纳诊断的最小实现范围；算法实现交付代码/合成检查/冻结，消融实验独立审查。全16配对、11复用、5恢复≤15新评分；建议60CPU/120wall/64MiB/0交互仅规划，未开reservation或运行。当前仅H036问题细化，不新增生产算法H；后续从H038/E165核对分配。

2026-10-23 用户要求从额度中断处恢复合作验证；主控接受最终包、登记单次60CPU/120wall/64MiB并完成限定SSH准入。E164现COMPLETE_VALID / EXPLORATORY_ONLY：[最终报告](../experiments/exp164/analyses/a001/report.md)、[独立审计](../experiments/exp164/analyses/a003/audit.json)、[主控分析](appendix/H036/COORDINATOR_E164_SYNTHESIS_20261023.md)。5fit/10score/16逻辑配对/0交互；UG−V全零，UR−UG旧留出KL5改善，新留出收益2改善/3退化/11不变，不能直接释放gate升级。观测CPU7.929755秒/墙钟9.350686747秒，含footer预留上界12.929755/19.350686747，精确完整/未测运维成本UNKNOWN。原核验器映射错误以分版纯离线副本修复，原失败/原始工件保留，无campaign重跑。额度关闭、余额退役、无作业/SSH；H036仍INCONCLUSIVE。下一问题只记录训练接纳/留出共同收益与新行为代价边界，未选新方法或分配E/额度；后续编号仍从H038/E165核对。

2026-10-24 用户明确授权下一轮设计与验证并在额度中断后要求恢复。主控预留 E165（H036 同问题细化）：固定 C→U 的 S0/S1 二分缩步，只切换原 C0 新训练 KL 界是否参与停止，另列 S1−D；全16、0fit、≤336训练oracle/≤33评分。已接受[算法设计](appendix/H036/NEXT_DESIGN_20261024.md)与[实现合同](appendix/H036/COORDINATOR_E165_CONTRACT_20261024.md)，旧 E163/E164 结论不改。当前仅实现/静态冻结，计划120CPU/240累计受控wall/128MiB，未开运行reservation。全部检查/科学计算/统计审计在获准远端执行；后续编号从 H038/E166 核对。

2026-10-28 用户规定主控、算法实现、消融实验（1）之间停止直接消息，由用户手动转达，分工不变。E165源码36项与CHECKS传输42项完成静态身份核对，方法窄审查通过；主控登记120CPU/240累计受控wall/128MiB独立预留，仅授权[CHECKS准备](../experiments/exp165/preparation/coordinator_prepare001.json)的容量检查与白名单传输。原历史输入路径/哈希仅是元数据，CHECKS准备不读历史模型载荷；执行授权仍false，待用户转交最终spec和就绪证据后主控再决定。未SSH、未运行检查。

2026-11-10 根据用户提供的 V2 修订稿完成有范围限制的[源码与证据静态审计](V2_ARCHITECTURE_AUDIT.md)：当前可复用多条路径中的策略、门控、路由和工程组件，但未见修订稿目标动态树和双门控的完整集成；原始 V2 prompt 与外部 first_version 的一手证据尚缺。E165 冻结测试存在一处与实现的转后回退规则相矛盾的静态断言，尚未运行验证；不修改历史快照。此条不更改任何实验结论、预算或执行授权；无本地研究运行、SSH 或向其他对话发消息。
