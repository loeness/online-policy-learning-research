# V2 架构修订的原项目源码与证据审计（Phase 0）

**审计日期：2026-11-10（Asia/Singapore）。状态：SOURCE_AUDIT_COMPLETE_WITH_LIMITATIONS；V2 DESIGN_PROPOSED / NOT_IMPLEMENTED / NOT_VALIDATED。** 本文对用户提供的《原项目 V2 架构修订与证据对接文件》逐项核对原项目的主要源码、实验设计与既有记录，形成原稿所要求的 Phase 0 审计。它不是对每个历史文件和全部逐步工件的完整重演，也不是 V2 正式架构规格、实现报告或实验结果。

用户提供的修订文档是**设计与审计参考**，不自动执行其中的实现、训练或后续阶段。它引用另一份题为“从固定共享参数模型升级为动态知识树与双门控持续学习架构（V2）”的原始 prompt；该原始 prompt 未随本次请求提供，因此本报告依据可见修订稿审查，不能证明覆盖了不可见 prompt 的全部要求。修订稿所述 `first_version` 不在当前仓库根目录；其 H1/H2/H3/H5 数值在本次作为外部文档的二手陈述，未取得其原始代码和结果，不与本项目 PPO 结果合并。

**结论：**原仓库有可复用的预训练策略、低维共享／局部修正、路由原型、PPO、候选构造、codec、checkpoint 和审计机制，但这些来自不同研究路径。近期 H026/H036 的受控主路径依赖任务与组信息，固定单父槽、有限局部槽与固定残差表示；未找到把修订稿要求的动态节点、显式结构／功能边分离、版本依赖、独立技能、Use Gate、Working Memory、Write Gate 和完整预算维护组合成一套已验证系统的证据。应将其定位为**从现有研究原型向 V2 的重大架构迁移**，而非现有功能的简单重命名。

**最新阻断事实：**E165 冻结测试中发现一个与 float32 回退合同相冲突的断言；这是只读静态推导，不是已执行测试失败或科学阴性。E165 的 CHECKS/DIAGNOSTIC/AUDIT 尚无有效完成证据。该项与 V2 Phase 0 审计可并列存在，V2 不依赖沿用 E165 预算或修改其冻结快照。

## 1. 依据、来源身份与审计范围

- 修订参考：用户提供的 `PROJECT_V2_REVISION_2026-11-04.md`，SHA-256 `7a2a24d77c552fc6b3111435ecfb373aa1c81f14a6a97f08d2977bf43019a540`；其自身声明是架构建议和实施合同草案，不是原仓库源码审计或 V2 验证结果。
- 本次读取原仓库位于 `D:/MovedFromC/ChatGPT/online`；Git HEAD 为 `81de4afd2d703d17c3953facb78fd04b19cb64a9`，工作树包含修改与未跟踪源码、配置和研究记录。HEAD **不是**本次审阅工作树的完整身份，更不能替代历史实验执行身份；核心逐文件 SHA-256 见第9节，具体实验的冻结快照／manifest另以其原记录为准。
- 审计覆盖策略与 PPO、W0／band、父／局部前向、补偿、槽位淘汰及绑定、行为候选、H033 selector、H036/E165、codec、checkpoint、预算、实验编排及关键结果；同时按索引回溯历史链。未全面逐行检查所有模块、全部数据／日志或全部历史工件。未重新运行任何研究实现、单元／合成／冒烟检查、统计审计或训练，也未连接服务器。
- 状态用 **已实现／已有局部机制／待修改／未见完整实现／未知** 区分。搜索未发现并不构成全仓库绝对不存在的数学证明；一个原型存在也不证明它已与近期保存流集成，更不证明性能。
- 仓库中历史文档 [KNOWLEDGE_V2_STRUCTURE_AND_ALGORITHM](../docs/KNOWLEDGE_V2_STRUCTURE_AND_ALGORITHM.md) 使用过“V2”名称，指的是受控 C1 的 `evidence_router_nonblocking`、最多6个扁平修正槽和周期性编码原型；本文的 **目标 V2** 专指本次修订稿中的动态知识树与双门控，不把旧名称当作目标 V2 已实现。

## 2. 必须优先修正的关键认识

1. **无标签路由不是零起点，但没有与新架构等价。** 既有 [TaskFreeRouter](../src/online_rl/task_free_router.py) 和 [EvidenceRouter](../src/online_rl/evidence_router.py) 有基于已观测移动事件、有限窗口、支持／歧义以及有限槽分配的组件。它们不是同时评估真实增量收益、安全试用、可验证继承的完整 Use Gate；近期共享保存流明确提供 task ID、group 和 bout 边界（[运行器](../scripts/run_storage_decision.py)）。两类研究结果不能拼成无标签端到端证据。
2. **已有树状／共享类不代表动态知识树。** `TreePolicy`、C1 固定路径和 parent/local 加法分解是旧原型或受控结构；没有证据证明它们提供动态独立技能、节点依赖版本、原子提交和安全重父化的目标 V2 合同。C1 中相同加法分解的 tree 与 grouped-flat 精确等价，不能将共享分解收益归为树拓扑。
3. **历史 gate 不等于修订稿的目标双门控。** 旧 Stage 2 已有共享知识复用 gate 和共享写入 gate，但作用于固定共享模块，未与目标动态树集成；H033 学的是操作前 keep／replace 排序诊断；H036 的 KL gate 在候选搜索／转后复查。这些均不能证明动态树上当前状态的调用收益／风险 Use Gate 与长期知识提交 Write Gate 已有效。H033/E162 和 H036/E163 没通过各自预设推进条件。[Stage 2 方法](../docs/RETENTION_STREAM_STAGE2_METHOD.md)。
4. **真实预算不能简化为 actor 字节。** H036 actor 固定89 B，而完整 learner 包上限262,144 B；序列化 checkpoint、foundation、证据和优化器等另计。目标 V2 还需纳入节点、门控、路由、版本依赖、工作／重启状态和维护操作的真实成本。
5. **附件的外部 H5 负结果值得作为警示而非本仓库结论。** 其报告的保守门控表现差来自另一套表格项目，当前仅有二手描述；不能拿其数值与本项目 PPO 合并，亦不能据此声称已验证任何新 Use Gate。

## 3. 原实现的主要数据流与源码清单

以下是按源码函数逐项定位的**审计导航**；具体时间顺序、条件及依赖在第4–6节说明。

| 范围 | 文件／关键入口 | 当前作用与信息流 | 目标 V2 处置 |
| --- | --- | --- | --- |
| 基础 actor、策略 | [policy.py](../src/online_rl/policy.py)、[c1_policy.py](../src/online_rl/c1_policy.py) 的 `TreePolicy`／`C1ResidualPolicy` | 冻结来源策略、特征与修正路径；具体维度／结构随实验策略类变化 | 可封装 W0/PPO 基础；不能要求所有新技能都落入原残差基底 |
| 源基与局部／共享参数 | [shared_write_policy.py](../src/online_rl/shared_write_policy.py) `SharedWritePolicy._correction`；[write_decision_policy.py](../src/online_rl/write_decision_policy.py) `FlexibleSharedPolicy._correction` | 在当前路由槽中按 local 和共享组组合修正；是固定张量／槽语义 | 提取可选残差路径；独立 actor/head、动态大小须新增 |
| 路由与部分标签无关原型 | [task_free_router.py](../src/online_rl/task_free_router.py) `observe`／`route`；[evidence_router.py](../src/online_rl/evidence_router.py) | 用过去动作与实际移动建立有界原型，未知时可未解析；最多固定槽并有替换机制 | 可作为候选检索原型；不等于修订稿双证据 Use Gate，需重新验证时间边界与集成 |
| 固定共享模块上的两道 gate | [run_retention_stream.py](../scripts/run_retention_stream.py#L264) 中 `mean_reuse_gate`／`retention_gate` | 检查是否使用旧共享知识；后者另在共享写入前检查历史 prototype 与新 rollout | 可审视已有证据和门控方式，但必须重新定义并验证动态节点上的调用与持久提交 |
| 父槽安装和补偿 | [storage_decision.py](../src/online_rl/storage_decision.py) `effective_matrices`、`compensated_single_parent_install`、`apply_action` | 用 physical groups／logical tags 确定共享生效对象；安装时补偿已驻留局部矩阵 | 数值补偿可借鉴；固定单父与隐式相加须改为显式继承及版本依赖 |
| 当前受控保存流与淘汰 | [run_storage_decision.py](../scripts/run_storage_decision.py) `_prepare_task`、运行循环及 `_snapshot`／恢复 | 测试调度提供 task/group/bout，按任务与组进行缓存分配和继承；保存状态用于分支评估 | 作为有标签历史基线保留；无标签 V2 不得直接把提供的真实标签喂给门控 |
| 行为候选 | [retention_behavioral_candidate.py](../src/online_rl/retention_behavioral_candidate.py) | 从有限来源观测与教师行为构造九系数候选 | 保留为可选候选生成或对照，不作为强制所有技能的能力边界 |
| H033 特征及诊断 | [h033_inputs.py](../src/online_rl/h033_inputs.py)、[h033_diagnostic.py](../src/online_rl/h033_diagnostic.py) | 从允许的历史／教师数据构造固定特征，整 seed 留出评估候选操作差 | 历史特征与防泄漏检查可借鉴；不可把其弱于预设门槛的排序器作为已有效部署 gate |
| H036 旧行为保护 | [retention_protected_candidate.py](../src/online_rl/retention_protected_candidate.py) `protected_objective_and_gradient`、`fit_protected_candidate`；[retention_protected_inputs.py](../src/online_rl/retention_protected_inputs.py) `prepare_inputs`、`encode_and_score_stage0` | 冻结教师／有限观测、双 KL 加 ridge、有限优化与新训练界、wildcard 安装编码及 fresh-slot 评分 | 作为候选生成／旧 prior 保留诊断；KL不可替代在线 reward，不适合强制所有冲突技能融合 |
| E165 有限搜索 | [retention_budgeted_search.py](../src/online_rl/retention_budgeted_search.py) `search_budgeted_pair`、`lock_budgeted_pair`、`seal_budgeted_cohort` | 对历史 C→U32 进行固定折半与训练界选择，16项锁定后评分 | 历史诊断单独保留；发现冻结测试断言矛盾，未运行，不能作为 V2 开发依赖或结果 |
| actor codec 与重启状态 | [write_decision_policy.py](../src/online_rl/write_decision_policy.py) `encode_actor`／`decode_actor`；[compression.py](../src/online_rl/compression.py) `export_payload`／`save_payload`／`load_payload` | 支持已有 actor 数据编码与 payload；H036另计完整 checkpoint、foundation、actor及evidence | 保留 codec/序列化基础；动态节点、路由、门控、依赖与增量/峰值成本需新的完整定义和实测 |
| 实验、统计与成本 | [retention_stream_value.py](../src/online_rl/retention_stream_value.py)、[run_retention_candidate.py](../scripts/run_retention_candidate.py)、[E163报告](../experiments/exp163/analyses/a002/report.md)、[E164审计](../experiments/exp164/analyses/a003/audit.json) | 共同前缀、受控动作、指标／保存及有限资源记录 | 复用理念和已验证部分；V2须重新冻结无标签协议、强对照、版本/状态与真实成本 |

## 4. 当前与目标 V2 的差距矩阵

| 修订稿目标 | 当前真实状态 | 判定与下一步 |
| --- | --- | --- |
| 动态节点数量、深度与独立技能 | 近期路径是固定维度 parent/local；旧扁平路由原型有有限个上下文槽及分配／淘汰。 | **待修改／未见完整实现**。动态分配接口不等于动态单父树、可变大小技能或独立 encoder/head。 |
| 结构父边与功能继承边分离 | 近期以 physical/logical group 和 wildcard 决定共享矩阵是否相加，并使用局部补偿。 | **待修改**。必须显式建模“不继承”和精确父版本，不可把树形关联自动当作功能调用。 |
| 节点版本、旧子保护与回滚 | 存在 slot generation、checkpoint／恢复和 actor 编码，但这不等于节点依赖图或可原子回滚的父版本升级。 | **未见完整实现**。需定义版本依赖、双版本峰值、回归失败与原子提交；当前状态保留为历史基线。 |
| 无标签候选检索与 Use Gate | 独立路由原型依据已观察事件工作；近期 H026/H036 受控流直接获得组及切换信息。 | **已有局部机制，待集成验证**。适用性与真实增量收益／风险分开估计；不可把路由支持或未来 observer 回报冒充候选收益证据。 |
| 工作记忆与持久树分离 | 在线 PPO 更新与共享参数、checkpoint／证据存在，但当前固定父参数可直接参与训练／安装，缺少修订稿要求的统一持久提交边界。 | **待设计／实现**。明确工作策略、未刷新 rollout、优化器与重启恢复，再定义何时形成和提交候选。 |
| 独立 Write Gate | 现有动作选择、H033预测和H036训练接纳针对不同对象，不是候选持久提交的完整收益、稳定、冗余、冲突、成本联合合同。 | **未见完整实现**。候选验证数据必须动作前可用；结果不足需能 defer／discard／独立分支。 |
| 节点 Delete/Merge/Compress/Re-parent | 已有有限槽淘汰、父安装、actor 压缩和完整字节检查。 | **部分基础可复用**。未建立保留依赖子节点、影子版本回归与操作回滚的完整节点维护接口。 |
| 完整持久包及计算预算 | H036 定义并测得完整包与进程／候选成本的一部分；历史计量有已披露 UNKNOWN。 | **部分基础可复用**。新的节点编码、优化器、critic、门控统计、replay、版本依赖和临时双版本须真实计量；理论 `nbytes` 不足。 |
| 严格因果与时序 | 旧无标签路由先根据历史路由、动作后 observe；受控保存流另有 task/group 标注与 observer 分支。 | **必须逐接口隔离**。对每条新链证明当步奖励、下一状态、真实任务标签、未来日程和反事实奖励不进入该步决策。 |

## 5. 版本化源码事实与风险说明

### 5.1 W0、任务先验和 PPO

以近期保存流程为例：固定来源 actor，使用 source basis、三个 band 的局部修正和共享修正；`SharedWritePolicy` 与 `FlexibleSharedPolicy` 通过槽／组决定共享是否参与。W0 是必要基线和 fallback 候选，但修订稿要求 V2 支持独立技能，因此原 3×3 固定输出基应作为一类可选参数化，而非 V2 唯一表达空间。

具体运行器在每个 bout 读取 task/group，调用 `_prepare_task` 后采样真实环境奖励并形成 rollout/evidence；各实验的 PPO 更新与 actor codec 约定以对应冻结配置为准。[保存流说明](../docs/RETENTION_VALUE_STREAM_METHOD.md)记载：普通 rollout 为256步、受控 bout为512步；离散保存臂冻结共享父，直接 shared PPO 臂对其更新；codec round-trip 后清 actor Adam moments，而 critic optimizer state 保留。该 shared PPO 是共同 harness 中的强对照，不是完全未改动的外部 PPO/SOTA 复现。不能把此状态清理通用化到所有历史实验。

### 5.2 路由、缓存和父局部关系

[task_free_router.py](../src/online_rl/task_free_router.py) 的类注释和 `observe`/`route` 接口使用归一化观测、动作与实际移动，默认最多6槽、32事件窗口；零移动被删失，支持、矛盾与未解析控制分配／回退。它为未知情境检索提供候选基础，但它仅解决特定可观测动力学证据问题，不能推成可验证技能收益的通用 gate。

[run_storage_decision.py](../scripts/run_storage_decision.py) 明确注明该受控实验提供 task ID、group 和 bout 边界；`_prepare_task` 使用 task/group、slot generation、parent group 与 wildcard 决定是否继承。PPO row／evidence在动作后记录 transition；未来分支在实验评价中使用，不得作为在线决策输入。不能因独立无标签路由原型存在就声称这条历史保存流已经无标签。

`compensated_single_parent_install` 的固定数组／组绑定及 residual compensation 有明确工程价值，但它没有把结构上的父节点与功能依赖分成带版本的独立实体。修订稿要求安全的独立分支、父升级和 re-parent/delete 时保留子行为，须新设合同并在受控真实编码路径验证，而不是把现有 `parent_group=-2` 重命名为节点树。

### 5.3 H026、H033／H036／E165 的正确归属

信息可见性须按版本区分：H026 的 fitter 不直接接收 future schedule 或未来 reward，但其输入含末位为 `route_id` 的13维策略观测，来源教师及样本区分两个 owner；当前特征提取器接收完整 observation，而其 W0 actor 的 `forward_actor` 只读取前4维，不能据此说该特征直接使用 route ID。教师行为及受控路由仍由槽位条件决定，因此整个协议不是无槽位身份信息的未知情境路由。[观测构造](../scripts/run_knowledge_loop.py#L175)、[行为候选特征](../src/online_rl/retention_behavioral_candidate.py#L78)、[actor 切片](../src/online_rl/policy.py#L36)、[候选输入](../scripts/run_retention_candidate.py#L790)。在另一条 H036/E165 冻结特征路径中，只使用 observation 前4维，训练视图不含 holdout 数组；owner、seed、logical case 仍作为加权或身份元数据。不能将两条路径的输入合同或整个协议的信息条件混为一谈。[冻结特征](../src/online_rl/h033_inputs.py#L201)、[训练视图](../src/online_rl/retention_acceptance_diagnostic.py#L201)。

H033 是在已发生历史上预测特定操作增量的诊断，其固定三特征模型在 E162 未达推进标准；它与目标 V2 的 Use Gate（当前调用适用性和真实增量／安全）或 Write Gate（是否持久提交）不等价。[H033 结果](records/H033.md)。

H036 的旧 teacher 是旧父 `local=0` 的 keep 初始化 prior，不是完整旧任务策略。其 closure 对完整 `K_new+λK_old+ridge` 搜索；接纳只维护符合 C0 新训练 KL 限制的 incumbent，转float32再检查，不能叫精确受约束求解。[方法](appendix/H036/METHOD_V1.md)。E165 在线段上规定 S0/S1，属于有限诊断；两版本均有 post-cast 复查，搜索阶段放开约束不意味着最终保留越界点。

**本次可从冻结源码直接复现的静态矛盾：**[测试](appendix/H036/delivery_budgeted_v1/source_snapshot/tests/test_retention_budgeted_search.py#L614) 将 S0 的搜索选中点 X=3（新 KL 4，C0界3）当作最终候选3；[实现](appendix/H036/delivery_budgeted_v1/source_snapshot/src/online_rl/retention_budgeted_search.py#L1353) 对该点作相同转后复查后应回退C0=0。需要算法侧在新版本中修正测试、保留旧冻结包、重绑 manifest；这不是已执行失败。源码中 S0/S1 都有 witness 字段，但合同正式见证只计算最终 S1；实验报告必须按合同区分。

### 5.4 Codec、完整状态及生命周期

`encode_actor`/`decode_actor` 是既有 actor 编码实现；H036 `encode_and_score_stage0` 完整安装、编码、解码后再计算 fresh-slot KL、驻留漂移和实际 bytes，并在 checkpoint 副本中清 local/shared actor optimizer state、核对 critic state 未变。其完整包为 foundation bytes + serialized checkpoint + actor + 原始 evidence。一次 `torch.save` 的真实序列化开销与内存峰值，与 actor 参数量、临时数据、可压缩理论大小是不同量。

对 V2 不能仅保存节点张量，还要定义及测量树边、版本引用、父子订阅、router／gate 统计、技能 encoder/head、必要 replay、工作策略及 optimizer/critic/RNG 状态，以及编码前后、影子版本并存和重启时的峰值；具体哪些状态持久化和恢复后如何行事尚无目标 V2 实现依据。需使用实际序列化字节和恢复行为，而非将旧89 B actor大小直接推广。

现有 checkpoint 的可恢复性也有边界：E161 保存 learner、模拟器及 driver 状态并以语义 hash 核对，支持固定共同前缀的受控分支；旧 V2 写流的 learner restore 不恢复环境轨迹；selected-feedback 写流保存 pending feedback 等状态，但未见相应的完整 resume 入口。它们不能自动证明任意时间点中断后能恢复整个实验、环境和预算账本。[E161 快照](../scripts/run_storage_decision.py#L180)、[旧写流](../scripts/run_write_stream.py#L694)、[selected-feedback 写流](../scripts/run_retention_value_stream.py#L414)。

## 6. 既有研究证据对 V2 决策的约束

| 已有证据 | 可以支持 | 对 V2 不可过度推断 |
| --- | --- | --- |
| H005–H006：相同共享加法分解下 tree 与 grouped flat 精确等价 | 共享分解及任务先验值得保留为基线 | 树容器／拓扑天然有增量；V2 应有同字节、同参数、同信息的扁平对照 |
| H007–H009：跨任务格51正39负，路由/整合增量有限 | 局部学习和共享存在兼容性问题 | 相似表示或同组即安全；不能无条件父子叠加 |
| H010–H013：受控容量竞争中 active shared 81.28%，独立32.09%，frozen shared 53.68% | 共享参数可携带在线经验与恢复信号 | 同实际字节 Pareto 胜出；共享 actor 230 B、独立204 B，且任务先验与组件条件有限 |
| H019/H025：保存操作效果随未来复现比例改变；E128 匹配拟合量后仍观测交互 | 未来环境会影响长期保存价值 | 部署门控已能预见未来；未来 schedule 与 observer 反事实不可免费作为输入 |
| E161：behavioral−mean +0.001591，下界+0.000045；behavioral−keep +0.001301，下界−0.001132，均小于+0.002点估计门槛 | 有限条件下有探索信号，未来分布体现适用边界 | 独立确认通过；H026仍 INCONCLUSIVE |
| E162：G_min +0.000720、折稳定1/4 | 当前过去信息／排序问题可被直接诊断 | 固定 ridge 达到升级条件、所有过去信息都无用；H033 INCONCLUSIVE |
| E163：8/16 < 预设12/16，旧KL均值1.17558→0.26965，新收益0.12650→0.08781 | 当前有限旧／新行为保护有权衡 | 固定H036 v1胜出或KL等于奖励；H036 INCONCLUSIVE |
| E164：UG−V全16零；UR−UG旧KL −0.221881，新收益−0.016072（全16均值） | 指向5个已恢复候选的接纳／新旧取舍 | 普遍应删除gate、固定容量不够、可按留出挑两个成功候选或已有奖励结论 |
| E165：方法冻结，当前有测试断言矛盾，远端检查／诊断无完成回执 | 给出可检验的有限候选路径及技术待办 | 已通过测试、存在可行见证、成功或失败科学结果 |

细节见 [项目结果总览](PROJECT_ALGORITHM_EXPERIMENTS_2026-11-03.md)、[E161](../experiments/exp161/analyses/analysis001/report.md)、[E162](../experiments/exp162/analyses/analysis001/report.md)、[E163](../experiments/exp163/analyses/a002/report.md)、[E164](../experiments/exp164/analyses/a001/report.md)。外部 `first_version` 的 H1/H2/H3/H5 是修订稿转述、未独立核验；不得与本表数据合并。

## 7. 对修订稿的审计结论与实施建议

**可接受为后续 V2 设计问题，而非已验证方案：**保留 W0 作为基线/回退，允许独立技能与可选父继承；结构边和功能边分开；调用决策和持久提交分开；明确工作策略、节点版本与完整容量；冲突时允许隔离。修订稿对时序／未来信息、平行强基线和父版本回归的要求与现有失败证据相吻合。但这些是可执行合同应解决的问题，不代表具体门控估计器或阈值已经成立。

**需要先明确而不能直接开跑的设计问题：**

1. 何为 V2 节点的能力与状态边界：动作／观测空间、独立 actor/head、版本引用、共享父的选择性组合，以及旧节点与新节点的可重启格式？
2. Use Gate 的真实收益比较如何收集：可用试用预算、安全约束、选择偏差、样本支持、反馈滞后、持续漂移和 abstain 冷启动；不能用未执行动作的反事实奖励作实测。
3. Write Gate 的独立验证如何形成、谁付交互与统计成本、如何避免 candidate 选择与评估重用同一数据，以及重复／冲突／不足证据分别触发何种动作？
4. 父更新、重新绑定、压缩、删除的回归范围与原子性：父版本绑定、未订阅子节点不变、依赖子节点保活、影子版本共存期间峰值字节如何计？
5. 无任务标签与已知边界／oracle 路由分别是什么实验设置？如何保证 gate 不读真实任务ID、group、未来 schedule、当前动作之后的 reward、observer counterfactual？
6. 各基线如何做到相同完整字节、交互、训练和检索开销，并允许强扁平库合理使用预算？如何比较动态树、同分解分组flat、直接shared PPO和合理独立学习？

**建议顺序：**以本报告作为源码审计起点；先补齐修订稿引用但未提供的原始 V2 prompt 和 `first_version` 原始来源，明确哪些外部结果可独立追溯；然后由主控与算法实现围绕上述问题制定可审查的 V2 合同和迁移设计，再根据用户安排进入实现、独立检查和有限验证。修订稿的 B0–B5、附加强基线和多场景是待权衡的实验草案，不代表必须一次启动全部组合或已获资源批准。本次请求只进行审查，不实现 V2、不创建或运行新实验。

修订稿提出的对照各自回答不同问题；在设计合同和计算预算冻结前，其配置及运行均为**待定**：

| 对照 | 作用 | 必要的解释边界 |
| --- | --- | --- |
| B0：W0 | 基础策略与回退 | 不能只拿它作为完整 V2 的唯一对照。 |
| B1：固定 parent/local | 比较已有受控共享机制 | 历史版本原样保留，明确真实任务／组信息和完整预算。 |
| B2：动态树、无双 gate | 分离动态表示本身 | 固定无 gate 的启用和写入规则，防止把无规则配置成弱基线。 |
| B3：B2 加 Use Gate | 检查调用选择、拒绝与冷启动 | 计入试用成本，严格限制操作前可用信息。 |
| B4：B2 加 Write Gate | 检查长期提交选择 | 与 B2 使用匹配候选和证据预算，隔离误保存及遗漏。 |
| B5：B2 加双 gate | 检查组合效果 | 对 B2–B4 和同预算强扁平库比较，不能单凭 B5−B0 归因。 |

同一候选库、相同信息条件和可比完整资源是有效归因的前提；还应纳入同分解 grouped-flat、独立技能扁平库及相关固定策略。真实任务路由只能作为单列 oracle 上界，不能作为无标签方法输入。各对照是否进入首轮、配对与统计门槛、交互和成本上限均须主控与算法实现先确定，再由消融实验独立审查；不将此表视作实验执行命令。

交付状态：本文为有范围限制的 Phase 0 静态审计；`V2_ARCHITECTURE_SPEC.md` 和 `V2_MIGRATION_PLAN.md` 在本次审阅的仓库文件清单中未发现，尚不能视为已冻结。修订稿另列的实现报告、模块映射、测试规格、实验计划、预算规格及状态 JSON，同样没有由本次审计生成；后续只有发生相应工作时才如实填写阶段及结果。

## 8. E165 保留事项与当前交接

E165 是 H036 的独立历史诊断，不是 V2 建立的前置条件。既有预算账本 `resource_authorized=true` 仅表示已预留并允许指定准备范围，`execution_authorized=false`、`active_job=null`；本地可见最新准备回执为 2026-10-29 的 `FAILED_CLOSED`，人工 SSH 认证成功不等于受控 helper 容量准入或已成功传输。旧 README／准备模板可能早于科学冻结、准备尝试及测试问题；实际操作以回执、具体版本及账本综合判断。本次未访问远端，无法判断本地材料以外有无更新。若用户之后继续 E165，应先由算法实现负责测试修订与新身份，验证方独立审查并按阶段准入，不能将本次源码审计当作远端测试。

历史成本同样需按实际记账：E162 的监督器没有累计解释器子进程 CPU，完整 CPU 用量为 UNKNOWN，其预留额度已关闭；不能拿已知下界当作完整耗时。E163 首次失败与 E164 运维部分亦保留各自 UNKNOWN，不与本轮或 V2 的预算混同。[E162 成本记录](../experiments/exp162/analyses/analysis002/README.md)。

三方分工保持：主控决定问题、资源、范围及接受；算法实现拥有算法／结构规格、代码与冻结交付；消融实验（1）拥有独立审查、获准远端执行、统计审计和资源证据。按用户要求，两个对话与主控之间的信息由用户手动转达；本次未发送消息。验证交付及主控和算法完成同轮分析后停止，不自动开启下一轮。

## 9. 本次审阅的部分源码身份

以下 SHA-256 只标识本次读取的**工作树文件字节**；不证明对应历史实验运行时就是该版本。更多版本依赖、配置及输入应以各 E 的 manifest 和 run 记录为准。

| 工作树相对路径 | SHA-256 |
| --- | --- |
| `src/online_rl/policy.py` | `edf1d024abdfa00dfea262255a358e7ce5dfb92c1e003b2642714a0a56067c1d` |
| `src/online_rl/write_decision_policy.py` | `84c9e9653cf8e6135154e4a4e1f851aa35fc8ceebd6eb0336af2ace31239654f` |
| `src/online_rl/shared_write_policy.py` | `8b241f7760b785e990c72335b73fb489bb43b8f67e75ac92e80418fabc47d3ee` |
| `src/online_rl/storage_decision.py` | `d732b97e7375e75d467bf019e74e7135ef56e054165857c64d83b3a3a1dd7068` |
| `src/online_rl/retention_behavioral_candidate.py` | `5c57f2913e33e8fd7553550c0c98eeff2a000c4ce67193d5d7535e842daf1d71` |
| `src/online_rl/retention_protected_candidate.py` | `0cabd3169650c3cd01ed719414f7dbe1f8bff156e5e84b834c71b56698c912d5` |
| `src/online_rl/retention_protected_inputs.py` | `d48c9c9cac14585c0c8e5ce43cff899b362652343738715975481cd0c4835316` |
| `src/online_rl/retention_budgeted_search.py` | `87a67fa6823eb349808eff3a65a3e3464fda5b0f3a6d212b8dd211ac7cedc438` |
| `src/online_rl/h033_inputs.py` | `4a2939c0616a0f522048553331f45679efb8c1315ad7f228aef85beee80e71cd` |
| `src/online_rl/h033_diagnostic.py` | `b4d2457df559b9cecf10fa385c8c3879038b8aa1aca04076f4f77a11017e5c67` |
| `src/online_rl/task_free_router.py` | `9dedd5827a11366467b64286b7f99fe0f78f97d25788258cad5b6dc1c06928c8` |
| `src/online_rl/evidence_router.py` | `10d78b18d7393b44353ab99fdc34b53fcb8b99e31e06ffd734e42c249dfef278` |
| `src/online_rl/compression.py` | `bc62a863b5ddafeb8deb39b2c1a4b4b0aaf98313d60083249fc4bf9572944a7a` |
| `scripts/run_storage_decision.py` | `6e52370cf4916ff23a6f4a2b54a84fd9c7aa4df7e52fe4c38956fb2945d84e22` |
| `scripts/run_retention_candidate.py` | `9b1e6c0123022943010d9a1e42c237aea095a71bc82fbfa480e7dbd7a349670f` |
| `scripts/run_knowledge_loop.py` | `5ed6d081c0e3381d2e22d9bcf280600909ac1f281c9e82f9c31067419fc26eb7` |
| `scripts/run_write_stream.py` | `fc2691c5664fbb0a6cb04d59eceac92867826c0b7732640334033d5c85e5214b` |
| `scripts/run_retention_stream.py` | `6c9edf8fe3a76c6248fba52feca67974718b64437e07537737ccbf2d2e7a0723` |
| `scripts/run_retention_value_stream.py` | `95656986d6f3edef83e1ce6936a01c2b99bb3e5e24b7c1e6ccc272890f6c403d` |
| `experiments/exp162/preparation/execute.py` | `e1d1ce7e0c32956294f5502cb4e947b7118c50f2f41bb4d4d7120a41b7e7482d` |
| `research/appendix/H036/delivery_budgeted_v1/source_snapshot/tests/test_retention_budgeted_search.py` | `981e51a31ed823f002584ede36459f10be5575533610c24ef66679721a1f9ec4` |
| `research/appendix/H036/delivery_budgeted_v1/source_snapshot/src/online_rl/retention_budgeted_search.py` | `87a67fa6823eb349808eff3a65a3e3464fda5b0f3a6d212b8dd211ac7cedc438` |
