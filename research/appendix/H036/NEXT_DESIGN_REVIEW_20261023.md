# H036 同轨迹接纳诊断：具体设计独立审查

日期：2026-10-23（Asia/Singapore）。状态：**DESIGN_REVIEW_PASS_WITH_FREEZE_CONDITIONS / 非执行 READY**。

审查方：消融实验（1）。[早期预审](NEXT_REVIEW_20261023.md)的PENDING_DESIGN日期语境保留；本文件审查已经交付的具体方案，不改算法方设计、原v1、E163结果或预算。主控仍负责设计接受、实现范围、新E及额度和最终准入。

## 结论与审查范围

**未发现科学方法或静态资源可行性的实质阻断，建议主控接受受限实现范围。** 全16逻辑配对、11条准确复用、5条缺失U参数各恢复一次，最多15个评分包，足以回答“同一已访问轨迹上两级接纳规则如何影响有限交付候选”。它不检验换搜索器、全轨迹可行性、一般表达能力或reward收益。

以下身份链、停止状态与评分adapter须在精确METHOD/HANDOFF、合成检查和新冻结中落实；属于实现/冻结验收条件，并非已验证完成。当前无新fit、评分、prefix反序列化、SSH、传输或计量预检。

| 被审查材料 | 本次身份 |
| --- | --- |
| [NEXT_DESIGN_20261023.md](NEXT_DESIGN_20261023.md) | 17411 B；SHA256 `d83e0b4ea8a0a79e867c063bc66b46bec79fa92125e3f981a80392a717bb7bff` |
| [NEXT_SCALAR_REPLAY_20261023.json](NEXT_SCALAR_REPLAY_20261023.json) | 83319 B；SHA256 `7d9e230971a18ad259e38204b3038c0d60d5193b8082ea496936595d8e3681ea` |
| 冻结数值与评分实现 | [原candidate模块](delivery_v1/source_snapshot/src/online_rl/retention_protected_candidate.py)、[原输入/评分模块](delivery_v1/source_snapshot/src/online_rl/retention_protected_inputs.py) |

本次核对了新标量附录的48份JSON引用大小/SHA均对应既有已验证镜像，16行及声明的11/5身份划分一致；没有重复旧1876项审计，也没有重新计算其辅助科学统计。另以s26201样例核对字段可串联性，不能把样例检查称为全部11项的冻结核验。

## 1. 十一条复用的证明成立，须落实完整身份链

原helper哈希包含dtype、shape及连续数组字节（candidate模块457–463行）。closure和best点按float64哈希，最终候选按float32哈希（625、650、747、751、755行）。因此同一语义下F64/U64哈希相同，配合原`fallback_applied=false`，支持最终float32点及其原评分相同；原转后gate路径见665–679行。不是仅凭L数值近似相等复用。

新冻结及准入必须逐条绑定以下链，不从11这个计数推定文件身份：

`重放F/U helper hash → 原best_float64_candidate_sha256 → 无fallback且原selected_float32_candidate_sha256 → 原candidate.json的scientific hash → 原metrics的candidate hash / prepared_id / λ=1 / wildcard → 89 B actor原文件SHA及原部署/评分准确引用`。

`candidate.f32`文件SHA与上述helper hash是两种身份，不能互换。原`candidate.json.raw_file_sha256`核对原36 B文件字节；按固定`<f4`、`(3,3)`、C顺序重算helper hash再对接`scientific_candidate_sha256`。原checkpoint/evidence/foundation、配置及评分相关源码依赖也须准确引用。新总snapshot可因新增模块而有新identity；要求旧科学评分依赖保持原字节，不要求整个新source identity等于旧包。

11条是原结果复用，不是新测量；V/UG/UR在这些项中是准确别名。计费中只计本次身份核验/编排，不抹去历史计算；统计单位仍16个prefix。

## 2. 五条恢复可证同一观测路径，停止状态也须匹配

R由全16条训练轨迹的F/U身份不同且U参数缺失决定，为26204/26206/26209/26212/26213。它与5个原C0恰好重合，不依赖留出结果选择。各一次原λ=1路径恢复，同时记录F/U，不是五条新优化规则或额外λ搜索。

设计规定按序逐项匹配closure数量、index、candidate hash及fullL/newKL/oldKL/ridge，要求F64/F32回到原身份；该要求能检查已观测点和目标量的重现。冻结时再明确核对**C0身份、iterations和termination_reason**，以及所有数值分量/非有限处理与原预算语义；不对差异临时加容差。额外计时、资源采样和日志不要求等于旧值。

原日志未保存逐点梯度，故不能声称“历史梯度逐点比对通过”。梯度/目标调用一致性由原helper的准确源码身份和新旧training-loop合成parity检查支持；真实恢复以记录点序列和标量的精确匹配为证。移除提前holdout诊断不应改变训练数值，但是否重现仍由获准的单次恢复检查，当前不先试跑。

任一输入、依赖、轨迹或停止状态不匹配，整批`REPLAY_MISMATCH`并停止于新增反事实留出评分之前；保留成本和部分证据。不得扩成16次恢复、把不匹配路径当成新结果、自动补跑或更改比较。

## 3. 两级gate比较的归因范围恰当

| 比较 | 受控含义 | 不得声称 |
| --- | --- | --- |
| UG−V | 同一λ=1轨迹上，保持float32转后gate开启时，closure候选接纳变化的作用 | 全局最优、另一搜索器的效果，或转后gate也已移除 |
| UR−UG | 固定U64选择后，转后新训练KL gate对交付的作用 | 纯float32舍入效应；U本来就可能明显越过C0训练界 |
| UR−V | 两级处理一起改变后的有限交付差异 | 单一gate独立效应或完整2×2交互 |

同轨迹U只旁路记账，不能改变optimizer点、完整目标/梯度、line search、预算、F及RNG。全部16的训练规则与float32处理先锁定，后评分；11条已有留出值本来已知，不能称未见数据。新模块training-only接口不得接收这些性能值；prepared/input哈希含数据身份不等于允许用holdout选择。

现设计只选U完整L最小点；即使它没有新增通过项，也不能否认其他未保存点可能有折中，更不能否认函数类。原四条件可逐点作反事实标签，E163仍8/16，原12/16不转为新推进线。UG/UR的新增通过、丢失通过和绝对旧KL/新收益变化都须同时报告。

## 4. 原评分器可复用，adapter必须满足准确legacy envelope

原`encode_and_score_stage0`会核对fit format、prepared_id、λ、有限float32的3×3 candidate及诊断中的candidate hash（inputs模块851–877行）。因此不能直接传裸candidate，也不能使用一个未经说明的自定义format。最小适配语义应明确包含：

- `format = h036_fit_v1`、匹配的`prepared_id`、`lambda_value = 1`；
- 有限、准确float32 `(3,3)` candidate；
- `diagnostics.candidate_sha256`为原helper对该candidate的哈希。

这是给原评分器的最小评价envelope，不冒称UR经过原v1优化/接纳。`diagnostic_only`、V/UG/UR来源、cast前后身份、B结果和锁定信息放在独立probe元数据；不伪造原optimizer/CPU/closure日志。评分器原样运行原安装/补偿/完整codec/fresh local=0与真实包计量，原fit未改。

新冻结至少用一个合成锁定候选验证完整adapter调用；同时覆盖别名、跨prepared/候选篡改拒绝。没有必要重新跑原23项或1876项审计。真实R的V锚点核对原actor字节及准确fresh训练/留出字段；资源/计时字段与整个JSON文件SHA不要求复现。新部署字节仍须保存并独立计实际容量，不能因原包合格省去新包检查。

## 5. 静态预算可行，未构成额度

5次恢复、最多15个完整评分包且无λ=0重拟合，在原单线程CPU实现与既有环境下，**60 CPU秒/120墙钟秒/64 MiB建议上限静态可行**。历史完整worker约7.79 CPU秒是冷启动参考；R约0.1714秒的fit分项不包括启动，不能冒充整轮成本。新增记录、全锁定、准备、预检、评分、序列化、报告、失败/收尾全部付费。

[执行计划](NEXT_EXECUTION_PLAN_20261023.md)给出约32.25 MiB的保守峰值工件包络（原始约10.75 MiB加归档/暂存副本），在64 MiB内留约31.75 MiB余量。实现冻结后须用实际源清单、日志格式/条数和暂存路径替换估计；不安装/复制Python环境。0.5 GiB是进程内存规划值，不是工件额度、实测新峰值或learner容量。影子F/U/C0和prepared是临时诊断状态，完整learner仍遵守89 B/262144 B原布局。

若采用60/120建议，可在cap内部规划55CPU秒/110墙钟秒停止及5CPU秒/10墙钟秒收尾；由主控登记并冻结后生效。新进程、fork前基线、wait4分项/最终差分、同campaign零模型预检及观测值/尾部上界区分沿用已验证原则，但新入口/路径/版本需重新接受。当前不执行任何预检来证明估计。

## 交回主控

可接受实现范围为算法方提出的新模块/config/合成检查和新冻结目录；本审查没有要求增加第四臂、其他候选、结构或新优化器。请主控接受具体范围后交算法方实现，并将上述身份、停止状态和adapter字段写入精确方法与冻结交付。随后实验方再审查实际实现；新E/数值预算/命令/远端中性目录和正式准入未齐备前，均不是执行READY。

本轮仅设计与静态源码/JSON身份阅读、文档更新；E163两个reservation关闭、累计完整CPU/墙钟UNKNOWN原样保留，未触及旧数据、预算、科学源码或共享H/索引。

### 后续主控接受

本次具体审查已发回主控。随后收到并读取[方向接受/实现范围](COORDINATOR_ACCEPTANCE_IMPLEMENTATION_20261023.md)及[接受记录](COORDINATOR_DESIGN_ACCEPTANCE_20261023.json)，E164已预留而资源未授权。本审查列出的身份链、停止状态和adapter要求继续作为实现/冻结验收条件，执行计划已同步这一已知状态；不因此宣称READY或开始真实恢复/评分/远端准备。

### 2026-10-23 E164 科学冻结后的验证方准备更新

主控已接受科学冻结 source identity `baae9a6a766518105a6d5b2e9bc64fef3488ff581870931cf83ff2d7b23a4c9a`（manifest `76073211ec4ff37af2677e0d3cae33c87e46a703a25c306ccb0257c03a2ef379`；27 个文件 / 762031 B）。本地 runner 已绑定冻结 replay/diagnostic/scoring API，report 已绑定 `e164_logical_scores_v1`；两者的 binding metadata 均为 `BOUND_FROZEN_SOURCE`。这满足本审查要求的接口/科学版本对应关系，但不代表不断编辑中的编排文件已形成最终执行包身份。

准备元数据列出 308 个 reference-map 项、40 个 transfer-manifest 成员；只记录其数量，不把当前编排 SHA/大小当最终身份。窄范围静态复审确认 runner 的固定输入路径检查与早期失败记录修复。仍须等待最终编辑后的定向独立审计和所需合成工程检查，并由主控单独决定数值资源授权与正式准入。既有 9 项冻结合成检查及 1876 项旧审计保持原结果，不重跑；目前没有科学调用、真实拟合/评分、SSH、传输或资源 reservation。


### 2026-10-23 最终本地交付

实验方已完成精确执行绑定、定向静态复核及五组人工 JSON 工程检查，状态更新为 **READY_FOR_RESOURCE_DECISION**。最终身份和命令见 `experiments/exp164/preparation/validation/READY_FOR_RESOURCE_DECISION.json`；旧 pending 段落是当时准备状态。仍无资源/执行授权或真实运行证据，需主控新单次准入后才能 SSH/传输和运行；不重跑科学9项或旧审计、不继承E163额度、不自动重试。


### 2026-10-23 E164 固定验证完成

E164 单次获准 campaign 已完成并独立审计为 **COMPLETE_VALID / EXPLORATORY_ONLY**：5 次原轨迹恢复、10 个新评分包、16 个逻辑配对、0 新交互/GPU；279 文件 / 5213330 B 精确镜像。UG−V 全16为零；UR−UG 在恢复5项旧KL全改善、新拟合收益3降2升。最终原始/统计/审计及成本见 `experiments/exp164/FINAL_VALIDATION.json` 和 `analyses/a001/report.md`，审计器字段映射的事后修正及全部原/中间失败均保留。唯一reservation关闭、作业与SSH结束，无自动后续或额度结转。科学解释及后续消融/算法实现由主控决定；本条不启动下一阶段。
