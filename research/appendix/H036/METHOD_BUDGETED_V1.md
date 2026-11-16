# H036 / E165 方法规格 v1

日期：2026-10-24（Asia/Singapore）。依据[主控合同](COORDINATOR_E165_CONTRACT_20261024.md)，不覆盖[原设计](NEXT_DESIGN_20261024.md)。交付阶段为 **SOURCE_FROZEN / REMOTE_CHECKS_PENDING**；源码冻结以本目录`delivery_budgeted_v1/manifest.json`存在且身份核对一致为准。本机没有运行研究实现、测试、拟合、评分或审计。最终源码与输入身份由该manifest及`anchor_manifest.json`绑定。

## 固定问题与数据

同一九系数父、旧keep初始化prior和新教师，在C0新训练KL界内能否用有限C→U搜索交付原D未交付的可行旧保护改进？本轮是已见数据的探索性存在性诊断，不是整个九维可行域证书、生产保存规则或reward确认。

固定原16个seed，全部保留；新owner各前96/128行训练，旧FIFO最新128行的前96训练，后32行分别留出。完整teacher/绑定/epsilon=.05/temperature=2/归一化/mean64沿冻结v1。输入准备可生成固定teacher，但搜索只接受`retention_acceptance_diagnostic.TrainingView`，不得接收prepared dict、holdout索引/观测/教师/性能或未来信息。

C/D为E163最终float32候选；U32为E164 UR的float32候选（11项按已验收身份准确引用D，5项读取实际UR文件）。C0=`float64(float32(M))`。输入与端点只读精确manifest；缺失、身份不符、越界路径或非有限停批，不补fit或从hash恢复系数。`load_anchor_references`只将训练预期带入`AnchorSet`，不带原metrics中的holdout值。

## Oracle、算术顺序与复用预期

原完整目标保持`L1=K_new+K_old+R`，`R=1e-4 mean((X-M)^2)`，lambda=1，ridge一次；新owner各占一半，旧行均权。M沿原float64求均值；C0先float32再float64；C、D、U32先读原float32再准确转float64。`X_j=C64+2**(-j)*(U32_as_float64-C64)`，j依次0…15，不改变运算次序或改用U64。

唯一界`b=K_new^tr(C0)`，经济容忍量0，原数值容差`1e-12`。C为性能参照，不用C另设界。完整L相等不替换D；不按近似L或holdout排序。

基准核对必须使用原post-float32训练值：D的`diagnostics.json:final_metrics.new_train_kl/old_train_kl`、`final_float32_ridge`、`final_float32_fullL`；C比较对应new/old/ridge分项，**不把C的lambda0 fullL与本轮L1(C)硬比**。C0比较原D初始化训练分项及initial_fullL。预期标量与其来源文件SHA均在anchor_manifest中；这里的核对是将来远端执行内容。

单次完整训练oracle评估包括原目标/解析梯度及规定KL分项计算；原接口伴随计算的梯度成本同样计费。cache key必须绑定prepared_id、training_hash、目标版本、dtype和参数原字节，不能只用loss或seed。每prefix最多3基准点、16个不同试探点、2个postcast点，总真实cache miss≤21；全cohort≤336。logical request、cache hit、cache miss、底层目标/梯度/KL helper调用分别报告。

fallback C0、保底D、最终float32分项与见证只复用已计的基准/postcast cache，不追加隐形oracle。336只约束DIAGNOSTIC真实prefix；CHECKS合成调用另列、同样扣本轮CPU。AUDIT只重算已存参数字节、trace、锁和指标，不调用模型oracle/scorer。

## 两臂与固定交付

S0返回第一个满足`L1(X_j)<L1(D)`的点；S1还要求`K_new(X_j)<=b+1e-12`，不满足则继续缩步。每臂最多16次，未找到保留D。目标、线段、顺序、上限和最终复查均相同，唯一开关是约束是否参与搜索停止。S1−S0为主比较，S1−D防止把S0退化当成新收益。不是完整网格择优或九维约束求解。

选中float64点转float32，用原新训练界复查，失败回C0；不继续搜索、改选更早点或增加转后L/holdout gate。搜索可在内部完成postcast计算，`lock_budgeted_pair`只验证并封装，不再运行第二轮oracle。trace、原始cast、最终参数、分项、停止索引/原因、fallback和计数全部可追溯。

见证仅按最终float32定义：字节不同于D、满足原界、`L1(S1)<L1(D)-1e-8`且`K_old(S1)<K_old(D)-1e-8`。报告阈值不参与选点；候选若只是原D、但S0变差，不能记新见证。见证不等于部署、留出或reward收益。

## 全cohort锁定与评分

全16条S0/S1的训练选择、cast和字节别名先完成，由`seal_budgeted_cohort`绑定全cohort锁、源码身份和端点manifest身份。写出锁定清单与SHA后才评分。缺seed、锁或参数被修改、prepared不符均拒绝。

评分先执行固定26201 D锚点；原D metrics与actor引用必须先由精确manifest核验。只比较规定科学字段/原actor字节身份，不要求新计时或ZIP序列化资源元数据相同。锚点成功才发receipt，S0/S1评分须核验该receipt和全cohort permit。

adapter明确是diagnostic-only候选，通过原`h036_fit_v1`最小接口调用冻结`encode_and_score_stage0`，不伪造旧优化轨迹。wildcard、补偿、完整actor int8编码、fresh local=0、actor optimizer清理及critic保留沿原方法。actor89 B，完整持久包≤262144 B；部署bytes逐字节保存，不重复序列化。临时端点、搜索cache和诊断均不进入learner。

返回原pre-codec/encoded/W0的train/holdout、resident漂移、容量与部署bytes。另报告encoded新训练KL相对原b的signed excess、非负违反量、原容差下是否满足。**不生成独立encoded C0参考臂或比较**（合同明确删去原设计此项）；S0/S1若最终回C0，仍须正常评分或准确别名复用，它属于既定32臂，不能跳过。编码后诊断不回写选择/回退。旧C/D是准确引用的性能参照，身份一致才可复用。一个锚点及最多32个新臂包，总上限33；runner可按同prepared、相同候选和同评分身份去重，但不能仅因loss相近复用。

## 状态、检查与停止

本轮没有本机runtime通过声明。只允许消融实验在主控指定robot项目子目录运行CHECKS、DIAGNOSTIC、AUDIT，按各段一次的预算/命令/冻结版本执行。合成检查不得读取真实prefix或调用fit。代码身份、精确调用和预期receipt见[HANDOFF](HANDOFF_BUDGETED_V1.md)。

无见证只说明有限线段/分辨率/规则未找到，不推导必须增容量或放宽预算。全部失败与未知成本保留；无自动重试、补网格、补seed、换方向、第二求解器或奖励分支。验证交付及主控/算法完成本轮分析后停止。
