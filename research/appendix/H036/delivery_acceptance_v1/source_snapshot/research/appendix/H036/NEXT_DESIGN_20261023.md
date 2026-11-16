# H036 下一设计：同轨迹接纳诊断与有限候选恢复

日期：2026-10-23（Asia/Singapore）。状态：**DESIGN_FOR_COORDINATOR_REVIEW，未实施、未准入真实计算**。

依据[主控合同](COORDINATOR_NEXT_CONTRACT_20261023.md)、[有效结果集成](COORDINATOR_E163_SYNTHESIS_20261023.md)和实验方的[NEXT_REVIEW](NEXT_REVIEW_20261023.md)／[NEXT_EXECUTION_PLAN](NEXT_EXECUTION_PLAN_20261023.md)。E163保持COMPLETE_VALID / NO_ADVANCEMENT、8/16；固定v1停止。本文是同一问题的结果后诊断细化，不提出生产保存规则或结构扩展，不改原门槛及账本。

**建议接受一套两段式方案：先用全16条既有D轨迹做训练侧标量重放；随后仅为5个缺失的反事实候选恢复系数，进行全16条配对诊断。每条恢复只跑一次原λ=1搜索，同时记两个incumbent；不另跑“去gate搜索”，不重拟合λ=0，不保存或评价全部closure系数。** 两段分属不同权限：标量核查已完成；参数恢复及新评分仍需新E、冻结与正式预算准入。

## 1. 本轮已能回答的部分

冻结源码的closure向L-BFGS始终返回同一个完整目标与梯度。`new_train_kl_gate`只决定是否更新返回用incumbent，不将incumbent回写搜索参数。因此应区分**搜索实际访问了什么点**和**最后保留什么点**；不能说过滤阻止优化器访问被拒点。有限L-BFGS也没有全局最优证书，epsilon混合softmax不能直接套用普通线性logit交叉熵的凸性结论。

对全16条D轨迹按第2节精确规则做的只读重放见[NEXT_SCALAR_REPLAY_20261023.json](NEXT_SCALAR_REPLAY_20261023.json)。该附录保存48份输入JSON的路径、大小、SHA和逐prefix标量/哈希；没有读取模型字节、恢复系数或调用数值策略。

- **11/16的无closure过滤最小完整目标点，与原过滤后最佳float64点的哈希完全相同**，不是仅目标近似相等。原16次D均无float32 fallback，故这11项的三个最终诊断结果可精确复用原D评分。
- 另5项的无过滤点与原点不同，完整目标严格更低；差值如下。其参数未保存，不能从哈希恢复float32或编码留出表现。

| Seed | 无过滤点closure index | 原过滤最佳L − 无过滤最佳L |
| --- | ---: | ---: |
| 26204 | 71 | 1.2596972796598378 |
| 26206 | 69 | 0.20660957205153935 |
| 26209 | 67 | 0.40643893528740932 |
| 26212 | 68 | 0.22628735431808283 |
| 26213 | 75 | 0.59840403933394826 |

两条辅助定位足以说明为什么不能把问题归为单一gate故障：7/16的C0新训练界比“保留C训练收益一半”的对应界更严格；6/16存在被gate拒绝、完整L低于C0且满足训练侧旧/新比率的点。但是这些点不是下面无过滤完整L最小点的同义词，也没有其留出表现。本方案不额外选择这些点。

对**已经选定的原C/D**，训练对应条件与留出条件的交叉计数为：训练通过→留出通过7、训练通过→留出不通过4、训练不通过→留出通过1、两者不通过4；编码前留出和最终编码留出在这个2×2表中相同。这已显示所选候选存在训练—留出失配，无须重拟合才能再确认。以上不是新成功指标，也不重做原失败分类。

## 2. 唯一搜索路径与精确选点

继续沿用原teacher、各128行/96训练32留出、owner等权、C0、λ=1、单次ridge、L-BFGS设置与最多256次closure。记训练目标：

`L(C)=K_new^tr(C)+K_old^tr(C)+1e-4·mean((C−M)^2)`。

`C0=float64(float32(M))`；初始化另作index=0，不算closure。按closure index 1…m处理**全部有限**目标评估点，包括line-search试探点；不只保留optimizer最终接受步。非有限是技术错误，不能跳过。

两种incumbent均从C0及其L开始，按相同顺序更新，相等均用后点替代，排序不加新容差：

| 名称 | 可进入集合 | 更新规则 |
| --- | --- | --- |
| F64 | C0及满足 `K_new^tr(C)<=K_new^tr(C0)+1e-12` 的closure点 | `L<=当前F的L`时更新 |
| U64 | C0及全部有限closure点 | `L<=当前U的L`时更新 |

U只做影子记账，不改变参数、梯度、目标、line search、停止、F或RNG；不是另一条优化。现存标量足以确定这两个点的索引、L、训练KL和哈希，不能确定未保存的U系数。全部16条的第一条closure与初始化L精确相等；按`<=`处理并记录index=1，不以loss相等替代候选身份核验。

## 3. 分开两个gate的三个有限诊断结果

定义 `Q(X)=float32(X)`。令 `B(Y)` 表示按原函数计算的 `K_new^tr(float64(Y))<=K_new^tr(C0)+1e-12`。B不计算留出或编码指标。

| 标签 | float64选择 | float32后处理 | 识别的比较 |
| --- | --- | --- | --- |
| V | F64 | Q(F64)通过B则保留，否则回Q(C0) | 原v1控制 |
| UG | U64 | Q(U64)通过B则保留，否则回Q(C0) | UG−V：**保持转后gate开启**时，closure接纳变化的作用 |
| UR | U64 | 直接保留有限Q(U64)，只记录B结果，不执行该回退 | UR−UG：**无closure过滤选择下**，转后gate的作用 |

UR是诊断探针，不是获准部署的新算法。UR−V包含两个处理，禁止归因于单一gate。本方案不估计完整2×2交互，不增加第四个臂；不改float32前后的完整目标排序，也不按留出、编码结果或原gate通过数回退。

三个标签最多对应三个不同float32矩阵，重复点按精确字节及相同prepared/绑定/科学版本去重评分。所有点使用原wildcard、补偿、**完整actor** int8编码和fresh local=0评价；每次从相同prefix独立副本开始。resident漂移另报。每个实际评分包都核实actor89 B、完整持久包≤262144 B。

## 4. 为什么最多5次拟合，分析仍覆盖全部16条

恢复集合R由**全16条训练轨迹的U/F身份不同且U参数缺失**确定，固定为26204、26206、26209、26212、26213。它不是按C0标记、留出失败或反事实收益挑选的子样本；恰好与原5个D返回C0重合不作为选择规则。

其余11项已有U64=F64的哈希证明，且原转后gate未回退，因此V=UG=UR的最终float32字节和评分相同，可引用原D的候选/部署/metrics身份。它们仍是全16个配对单位，不能只报告恢复的5项。C一律准确引用原λ=0固定评分，不增加λ=0拟合。

对R仅恢复各一次原λ=1路径，同时保留F64和U64。必须先验证：输入/配置/依赖身份；新轨迹的closure数、index、有序候选哈希和原有完整L/newKL/oldKL/ridge标量与E163对应轨迹精确相符，F64/F32候选身份一致。CPU/wall、内存、额外日志不要求相同。任何不匹配均为`REPLAY_MISMATCH`，在新反事实留出评分前停止整批；不临时设容差、补跑或扩为16次恢复。

之后先完成全部R的训练选择、float32处理和**全16条V/UG/UR锁定清单**，写出清单SHA与事件，才开始新增候选的holdout评分。准备中可生成固定教师；搜索只接收训练行，不能先运行原fit的初始/最终holdout诊断。原11项的已知留出作为引用，不能假称人或数据从未见过；程序的恢复与选点函数不得接收这些值。

R的V逐项重新评分作为编码/评分锚点，核对原actor字节及fresh训练/留出值；有不符停止。UG/UR仅按已锁定字节评分，相同字节复用同一结果。上界为**5次fit、15个独立评分包**；没有全closure评分，没有奖励、主动探测、新seed或参数搜索。新实验若只执行5次恢复，报告须同时区分“5次实际新计算”和“16条逻辑配对”；复用项不冒称新测量。

## 5. 输入、API和源码范围（接受后实施）

现阶段仅本文与标量附录。建议后续全部新增文件，原工作模块和`delivery_v1`都保持不动：

| 文件 | 精确责任 |
| --- | --- |
| `src/online_rl/retention_acceptance_replay.py` | 标准库纯标量重放、全16身份与复用/恢复清单；不import模型或读取checkpoint |
| `src/online_rl/retention_acceptance_diagnostic.py` | training-only view、原训练搜索的双incumbent记账、float32处理、锁定及调用原评分器 |
| `configs/h036_acceptance_diagnostic_v1.json` | 固定全16、R、F/U/B语义、原v1及标量引用身份；不新增λ或优化超参 |
| `tests/test_retention_acceptance_diagnostic.py` | 第7节必要合成检查 |
| H036附录中的`METHOD_ACCEPTANCE_V1.md`、`HANDOFF_ACCEPTANCE_V1.md`、`delivery_acceptance_v1/` | 被接受方法、精确调用、新冻结源/dirty patch/哈希和检查；不覆盖旧交付 |

API规格如下，实验runner不补科学语义：

1. `replay_existing_trace(initial_train, closure_records, recorded_best64, original_cast_status) -> ReplayDecision`：纯输入值，按第2节返回F/U索引、训练量/哈希、复用资格；不接收holdout、原gate结果或C0布尔标记作选择。
2. `make_training_view(prepared) -> TrainingView`：从原准备接口复制new192行与old96行的features/教师、owner权重、M/C0及prepared/输入身份；不含holdout观测、教师或性能。原三文件SHA、代次、绑定与行序检查全部沿用。它不评估学生holdout。
3. `fit_selection_trace(training_view) -> SelectionTrace`：λ固定1；复用原目标/梯度和数值helper，在新模块中严格保持原训练loop、返回梯度/目标与预算处理，只加入U旁路记录。保存原有全部closure**标量和哈希**，最终仅保留C0、F64、U64系数与来源index，不保留整条系数轨迹。禁止monkeypatch原模块或torch全局对象。
4. `lock_acceptance_probes(training_view, trace) -> LockedProbes`：仅训练输入；按第3节产生V/UG/UR、实际cast检查/回退与别名清单。候选锁定后不可修改；trace与probe列表分别绑定SHA。
5. `encode_and_score_locked_probe(prepared, locked_probe) -> score`：检查prepared/trace/候选身份；以明确的diagnostic-only候选adapter调用原`encode_and_score_stage0(..., binding='wildcard')`。adapter只声明candidate、λ=1、prepared_id、候选哈希与选择来源，不伪造原fit的优化诊断。原评分器/损失/teacher/codec不改。

新训练loop与原fit的差异仅是省去选择前的holdout诊断和增加U记账；训练数值调用顺序、初始化和梯度返回必须合成逐点一致。这样不需要给旧科学模块增加回调，也避免原fit在锁定前计算holdout。新模块不是新的生产保存器。

### 必须交付的数据字段

- `SelectionTrace`：seed、输入/prepared/原版本身份、λ、C0、F64/U64各自origin index、fullL/newKL/oldKL/ridge、完整候选哈希、有序closure记录与path摘要、终止/预算状态、真实调用/资源计数。`best`只指相应已访问集合的最小L。
- 仅C0/F64/U64保存3×3 C-order little-endian float64原字节（每点72 B），probe保存同布局float32原字节（每点36 B）。记录dtype/shape/字节序、文件SHA及原helper语义的候选哈希，两种哈希不混用。相同点可用准确引用去重。
- `LockedProbes`：每标签选择规则、float64来源/path/index、原始cast哈希、新训练KL/B判据、是否回C0、最终float32哈希、别名及`diagnostic_only=true`。实验方保存全16锁定清单后才调用新holdout评分。
- 评分原样返回pre-codec/encoded/W0的train/holdout、fresh/resident、容量、原部署bytes与资源；数据与候选身份贯穿。所有teacher和影子状态只存诊断工件，不进入learner。

输入白名单为：已哈希的E163标量JSON、原C/D候选/评分/部署准确引用；R对应的三份原prefix文件及原foundation准确引用，仅在新准入后用于恢复与完整包核验。observer、simulator、未来日程/reward、完整历史日志、淘汰source参数均不进入算法。源、输入位置可搬迁，SHA不变；真实prefix反序列化/新策略评价都属于将来付费计算，不能称为免费读取。

## 6. 要报告什么、什么结果改变决定

配对单位始终是prefix，不把closure或多个probe当独立样本。原E163的12/16不自动成为新实验成功线；本诊断没有算法升级或奖励准入开关。

**主要新诊断**为第3节两个相邻受控比较：UG对V、UR对UG的全16配对表，分别报告改善/退化/不变数，以及旧KL和新收益的绝对差。可按原四个holdout条件记录每个锁定点的“相同条件下是否通过”，但这些是新反事实诊断标签，不重写E163结果、不合并成新“最佳臂”。

| 观察 | 支持／削弱范围 | 下一决定 |
| --- | --- | --- |
| UG相对V出现新通过点 | 在保留转后gate时，closure接纳影响了该轨迹的可交付折中；同时报告丢失的原通过点 | 交主控评估是否值得登记新的训练选择规则；不直接部署或奖励验证 |
| UR有新通过点，UG没有对应改善 | 必须放开转后gate才呈现该点；不能称只改closure有效 | 先评审两个约束的明确方法合同，不能暗中把两项合成单一修复 |
| U32在编码前训练条件成立、相同点的编码前holdout条件不成立 | 这个训练选点存在迁移失配；不需以codec解释它 | 单纯去gate不足以解决该点，不自动加容量或新损失 |
| 同一锁定点pre-codec与encoded结果改变 | 该点的codec效应，可与迁移分别定位 | 只登记该点证据，不因此自动改codec |
| UR/UG均无新增通过点，或收益伴随新的失败 | 削弱“释放这条完整L最小点即可修复v1”的理由／显示取舍 | 停止这套有限探针，交主控决定是否另问目标/数据问题，不继续补候选 |
| 重放、输入、有限性、容量、时序或计量错误 | 无法回答科学问题 | INVALID/INCOMPLETE，保留成本与部分证据，不免费重试 |

同时以 `G^s=K_new,W0^s−K_new^s`（s=train或holdout）分别计算训练和留出对应条件，训练旧界`.75*K_old,C^tr`、新界`.5*(K_new,W0^tr+K_new,C^tr)`；原数值guard仍1e-10。比较必须在同一精度/编码阶段进行，不把float64 closure训练值直接与encoded留出差当成纯泛化效应。

已知6条轨迹上的其他训练可行点，本方案不恢复或评价；它们的编码留出仍UNKNOWN。因此即使U失败，也不能声称整条轨迹没有可交付折中，更不能否定函数类。也不比较另一种搜索器、投影/约束优化或全局可行性。若未来要把UG/UR发展为独立算法假说，由主控先分配H并登记；当前仅是H036诊断。

## 7. 合成检查、成本与准入

接受方向后只做必要合成检查：

1. 新training-only loop的F与原fit在λ=1的closure哈希/标量序列、F64、F32、终止及合法预算耗尽一致；原λ=0接口不改。覆盖精确平局、C0、非有限错误；记录不反馈搜索或消耗RNG。
2. 人工轨迹分别产生三种标签相同、仅closure选择不同、U32被转后gate回退的情况；禁止把两gate合并。检验字节/dtype/索引/摘要与别名。
3. 改变holdout、原E gate通过标记或future字段不能改变恢复集合、F/U系数字节、float32处理或停止；输入provenance身份应如实随原数据变化，不要求其哈希不变。training-view/选点API拒绝这些评价字段。
4. 复用11项的证明与5项恢复清单按哈希而非L近似相等；锁定后修改候选或跨prepared调用必须拒绝。对一个合成锁定点复用原完整评分链即可，不重复旧23项/1876项审计。

结构不增加：仍一个九系数父、原local、绑定、buffer和优化器状态；持久learner增量0 B。每条恢复额外最终float64系数最多216 B、probe float32最多108 B；连同影子副本/元数据按真实内存计量。标量日志≤5×256条，不持久化完整系数轨迹。5条prepared可为“先全锁定、后评分”临时保留，或按固定输入重新准备并核对身份；不以省内存改变输入。

历史16次D的fit分项CPU合计约0.5571 s，其中R五次约0.1714 s，**均不含整个启动/计量/评分/IO**。E163完整科学worker约7.79 CPU s可作冷启动参照。下一方案为5次恢复、最多15次完整评分/序列化，规划估计10–25 CPU s、15–45墙钟s，属于未实测估计；建议主控评审独立单次60 CPU s/120墙钟s、单worker单数值线程/0GPU/0新交互、64 MiB新增工件/cache/temp的规划上限，**不是现有授权**。如验证方静态估计不能覆盖全部启动/预检/失败/收尾，先退回预算，不先试跑。

15个包按cap上界约3.75 MiB；完整标量/源/清单及传输/归档双份计入峰值，64 MiB仍需实验方正式逐项估算。参考进程生命周期峰值约348 MiB，规划应预留约0.5 GiB进程内存并实测，不能把裸系数字节当真实峰值。所有准备、fit、cast复查、评分、序列化、日志及出错分项计数，不把共享或复用称为无成本。

主控当前只需决定：**是否接受这套“全16逻辑配对、5次缺失参数恢复、三个gate分解探针”的范围及新增文件实现。** 接受后算法方实施、完成上述合成检查、交METHOD/HANDOFF和新冻结包；实验方独立审查，主控再登记新E和预算及正式准入。新远端目录由新E确定，仅robot既有项目精确白名单；当前没有SSH、传输、真实fit、计量campaign或E163余额复用。
