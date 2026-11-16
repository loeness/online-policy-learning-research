# H036 接纳诊断：方法规格 v1

日期：2026-10-23（Asia/Singapore）。依据[已接受设计](NEXT_DESIGN_20261023.md)及[实现合同](COORDINATOR_ACCEPTANCE_IMPLEMENTATION_20261023.md)。本方法用于E164探索性诊断；实现/合成检查不构成真实运行授权。E163结论、原v1及其冻结包保留。

## 数据、搜索与选择

真实恢复输入仅为原16项中26204/26206/26209/26212/26213的三份prefix文件，必须匹配冻结清单。原`prepare_inputs`负责三文件SHA、foundation身份、代次、owner、绑定、顺序与96/32检查。原teacher、W0/source_basis、temperature=2、epsilon=.05、wildcard、codec与容量均不变。

`make_training_view(prepared)`复制新组两owner各96训练行、旧组96训练行、相应features/教师/权重、M/C0及来源身份，返回冻结`TrainingView`。它不携带holdout观测/标签/性能；原始输入哈希可绑定包括留出在内的数据身份，不能用作选择特征。训练数组只读，运行前核验其摘要；不调用环境、PPO或RNG。

`fit_selection_trace(view)`固定lambda=1，目标为原`protected_objective_and_gradient`：新KL对owner等权、旧KL均值、ridge=`1e-4 mean((C−M)^2)`只计一次。C0=`float64(float32(M))`。L-BFGS原设置：lr1、history10、64iter、256closure、strong_wolfe、grad_tol1e-9、change_tol1e-12。初始化调用独立计量，非有限目标/梯度/KL为错误；只有原closure预算耗尽允许返回已见最佳。

新模块复用原目标、梯度和数值helper，保持原训练loop的点、调用顺序与返回值；省去原fit中选择前的holdout诊断。每个closure先计算原完整L/梯度和训练KL，再分别记账，最后向优化器返回原L/梯度。U不反馈参数、line search、停止、F或RNG。

- F64：以C0初始化；点须 `newKL<=newKL(C0)+1e-12`，且 `fullL<=当前F.fullL` 才更新。
- U64：同初值；所有有限closure点中 `fullL<=当前U.fullL` 时更新。
- 精确平局取后点，不加排序容差。包含line-search试探评估，不能只取optimizer接受步。C0为index0，closure从1连续编号。

所有原closure训练标量/哈希保留，最终仅存C0/F64/U64系数。它们是在对应有限访问集合中的最佳点，不是全局解；没有保存的其他点不补造表现。

## 两道gate及三个探针

`lock_acceptance_probes(view, trace)`先验证trace完整性、来源及训练数据，再产生：

| 标签 | float64来源 | 转float32后 |
| --- | --- | --- |
| V | F64 | 原新训练KL复查不通过则回C0 |
| UG | U64 | 相同原新训练KL复查，不通过则回C0 |
| UR | U64 | 保留有限cast结果，仅记录复查是否通过 |

复查始终使用cast后float32矩阵转float64及原训练函数，界为`newKL(C0)+1e-12`，不使用holdout或编码评分。UG−V隔离保持转后gate时的closure接纳变化；UR−UG隔离固定U选择后的转后训练KL约束，**不是纯舍入效应**，U64本身也可能越界。UR−V包含两项处理，不估计完整2×2交互。

每个探针记录原始cast、B结果/回退、实际候选来源index、最终科学helper哈希与原字节SHA。相同最终字节为评分别名；不以loss近似相等合并。锁定摘要覆盖全部候选内容与选择元数据。C0/F64/U64用C-order little-endian float64 `(3,3)`，每点72 B；探针为同布局float32，每点36 B。科学helper哈希为`sha256(dtype.str + NUL + repr(shape) + NUL + raw_C_bytes)`；文件SHA仅覆盖原字节，两者不互换。

## 标量重放与完整身份复用

`retention_acceptance_replay`仅用标准库处理训练标量与身份；不import模型依赖。它按相同顺序/精确平局规则重放F/U，检查记录的原F、无转后回退及最终float32身份。

全16逻辑配对固定为26201–26213、26215、26216、26218。R由训练侧F/U身份不同且U系数缺失确定，固定为上述5项；其余11项同一F/U64及原无fallback，故V/UG/UR最终矩阵、评分均准确复用原D。C全部复用原lambda0评分。不按留出通过、C0标记或新反事实结果选R，也不将5项当全体。

[ACCEPTANCE_BINDINGS_V1.json](ACCEPTANCE_BINDINGS_V1.json)逐项绑定：原prepared_id与三文件/foundation；原科学source identity与依赖单文件SHA；C/D候选JSON、36 B原候选、科学helper/原文件SHA；metrics的prepared/lambda/wildcard/candidate身份；原89 B actor、checkpoint、evidence及持久引用；run002 provenance与镜像。该文件保存准确引用，未复制历史模型。路径是放置来源，key/SHA/bytes绑定文件；搬迁时由验证方显式解析清单，不能重新计算后把当前值当expected。

新冻结包的总source identity会改变，所有旧科学数值/评分依赖必须仍与v1逐文件一致。真实执行时对允许引用文件再次按显式映射核对字节，不能把本机绑定检查当远端已通过。

## 恢复匹配、全体锁定与锚点屏障

`assert_replay_matches(expected_diagnostics, actual_trace)`核对C0、closure数量/顺序/index、逐点完整L/newKL/oldKL/ridge/候选hash、原F64、cast/F32/回退、iterations及termination。已有日志未保存逐点gradient，不能声称历史梯度逐点比对；梯度路径靠共享helper身份和合成parity支撑。任一匹配失败为`ReplayMismatchError`，整批停止，不加容差、补跑或扩为16恢复。

`build_recovery_manifest`与`lock_cohort`形成全16×V/UG/UR的JSON锁定清单，其中11项引用原D，5项绑定新不可变锁。全体锁定摘要保存后，评分按两个阶段进行：

1. `make_anchor_permit(cohort)`仅允许R中的V。先完成全部5个V锚点，保存新score和原deployment bytes。
2. `confirm_anchors`核对五个V与旧D准确metrics字节引用、prepared/candidate/绑定、原actor字节及fresh pre-codec/encoded/W0训练/留出字段。CPU/resources及整个新metrics文件SHA不要求复现；新包仍独立核实实际大小。五项全部通过后才生成counterfactual permit，允许UG/UR。

`encode_and_score_locked_probe`首先检查锁和真实候选字节、prepared及stage permit，拒绝跨锁、篡改、缺全体锁或缺锚点阶段。阶段摘要是防误编排屏障，**不授予资源或远端权限**。runner另负责准入、一次性任务计数及max5fit/max15评分的限制。

## 原科学评分与部署字节

适配器传给原`encode_and_score_stage0`的envelope精确为：`format='h036_fit_v1'`、匹配`prepared_id`、`lambda_value=1`、有限float32 `(3,3)` candidate，以及`diagnostics.candidate_sha256`。没有伪造原fit的optimizer统计。`diagnostic_only`、V/UG/UR、trace/锁、gate与选点来源单独返回。

原评分器在独立工作副本执行补偿安装→完整2local/1shared actor的89 B int8编码→解码→fresh local=0的train/holdout KL；resident漂移另算。实际包为foundation+实际torch ZIP checkpoint+actor+原evidence，硬上限262144 B。teacher、影子点、trace和诊断不进入learner，持久学习结构增量0 B。

原返回`deployment.actor_bytes/checkpoint_bytes/evidence_bytes`若保存须逐字节写入；checkpoint不能再torch.save。不同探针不能继承前一评分的状态。相同最终候选/同prepared与绑定可引用同一评分；仍保留三个标签及别名。

## 计数、失败与解释

报告准备、训练搜索、初始化、closure、cast复查、锁定、评分、序列化和IO/统计的实际分项及整个campaign。lambda1每完整目标调用内部有两个objective helper调用；closure数不等于objective次数。记录临时teacher/optimizer/候选数组字节及进程生命周期峰值的不同口径；方法内计时不能代替完整生命周期计量。

正常完成的单条新trace若有m次closure，objective helper调用为`2m+4`（初始化2、closure每次2、最终2），训练KL诊断调用为`2m+5`（初始化2、closure每次2、F转后1、最终2）；没有holdout诊断调用。随后一次`lock_acceptance_probes`对V/UG/UR各做一次cast训练KL复查，共3次，当前没有把UG/UR相同来源的计算假记为共享。每个实际完整评分沿原接口另计6次prior概率遍历、1次resident概率遍历及7次KL行遍历；完整序列化也计成本。锁/permit的SHA和JSON校验另由runner计时，不能漏在campaign之外。

合法C0、closure预算返回及转后回退保留；非有限、输入/轨迹/锁定/锚点/容量/阶段错误整批停止，保留部分证据及成本，不当作机制阴性，不自动重试。原12/16不是本诊断的新成功开关。主要报告全16的UG−V、UR−UG配对改善/退化/不变，以及旧KL、新收益绝对差；同精度/编码阶段比较训练与留出，不用holdout选点。未保存的其他训练可行点仍UNKNOWN，无新增通过不能否定轨迹/函数类兼容性。

当前仅本地受限实现与合成检查。新E164的60CPU/120wall/64MiB等仍是规划；冻结接受、独立审查、资源授权及正式准入完成前，不进行真实恢复/评分、prefix反序列化、SSH、传输或计量预检。
