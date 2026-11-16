# E164 · H036 同轨迹接纳与float32复查诊断

- 预留日期：2026-10-23（Asia/Singapore）；主H：[H036](../../research/records/H036.md)。
- 状态：**COMPLETE_VALID / EXPLORATORY_ONLY**；单次验证及独立审计完成，额度关闭，无后续执行授权。
- 依据：[精确设计](../../research/appendix/H036/NEXT_DESIGN_20261023.md)、[主控接受及实现范围](../../research/appendix/H036/COORDINATOR_ACCEPTANCE_IMPLEMENTATION_20261023.md)、[独立预审](../../research/appendix/H036/NEXT_REVIEW_20261023.md)。

## 问题、受控比较与证据

检验在同一原lambda=1搜索轨迹上，closure接纳和float32转后新训练KL复查分别怎样改变交付候选；再描述训练与留出差异。不是新生产规则或奖励实验，不改变E163的8/16、12/16门槛、失败、数据或停止决定。

固定原初始化、目标、梯度、优化器/closure上限、数据/教师、绑定及codec，记录F64/U64。V使用原过滤及转后复查；UG使用U64但保留转后复查；UR使用同一U64，不按转后复查回C0，仅记录其结果。主要是全16个prefix的UG−V、UR−UG配对表，报告旧KL/新收益的绝对变化及改善/退化/不变数。UR−V包含两个处理，不作单一归因。不以closure或探针作为独立seed。

候选完全依训练规则确定并锁定，随后按同一teacher/split/精度/编码阶段评价；禁止holdout选择、原reward标签、事后最佳臂或临时新增候选。原条件下的探针通过标签仅作探索性诊断，**不自动继承12/16作为新升级开关**。本E没有直接奖励验证/部署准入。

全16逻辑配对中，11项依完整float64 hash相同、原无回退和完整prepared/输入/评分/部署身份复用原D；C全部引用原固定评分。只对26204/26206/26209/26212/26213各恢复一次原lambda=1轨迹，最多5次fit、15个新增评分包。恢复集合按缺失候选身份确定，非按留出好坏选样本；所有16项保留。原标量分析已见数据，标记为设计支持的结果后分析，不追称独立确认。

## 预定停止与决定

- 恢复轨迹的closure顺序/数目、候选hash、训练标量或原F身份任何不匹配，在新增反事实留出评分前整批REPLAY_MISMATCH并保存，不换容差、不补跑或扩至16次。
- 全部V/UG/UR锁定清单写出后才新评分；原V锚点评分与部署身份须匹配。输入、代次、有限性、容量、锁定/防泄漏或计量无效时INVALID/INCOMPLETE，不当作机制阴性。
- 有效有限计划完成即停止；合法C0、转后回退和预算返回按冻结语义保留，不能删样本。
- 相邻比较出现新增通过同时报告丢失通过、完整旧/新变化及训练/留出差异，只支持考虑下一方法合同。无新增改善或显著取舍则削弱释放这两个有限选点可修复v1的理由；不推出整条轨迹或函数类无兼容解。
- 无自动新fit、lambda搜索、新seed、结构扩展、StageA或下一阶段。

## 实现与独立验证准入

算法实现拥有新增诊断模块、必要合成检查、METHOD/HANDOFF及delivery_acceptance_v1；原v1源码/快照受保护。消融实验只在主控接受冻结实现、独立审查通过、具体预算与精确执行/输入白名单齐备后运行。真实模型输入不用于工程试跑。

新源码、配置、48份标量引用、旧候选/评分/部署、5份prefix三文件及foundation等按需逐项manifest绑定。精确SHA/命令/cwd/env、远端输入allowlist和部署包引用待冻结交付/独立编排补齐；任何缺项不得从本草案猜测开跑。

规划后续SSH工作在robot既有项目内的新`/data/robot/p001/e164`，读取E161/E163必要固定文件仅按准确清单；当前不授权创建远端目录、传输、预检或运行。继续排除本机指令、凭据、Git历史和无关旧工件，不访问同事资料/进程、不sudo或改系统。

## 预算与工件

唯一账本：[budget.json](budget.json)。建议单次60 CPU秒/120campaign墙钟秒、0交互、0GPU、1 worker/数值线程、64 MiB新增工件/cache/temp，项目总量100 GB；当前**PLANNED_ONLY / execution_authorized=false**，没有reservation。E163额度不复用。候选恢复/评分、预检、启动、加载、序列化、失败与收尾都必须计入新campaign；运维单列，观测值与footer上界分开。

原始输出和分析将分别新建runs/run001、analyses/analysis001；当前NOT_CREATED/NOT_RUN。保存选点字节及hash、训练轨迹、锁定事件、评分/原deployment bytes与真实包大小、计量/PID/wait4和原始失败证据。无需预先生成图；配对表足以回答本E。

## 2026-10-23 科学实现冻结接受

主控接受[方法](../../research/appendix/H036/METHOD_ACCEPTANCE_V1.md)、[交接](../../research/appendix/H036/HANDOFF_ACCEPTANCE_V1.md)和[冻结包](../../research/appendix/H036/delivery_acceptance_v1/manifest.json)。source identity为`baae9a6a766518105a6d5b2e9bc64fef3488ff581870931cf83ff2d7b23a4c9a`，manifest SHA为`76073211ec4ff37af2677e0d3cae33c87e46a703a25c306ccb0257c03a2ef379`；27项源码/配置/记录共762031 B精确核验，16项原科学依赖仍与v1一致。冻结副本9项必要合成检查通过，15个项目模块全部从冻结src导入；未重跑旧23/1876审计。

完整16prefix/32原C/D的输入、候选、评分及部署元数据链核对通过；新源码快照排除本机指令、Git/凭据及历史模型原始副本。详见[主控接受记录](preparation/coordinator_acceptance_v1.json)。这只接受工程实现与身份，不是新的机制证据。

消融实验正在把本地runner/report/独立验证映射到该冻结API并完成定向审查。E164仍0真实恢复/新评分/新交互，无SSH/传输；60CPU/120wall/64MiB仍规划，资源与执行开关均false。具体执行包可审阅、资源授权及正式准入完成后才运行。

## 2026-10-23 额度中断后恢复与本地执行包接受

用户直接要求恢复两个对话的合作验证。主控接受[最终执行包](preparation/validation/READY_FOR_RESOURCE_DECISION.json)：23项准备工件、40项传输文件/1177074 B的身份复核通过，orchestration identity为`8b2cf8c5a45c009a4955cac9303eb2a2d810f73944c4a0bdc285b9f3021dd3e0`；科学冻结不变。155项静态检查、5组人工JSON工程检查通过，未重跑9项科学检查或旧1876审计，未作真实拟合/评分。

在该继续验证请求内，主控为一个新E164 attempt分配60 CPU秒/120campaign墙钟秒/64 MiB、单worker单数值线程、0GPU/0新交互、最多5fit/15完整评分包；用户没有另行逐项报出这些数值，数值为主控采用的已审查有限包络。登记`acceptance_diag_campaign001`，不结转E163。计量范围为fresh supervisor出生至启动、同campaign预检、worker工作与收尾；55CPU/110wall停止并保留5CPU/10wall及65536 B。SSH、父shell、传输和容量准入属于外部运维，elapsed分列，未测CPU为UNKNOWN。

[准备指令](preparation/coordinator_prepare001.json)仅准许robot在既有项目范围内作精确路径/容量/输入准入和白名单传输，新写入只在`/data/robot/p001/e164`。动态容量、inode、项目用量和spec基线须实测；当前预算resource_authorized=true、execution_authorized=false。验证方返回重绑定spec与传输身份后，主控才发单独的运行指令；现在不能启动campaign预检、模型准备、fit或score。无自动重试或后续阶段。

后续远端准入完成：40/40传输与308/308引用的路径/字节/SHA匹配，实测项目约2.92GB、目标可用约956.7GB，空间/inode相容。runtime spec仅三个准入字段改变，其余相同，新SHA为`40a1ac53ec03b2ac0eb0cea6004f962fccdc3d342490d2b67eef1b698f2dd303`。主控发布[单次启动指令](preparation/coordinator_start001.json)，resource/execution均true；55CPU/110wall停止与既定footer保留不变，同campaign预检须先于科学worker。尚无结果，不自动重试或进入下一阶段。

## 2026-10-23 最终结果与主控决定

[最终交付](FINAL_VALIDATION.json)、[统计报告](analyses/a001/report.md)、[独立审计](analyses/a003/audit.json)、[成本](analyses/a001/cost.json)和[主控集成分析](../../research/appendix/H036/COORDINATOR_E164_SYNTHESIS_20261023.md)已完成。5次fit均精确恢复，16逻辑配对先锁定，5个V锚点通过，实际10个新评分包；279原始文件/5213330 B逐SHA镜像，0新交互、0GPU。256配对指标及汇总独立重算一致，最终审计COMPLETE_VALID、errors=[]。

UG−V在所有视图/划分的全16项均精确为零：五个恢复项UG未过转后新训练KL复查，回退到V的相同字节。UR−UG的编码后旧留出KL均值−0.221881294348，5改善/0退化/11不变；新留出G_new均值−0.016072061517，2改善/3退化/11不变。新训练G_new在恢复五项均退化。这支持旧保护与新拟合的有限权衡诊断，不建立奖励收益、统一改善或生产规则。

观测小计7.929755 CPU秒、footer前9.350686747秒墙钟；含预留收尾的保守上界12.929755/19.350686747，预留不作实耗，精确完整footer及外部运维CPU仍UNKNOWN。单次reservation已CLOSED_COMPLETED_VALID_NO_RETRY，active job/reservation均空、resource/execution均关闭，剩余额度退役。远端E164共8956426 B、项目总量2926012562 B，SSH和项目作业已退出。

原冻结核验器的字段/映射错误经主控审阅后在独立a002/a003副本纯离线修正；没有重跑campaign或改科学源码/结果/阈值。原INVALID/INCOMPLETE与所有版本链保留，详见audit及coordinator_audit_correction001/002/003。H036保持INCONCLUSIVE；本E结束，不启动StageA、新seed、lambda扫描或结构扩展。下一问题只记录为训练接纳与留出共同收益如何协调，尚无新方法/E/额度。

[算法侧分析](../../research/appendix/H036/ALGORITHM_E164_ANALYSIS_20261023.md)已完成并获主控验收。验证交付、主控分析、算法分析三项齐备；按用户最新规则，本轮到此停止，下一轮实验验证等待用户新指令。
