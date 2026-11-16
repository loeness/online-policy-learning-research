# E163 · H036 编码后旧/新行为兼容性筛查

- 注册：2026-10-22（Asia/Singapore）；主 H：[H036](../../research/records/H036.md)。
- 当前状态：**COMPLETED_NO_ADVANCEMENT**；run002完整有效，32次拟合、8/16通过，未达预设12/16；run001保留ACCOUNTING_INVALID。
- 当前执行位置：用户于2026-10-22指定 SSH，robot 账号、既有 `/data/robot/p001` 项目内的 `/data/robot/p001/e163`；方法与数值预算不变。
- [最终报告](analyses/a002/report.md)、[独立审计](analyses/a002/audit.json)、[逐prefix表](analyses/a002/per_prefix.csv)已完成。两次单次额度均关闭；累计完整CPU/墙钟因run001仍UNKNOWN，无自动后续重试或Stage A授权。
- 主控合同：[COORDINATOR_CONTRACT_20261022.md](../../research/appendix/H036/COORDINATOR_CONTRACT_20261022.md)；来源设计：[提案](../../research/appendix/H036/DESIGN_AND_VALIDATION_PROPOSAL_20261022.md)。

## 问题、方法与控制

问题：同一个九系数、89 B actor 的父模块，加入旧 keep 初始化先验行为项后，能否在编码后减少旧 holdout KL，同时保留部分新来源拟合收益？这是奖励筛查之前的必要机制检查，不验证未来奖励或一般表示兼容性。

固定 lambda=0 的候选 C 对固定 lambda=1 的候选 D；两者均 wildcard，复用相同旧/新行、教师、初始化、正则、优化上限、补偿、原 codec 和完整持久布局。B 与 C 共用 lambda=0 候选；本 E 不执行 A/B/C/D 的在线分支。具体目标、128=96+32 时序划分、回退与实现要求以主控合同和随后接受的 METHOD_V1 为准。

固定 16 prefix：26201–26213、26215、26216、26218。每个 C/D 各拟合一次，共32次；新增环境交互0。算法只用 E161 prefix 的 checkpoint.pt、actor.bin、evidence.npz 与固定配置，不读取未来标签、原完整日志或已淘汰 source 参数。输入不足/身份错误整批停止，不删 seed 或补行。

## 预设主要指标与决定

评价学生为完整 actor 安装/编码/解码后的 parent，在 fresh slot local=0 时的策略。旧教师为 P_old 的 keep 初始化策略，新教师为两当前 owner 的有效策略；新 KL 两 owner 等权。

令 G_new,a=K_new,W0−K_new,a。每 prefix 同时满足 K_old,C>1e-10、G_new,C>1e-10、K_old,D<=.75 K_old,C、G_new,D>=.5 G_new,C 才通过。**至少12/16**通过为 MECHANISM_SCREEN_PASSED_EXPLORATORY；其余有效完整结果 NO_ADVANCEMENT。无效或未完成结果另报 INVALID/INCOMPLETE。

近零/无新改善的 prefix 计不通过，不改分母。不使用 holdout 选参/挑候选，不以训练或编码前 KL 挽救结果。通过只支持提出在线奖励筛查设计；失败停止这个固定 lambda/拟合版本，不扫 lambda、不更换编码、不推断函数类不可能兼容。完整报告各 prefix 及失败、回退和成本。

K_new,W0 是同一新holdout、相同温度/epsilon下local=0且无父的W0学生KL。1e-10仅为数值guard；绝对KL与增益完整报告，通过比例门槛不建立实用量级。合法C0/预算耗尽回退按固定门槛评分并显式标记，不重跑或删除；若由回退通过，不归因于已学到的保护更新。编码评分的科学接口由算法实现冻结：完整actor编码bytes→解码父→评价副本local清零→冻结教师KL，不由实验方猜实现。

## 所有权与执行前置条件

算法实现负责新的 fitter、输入/编码评价科学接口、合成正确性检查和冻结源码/配置交付。消融实验（1）负责本 E 内的独立有效性审查、runner、统计、计量和证据；不能在执行时补全或改算法语义。主控接受冻结版本后才更新执行状态；登记本 E 不构成当前执行授权。

执行前：逐项绑定代码/输入身份、检查旧/新数据和代次、核实 C/D 公平性与89 B/262144 B上限、冻结精确 argv/cwd/独立输出路径。以零模型预检真实解释器及子进程生命周期 CPU 的覆盖和限额处理；这项执行预检计入本 E，不用包内阶段时长替代完整成本。

## 资源边界

首次单次额度：既有服务器项目内单 worker、Torch/OMP/MKL/OpenBLAS各1线程、0 GPU、0交互、累计CPU120秒、campaign墙钟180秒、新输出/缓存/temp合计256 MiB。该次尝试已在预检停止，完整CPU/墙钟UNKNOWN，原reservation已退役；唯一账本为 [budget.json](budget.json)。修复的同额上限不构成新授权。首个计量预检至最终报告计 campaign，包括加载、失败、统计与执行核验；初始只读设计审查与算法合成开发检查另记工程工作，不冒充本 E 实耗。

临时目录/适用库缓存全部指向获准项目存储，HOME不改；使用既有项目环境，不安装依赖或修改系统。实际拟合前检查授权目标卷容量、inode和项目100 GB上限。超限/错误时保存部分证据并返回，不自动重跑或释放旧实验额度。瞬时教师与优化工作区报告峰值内存，不作为免费持久状态。

## Git + Data Paths + Figure Paths

| 项 | 当前绑定 |
| --- | --- |
| 科学方法 | [METHOD_V1](../../research/appendix/H036/METHOD_V1.md) + [HANDOFF_V1](../../research/appendix/H036/HANDOFF_V1.md)；父SHA 81de4afd2d703d17c3953facb78fd04b19cb64a9 + [21文件dirty snapshot清单](../../research/appendix/H036/delivery_v1/manifest.json)，source identity 0685e20c095b4c49c12d04041e3857f19b584c062dbe3eea456af1561fa661ea |
| 固定输入 | E161 run001/producer/cases/ 的16份prefix、48文件、2861895 B；准确allowlist/大小/哈希已在交付manifest核验，远端还须再次核验，不改E161输入 |
| 命令/环境 | 首次[冻结编排](preparation/validation/execution_freeze001.json)与修复版[freeze](preparation/validation/repair002/freeze.json)分别保留；run002已获[单次正式授权](preparation/coordinator_start002.json)，仍须动态远端准入和同campaign运行预检 |
| 原始输出 | [runs/run001](runs/run001/README.md)：仅计量预检与失败原始证据，无科学worker或拟合；[run002](runs/run002/README.md)：250文件原始镜像、32次拟合与完整评分，字节核验通过 |
| 分析 | [analyses/a001](analyses/a001/report.md)：失败结算、成本与独立审计；另见[a002有效结果](analyses/a002/report.md)，8/16、NO_ADVANCEMENT |
| 图 | **NOT_APPLICABLE**：完整配对表足够回答本轮问题 |

E161/E162 科学输出、本机指令、原冻结交付受保护，不纳入新源码快照或变更。新 snapshot 显式排除本机指令与凭据。真实结果出现前保存不可变的最终协议和预算注册副本；任何后续技术修复追加且保留原证据。

Stage A 及新 seed 确认没有在本 E 授权，不自动启动；其候选规模不作为资源余额或既定实验。

## 2026-10-22 用户指定 SSH 与职责交接

用户明确要求等待算法实现后交给消融实验（1），经SSH验证后由主控分析总结。此为E163执行位置/协调授权的更新，科学方法、16个prefix、32次拟合与12/16门槛不变；原始本地草案已保存在[registration_v1](preparation/registration_v1/)。当前仍需冻结交付与独立审查，不立即开跑。

仅以robot使用既有项目环境和必要项目输入。新增工作在 `/data/robot/p001/e163`；E161历史prefix只按逐文件清单读取，不修改。传输仅包含本E必要冻结源码/配置/输入白名单，不含本机指令、凭据、Git历史或全部旧输出。校验resolved路径及symlink边界、传输前后文件身份，并保存远端原始证据与本地镜像映射。只管理本E记录的进程句柄，不访问同事目录/进程或使用sudo。

消融实验（1）可先准备独立协议、计量与传输清单；在主控接受冻结实现之前不开始真实拟合或计量campaign。计量预检至最终报告必须落在180秒campaign和120累计CPU秒内，不能用等待代码交付消耗或重置运行时钟。实施时冻结精确远端命令、env与输入路径。

## 2026-10-22 科学冻结交付已接受

主控接受v1的规格与身份，21/21源码配置、48/48输入和参考文件匹配；冻结副本23项合成检查通过。见[交付验证](../../research/appendix/H036/delivery_v1/verification.json)与[主控接受记录](preparation/coordinator_acceptance_v1.json)。未重跑合成检查或真实拟合。E163仍等待实验方独立有效性审查、SSH身份/容量准入及计量编排，budget执行开关关闭；只有最终准入后才开跑。

## 2026-10-22 READY 验收与正式单次启动授权

主控验收[READY](preparation/validation/READY.md)、科学审查PASS及runner复审PASS，核对编排与run spec字节身份一致；资源/路径复核没有实质阻断。远端49项输入/foundation、33项依赖已核验；源码与编排白名单已传输，不含本机指令或历史原始输出拷贝。原准备阶段状态保留日期语境，不作为当前执行开关。

正式授权为 **E163_RUN001_20261022_APPROVED**，仅绑定run spec SHA256 `6e7604856c28ce33218f732eb893c1cc56827cab7eb4e51abee21b994f4491e8`、科学identity `0685e20c095b4c49c12d04041e3857f19b584c062dbe3eea456af1561fa661ea`、编排identity `0cb8bb54683809acd20b8303a7a73ff578759a62875d4627bbe9be778f242168`。精确命令/cwd/env见[execution_freeze001](preparation/validation/execution_freeze001.json)，授权载荷及证据绑定见[主控启动记录](preparation/coordinator_start001.json)。消融实验（1）负责创建唯一授权文件并执行，不由主控另开并行任务。

启动前刷新目标容量/项目上限、确认原冻结身份及run001不存在；任一不符停止。随后仅一个campaign：同一时钟内先做生命周期CPU与停止预检，再执行16×2=32次拟合和完整报告。120 CPU秒/180墙钟秒不变，115 CPU秒/170墙钟秒停止阈值给5 CPU秒/10墙钟秒收尾；报告区分观测值与保守尾部上界，不将保留额冒充精确实耗。SSH/shell准入运维另记，不冒充监督器进程生命周期计量。

输出限额256 MiB、项目100 GB、单worker单数值线程、0GPU/0交互、12/16门槛不变。保留原始及部分失败证据，无自动重试；StageA、新seed和完整流无授权。执行期间实际计量与任务句柄字段由验证方单独维护，主控只更新科学索引和后续决定。

## 2026-10-22 run001：计量预检失败，机制未评价

首个CPU生命周期子进程正常退出，但RUSAGE_CHILDREN与该campaign记录的wait4总量不一致，监督器按设计停止；科学worker和停止预检均未启动，实际fit=0、新交互=0，12/16门槛没有计算，不能记成0/16或NO_ADVANCEMENT。

启动脚本末行exec替换了已有SSH shell。原始日志首条campaign_started已有约67.336秒进程年龄，与沿用旧PID起点及历史子进程计数一致。已记录预检子进程wait4 CPU=0.12207秒可作为该子进程成本；原状态中的0.268236 CPU秒、67.5276秒墙钟及加尾部保留额的数值混入历史，不能当作精确E163实耗或据此计算可用余额。CPU/墙钟完整口径和预算结算以验证方最终账本披露为准，不覆盖原始读数。

证据：[原state](runs/run001/remote/run001/state.json)、[meter日志](runs/run001/remote/run001/meter.jsonl)、[wait4记录](runs/run001/remote/run001/wait4.jsonl)、[失败观察](preparation/validation/failure001_observation.json)。主控已要求仅在独立repair002中准备全新监督器进程、父shell显式wait及启动基线记录的最小修复；不改H036冻结源码，不在本轮再SSH试跑。首次证据、编排与授权保留，第二次执行尚未授权。

## 2026-10-22 失败结算与 repair002 主控验收

[最终失败报告](analyses/a001/report.md)及[独立审计](analyses/a001/audit.json)确认ACCOUNTING_INVALID；完整campaign CPU/墙钟均UNKNOWN，原stage0_campaign001额度退役，不计算可复用余额。科学worker、prefix反序列化、真实拟合与交互均0，12/16门槛未评价。

[repair002](preparation/validation/repair002/README.md)已冻结并通过独立静态复审，主控接受[修复身份与边界](preparation/coordinator_repair002_acceptance.json)。修改限于新监督器子进程、父shell显式wait和授权后fork前资源基线；科学代码、输入、配置、32次拟合和门槛不变。静态通过不代替Linux运行预检。修复未SSH传输或执行；追加单次CPU120秒/墙钟180秒/0交互预算已向用户提出，未获答复前不开新reservation或启动指令。StageA及其他后续阶段仍无授权。

## 2026-10-22 用户追加单次预算与 run002 正式授权

用户明确回复“授权追加一次重试预算”。主控新增stage0_campaign002，独立上限CPU120秒/墙钟180秒/0交互；首次reservation及UNKNOWN成本保留，既不重置也不计算可复用余额。两次授权额度之和仅是行政记录，不能代替总实耗或续跑余额。新尝试仍遵守单worker单线程、0GPU、E163新增工件/缓存/temp合计256 MiB及项目100 GB限制。

正式指令 **E163_RUN002_20261022_APPROVED** 绑定修复run_spec SHA256 `d5764b8091aba95b7223ee7cdcb18d752e123d17063bb0c936775753f6e8189d`、编排identity `b8df209ce3d89df2a4847b2b7e2da380e8d01b819e1f12f8ccde70782ad1353b`及原科学identity。完整载荷与条件见[启动记录](preparation/coordinator_start002.json)，旧账本保存于[启动前副本](preparation/budget_before_start002.json)。此前false模板和验收记录保留其历史时间语境，不覆盖为已运行。

仅由消融实验（1）按逐文件白名单传输新编排至o002，输出新建run002；复核固定s001源码/E161输入及foundation身份、resolved路径、动态容量/inode和项目总量后才启动。精确入口 `sh /data/robot/p001/e163/o002/launch_command.sh`。原o001/run001不覆盖；本机指令、凭据和Git历史不得传输。

从全新监督器进程起点计同一campaign，先通过生命周期CPU和停止预检，再执行固定32次拟合与完整报告；115CPU秒/170墙钟秒停止阈值为上限内收尾留额。失败保留证据并停止，无第三次自动重试、lambda搜索、StageA或新seed授权。返回后由主控评估科学门槛及下一决策。

## 2026-10-23 run002 最终交付

固定32次拟合完整结束，独立审计COMPLETE_VALID；8/16通过，低于预设12/16，结果NO_ADVANCEMENT。完整[报告](analyses/a002/report.md)、[配对表](analyses/a002/per_prefix.csv)、[逐fit诊断](analyses/a002/per_fit.csv)、[成本](analyses/a002/cost_accounting.json)及[结算](preparation/validation/run002_settlement.json)已保存。原始证据250文件、6875913 B逐字节核验；不改冻结源码或历史结果。

run002观测CPU小计8.625535秒、墙钟10.011705487秒，含收尾保留额上界13.625535／20.011705487秒；累计E163完整成本因首次计量缺口仍UNKNOWN。预算执行开关关闭，active_job=null，无自动第三次或后续阶段。全局H记录、研究索引及最终研究决定交主控集成。

2026-10-23 主控已接受[最终结果](preparation/coordinator_result002_acceptance.json)并完成[集成分析](../../research/appendix/H036/COORDINATOR_E163_SYNTHESIS_20261023.md)：H036为INCONCLUSIVE，固定v1停止。下一问题先审查完整双目标搜索、新训练KL接纳规则与留出表现的关系；当前失败在编码前已存在，不优先改codec。该问题登记不构成新实验授权，所有原始证据和预设门槛保持不变。
