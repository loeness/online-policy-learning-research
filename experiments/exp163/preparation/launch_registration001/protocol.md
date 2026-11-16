# E163 · H036 编码后旧/新行为兼容性筛查

- 注册：2026-10-22（Asia/Singapore）；主 H：[H036](../../research/records/H036.md)。
- 当前状态：**AUTHORIZED_SINGLE_CAMPAIGN_PENDING_START / NOT_RUN**。
- 当前执行位置：用户于2026-10-22指定 SSH，robot 账号、既有 `/data/robot/p001` 项目内的 `/data/robot/p001/e163`；方法与数值预算不变。
- 科学与编排审查、远端身份核验已完成；主控已授权绑定冻结run spec的单次campaign，启动前刷新容量，通过同一campaign计量/停止预检后才拟合。尚无真实结果。
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

独立额度：既有服务器项目内单 worker、Torch/OMP/MKL/OpenBLAS各1线程、0 GPU、0交互、累计CPU120秒、campaign墙钟180秒、新输出/缓存/temp合计256 MiB。当前未执行，唯一账本为 [budget.json](budget.json)。首个计量预检至最终报告计 campaign，包括加载、失败、统计与执行核验；初始只读设计审查与算法合成开发检查另记工程工作，不冒充本 E 实耗。

临时目录/适用库缓存全部指向获准项目存储，HOME不改；使用既有项目环境，不安装依赖或修改系统。实际拟合前检查授权目标卷容量、inode和项目100 GB上限。超限/错误时保存部分证据并返回，不自动重跑或释放旧实验额度。瞬时教师与优化工作区报告峰值内存，不作为免费持久状态。

## Git + Data Paths + Figure Paths

| 项 | 当前绑定 |
| --- | --- |
| 科学方法 | [METHOD_V1](../../research/appendix/H036/METHOD_V1.md) + [HANDOFF_V1](../../research/appendix/H036/HANDOFF_V1.md)；父SHA 81de4afd2d703d17c3953facb78fd04b19cb64a9 + [21文件dirty snapshot清单](../../research/appendix/H036/delivery_v1/manifest.json)，source identity 0685e20c095b4c49c12d04041e3857f19b584c062dbe3eea456af1561fa661ea |
| 固定输入 | E161 run001/producer/cases/ 的16份prefix、48文件、2861895 B；准确allowlist/大小/哈希已在交付manifest核验，远端还须再次核验，不改E161输入 |
| 命令/环境 | 精确argv、cwd、依赖和解释器身份 **PLANNED**；不得以本草案推测命令开跑 |
| 原始输出 | **PLANNED**：runs/run001/ 的预检、准备数据身份、32次拟合/编码诊断、逐prefix结果、日志和provenance；目标必须新建且唯一 |
| 分析 | **PLANNED**：analyses/analysis001/ 的16行门槛表、report、机器可读指标、审查与来源；未执行 |
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
