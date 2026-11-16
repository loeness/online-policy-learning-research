# E163 · H036 编码后旧/新行为兼容性筛查

- 注册：2026-10-22（Asia/Singapore）；主 H：[H036](../../research/records/H036.md)。
- 当前状态：**REGISTERED_PENDING_IMPLEMENTATION_AND_FROZEN_REVIEW / NOT_RUN**。
- 这是运行前草案，尚无冻结实现、完整执行身份、命令或真实结果；不准直接开跑。
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

独立规划额度：本地单 worker、Torch/OMP/MKL/OpenBLAS各1线程、0 GPU、0交互、累计CPU120秒、campaign墙钟180秒、新输出/缓存/temp合计256 MiB。当前未执行，唯一账本为 [budget.json](budget.json)。首个计量预检至最终报告计 campaign，包括加载、失败、统计与执行核验；初始只读设计审查与算法合成开发检查另记工程工作，不冒充本 E 实耗。

临时目录/适用库缓存全部指向本 E，HOME不改；无安装或远端访问。实际拟合前检查目标卷容量。超限/错误时保存部分证据并返回，不自动重跑或释放旧实验额度。瞬时教师与优化工作区报告峰值内存，不作为免费持久状态。

## Git + Data Paths + Figure Paths

| 项 | 当前绑定 |
| --- | --- |
| 科学方法 | 原设计 + 主控合同；新 METHOD_V1 / 完整Git SHA、dirty snapshot、源码/配置哈希 **PENDING_IMPLEMENTATION** |
| 固定输入 | E161 run001/producer/cases/ 的16份prefix；准确allowlist/大小/哈希 **PLANNED**，冻结交付及运行前复核，不改E161输入 |
| 命令/环境 | 精确argv、cwd、依赖和解释器身份 **PLANNED**；不得以本草案推测命令开跑 |
| 原始输出 | **PLANNED**：runs/run001/ 的预检、准备数据身份、32次拟合/编码诊断、逐prefix结果、日志和provenance；目标必须新建且唯一 |
| 分析 | **PLANNED**：analyses/analysis001/ 的16行门槛表、report、机器可读指标、审查与来源；未执行 |
| 图 | **NOT_APPLICABLE**：完整配对表足够回答本轮问题 |

E161/E162 科学输出、本机指令、原冻结交付受保护，不纳入新源码快照或变更。新 snapshot 显式排除本机指令与凭据。真实结果出现前保存不可变的最终协议和预算注册副本；任何后续技术修复追加且保留原证据。

Stage A 及新 seed 确认没有在本 E 授权，不自动启动；其候选规模不作为资源余额或既定实验。
