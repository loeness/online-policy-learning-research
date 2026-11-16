# H036 · E162 后的算法、结构与验证提案

日期：2026-10-22（Asia/Singapore）。作者：算法实现；用户请求在阅读主控与消融实验最新进度、核对论文后，提交主控选择下一验证。本文是 [H036](../../records/H036.md) 原假说的日期化方法细化，不改原假说或状态，不分配 H/E，不改 E161/E162，不授权运行。算法源码未修改；主控接受方向后，算法实现仍须实现、检查并交付新的冻结版本，消融实验（1）才有条件独立审查和执行。

## 1. 建议的下一问题

**先检验：在现有单父表示和固定存储下，旧初始化先验与新来源行为能否兼容整合，并转化为未来奖励改善。** 暂不增加选择器，不增加 parent，不改情境摘要、路由或 codec。

证据链：

- E161：behavioral−mean +.001591、behavioral−keep +.001301，联合确认与 .002 实际量级门槛未过。old-major 下 behavioral−keep −.005452，new-major 下 +.008054。这是日程收益取舍，尚不是已经定位的遗忘原因。
- E162：三特征 ridge 相对最强简单频率规则 +.000719706，稳定性仅 1/4 折，NO_ADVANCEMENT。不能继续在同批数据上换特征、调 α、补种子来争取通过；也不能推成所有操作前信息无用。
- 本提案检验不同机制：修改被存储内容所承担的行为目标，以及旧组能否实际调用该内容。它不预测不可见未来，不由 E162 的负结果自动获得执行授权。

来源：[E161 结论](../../records/H026.md)、[E162 主控集成](../../records/H033.md)、[E162 统计报告](../../../experiments/exp162/analyses/analysis001/report.md)。

## 2. 当前结构到底限制了什么

记冻结 W0 的 logits 为 z0(s)∈R⁴，H∈R^(4×3) 是正交动作对比基，phi=[h(s);1]。源码的 `source_basis=Hᵀ[W0_head|b0]`，因此：

`q(s)=source_basis phi=Hᵀ z0(s)`；

`z_C(s)=z0(s)+H C q(s)=[I+H C Hᵀ]z0(s)`，C∈R^(3×3)。

策略再经过 temperature=2 的 softmax 和 epsilon=.05 的均匀探索混合。局部矩阵是每槽、每 band 一个 3×3，共 27 系数；父矩阵只有 9 系数，对所有 band 共用。同一个 C 可以随 q(s) 变化输出不同动作，但**没有独立的 group 或 band 条件矩阵**，也没有新增状态特征。若两个状态的 q 相同，则任何这种 parent 的动作分布相同；若任务确需区分它们，该函数类无法做到。特别地 q=0 时这类残差不能产生非零动作对比。当前实验尚未证明这些限制实际造成失败；更换 source basis 若增加可学习状态，也必须付容量成本，不能当作免费的编码修复。

`new-bound` 只改变哪个组在 local 清零后继承 C；`wildcard` 让旧、新组继承同一个 C，不使 C 自动获得组条件。组身份在当前受控协议中确实可用，但尚未作为 wildcard parent 的函数输入。不能把“没有未来日程信息”和“当前可见组信息没进入父函数”混为一谈。

在同一输入上对两个不同教师做 forward KL 的最优自由分布是加权概率混合，仍有加权 JS 残差；受限的九系数模型还可能有额外投影误差。这只是条件性数学说明，不能用它断言现有旧、新观测支持已经冲突。epsilon 混合下的现有拟合也不享有这里未证明的全局凸性；单次 L-BFGS 没找到好解，不等于证明不存在兼容解。

代码依据：`src/online_rl/c1_tensor_controls.py:117`、`retention_behavioral_candidate.py:103`、`shared_write_policy.py:114`、`storage_decision.py:172`。

## 3. 文献借鉴：保留什么，拒绝直接迁移什么

| 原始来源 | 与本问题相关的机制 | 本项目采用/不采用的界限 |
| --- | --- | --- |
| [Policy Distillation，Rusu 等，2015/2016](https://arxiv.org/abs/1511.06295) | 原文将 DQN 的 Q 值软化后蒸馏；多任务版本还使用任务专用输出 | 借鉴 teacher→student 行为拟合；这里教师必须是 categorical actor 动作分布，不能把 Q-softmax 或不同任务头偷偷混入当前策略 |
| [Progress & Compress，Schwarz 等，2018，§3–4/式(2)](https://proceedings.mlr.press/v80/schwarz18a.html) | 新策略蒸馏与 online EWC 旧知识保护共同作用 | “蒸馏+保护”本身不是新颖贡献；H036 以已存在旧观测约束旧 prior 的函数，非原文 Fisher 参数惩罚，也不宣称完整复现 |
| [CLEAR，Rolnick 等，2019，§3/式 policy-cloning](https://arxiv.org/html/1811.11682v2) | 在回放状态上约束历史动作分布，并配合新交互和离策略学习 | 借鉴有界观测上的函数保护；本轮不引入 V-trace、value cloning 或 PPO 回放，不把历史 behavior_probs 与当前 keep prior 混成同一教师 |
| [Distral，Teh 等，2017，§2–3](https://arxiv.org/html/1707.04175v1) | 多个任务策略与共享行为先验通过 KL 联合训练；共享的是行为中心 | 提醒共享先验与任务专有残差需要分工。H036 只是一次受控整合，不新增共同训练流程，不把现有 logit 加法直接宣称为 Distral 复现 |
| [Continual Policy Consolidation for Lifelong Robot Learning，He 等，2026，v2，§4](https://arxiv.org/html/2601.22475v2) | 独立 RL teachers、任务上下文码、轨迹回放及可扩展 MoE student；学生拟合连续动作 | 与本项目最接近的是技能获取与整合分开。其上下文、增长容量和轨迹存储都需付费；本轮不照搬 MoE、DPP 回放或连续动作 MSE，不继承其低遗忘结论 |
| [DMPEL，Lei 等，2026 v3](https://arxiv.org/abs/2506.05985v3) | 增长的低秩专家库、router 与旧系数回放 | 是后续条件组合结构的参考；系数保持不等于动作保持，专家增长不等于固定总容量，本轮不增加 router/expert |

2026 CPC 的 v1 旧标题为 *Continual Policy Distillation from Distributed Reinforcement Learning Teachers*；本文引用固定 v2 新标题，避免把两个版本当两篇证据。仓库已登记 CPC、P&C、CLEAR 和 Distral；此次复核其具体借用边界，不冒充首次引入。DMPEL 是本次补充的条件结构参考，正式引用索引归主控整合。上述论文不构成 H036 已有效的证据。

可争取的贡献只能来自：**有限完整存储、真实可用教师与调用规则下，反复在线经验形成的知识是否带来可归因的保留/恢复收益**。通用蒸馏、回放、树或 MoE 本身都不是这里的新贡献。

## 4. 最小算法规格：同一九系数候选的双目标整合

### 允许输入和保护对象

仅同一 prefix 的 `checkpoint.pt`、`actor.bin`、`evidence.npz` 和冻结算法配置。核对 seed、来源槽位、generation、当前逻辑组与物理绑定；不读 observer 的旧 source 矩阵、未来日程/回报或完整历史作为方法输入。

- 新观测：两个当前新 resident 的 sidecar 各最新 128 行；owner 内按时间顺序前 96 行训练、后 32 行仅诊断。教师 T_j(s) 使用该 resident 的**当前有效 local+parent** 系数，按观测 band 取值。
- 旧观测：checkpoint 旧组现存 FIFO 的最新 128 行，前 96 行训练、后 32 行诊断；不足 128 行则该批不符合本规格，停止审查，不删 seed 或从原日志补行。
- 旧教师 T_old(s)：当前已编码/解码旧父 P_old，旧组新槽、local=0 的 **keep 初始化策略**。不是已淘汰 source 的完整策略，也不读取存储的 advantage、return、behavior_probs 作教师。仅用 s=obs[:4]，旧 route 和当前新 local 不参与该教师。
- 学生 p_C(s)：local=0、parent=C 的同一 actor 函数；旧、新数据都使用同一冻结 W0/source_basis、temperature 与 epsilon。

每 owner 和旧组内部逐行等权，定义训练目标：

`D_new(C)=.5 Σ_{j=0,1} mean_train_j KL(T_j || p_C)`；

`D_old(C)=mean_train_old KL(T_old || p_C)`；

`M=mean_{2 owners × 3 bands} E_new`；

`L_lambda(C)=D_new(C)+lambda D_old(C)+1e-4 mean((C−M)^2)`。

只比较 lambda=0 与 lambda=1.0。旧项总权重等于两个新来源合计权重；各项先独立归一化，不再按 384 总行数稀释其中一个来源。lambda 在结果前固定，不扫参。ridge 只加一次；若复用两个都自带 ridge 的旧 objective helper，必须显式减掉多算的 ridge 及其梯度。

### 优化、选择和回退

M 以 float64 求均值，C0=float32(M) 后转 float64 优化；旧父不参与初始化或正则中心。L-BFGS lr=1、history_size=10、strong_wolfe、tolerance_grad=1e-9、tolerance_change=1e-12，最多 64 iteration/256 closure evaluation，C/D 一致。

沿用 H026 的候选接收过滤：仅在有限、`D_new(C)≤D_new(C0)+1e-12` 时，允许以不高于已见最优的**本臂完整 L_lambda** 替换候选；相等允许后点替换。L-BFGS 的搜索目标仍是完整 L_lambda，这不是宣称已实现精确约束优化。预算耗尽返回已见最优；float32 后若违反原新训练 KL 门槛则回到 C0；其他输入/优化异常报错。holdout 不能选择候选、lambda 或初始化。

lambda=0 必须与原 H026 fitter 在同一合成输入上复现相同候选语义；B/C 直接共享同一 float32 C0-fit 结果，不能各拟合一次造成非绑定差异。lambda=1 的不同 closure 次数或 CPU 必须实报，优化**上限相同**不等于实际成本相同。

## 5. 四臂和实际保存规则

| 臂 | 保存内容 | parent_group | 本轮回答什么 |
| --- | --- | ---: | --- |
| A | incumbent P_old | 0（旧组） | keep 的实际价值参照 |
| B | lambda=0 的新行为候选 | 1（新组） | 原替换机制的本轮参照 |
| C | 与 B 完全相同的候选 | -2（wildcard） | C−B 隔离扩大绑定的影响 |
| D | lambda=1 的双目标候选 | -2（wildcard） | **D−C 隔离旧行为项** |

不要用高层 `mean` 动作实现 C/D：该动作平均 P_old 与候选，语义不同。C/D 走 `compensated_single_parent_install(..., parent_group=-2)`：先保留各 resident 的编码前有效矩阵，再扣掉其新继承父矩阵实现 local 补偿；淘汰后新槽 local 清零，旧、新组均按 wildcard 继承。

codec、优化器与随机状态复用 E161 的确切路径：A keep 在操作点不重新安装/编码，直接复用 prefix actor bytes；B/C/D 安装后 `_encode_live_actor`，再清 local/shared actor Adam 状态。所有臂以后每个 256 行 rollout 的 PPO update 后都 codec 并清 actor Adam；critic 权重和 Adam 状态保留。每个配对 branch 恢复同一 prefix，并按相同 future seed 重置 action/behavior/Python/NumPy/Torch RNG，环境状态沿原机制恢复并校验。C/D 不保存旧教师副本、lambda 自适应状态或额外 optimizer state。分叉后各自更新，不能混用分叉后的教师。D−A 是完整保存动作的净比较，包含真实安装与编码代价；不把它单独归因于旧 KL 项。

公平条件：C/D 使用完全相同的旧/新行、教师、初始化、正则、优化上限、representation、安装和调用规则。两臂均保留已有完整 bounded evidence；取 128 旧行只是其视图，不增加在线容量，不挤占 sidecar 两来源各 128 行。lambda=0 不把未用的旧缓冲空间换作其他优势。teacher/拟合数组与 L-BFGS workspace 为临时内存，也需计峰值。

actor 仍为 89 B；完整 learner package 上限仍为 262144 B。C/D 必须分配相同持久布局并报告真实序列化字节；若 metadata 或最终轨迹导致文件大小略有差异，须披露并证明没有一臂使用更大容量，而不是宣称实测字节已经相同或用 padding 隐藏差异。

## 6. 编码后的可证伪预测

H036 的预测是 D 相对 C 改善**编码后**旧 holdout prior KL，并保留部分新 holdout 拟合收益，最终在共同未来分布上改善回报。浮点训练 loss 下降是目标定义的后果，单独不能作为推进证据。

必须记录：新/旧训练和 holdout KL、候选/数据哈希、优化回退与次数、编码前后 KL、每个 resident 的编码漂移、实际 bytes、CPU/wall/内存。安装补偿在编码前须按 `atol=rtol=1e-6` 核对有效矩阵；编码后只报告实测误差，不假装 int8 仍严格不变。所有保护/拟合 KL 的学生是**实际解码父矩阵、fresh slot local=0** 的策略；不能拿补偿后的 resident 全策略作学生，把编码前本应保留的近零漂移误当成知识保护成功。resident 漂移另报。

旧组重分配 local=0、旧路由 id 改动不影响 old prior、新 owner/band 教师正确、lambda=0 对原 fitter 一致、旧项梯度正确且 ridge 不重复，是实现后最小正确性检查。技术检查仅用合成输入，不冒充 H036 实验。

## 7. 建议验证顺序与停止规则（待主控结果前接受）

### 第一步：接受合同，再实现与冻结

主控接受或返回本提案，登记本轮 H036 细化与所需 E；不从 E162 余额续跑。算法实现负责新增 `src/online_rl/retention_protected_candidate.py`、`retention_protected_inputs.py`、最小检查与固定配置，尽量不修改已冻结 H026 模块。接口合同为：

- `prepare_inputs(prefix)`：仅读三个允许文件，完成身份/codec/来源代次与行序核对；选择规定旧/新行，返回冻结 features、两类教师、划分和输入哈希，不接收 outcome 参数。
- `fit_protected_candidate(prepared, lambda_value)`：只允许 0/1，返回 float32 3×3 候选及完整训练/holdout、次数、回退、时间/临时内存诊断，不安装、不更新 learner 或 RNG。
- `prepare_install(local, shared, physical_groups, logical_tags, candidate, binding)`：返回补偿后的数组、parent_group 和有效矩阵检查；binding 仅本合同 new/wildcard，不擅自把 candidate 与旧 parent 平均。实际编码/更新状态由 runner 沿原 E161 路径调用。

实验 runner 由消融实验（1）拥有，不能边跑边补算法语义。完成源码/配置哈希、dirty snapshot 和合成检查后再交审。本段的文件/API 是待实现合同，不能被描述为现成可运行版本。

### 第二步：Stage 0 零新交互的机制筛查

复用固定 16 个 E161 prefix，C/D 各拟合一次；每 prefix 的 B/C 共用 lambda=0 候选。只从保存状态计算编码后的旧、新 holdout KL，不读取未来奖励调整方法。建议单 CPU worker、1 Torch thread、0 GPU，上限 120 累计 CPU 秒/180 墙钟秒/256 MiB 新输出；全部是新预算建议，当前未授权。

建议机制门槛：令 K_old,a 和 K_new,a 分别为臂 a 编码后、fresh local=0 的旧/新 holdout KL，G_new,a=K_new,W0−K_new,a。至少 12/16 prefix 同时满足 `K_old,C>1e-10`、`G_new,C>1e-10`、`K_old,D≤.75 K_old,C` 和 `G_new,D≥.5 G_new,C`。即旧 KL 降低至少 25%，并至少保留 C 相对无父 W0 的新拟合改善量的一半。近零或无改善的 prefix 计为不通过，不删去它，不按结果改分母。

这是判断“旧保护是否有可见作用，且没有吞掉全部新拟合”的建议筛查门槛，不是奖励阈值、统计置信结论或兼容性证明；主控可在任何本轮结果出现前明确返回/改定。Stage 0 使用的 16 个旧 prefix 包含后续 Stage A 的四个，两个筛查关口不独立，不能把它们当两次验证。若失败，停止该固定 lambda/拟合方案；不能自动扫 lambda、修改编码或宣布表示不可能兼容。

### 第三步：Stage A 最小在线机制筛查

仅 Stage 0 达到已接受门槛并获得单独交互授权后进行。使用事先固定四层各首个 seed：26201、26202、26204、26207；不按 KL 或历史奖励挑选最好的 prefix。共享旧 prefix 作为已付费输入，披露其原始成本，不再产生前缀交互。

所有 A/B/C/D 都在本轮相同机器/依赖/runner 重跑，不拿服务器旧 A/B 与本机新 C/D 直接比较。未来只取 old_major/new_major，各 1 replica、6144 步，保持 E161 的完整日程与在线更新规则，不加 balanced 或额外评估。旧 seed/prefix 已参与过结果后设计，所以即使购买新 C/D 分支，本轮仍是探索性筛查。

新增计划交互 = `4 prefix × 4 arms × 2 futures × 1 replica × 6144 = 196608`。建议本轮硬交互上限同为 196608，CPU/wall 各 900 秒，单 worker/单 Torch thread、0 GPU、新输出上限 512 MiB；失败保留，不能额外免费重跑。真实资源上限仍须主控单独接受，未核实的本机速度不当作实测预算。

主评分：每 seed 等权平均两个未来的 reward/step。建议进入确认设计的筛查条件为 D−C 均值>0 且至少 3/4 seed 为正，D−A 均值≥0；另要求 old-major 的 D−C 均值>0、new-major 的 D−C 均值不得低于 −.002，防止仅靠新日程收益抵消旧日程受损而宣称保护成功。两种日程都仍是完整混合日程，不称纯旧/新任务效应。四 seed 仅作机制筛查，不能建立稳定性能结论。C−B、D−B 和所有边界完整报告，不作事后挑选的救援指标。

### 第四步：条件性独立确认，另行登记

只有正向筛查才值得由主控制定新 seed 确认；不得自动触发。规模示例是 16 个未见 seed（家族×历史四层各 4），同四臂、两个未来、两个 replica、6144 步。新前缀仍每 seed 29696 步，则计划总交互为：

`16×29696 + 16×4×2×2×6144 = 2048000`。

这只是可审查的规模/成本公式，不预断样本充分或分配 seed；新 seed 必须先查全项目使用记录。正式样本量、CPU/wall、主估计量和不确定性分析由主控与消融实验在看确认数据前固定。主要主张必须同时面对 D−C（旧项增量）和 D−A（超过保留旧知识的实际价值），不能只赢一个较弱新替换对照。建议沿用 .002 reward/step 的实际量级要求作为讨论起点，但是否及如何用于联合推进须主控明确决定。若延续“旧日程改善且新日程损失受限”的主张，还必须结果前固定 old-major D−C 的方向检验与 new-major 的非劣性界及分析，不能仅由等权平均值代替。该确认最多覆盖同一 W0/任务族、已知身份与边界，不覆盖其他 W0、未知任务识别或一般开放世界在线学习。

不直接进入反复保存完整流；如果确认有收益，下一阶段仍需同期 direct shared PPO、强独立基线与完整容量/交互核算。这些不塞进本轮最小筛查矩阵。

### 成本计量前置条件

E162 的 Windows venv 启动器漏计不能重演。实验负责人必须先用零模型检查确认计量覆盖实际 Python 解释器与所启动子进程，按已记录项目句柄核算生命周期 CPU；包括启动/加载、失败、重试和统计，包内计时不能代替 campaign 成本。任何无法可靠执行资源上限的计量条件都应在真实运行前解决；不靠事后补测伪造原成本。无远端操作授权，E161 的服务器授权不延伸到本提案。

## 8. 结构升级的条件，而不是当前的新模块

直接增加第二个 9 系数 parent，在当前 codec 下 actor 为 `89+(9+4)=102 B`，违反现有 89 B 合同。把相同加法改画成树也不改变函数类；H006 已有树/分组加法等价证据。因而本轮不宣称“加模块即可解决”。

若 H036 出现可重复的旧、新代理取舍，先区分数据覆盖、优化、量化和函数限制；只有有证据指向条件函数不足时，才进入已登记 H031 的固定预算结构合同。届时至少说明：运行时条件从何来、增加的父/路由字节从 local 容量还是编码精度让出、对应控制如何隔离这些代价、平面分组与树/DAG 是否实为同一函数。任何具体新结构算法应由主控另登记独立可检验假说后设计实现。

论文给出的 context code、增长 expert 和历史回放是各自付费的部件，不能被当作当前结构已经免费具备的能力。当前受控 group ID 可以支持未来条件结构研究，但不能据此声称未知任务识别已解决。

## 9. 本次交接结论

推荐主控先选择 H036 的**同绑定双目标行为整合**作为一个新机制问题，保留 A/B/C/D 的可归因结构，采用零新交互筛查后再决定是否购买未来分支。H033 v1 停止不变；H035、扩容 MoE、DAG 和新 selector 继续不启动。

本次完成文献与源码研究、精确设计和验证步骤；未修改算法、未运行新拟合/环境、未创建实验目录、未分配预算。主控接受本设计后，算法实现仍承担代码、必要检查和冻结交付责任，不能把本文直接当成消融实验可开跑版本。
