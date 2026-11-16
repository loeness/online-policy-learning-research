# E161 · H026 行为候选的独立确认与预设未来日程边界

- 主 H：[H026](../../research/records/H026.md)。提出/设计日期：2026-10-21（Asia/Singapore）。
- 提出者：Codex；依据用户对下一研究主线和详细验证方案的请求，以及 H026/E130 的既有信号。
- 执行状态：`COMPLETED`；协议状态：`FROZEN_BEFORE_RUN`；科学结果：`INCONCLUSIVE`。
- 原草案预算在 2026-10-21 用户要求按本案使用服务器展开验证后获准执行：本轮上限为 2500000 次实际环境交互、累计 3600 CPU 秒、campaign 2700 秒墙钟（均含失败/重试），2 个单线程 CPU worker，不使用 GPU。固定方案完成后停止，不自动启动后续完整流。
- 本轮已完成；下列方案保留结果前设计，末节记录实际结果。结果前协议和配置的原始字节保存在 s001.zip，源字节以 preparation/source-manifest.json 及远端 s001/bundle-manifest.json 绑定。授权目标为 `robot@100.111.191.97:/data/robot/p001`，项目存储总上限 100000000000 B；仅传输本次必要代码/配置/输入白名单。预算实时账本为 [budget.json](budget.json)。
- H026 仍为 `INCONCLUSIVE`。本 E 延续同一候选内容问题，不新建一个假说来改写原始预测。

## 1. 下一研究主线及本轮决策

沿 H026 的已执行证据继续，先用一次固定规模的新种子确认决定是否值得投资候选内容。H035 是有界递归情境摘要的独立 `PROPOSED` 问题，编号不是时间或依赖顺序；本轮不引入路由、摘要、动态树或 DAG。

冻结待确认的条件为 `dynamics / late`。原六种子、多条件探索中，此条件的两个主要对比均为正，但尚未经独立确认。选择该条件的原因和旧结果必须公开；新种子才构成本次确认数据，旧 E130 不合池进入主检验。

问题：在相同 W0、九个共享系数、来源观测和最终 int8 编码下，行为拟合替换是否在原两种未来日程等权的分布中，同时优于参数平均替换与保留旧父知识？预设中间日程用于检查附近边界。

支持证据：两项 seed 配对奖励差均为正且达到下文门槛。削弱证据：不重复、只改善 KL、只优于一个弱端点，或只有事后选中的日程有效。局部失败不等于整个有限持久知识研究问题被否定。

本轮不声称候选在所有未来都更好，也不声称能只依据过去选对操作；原 old-major 下的负效应必须单独报告。

## 2. 冻结项与受控改变

参考版本：[H026 原协议](../../docs/RETENTION_CANDIDATE_PROTOCOL.md)、[原配置](../../configs/retention_candidate.json)、[E130](../exp130/README.md)、[原审查](../../docs/RETENTION_CANDIDATE_REVIEW.md)。

| 项目 | 本轮方案 |
| --- | --- |
| 关系 / 状态 | 仅 dynamics / late；不重新搜索四个 relation×state 条件 |
| W0 / 表示 | 沿用原 foundation 与冻结 source basis，固定 temperature=2、epsilon=0.05 |
| 容器 | 2 local slots，各 27 系数；1 parent，9 系数；actor 89 B |
| 完整持久包 | <=262144 B；底座、learner、optimizer、RNG、证据与 sidecar 均按原口径计费 |
| 前缀 | 29696 步；沿用四个 warmup cycle、旧组写入规则及 history 模板 |
| 来源观测 | 每来源最后 rollout 的 128 行，按 owner/slot/generation 核对；不从全日志回补 |
| 参数均值候选 | 两来源×三个 band 的有效矩阵等权平均 |
| 行为候选 | 原 forward KL + 1e-4 系数均方正则；75/25 时间划分；L-BFGS 64 iterations / 256 closures |
| 保存操作 | keep、replace_mean、replace_behavioral；不执行 merge/none |
| 在线更新 | 每 256 步 rollout；batch 64、最多 4 epochs；local/critic LR=.003、parent LR=0 |
| 其他 PPO 参数 | gamma=.98、GAE=.95、clip=.2、target KL=.06，保留原 KL 提前停止语义 |
| 编码 / optimizer | 原 int8 编码、补偿和 actor Adam 清理规则；不同时修编码或特征归一化 |
| 淘汰 / 物理回合 | 自然 LRU 继续；切换任务不强制重置回合；新分支恢复同一完整 learner/模拟器 |
| 任务信息 | 继续提供 task/group/boundary；本轮不检验无标签辨识 |

三个臂共用每 seed 的一个前缀。先保存完整分叉状态，再安装对应操作；父知识改变时按原实现补偿驻留 local，记录编码前后漂移，随后父参数冻结、local PPO/critic 继续。真实写入与 keep 的编码差别保留，因为它属于待确认的实际实现。

## 3. 样本、日程与独立性

- 固定 **16 个未使用的部署种子，旧家族 A/B 各 8 个**，不依据本轮奖励增加或删减种子；同时冻结每个 seed 到旧家族和历史 regime 的完整映射。
- 原草案候选区间为 `26201–26216`（历史草案，已被本条冻结清单取代）。实际冻结为 `26201–26213, 26215, 26216, 26218`，完整映射见 [seed-map.csv](preparation/seed-map.csv) 和 [seed-audit.json](preparation/seed-audit.json)。按升序候选池 `26201–26300`，每个 A/B × old/new-history 格取最先 4 个未使用 seed；26214 和 26217 因各自格已满而跳过，未观察本轮结果。四个联合分层各 4 个，A/B 各 8 个。查重覆盖 25535 个项目配置/源码/Markdown/JSON/YAML/TOML 文件；未扫描二进制和逐步 CSV。未发现已实际使用这些 seed 的运行；无冲突的结论限于该覆盖。
- 每 seed 一个 dynamics/late 前缀；每未来日程 **2 个随机续程**；各臂使用配对的未来随机流，保留原 RNG 派生公式并核查范围冲突。
- 当前实验固定旧任务族和一个 W0。新种子提供新交互轨迹/初始状态，不等于验证新任务族、独立预训练底座或新表示。

原未来日程来自 `scripts/run_retention_mechanism.py` 的四个 3-bout block，每 bout 512 步，总 6144 步；每 block 为 minority 一次、dominant 两次。因此原 old_major/new_major 的旧组占比为 **2/3 与 1/3**，不是 3/4 与 1/4。

| 未来日程 | 旧组 bout 数 / 总数 | 地位 |
| --- | ---: | --- |
| old_major | 8/12 | 主 estimand 的第一个组成部分 |
| new_major | 4/12 | 主 estimand 的第二个组成部分 |
| balanced | 6/12 | 预设次要边界，不参与主成功门槛 |

balanced 沿用原 block 内模板及 variant 循环，让四个 block 的 dominant 固定交替为 0/1/0/1。两个 replica 保持同一任务日程，仅改变续程随机流。完整 bout 清单已在执行前保存于 preparation/schedules.json，随源包冻结。它检验预定义日程边界；时间顺序与比例共同定义该环境条件，不能把曲线当成只改变独立同分布复现概率的因果剂量曲线。

各 seed 内三臂×三日程×两 replica，共 18 分支；总计 **16 前缀、288 分支**。future、replica、嵌套 horizon 和逐步日志都不是额外独立种子。

## 4. 指标、统计与事先门槛

主指标是保存后 6144 步累计奖励除以实际步数，越大越好。prefix 成本与奖励另报；不把窗口改成观察结果后最有利的一段。

对每个 seed，先在同一日程内平均两个 replica 的配对差，再把 **原 old_major/new_major 两日程等权平均**，得到：

1. D1 = replace_behavioral − replace_mean。
2. D2 = replace_behavioral − keep。

balanced 绝不进入上述主均值。旧 E130 的数据不加入新样本。

旧家族 A/B 各 8 个 seed，主目标为两家族等权。每比较的估计量是 `(mean_A+mean_B)/2`，标准误为 `sqrt(s_A^2/8+s_B^2/8)/2`，自由度使用 Welch–Satterthwaite 公式，避免旧数据中明显不同的家族方差被强行合并。

主张是 **两个增量同时为正**，采用交集并集检验（IUT）：每比较计算单侧 95% Welch-t 下界，只有两项下界均 >0 才通过联合方向判据。该联合主张按每项单侧 alpha=.05 控制整体假阳性率；t 推断依赖分层 seed 均值近似，非无假设保证。逐项 95% 双侧区间用于报告，不称为两个区间的同时 95% 覆盖。另报分层 seed bootstrap 为敏感性分析，不在方法间事后挑有利规则；实现、重采样 seed/次数和分层清单执行前冻结。

以下门槛是 **2026-10-21 的新方案**，不是对原探索协议门槛的追认：

| 观察 | 决策 |
| --- | --- |
| 两个单侧 95% 下界均 >0，且两个点估计均 >=.002 reward/step | 允许进入有强基线的完整流验证；仅确认固定条件/分布下候选增量 |
| 两个单侧下界均 >0，但一个或两个点估计 <.002 | 可称小的条件性收益；不自动进入大规模算法实验 |
| 点估计为正但方向判据未过，或只一项通过 | INCONCLUSIVE；按原规模停止，不临时补 seed 或改主条件 |
| 方向明显不重复，或完整性/容量失败 | 不升级旧信号；保留负结果/失败，重新判断该候选机制是否值得继续 |

点估计 >=.002 不证明真实效应下界 >=.002；只有对应区间下界也超过 .002，才可作那种更强表述。没有拒绝零假设也不等于效应为零。

固定 N=16 是一次有预算上限的确认方案，不保证对 .002 或两个联合门槛有足够功效。旧六 seed 有选择偏差，样本规划不能将其效应当成真值；不得根据新数据的显著性扩充 N。只读规划从原 `outputs/retention_candidate_completed_report/analysis.json` 的 dynamics/late 逐 seed 结果获得 SD：D1 的 A/B 约 .005808/.001648，D2 约 .006232/.001101（每层旧 n=3，不稳定）；按 8+8 配置粗估新 SE 分别为 .001067/.001119。此计算仅用于样本规划，不是新增效力证据。计算、精确旧输入哈希、逐 seed 数值及近似功效局限已保存于 [power-plan.json](preparation/power-plan.json)，禁止用本轮中途回报重新估计 N。

按这些旧 SD 和单侧 .05 的近似规划，D1/D2 的 Welch 自由度约 8.12/7.44、t 临界值约 1.856/1.878；以 `(t_critical+z_.8)×SE` 粗算单项 80% MDE 约 .00288/.00304。它不保证联合成功概率；真实效应为 .002 时，点估计也超过 .002 的筛选本身约有一半通过机会，不能把本案写成对 .002 有 80% 功效的试验。次要边界区间均为描述性，不受主 IUT 的联合主张控制。

次要结果固定报告：三个日程各自的 D1/D2、逐 seed 结果、1536 步嵌套窗口、发生重新分配时首 256 步回报、训练/诊断 KL、编码前后动作漂移、实际父冻结核对、淘汰数、CPU/墙钟/optimizer 更新数、持久包与 source buffer 字节。三点边界曲线不用于事后寻找或宣称最优分布。

## 5. 实施顺序与技术验收

1. 核对 ID、seed 使用历史、参考 producer/config 的文件哈希和源版本，补齐授权预算与剩余量。
2. 仅改实验编排：支持三臂子集、late-only、balanced 日程和对应审计/报表；learner、候选构造、编码和 PPO 不改。现有 runner/reporter 有五臂、两状态、六 seed 的硬编码，当前不能只改 config 运行。预期涉及 `scripts/run_retention_candidate.py`、`run_retention_mechanism.py` 的 schedule/validation、`audit_retention_candidate.py`、`report_retention_candidate.py`、共享 `src/online_rl/storage_decision.py` 的 regime 校验、新实验配置及相关已有检查；具体 diff 冻结，保证旧两个 regime 输出不变。
3. 用既有工件和零环境调用的软件检查核对：三臂筛选不会隐式运行 merge；每个日程 12 bouts/6144 步；8/4/6 个旧组 bout；RNG 配对、owner/generation、输入不可变、恢复和成本计数正确。报告主效应必须显式只汇总原 old/new 两日程，不能把新 balanced 自动并入主均值；审计不能把任何非 old-major 一律当成 new-major。先跑相关现有检查，不做全仓库测试扩展。
4. 用已保存原始 case 的只读回放/状态核对确认端点语义；不把回放当新独立确认数据，不启动一个仅用于填目录的 pilot。
5. 冻结协议、resolved config、完整命令、执行 Git SHA、dirty diff/所需未跟踪代码、源文件哈希及分析版本。只有满足预算、存储与 provenance 前置条件才准入生产。
6. 每 seed 产生并收费一个 29696 步 prefix，保存模拟器/learner/RNG 的完整分叉身份；拟合两个候选一次，共用候选不重复训练来源。
7. 依冻结的执行顺序运行 18 条 6144 步分支；各臂从对应的相同完整状态出发。禁止把前一个分支的 learner/critic/RNG 带入下一个。
8. 每 rollout 核对 parent 冻结、更新数、容量；每 seed 核对分支覆盖、输入和源哈希、完整账本及成本。资源/实现错误可触发安全停止，奖励好坏不触发选取性停止。
9. 全部预定 seed 完成后做独立审计，再执行冻结分析，输出主比较和全部边界。技术 PASS 与科学通过分开记录。
10. 回写 E 实际状态、每次 attempt 的成本及 H026/index 的科学决策；失败/取消不覆盖。中断已落盘步数只当下界，重试另建 run。

首个实际生产 seed 仍属于正式设计；如果实现有错，保留该失败 attempt，修复/版本差异披露，不能用它调候选或把后续结果拼成未改动版本。若需要新增环境 pilot，先另登记 E、预算和排除规则，不能静默消耗本 E 的确认数据调参。

## 6. 交互、计算、存储预算（2026-10-21 已授权）

| 项目 | 精确计划交互 |
| --- | ---: |
| 16 个共同 prefix：16×29696 | 475136 |
| 原两日程：16×3×2×2×6144 | 1179648 |
| balanced：16×3×1×2×6144 | 589824 |
| 完整方案 | **2244608** |
| 仅原两日程方案（若在冻结前删去边界） | 1654784 |

不得在看到结果后才删除 balanced、删 seed 或切换规模。全案交互公式为 `N×(29696+3×3×2×6144)=140288N`。

本轮授权执行上限为 **2500000 次实际环境交互（含失败/重试）**、合计 **1 CPU 小时**、campaign **45 分钟**；2 个项目 worker、每 worker 1 个 Torch thread、不使用 GPU。用户已要求按本案展开验证；任一触及即停止新分支并保存现有证据，不能自动续额。额外分析 CPU 单列，不能隐藏在 RL 成本之外。

交互余量为 255392 步。最多允许一次因明确技术失败而进行的完整 seed 等价重试（140288 步），重试前必须连同失败 attempt 的实际交互重新核算，确认可在总上限内完成；余额不足则不启动。技术失败限定为崩溃、源/配置/恢复身份不匹配、漏分支、容量越限或明确完整性错误，不包含奖励差、KL 留出变差或区间不利。不使用备用 seed 替换合法负结果。触及上限或无法完成 16 个有效 seed 时，将执行状态记为 FAILED 或 CANCELLED、覆盖标为 INCOMPLETE，披露剩余数据与缺口，不能用子集冒称完成确认。

历史 E130 的 3391488 步对应 campaign CPU 1957.625 秒、墙钟 1025.364 秒；仅按交互比例外推，全案约 1296 CPU 秒、679 秒墙钟。此为同机同并发的粗略量级估计，late 比例、日志 IO 和机器状态都会改变耗时，不是实测或完成承诺。

原 `outputs/retention_candidate_completed` 的只读汇总大小为 1136513605 B。按前缀/分支及元数据比例粗估，本案 producer 输出约 **738–741 MB**，不含额外报告副本、重试和源备份；建议至少预留 **2 GiB 项目空间**并在实施前重估峰值。实际新输出/峰值均未测量；项目获批总 allowance 为 100000000000 B（监测上限，非文件系统强制 quota）。执行前聚合检查项目已有约 1.977 GB、目标卷约 957.66 GB 可用和 935250582 可用 inode；峰值按现有环境加本轮输出、缓存和临时副本预留至少 2 GiB，低于项目上限。生产前检查项目目标卷的可用字节/inode（平台适用时）并留出同意的余量。用户已指定远端 robot 账号、项目目录与 100 GB 上限，本轮在该目录执行；不使用 GPU 或 sudo。

## 7. Git + Data Paths + Figure Paths

以下均为来源或计划，不能当作已存在的新运行工件。

| 要素 | 状态 / 位置 |
| --- | --- |
| 设计阅读背景 HEAD | `81de4afd2d703d17c3953facb78fd04b19cb64a9`；只是本次阅读背景，非未来执行身份 |
| 新 producer 执行 SHA + dirty snapshot | `81de4afd2d703d17c3953facb78fd04b19cb64a9` + preparation/source-manifest.json、changes.patch 和 source bundle；保留当前必要 dirty/untracked 源字节，不能仅写 HEAD 冒充执行代码 |
| 原基准 provenance | [E130 run002](../exp130/runs/run002/provenance.json)、[旧协议 freeze](../../outputs/retention_candidate_protocol_freeze.json)；原执行缺口保持披露 |
| 主要输入 | `data/foundation.pt`；metadata 中 SHA256=`e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222`，运行前须重新核验实际字节 |
| 数据定义 | 原 C1/dynamics task generator、seed/task 映射与冻结 bout 清单；不重新预训练 W0，不混入旧结果 |
| 最终配置/完整命令 | `configs/e161.json` 已冻结；完整三阶段命令见下方远端执行冻结 |
| 工作目录/环境 | `/data/robot/p001/s001`；项目独立 `/data/robot/p001/env`，Python 3.12.15、torch 2.8.0+cpu；requirements-lock.txt 及准入记录绑定其余版本 |
| 新 raw/log/config/checkpoint | **PLANNED**：`experiments/exp161/runs/runNNN/producer/`；每次实际 invocation/attempt 独立目录 |
| 分析与软件来源 | **PLANNED**：`experiments/exp161/analyses/analysis001/`，记录精确输入/排除、分析源码及版本；不伪造空的完成产物 |
| 图 | **PLANNED**：`analyses/analysis001/figures/paired_seed_contrasts.png` 与 `future_schedule_boundary.png`；源数据为该 E 审计通过的逐 seed/分支奖励汇总，绘图命令和代码版本随分析保存 |
| 最终结果和预算剩余量 | **MISSING — 尚未运行** |

源快照采用明示的项目代码/配置/依赖 allowlist；只收集本次实际执行需要的文件，不把根目录或整个历史 outputs/Git 作为 source snapshot。

## 8. 通过后如何回到完整研究问题

本轮有明确终点：一次独立确认结束后必须作出 keep/停止升级/不确定的决策，不自动再复制同一矩阵。

若达到本轮升级门槛，下一项优先进入 **H024 的新一轮完整流验证**：保持候选构造与预算固定，先比较 keep、参数均值+固定替换、行为候选+同一固定替换、direct shared PPO、强独立/扁平 local。每法都从共同 W0 初始化并支付自己的完整历史交互，不能让强基线继承其他方法的已学状态。重用 H024 的流定义只作为参考，新完整流需另登记 E、冻结样本/判据并获预算。

该比较回答候选内容在重复在线保存中的价值，以及离散保存是否值得超过直接共享更新的复杂度；不能用一次保存的胜利宣称完整算法占优。独立基线应合理使用允许的完整字节，报告交互/更新/计算差异；不能用旧 H024 分数代替同期对照。

只有需要改进操作排序且存在可检验的过去信息时，才从 H033 检查动作对比/排序；MSE 变好不够，精确等价的重参数化也不是新机制。只观察 logger 分布的一步价值不能替代完整流策略验证。

编码/联合蒸馏、late 成因拆分、动态树/DAG 均不作为本 E 的自动后续。若编码偏差被认为值得干预，先登记单独可证伪问题，从同 prefix/同候选只改一个因素，固定净字节再看后续奖励；不能把本轮收益或失败直接归因于编码。

在已知身份下的保存机制有价值后，再按 H035/H029 单独移除标签、检验路由：窗口 vs 递归摘要固定下游策略，报告识别/切换延迟、全流回报、额外辨别交互和完整字节。H027/H028/H031 的完整 C2/C3/DAG 要等必要的容量与组织机制证据，不按编号逐个启动。

## 9. 引用与贡献边界

本 E 确认已有 H026 自定义候选，继承其 [Policy Distillation / Distral / Progress & Compress 的借用说明](../../research/records/H026.md#引用关系与借用边界2026-10-20-复核)，不新增方法新颖性主张。行为蒸馏已有先例；本轮增量在固定有限观测、实际持久字节和在线继续学习下的独立证据。

## 修订记录

- 2026-10-21：首次方案草案；预留 E161；零新增环境交互。静态审查后将较保守的 Bonferroni 同时区间方案改为直接针对“两个效应皆正”的 IUT，并采用 A/B 分层异方差标准误；改动发生在任何新增结果之前。执行版本、seed 查重、最终统计实现、精确存储与授权预算尚待完成，不能称为已冻结预注册或已获批实验。


## 2026-10-21 远端执行冻结（结果前）

- 源目录：`/data/robot/p001/s001`；输入：该目录 `data/foundation.pt` 与 `data/pretrain_source.npz`，后者 SHA256=`493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2`。两个输入均与 E130 相同；旧 E130 的执行 SHA 缺口保持披露。
- 输出：`/data/robot/p001/e161/run001/producer`；独立审计：`run001/retention_candidate_audit/main.json`；冻结报告：`run001/analysis001/analysis.json` 与 `report.md`，图名由冻结报告器生成并记录其源数据。原表中的图路径仅为草案预期，以实际图清单为准。
- shell 先 source 项目 enter.sh，再临时设置 `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`；HOME 不改，cache/temp 均由已有项目 config.sh 指向分配目录。
- 监督入口：`/data/robot/p001/env/bin/python -B scripts/launch_e161.py --source /data/robot/p001/s001 --run /data/robot/p001/e161/run001 --cpu-seconds 3600 --wall-seconds 2700`。入口依次执行 producer（`--config configs/e161.json --output .../producer`）、auditor（`--root .../producer --report .../retention_candidate_audit/main.json`）、reporter（`--input .../producer --output .../analysis001`）；记录完整 argv、自己的进程树句柄与累计 CPU，任一预算到限停止本项目进程组并保留证据。
- 执行前检验拒绝路径越界和 symlink，核验源包内每文件哈希及两项数据路径；审计与分析前后重复核验。传输不含机器本地指令文件、凭据、Git 目录或历史 outputs。
- 审查接收：三臂/late-only/balanced 编排、平衡查重及独立 Welch/IUT 分析已接收；修复审核发现的 CLI、审计路径与 CPU 跨阶段衔接问题。learner、候选 KL/L-BFGS、PPO、补偿和编码均未修改。
- 停止规则：一次固定规模确认后停止；技术通过与科学推进门槛分开。不得按回报增删种子，不启动 H035 或后续完整流。

- run001 已由监督进程 PID 1214095 启动；运行中精确状态见远端 run001/state.json，预算账本记录活跃句柄。平台检查未产生环境交互；一项旧多审计文件排序测试需要挂载不支持的 os.utime，本地通过，远端本次仅生成单个审计文件，无该时间戳修改要求。


### 2026-10-21 技术审计修订（训练未重跑）

run001 producer 已完整完成 16 个 prefix、288 分支和 2244608 交互，首次监督耗费 999.518698 CPU 秒 / 546.601151 秒墙钟（含失败审计）。首审共 2416 个路径错误：32 个 source_copy_missing、2368 个 artifact_missing、16 个 observer_prefix_path；均源于旧 auditor 强行把 `/` 转为 Windows `\`，在 Linux 上找错路径。原审计 main.json 完整保留。

一次定向修复仅将审计路径解析改为跨平台 separator/absolute/traversal/symlink 检查；21 项检查通过，一项 Windows 无权建立 symlink 的检查跳过，随后 Linux 上 separator、traversal 与 symlink escape 检查均通过。详见 preparation/analysis-repair.json。源包 s002 SHA256=`f0df678ee9b909b8af5d0e8a96e8b6f3a1dbb0f4653d265209ee037b09deebfd`；保留 s001。候选、learner、producer、输入和统计报表字节未变，不新增环境交互。

仅重做审计与同一冻结分析：新控制状态 `/data/robot/p001/e161/a002/state.json`，新审计 `run001/retention_candidate_audit/main002.json`，报告 `run001/analysis001`。控制器按原始 campaign 剩余墙钟和剩余 CPU 准入，不能重置总预算。当前科学结果继续等待完整技术通过。


## 2026-10-21 实际结果与停止决定

**执行 COMPLETED；科学 outcome = INCONCLUSIVE；独立技术审计 PASS（0 errors / 0 warnings）。** 完整 16 个新 seed、16 个共同 prefix、288 分支，实际 2244608 次交互；无环境重试、无 seed 排除、未合池 E130、balanced 未进入主检验。一次审计器跨平台路径修复与失败记录见上节；训练数据和统计门槛未变。

主比较按 seed 配对、A/B 等权和预定 Welch 规则：

| 对比 | reward/step 增量 | 单侧 95% 下界 | 双侧 95% 区间 | >=.002 点估计门槛 |
| --- | ---: | ---: | --- | --- |
| behavioral − mean | +0.001591365 | +0.000044580 | [-0.000316381, +0.003499110] | 未过 |
| behavioral − keep | +0.001300812 | -0.001131791 | [-0.001665413, +0.004267037] | 未过 |

只有第一项方向下界为正；第二项未通过，且两项点估计都低于 .002。因此既未通过联合 IUT，也未达到预定推进门槛。独立复算从 192 条主日程 branch/horizon 观测还原 16 个 seed 的配对均值和异方差标准误，完全匹配冻结报表；四个联合层各 4 个。见 [复算记录](analyses/analysis001/review.json)。

| 未来日程（旧组 bout 比例） | behavioral − mean | behavioral − keep |
| --- | ---: | ---: |
| old_major（2/3） | +0.001386 | -0.005452 |
| balanced（1/2，次要） | +0.001046 | -0.001136 |
| new_major（1/3） | +0.001797 | +0.008054 |

日程边界清楚：旧任务占多数时行为替换相对 keep 为负，新任务占多数时为正；这些逐日程区间为预设次要描述，不能据此宣称部署选择器有效。完整区间、逐 seed、1536 步和首 256 步恢复、KL、编码漂移、淘汰、更新量均见 [report.md](analyses/analysis001/report.md) 与 [analysis.json](analyses/analysis001/analysis.json)。两项训练前/拟合后平均 KL 为 .140321→.076371，诊断留出为 .163456→.116238；KL 降低不替代长期奖励判据。所有核验的 package 均 <=262144 B，actor 均为 89 B。

成本：监督程序计量首次 producer+失败审计 999.518698 CPU 秒，修正审计+冻结报告 82.018105 秒，工件哈希归档 4.932669 秒，合计 **1086.469472 CPU 秒（18.11 分钟）**。从首次启动至最终分析完成（含修复间隔）**832 秒（13.87 分钟）**；producer 本身墙钟 476.58 秒。零环境软件检查与 SSH 元数据查询为另记运维开销，不冒充 RL 交互。剩余交互 255392、计量 CPU 2513.530528 秒、campaign 墙钟 1868 秒；本轮已停止，余额不触发自动实验。唯一账本为 [budget.json](budget.json)。全程未用 GPU 或 sudo。

工件：远端 canonical `/data/robot/p001/e161/run001/producer`；本地完整镜像 [runs/run001/producer](runs/run001/producer)，3691 个工件、752637591 B，逐文件 SHA256 与压缩包校验通过。[来源](runs/run001/provenance.json)、[逐文件清单](runs/run001/artifact-manifest.json)、[镜像映射](runs/run001/mirror.json)、[最终审计](runs/run001/retention_candidate_audit/main002.json)。原失败审计 main.json 和两个监督状态均保留。远端整个项目约 2.905 GB，低于 100 GB 上限；不清理其他用户文件。

实际图：[PNG](analyses/analysis001/figure.png) / [PDF](analyses/analysis001/figure.pdf) / [SVG](analyses/analysis001/figure.svg)，由冻结 reporter 命令读取本轮原始账本生成；[analysis001 来源与图绑定](analyses/analysis001/README.md)。草案中两个预期图名未另行生成，表格已报告全部边界，不伪造已存在图。

**停止决定：不追加 seed、不修改窗口、不升级为完整流验证，也不启动 H035。** 结果保留“行为拟合存在较小的条件性正信号”和“keep 的适用边界”，但没有独立确认两项联合、达到预定量级的收益；不把未通过解释成小效应为零或否定整个有限容量在线知识问题。H026 维持 INCONCLUSIVE。
