# E162 · H033 v1 固定三特征的离线增量排序诊断

- 主 H：[H033](../../research/records/H033.md)；关联 [H026](../../research/records/H026.md)、[E161](../exp161/README.md)。
- 登记/主控接受日期：2026-10-21（Asia/Singapore）。执行状态：`REGISTERED_PENDING_INDEPENDENT_REVIEW`；科学结果：`NOT_RUN`。
- 授权依据：用户明确要求主控确定实验、资源与推进条件，并交由“消融实验（1）”独立审查后验证。本 E 仅为本地离线诊断，新增环境交互为 **0**，无 GPU、无远程连接或传输；不沿用 E161 余额。
- 优先级/阶段：P0 必要条件诊断 / Stage A。触发：E161 未确认候选的联合保存收益，且实现侧已查证存在有界操作前信息并交付冻结 v1。此次验证是否值得继续投入排序器，不重试 E161，也不实施 H036/H035。

## 问题、冻结方法与可证伪预测

问题：在 E161 指定的 old_major/new_major 等权目标下，固定三个操作前特征能否支持比四个简单规则更有价值的 keep/replace_behavioral 选择？

唯一方法语义为 [METHOD_V1.md](../../research/appendix/H033/METHOD_V1.md)，执行交接为 [HANDOFF_V1.md](../../research/appendix/H033/HANDOFF_V1.md)。方法规格不是此次新结果的独立确认：E161 已见结果，E162 是结果后探索复用。

冻结 recent_frequency_contrast、new_fit_gain、old_prior_loss；旧保护对象为旧任务 local=0 的 keep 父先验，replace 仅绑定新组。候选按原 H026 有界 L-BFGS 从允许的 prefix 状态重构，不调用环境、不从 observer 补入候选或旧 source 教师。候选 float32 编码前代理与原服务器最终候选不保证逐位相同，记录平台/哈希差异，不能择优选择特征版本。

16 个 seed 固定，family×history 四层各 4。四折每折训练 12 / 留出 4，标准化只用训练行；ridge alpha=1，预测严格 >0 才 replace，并列 keep。对照固定为 always_keep、always_replace、训练均值 intercept-only、频率规则。不改折、不调参、不增删特征或 seed。

每 seed 的目标 y_s 是 6144 步下 replace_behavioral−keep，先平均两 replica，再将原 old_major/new_major 日程等权平均。balanced 不参与拟合、标准化、阈值或主推进标准，仅对相同留出决策作次要描述。outcomes 包需 192 个单元（16×3 futures×2 replicas×2 arms），其中主目标使用 128 个单元；独立评价单位始终是 16 个完整 seed。

## 主指标、统计与推进/停止条件（预测前固定）

令 V_s(m) 为方法 m 在 seed s 的两个主日程等权所选 reward/step。对每个简单规则 b，报告配对差 d_s,b=V_s(ridge)−V_s(b)，以及 G_b=mean_16(d_s,b)。主指标为 **G_min=min_b G_b**，即相对四个简单规则中表现最强者的选择价值差；不按全体标签挑选每个 seed 的 baseline。

支持证据必须是选择价值，不是总体 MSE、拟合 KL 或方向预测准确率。预设 Stage A 推进筛查条件同时为：

1. G_min >= **0.002 reward/step**（即相对四个基线均至少达到此量级）；
2. 将每个原定留出折的四个 seed 单独汇总，至少 **3/4 个折**的 min_b mean_fold(d_s,b) >0。

以上是决定是否值得提出独立新 seed 确认的**操作性筛查规则**，不是显著性检验，也不保证总体误报率。四折训练重叠，不能当四次独立实验。通过仅记 `PROMISING_EXPLORATORY`，仍须主控另行接受新确认计划与预算，不直接升级完整流或部署算法；未通过记 `NO_ADVANCEMENT`，停止本三特征/固定 ridge 路线，不据此证明所有过去信息无用。

不确定性单独作条件性描述：冻结 16 个留出决策及对应配对值，在原四个 family×history 层内有放回抽样（每层 4，整体 16），10000 次，NumPy PCG64 seed=162033。报告每个 G_b 和 G_min 的 2.5%/97.5% percentile 范围。**不重拟合 bootstrap 模型；该范围只反映既定留出结果的重抽样波动，不包含训练不确定性，不宣称正式 95% 覆盖，不用于替代推进判据。**不报确认性 p 值。

另报告全部 seed 的目标、预测、决策/选择次数、四个 fold 的配对均值、balanced 所选差值、特征范围/训练标准化和实际成本。MSE 仅作诊断；不得用 balanced、某个家族或另一个 horizon 挽救主结果。不得与 E161/E130 合池以增加样本量。即使结果正向，也不证明仅观察所选反馈的部署学习器有同等能力。

## 独立审查、执行与职责

执行负责人：“消融实验（1）”。主控接受了 v1 的范围及本 E 协议；运行前仍需该负责人独立审查标签身份、操作时序、old-local=0/new-bound 语义、完整 seed 留出及比较有效性。

源包 19 文件和 52 输入引用的大小/哈希已由主控的只读身份核对匹配；28 项合成检查是交付方已完成的软件证据，不能代替真实输入/输出审查。复用有效检查，只补必要验证。

只写 `experiments/exp162/` 内的协议接收/审查、执行编排、独立统计、日志、provenance 和结果；固定科学方法的 source_snapshot、方法合同、E161 全部原始文件与冻结报告、在线策略、共享研究索引均受保护。遇到方法语义/实现错误，先返回主控交还“算法实现”，不在实验中猜测或修改算法。编排/统计错误最多一次有记录的定向技术修复，保留失败 attempt，且全部成本仍须落在本 E 上限内；奖励/筛查结果不利不是重试理由。

审查通过后执行 features -> outcomes -> crossfit，每步成功才进入下一步。输出必须为新路径，不覆盖。features 不接收 outcome 路径；outcomes 与 features 分离；crossfit 完整验证。缺数据、哈希不匹配、owner/generation/绑定错误、非有限数或 seed 无效时停止整批，不补值、不删 seed、不增设候选程序。

本机 Python：`D:/MovedFromC/ChatGPT/online/.venv/Scripts/python.exe`。
冻结源码根：`D:/MovedFromC/ChatGPT/online/research/appendix/H033/delivery_v1/source_snapshot`。
工作/输出根：`D:/MovedFromC/ChatGPT/online/experiments/exp162`。三步命令使用交接中的固定入口，将 expNNN 唯一替换为 exp162、输出指向 `runs/run001/`。全 argv、cwd、环境、源码和输入哈希随 run 保存。

单个 CPU 执行进程，Torch/OMP/MKL/OpenBLAS 线程均为 1；不启动环境、PPO、GPU、远程命令或安装依赖。设置适用 TEMP/TMP、MPLCONFIGDIR、XDG_CACHE_HOME、TORCH_HOME 到本 E 的 project-local temp/cache；不改 HOME。禁止向冻结源码写 pycache（PYTHONDONTWRITEBYTECODE=1）。

## 本轮资源与停止边界

上限由主控在用户此次验证请求范围内分配：**本地累计 CPU 300 秒、campaign 墙钟 600 秒、单进程单线程、新增输出/缓存/临时文件合计 256 MiB、新增环境交互 0**。预估三阶段及轻量统计为几十秒 CPU、远低于空间上限；属于估计，不是实测。

计时覆盖首个 features 进程启动到独立报告完成，含导入、候选重构、模型计算、失败/重试、统计与执行核对；外部计量进程 CPU，不能只引用包内阶段计时。启动前只读审查/准备时间另列运维范围。运行前检查本地目标卷可用空间并预留 256 MiB；Windows inode/quota 不适用项注明，不伪造检查。触及任一上限停止本 E 工作并保存证据；无自动扩额/新 E。资源唯一账本为 [budget.json](budget.json)。

## Git + Data Paths + Figure Paths

| 要素 | 绑定 |
| --- | --- |
| 方法代码 | 完整父 SHA `81de4afd2d703d17c3953facb78fd04b19cb64a9` + [交付清单](../../research/appendix/H033/delivery_v1/manifest.json) 与其 allowlisted dirty snapshot；交付 identity `eed12dc721e9d3a56e6d079cd51ecd4bb29f87c884f362f044d97ada8ba0351c` |
| 冻结配置 | source_snapshot/configs/h033_diagnostic_v1.json；SHA256 `d15c0ba858a96b48598e1734a973ec31404b904bd5057c65ee170192d01852db` |
| 输入 | [E161 producer](../exp161/runs/run001/producer) 的 16 份 checkpoint.pt/actor.bin/evidence.npz + [原 analysis.json](../exp161/analyses/analysis001/analysis.json)，SHA256 `813561c90e3219bc1463133c67aead68103849b465c7a72bbf7c20fc798bbaf8`；52 引用见交付 manifest |
| run | PLANNED：runs/run001/features.json、outcomes.json、crossfit.json、执行日志与 provenance；失败 attempt 独立保留 |
| 独立分析 | PLANNED：analyses/analysis001/report.md、metrics.json、per_seed.csv、原样保存的统计脚本和精确输入身份 |
| 图 | NOT_APPLICABLE：16 行配对表与四折结果足够回答此最小诊断，不要求额外绘图 |
| 协议冻结 | 运行前由执行负责人核对并记录本 README/budget 及方法包哈希；之后只追加实际结果，不改写本节前置判据 |

E162 wrapper 绑定交付 manifest 中尚为空的 experiment_id；保留原 manifest 原样，不为填 E 编号修改冻结交付。分析/编排新增文件同样记录执行身份，不包含本机指令或凭据。

## 当前决定与交接

主控已登记本 E 并准许在独立审查通过后按上述本地边界执行。此刻尚无真实特征拟合或排序结果。执行完成后返回：审查结论、完整来源、16 个留出决策、各 G_b/G_min、折稳定性、条件性波动范围、全部失败及 CPU/wall/空间账本；由主控更新 H 状态/研究索引并决定是否停止。
