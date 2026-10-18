# 三篇论文对“持续在线 RL + 有界持久知识”的决策记录

日期：2026-10-18
研究目标：W0 预训练后部署真实在线 RL，参数持续更新；在固定小容量内对局部/共享知识做形成、保留、整合或替换，并评估未来未知情境与切换下的长期价值。报告依据现有研究设置：2 个局部槽、1 个共享槽、old/new 候选；keep / replace / mean / none 后继续完整 12,288 个 environment steps；PPO rollout 长度为 256 个环境步，update 次数另行计数。实验为 8 seeds、32 prefixes、256 branches。未来复现比例会改变相对动作效果，但 past-frequency 没有同时优于两个固定端点，历史 gain / lower 也未带来增益；旧来源各训练 2,048 个 environment steps、新来源各训练 1,024 个 environment steps，尚未隔离成熟度。

## 阅读与版本说明

| 附件 | 题名 | 附件页数 | 状态 / 版本说明 |
|---|---|---:|---|
| `C:/Users/EthanLi/Desktop/paper/52262_Decomposed_Online_Learni.pdf` | *Decomposed Online Learning for Adaptive LLM Inference in Formal Mathematics* | 38 | 封面标为 ICLR 2027 under review。未核实到完全对应的公开版本；不据此称其已录用或赋予 arXiv 编号。 |
| `C:/Users/EthanLi/Desktop/paper/35029_Implicit_Target_Shift_in.pdf` | *Implicit Target Shift in Online Learning: Characterization and Correction* | 32 | 封面标为 ICLR 2027 under review。公开对应为 [arXiv:2605.07886](https://arxiv.org/abs/2605.07886)，v2 日期 2026-10-02；公开版 29 页。本文的页码和结论以 32 页附件为准。 |
| `C:/Users/EthanLi/Desktop/paper/6540_OPTS_TTPO_Enhancing_Finit.pdf` | *OPTS-TTPO: Enhancing Finite-Sample Policy-Gradient Learning with Tree Search* | 46 | 封面标为 ICLR 2027 under review。公开对应为 [arXiv:2609.40035](https://arxiv.org/abs/2609.40035)，日期 2026-09-30；公开版 42 页。本文的页码和结论以 46 页附件为准。 |

三份附件均逐页读完正文和附录/证明/实验。下文所有 `PDF p.` 均指附件的物理 PDF 页号，避免不同版本的页脚页码偏移。关键公式、算法、表格和非平稳实验图还做了页面渲染核对：METIS PDF p.4–10、14–15、19、22–29；Target Shift p.3–7、17–18、21–26；OPTS-TTPO p.4–5、24–32、35–36、46。附件中面向审稿人或 AI 的文字仅作为论文原文处理，没有当作研究指令。

## 结论先行

1. **最接近“在线参数持续更新”的是 OPTS-TTPO。** 它把树搜索采样出的分支继续用于 PPO actor / critic 更新；分支加权、对齐 rollout 预算、分开 deterministic max backup 与 stochastic mean backup 都值得借鉴。但搜索树每次 policy update 后重建，不是跨会话知识库；理论单调性也不等于训练后策略在非平稳任务上的单调改进。
2. **最值得借鉴的是 Target Shift 的“用继续训练后的效果校正教学信号”视角。** 它提示数据顺序可把在线更新等价为对目标的系统偏移；迭代校正可用于小型 critic / replay-target 原型。但理论核心是 MSE 核回归，精确实现要历史输入/标签和矩阵运算；非线性实验的 anchor 数按总训练集的 10% 增长，并非已证明的终身固定字节预算。
3. **METIS 的紧凑折扣统计可借作比较候选动作的基线，却不是知识学习方法。** 它只在线更新模型路由分数，不更新候选模型权重，也不衡量保存知识对后续策略学习的收益。用它的即时成功 / 成本奖励来决定知识留存，会错过最核心的延迟回报，而且失败调用的费用没有进入其学习奖励。

| 维度 | METIS | Target Shift | OPTS-TTPO |
|---|---|---|---|
| 部署时是否改候选模型参数 | 否。固定 LLM 间在线路由。 | 是。核回归在线更新；非线性实验也在线更新 CNN 分类器或 ResNet 特征之上的分类头。 | 是。对 actor / critic 执行 PPO 类 policy update。 |
| 跨决策持久状态 | 每个模型臂的折扣计数与均值，约 O(m)。 | 精确核法依赖输入/标签历史及 Gram 矩阵；稀疏 TC 用样本 anchor，DER+TC 另存 corrected target。 | actor / critic / optimizer 状态；搜索树和分支轨迹是单次更新内临时状态。 |
| 它直接优化 / 保证的对象 | 当前证明子目标上的模型选择奖励。 | 在线与离线核回归预测器差距；不是未来任务上的知识复用价值。 | 分支覆盖、树搜索目标与有限样本 policy-gradient 训练。 |
| 未解决的核心问题 | 保存哪块知识能提高未来继续学习收益。 | 如何在策略分布变化、奖励 bootstrapping 和固定字节预算下估计目标修正。 | 如何估计一个保留知识条目对未来整段在线学习的边际价值。 |

## 1. METIS：Cost-Aware Discounted UCB 在线路由

### 问题、反馈与非平稳性

METIS 用一个强模型先起草 Lean proof skeleton，将待填的 `sorry` 子目标拆成 proof obligations；更便宜的候选模型逐条完成子目标，Lean 编译器验证，最后合并证明。每轮对应“尝试解决一个子目标”，每个固定模型是一个 bandit arm（§3.1–4，PDF p.3–5）。反馈为成功位 (s_t(i)\in\{0,1\}) 和费用 (c_t(i)\)。只有 Lean 文件成功编译且所有 `sorry` 都消失才记成功；编译成功但 proof gap 未清零不算成功。

成本归一化为 \(\tilde c_t(i)=\min(c_t(i)/c_{ref},1)\)，奖励定义为

\[
X_t(i)=s_t(i)\bigl(1-\lambda\tilde c_t(i)\bigr),\qquad
\mu_t(i)=p_t(i)-\lambda q_t(i),
\]

其中 \(q_t(i)=\mathbb E[s_t(i)\tilde c_t(i)]\) 是**成功加权的费用**（§3.1，Eq. 4，PDF p.4）。因此失败请求不论成本多高都拿 0 分；失败费用虽计入论文报告的总费用，却不惩罚学习奖励。这一奖励定义不能直接移植到真实 RL，除非奖励本身明确覆盖全部交互与调用成本。

非平稳性来自 proof obligation 难度、API 价格、model serving 和 uptime 变化；算法把每个 arm 的均值建模为可能随时间变化的序列。消融与理论使用的是更窄假设：reward 在轮次间独立、均值向量分段常数、变化点数有限（§3.1–3.2，PDF p.3–4）。它不分析任意快变化、与 policy 更新相关的反馈，或候选知识内容随训练改变而使 arm 本身发生变化。

### 算法与保证

CAD-UCB 在每次选择前维护折扣次数与均值（§3.2，Eq. 2–3，PDF p.4）：

\[
N_t(\gamma,i)=\sum_{s<t}\gamma^{t-1-s}\mathbf 1\{I_s=i\},\qquad
\bar X_t(\gamma,i)=\frac{\sum_{s<t}\gamma^{t-1-s}R_s\mathbf 1\{I_s=i\}}{N_t(\gamma,i)}.
\]

先对每个 arm 做一次 warm-up；随后选最大
\(\bar X_t(\gamma,i)+2B\sqrt{\xi\log n_t(\gamma)/N_t(\gamma,i)}\)，其中 \(n_t=\sum_iN_t\)、\(B=1\)。\(\gamma=1\) 退化为普通 UCB；\(\gamma<1\) 使旧观测淡出。附录实现参数为 \(\lambda=0.5\)、\(\gamma=0.99\)、\(\xi=0.05\)、\(c_{ref}=0.0577\)（§C，PDF p.19）。特别是，实际实验的 \(\xi=0.05\) 不满足 Proposition 1 所需的 \(\xi>1/2\)；因此该 regret 定理不直接覆盖论文报告的这组实用参数。

Proposition 1 的期望 regret 界为 \(\mathbb E[Regret(T)]\le C\sqrt{T\Upsilon_T\log T}\)（Eq. 5，PDF p.4；证明 Appendix A，PDF p.14–15）。条件是：轮间独立的有界奖励、均值分段常数且至多 \(\Upsilon_T\) 个变化点、所有正的最优臂 gap 有下界 \(\Delta_{min}>0\)、\(\xi>1/2\)，并按 \(\gamma=1-\frac14\sqrt{\Upsilon_T/T}\) 选折扣率；常数依赖 arm 数、最小 gap 和 \(\xi\)。因此它是**当前臂路由 regret** 保证，不是策略梯度、继续训练收益、或存储内容未来复用的保证。附录证明没有放宽这些关键假设。

### 实证、baseline 和在线状态

METIS-DS 有 290 个 sublemmas、四个候选模型输出、约 13.4K 行 Lean code，覆盖九类数学主题（§4、Appendix C / Table 5，PDF p.5、17）。主 benchmark 为 FATE-M、miniF2F-v2s IMO 子集、Putnam 2025、DeepSeek-ProverBench；router baseline 包括 Random、kNN、MF、Graph、CausalLM，以及 no-routing / cheapest-model 等（§5、Appendix F，PDF p.6–9、22–23）。

| 附件定位 | 结果与解读 |
|---|---|
| §5.2 / Table 1，PDF p.6 | 子目标路由：FATE-M 上 METIS 89.3% / $98.24，对比 no-routing 的 93.3% / $250.56；miniF2F 上 METIS 85.0% / $75.18。正文说明 router 预算 $20 / problem、base no-routing 预算 $50 / problem，因此不是所有行都在同一预算上比较。 |
| Table 2–3，PDF p.7 | 完整问题 query-level routing 中 CAD-UCB 在 FATE-M 88.7% / $7.49；Putnam 2025 为 7/12、$295.96，对比 no-routing 5/12、$528.23。静态 query router 与分解式 routing 的任务粒度不同。 |
| Appendix F / Table 8–10，PDF p.22–23 | DeepSeek-ProverBench 所有方法均 88.9%，METIS $13.41，MF $13.11，故此项没有最低费用优势。FATE-M 表中 METIS 89.3% / $98.24；miniF2F 为 85% / $75.18，CausalLM 也是 85% / $82.46。 |
| §5.3 / Figure 4–5，PDF p.8–9；Appendix H，PDF p.26–29 | 动态模型目录 replay 先重放预计算模型输出，再中途把 DeepSeek-V4-Flash 成本提高 5 倍；uptime 实验重放 OpenRouter 每 5 分钟记录，重复 20 次。CAD-UCB 会改变模型调用占比。它们隔离了 router 的统计适应，但不是在持续变化的线上服务中实时生成 proof。附录中 full-provider trace 下 UCB 在 accuracy 上略高于 CAD-UCB，CAD-UCB 的成本及方差略低，显示 tradeoff 而非全面胜出。 |

静态 router baseline 使用 METIS-DS 适配数据训练；METIS 自身也先对每个模型 warm-up。因此“无需先验训练”不等于完全没有暖启动反馈。METIS 运行时不更新候选 LLM 权重，也不保存已学会的证明内容；它保留每个 arm 的折扣计数和均值，约 O(m) 数值。当前 proof skeleton、Lean state、subgoal prompt 和模型返回是任务期上下文 / 交互成本，不构成跨任务知识库。路由器本身的统计更新开销小；主要交互成本仍是强模型 draft、各子模型调用和 Lean 工具验证（§4、Appendix C–D）。

### 对本研究的可迁移点与边界

**可借鉴：**O(m) 的指数折扣统计是“需要极小持久状态来跟踪非平稳反馈”的干净固定端点；成本/收益应同一奖励里计量，且应对比 \(\gamma=1\)、折扣频率策略和固定策略。它也提醒我们，把分支当作相互独立 arm 前，应说明反馈是否独立且候选定义是否固定。

**不能直接套用：**METIS 在每轮看到一次当前证明的成功 / 费用，不估计一个 memory edit 对后续 12,288 个 environment steps 的在线 PPO、未来切换、策略回放比例或最终 return 的延迟效应。old / new candidate 也不是静态模型 arm：其训练成熟度、内容和 future-use distribution 都可不同。按当前冻结策略 reward 计算的 bandit reward，正是可能让“短期看起来有效但不能持续学”的候选被误选的机制。其 reward 还会漏算失败尝试成本。

**最小可归因实验建议：**先不训练 router。用相同 prefix 和相同的 future-context schedule 配对分支，随机分配一个共享槽动作为 keep / replace / mean / none；把 old/new 来源的成熟度对齐，固定完整 continuation 为 12,288 个 environment steps、rollout 长度为 256，并单独记录实际 PPO update 数及存储字节。报告终点 held-out return、后续切换后的 return、实际交互数、在线更新耗时和持久 bytes。交互预算匹配与计算预算匹配分开报告：一组固定 environment steps / rollout budget，另组固定 wall-clock 或计算预算；不要同时要求两者完全相等。只在完整 continuation 的实际收益和即时 frozen-policy reward 之间建立可复现关系后，再把动作视作 arm 测试折扣统计。这样可检验 METIS 类机制的适用性，而不会把即时奖励误称为长期知识价值。

## 2. Target Shift：在线更新被等价改写成目标偏移

### 问题设定与关键公式

论文比较样本序列上的线性 feature / kernel 回归与对同一批样本做 batch kernel ridge regression。在线 SGD 的更新顺序让样本之间产生单向、因果的作用；离线 Gram 矩阵则包含对称的双向作用（§3，PDF p.3–4）。给定顺序数据 \((X_n,Y_n)\)，在线预测器可写为一个用有向上三角 Gram 矩阵 \(K_U\) 的闭式解（Lemma 3.1，Eq. 5–6，PDF p.3–4）。作者定义有效目标

\[
Y_n^e=Y_n\left(\frac1\eta I+K_U(X_n,X_n)\right)^{-1}
(\gamma I+K(X_n,X_n)),
\]

在 \(\eta,\gamma>0\) 时，作者证明在线预测器与在相同输入 \(X_n\)、替换目标 \(Y_n^e\) 上训练的 offline kernel predictor 完全相同（Theorem 3.2，Eq. 8–10，PDF p.4）。新样本预测误差会按与旧输入的 kernel 相似度移动它们的 effective targets（Eq. 11，PDF p.5）；这使“在线偏差”可被解释成数据目标受到了隐式变形。

反解得到完整数据上的 exact corrected target（Eq. 12–14，PDF p.5–6）：

\[
Y_n^c=Y_n(\gamma I+K)^{-1}\left(\frac1\eta I+K_U\right).
\]

它能让在线更新精确匹配指定的 offline predictor，但矩阵依赖完整 \(X_n,Y_n\)，所以用于真正在线到达流时是非因果 / look-ahead 的。论文明确指出直接用 exact labels 的中间学习曲线会不稳定（§4.2，PDF p.8）。

因果版本把目标按块更新：冻结过去 corrected targets，只为新输入块最小化 online/offline predictor 差异并加 Tikhonov 稳定项 \(\gamma_o\)（Eq. 15，PDF p.6）；闭式新 target 如 Eq. 16–17，由在线残差项 \(C_{on}\) 与注入 offline-prefix 结构的 \(C_{off}\) 两部分组成。重排后的式子 Eq. 18（PDF p.7）显示两者如何共同改写新样本 residual。它仍要访问既往输入、真标签和相应 kernel 子矩阵，因果性不等于常数内存。

### 理论限定

任意数据顺序下，Proposition 4.3 给出在线 / 离线预测器 gap 的上界：\(\|f_{on}-f_{off}\|_H\le\frac1{\sqrt\gamma}\sum_i\|Y_i-f_{off}(X_i;X_{<i},Y_{<i})\|_F\)（Eq. 19，PDF p.7）。这是前缀 offline predictor 累积误差界，**不保证非 IID 下 gap 会缩小**。相较之下，未经修正的差距可能依赖有向 kernel 矩阵的最小奇异值；因 \(K_U\) 非 normal，该值没有统一下界（Appendix A.10，PDF p.21–22）。

论文的 IID 收敛只在 Appendix A.11 的额外条件下成立：有限维特征、标量输出、\(\|\phi(x)\|=1\)、协方差矩阵下界 \(\Sigma\succeq\rho I\)、可实现线性目标 \(y=w_*^T\phi(x)\)，并对样本独立同分布。Eq. 20（PDF p.7）及完整 Proposition A.5 / Eq. 100（PDF p.25–26）界的是**期望平方 RKHS gap**：为一个 \(O(1/n)\) 项加指数衰减项；不是一般非平稳深度 RL 的策略 regret，也不是对普通 gap norm 的 O(1/n) 声明。

### 实验、baseline 与实际内存

先在 MNIST NTK 上展示 vanilla online、exact correction 和 iterative correction：exact correction 可最终达到 offline NTK 表现但中途不稳定；iterative correction 曲线更接近 offline（§4.2，PDF p.8）。随后把机制启发式用于非线性网络（§5，PDF p.8–9；Appendix B–C，PDF p.27–32）：CIFAR-10 五个顺序 binary tasks 的 domain-incremental single-head 设置，以及 CORe50 50 类 / 9 个 task 的 class-incremental 设置。模型以 MSE 和 mini-batch SGD 持续更新；CORe50 的预训练 ResNet-18 feature extractor 冻结，只有两层 classifier head 更新。empirical NTK 在 CIFAR 每任务开始时更新，CORe50 每个 task epoch 后重算。非线性部分使用变化的 empirical NTK，是理论外的启发式。

| 附件定位 | 主要观察 |
|---|---|
| Appendix B Table 1，PDF p.27 | 10 seeds 的 CIFAR 与 5 seeds 的 CORe50。Vanilla SGD 为 57.19% / 13.59%；TC（10% anchor）为 64.57% / 22.54%；DER+TC 为 67.64% / 32.04%。但 CIFAR 的 ER / DER++ 为 69.49% / 69.90%，CORe50 的 ER / DER++ 为 28.29% / 27.59%，所以单独 TC 并未普遍胜过强 replay baseline；DER+TC 才在 CORe50 高于比较项。 |
| Appendix B.1，PDF p.26 | 普通 random-order CIFAR 单 pass 中，TC 没有清晰改善；multi-epoch 时优势才出现，支持其效果依赖结构化分布偏移。 |
| Appendix B.3 / Eq. 118，PDF p.27、32 | DER+TC 用 corrected target 替代 DER 的冻结旧网络输出；作者报告 CIFAR 与 CORe50 都高于 DER。它不能与“只靠 bounded knowledge library 改善参数学习”混为一谈，因为这个方法显式保留旧样本作为 replay。 |

精确 kernel 解要保留输入/标签历史并构造、求解 \(n\times n\) Gram 系统；朴素稠密分解随样本数增长，存储至少是 O(n) 样本加 O(n²) kernel 项，直接矩阵求逆约 O(n³) 运算。在线分块公式降低单次标签计算的范围，但仍需前缀状态 / anchors 来构造 kernel 交互。非线性实验对 sparse TC 设 \(M=0.1N_{train}\)：CIFAR 3000 个、CORe50 600 个 anchor（Appendix B Table 1，PDF p.27；C.3，p.31–32）。这约束的是固定 benchmark 上的样本数比例，随任务总样本增长而增长，不是终身不变的绝对容量。DER+TC 每个 replay slot 还保存一份 corrected teaching vector \(z_c\)；TC anchors 保存样本和真标签。论文没有报告上述 anchors / kernel 的统一持久字节预算。

### 对本研究的可迁移点与边界

**可借鉴：**评价知识编辑时不能只看当下冻结策略的 reward。Target Shift 直接展示“当前目标误差”如何改变后续在线更新的轨迹；可以把同一思想缩小到 critic/replay target，测其对继续 PPO 更新后的 held-out return 是否有因果影响。其稀疏 anchor variant 也给出可控的子集基线，但必须用真实字节数而非 anchor 个数做预算。

**不能直接套用：**RL 的 critic target 带 bootstrap、行为策略不断变、状态分布随 actor 更新，且动作价值和 policy-gradient 的目标并非固定 kernel MSE 标签。target correction 不会自动给出“保留共享知识 vs 替换后，未来能多学多少”的估计。若 full history 与 Gram computation 被删减，理论等价关系也不再成立；要把 sparse correction 当独立近似重新评估。

**最小可归因实验建议：**分成离线诊断和在线结果两步。在同一 W0、同一批已记录 transition 和固定小 anchor-byte cap 上，只比较普通 TD / return targets 与 iterative-corrected targets 对 critic value、target error 和 actor gradient 的影响；相同 rollout 在这一步只用于控制数据差异，不将其称作在线训练收益。随后从同一 W0 / prefix 配对启动两种真实在线 continuation；由于 critic / actor 更新后策略会改变，允许各分支产生自然不同的轨迹，可尽量固定外生 RNG streams，并比较 held-out return、切换后回报和 anchor bytes。交互预算匹配与计算预算匹配分别运行或分别报告，避免承诺自然轨迹相同或同时严格匹配 wall-clock 与 environment steps。先在固定 linear critic feature 上做，再扩大到非线性 critic；这隔离“修正 target 是否帮助持续更新”，不声称证明原论文理论覆盖 RL。

## 3. OPTS-TTPO：用树搜索分配 on-policy 采样预算并继续更新 PPO

### 方法与关键公式

论文针对有限 rollout 样本下的 policy-gradient 方差 / 分支覆盖问题。朴素地把树中每条轨迹都当独立完整序列会对共享 prefix 过计数；Branch Aggregation Lemma 要求每个 parent 的子节点及归一化局部权重 \(\alpha_{p,c}\) 在观察其 outgoing transition 前即可确定。路径权重递推为 \(W(c)=W(p)\alpha_{p,c}\)，将其用于 trajectory / token gradient、whitened advantage 和 TreeGAE 聚合，才恢复相应 chain-transition 的期望（§3、Appendix A，PDF p.3–5、11–22）。这构成 TTPG、TreeGAE 和 TTPO 的 branch-weighted policy objective。

OPTS 根据观测轨迹的 policy-relative advantage 定义 branch value：\(\Delta(s_t;\tau)=-\sum_{k=t}^{n-1}\gamma^{k-t}A^\pi(s_k,a_k)\)（Appendix B.1，Eq. B.1，PDF p.22）。其期望等于从该 state 把行为策略换成当前 \(\pi\) 的 value difference；确定性环境下它可写成 \(V^\pi(s_t)\) 减观测 suffix return。实际用 TreeGAE 估计 \(\hat\Delta\)，依实现除以 suffix length 的 \((n-t)^\xi\)（Eq. B.7，PDF p.23；LLM setting 使用未惩罚的版本）。

Algorithm 1（Appendix B.3，PDF p.24）要点：每棵树先有一条 on-policy rollout；做 TreeGAE backup；每轮取当前 greedy path，delta-backup 后选最高分且高于 baseline 的分支位置；最多取 B 个位置，单次 batched rollout 同时从这些位置采样新 suffix，并从初始分布补足未用 batch slot；接回树并重算 TreeGAE。确定性环境用 max backup；随机环境用 mean backup，降低在少量样本上取最大值导致的噪声偏好。TTPO 再用加权树数据做 actor / critic 更新（Algorithm 2，Appendix C，PDF p.32）。

自适应 rebranch 状态由已观察结果选择，违反上面的 prefix-measurable 条件；均匀 \(\alpha\) 只能校正 branch multiplicity，不能消除 posterior-selection bias。确定性 max-backup 还会产生前缀 credit：\(g_{max}=g_{mean}+\nabla L_{search}\)（Eq. 8 / Appendix C.1，PDF p.5、28）。论文给出的 gradient-bias 上界为 \(\epsilon_0+\min\{2pC,\,2\rho_SC+\epsilon_{backup}\}\)（Eq. 9，PDF p.5），是**控制 bias 而非证明 unbiased**；\(\epsilon_{backup}\) 衡量额外 max-backup prefix credit。

### 理论保证的准确范围

Theorem 1–2（Appendix B.4，PDF p.25–28）证明树扩张预算越大，搜索 induced policy 的目标值不下降；Theorem 2 的真实 return 单调性要求：确定性环境、固定且精确的 \(V=V^\pi\)、可积 return、非负 baseline、独立且完整的 on-policy suffix rollout、suffix-closed admissibility，max-backup 并在 tie 时保留 incumbent。这里的量是把树 greedy path 执行出的 search policy \(\pi_j^S\) return，**不是经过有限样本 PPO 更新后的新 actor 策略保证**。当 critic 学习误差、随机转移、distribution shift 或 continuing-learning history 出现，不能直接套用此定理。论文自己的结论也明确说单调性限于 deterministic + exact-value setting，而 posterior-dependent expansion 的 bias 只被界住，没有消除（§5，PDF p.9）。

### 实证、baseline 与预算

| 实验 / 附件定位 | 设置与结果 |
|---|---|
| Branch aggregation，§4.1，PDF p.6 | frozen step-400 Qwen3-1.7B PPO；16,384 prompts；32 个独立 chain groups 作 reference；对比 NaivePG 和 TTPG，额外 suffix 数 K={0,1,3,7,15}。NaivePG relative bias 从 0.1251 升至 0.4884；TTPG 保持 0.1221–0.1260。说明“advantage 无偏”本身不足以避免 branch prefix 被重复累计。 |
| Exact-value search，§4.2 / Appendix B，PDF p.6、24–28 | 32 个 depth-4 deterministic binary trees；每种模式 1,920 个相邻预算比较，精确 V、max backup、zero baseline 下都同时提升 \(J_\lambda\) 和真实 \(J\)。learned critic 的汇总也随预算提高，但不是每个 benchmark 都单调。 |
| Test-time scaling，§4.2 / Fig. 4，PDF p.6–7 | 902 个 prompts，Smax=3；相同 rollout 数 k=8–128。reward-guided correct-answer coverage 为 .7561→.8259，IID 为 .7295→.8082；value-guided cons@k 从 .6652 提到 .6785（k=128，k=64 峰值 .6796），IID 对应 .6552→.6685。这里提高的是推理时覆盖 / majority accuracy，不是跨 task knowledge retention。 |
| Policy learning，§4.4，PDF p.8–9 | 五个 MuJoCo 任务各 1M env steps / 10 seeds；OPTS-TTPO 相对 PPO 的 final-100-log tail-return 增益 11.2%–28.6%，任务等权 +18.7%。Atari-57 10M steps / 3 seeds，human-normalized IQM 从 .247→.255（全程均值）、.357→.374（末 100 点）；PPO 仍在 26 / 22 个任务胜出。 |
| LLM RLVR，§4.4 / Table 3，PDF p.8–9；Appendix D，p.35–36 | 四个 Qwen3 变体、每次 update 4,096 rollouts、400 update、Smax=3；另与同 rollout budget 的 DAPO / REINFORCE++ 外部结果比较。四个模型 macro avg@32 与 pass@32 最优；但 Qwen3-8B micro avg@32 中 OPTS-TTPO .6882 略低于 DAPO .6888。正文数据为训练 Math12k + NuminaMath、共 16,384 prompts；902 个 eval prompts 做 overlap audit，去除 exact normalized match 但未排除 paraphrase。 |

baseline 包括 PPO、DAPO、REINFORCE++、IID sampling、固定位置 branching，以及分支加权 / backup 消融。环境部署中模型参数确实持续更新；每轮的搜索 tree 则在本次 rollout / policy update 内使用，下一轮从当前 policy 重新生成。Appendix D.9 的快照只为分叉采样临时恢复 simulator state；预算把新采样 transitions 计为 interaction，tree 每次 policy update 重建，carry-over prefix 不进入 branching。论文没有把树当作跨更新的知识库，也没有给树知识条目持久字节指标。

### 持久与临时计算成本

训练后 actor / critic / optimizer 会延续，这是标准在线 RL 的持久状态；search nodes、actions、returns、advantages、环境 snapshot 和 rollout buffers 是本次更新的临时工作集。对 Qwen3-1.7B、Smax=3、8 张 H200 的 wall-clock composition（Appendix L，PDF p.46），每 step 合计 796.0 s：rollout 348.1 s（43.7%）、critic update 214.4 s（26.9%）、actor update 131.4 s（16.5%）、critic forwards 56.5 s（7.1%）、old log-prob 39.2 s（4.9%）；search selection 3.2 s（0.4%）、TreeGAE 2.7 s（0.3%）。树选择计算本身占比小，但 branching / 额外 rollout 仍有交互与模型生成成本。附件没有将 GPU 权重字节、optimizer checkpoint 字节、tree 峰值 RAM / GPU bytes、环境快照 bytes 换算为统一总容量。

### 对本研究的可迁移点与边界

**可借鉴：**(1) branch weighting 防止共享 prefix 因 fan-out 被重复计 credit；(2) 用当下观测做分支点选择时要单独标出选择 bias，不能把分支更多直接说成 unbiased；(3) tree 搜索收益需要与等 interaction / compute budget 的 IID 或固定分支比较；(4) deterministic max 与 stochastic mean backup 应分开检验。由于它确实在 PPO 中继续更新参数，是三篇中最接近本研究执行形态的算法。

**不能直接套用：**tree path return 是当前 search / policy 的短视估值；它没有测出将条目永久保存后对许多未来 policy updates 的可复用价值。理论最强版本需要确定性转移和 exact critic，LLM RL、Atari、一般在线 RL 不满足。paper’s matched rollout budget 也没有等价于本研究所需的“同一总持久容量 + 未来 switch 后性能”。树是临时采样结构，不能把 tree node count 当作 bounded persistent knowledge bytes。

**最小可归因实验建议：**固定一种知识库动作（例如每个 prefix 都 keep），其余 PPO 配置和 memory bytes 固定；配对比较 IID rollout 与 OPTS 分支采样，并完整继续 12,288 个 environment steps（rollout 长度 256，另报 update 数）。分开做 interaction-matched 比较和 compute / wall-clock-matched 比较，不要求分支因策略变化仍生成同一真实轨迹。可尽量使用固定外生 RNG streams；在预注册的未见 task 与一次切换后测 return、forgetting、更新速度和峰值临时内存。若 OPTS 只提高冻结策略的 coverage，却不提高继续更新后的跨任务 return，就不能把其 branch score 当作该共享知识的保存价值。

## 三篇合并后对下一步实验的判断

当前问题不是再证明“折扣在线统计存在”“共享可减少重复”或“branch search 会增加覆盖”，而是缺少一个**可归因的延迟价值标签**：在有限持久 bytes 下，某个旧 / 新 / 合成 knowledge state 被保留后，经历规定的持续参数更新和未来情境切换，最终让策略比替代动作多学到多少。三篇给出的机制分别覆盖即时 arm 反馈、kernel target bias、临时轨迹重用；都没有单独回答这个 estimand。

对下一轮最应控制的混淆项：

1. 把 old/new source maturity 对齐，或把 maturity 作为随机化区组；目前 old=2,048、new=1,024 个 environment steps，不能把来源价值与成熟度分开。PPO rollout 长度为 256 个环境步，update 数应另行报告。
2. 对每个动作分别做 interaction-matched 与 compute-matched 比较，并固定或报告临时 rollout/tree 上限、持久库 bytes；报告 bytes 和 FLOPs / wall-clock，而不把交互预算与计算预算当作可同时严格匹配的单一约束。
3. 将即时 frozen-policy advantage / reward 与完整继续训练后的 held-out return 分开报告；使用不同 future reuse / switch schedule，让训练中的处理动作不泄漏测试情境。
4. 先用同一 prefix 的配对随机化估计 keep / replace / mean / none 的真实继续学习效果，再拟合 past-frequency / discounted bandit 或学习 history predictor。比较对象至少包括两个固定端点与等容量 `none`；不要只比较多个自适应 estimator。

### 范围与证据边界

本文已覆盖三份附件 PDF 全部页面：METIS 1–38、Target Shift 1–32、OPTS-TTPO 1–46；包括证明、补充实验、implementation / hyperparameter、dataset / reproducibility、runtime appendix。上文的理论边界以附件中的假设和定理为准；公开版本页数另列仅用于版本识别。三份附件封面均标明 ICLR 2027 under review；本文没有把它们写作已录用成果。视觉核验优先用于公式、流程图、关键 benchmark 表和 nonstationary replay 图，未声称逐个像素复查所有普通文本页。
