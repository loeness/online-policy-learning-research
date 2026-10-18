# 三篇在线学习理论论文：完整阅读与可迁移性分析

阅读日期：2026-10-18

## 阅读范围与问题边界

本报告对应用户提供的三份匿名评审 PDF。来源文件现位于：

- C:/Users/EthanLi/Desktop/paper/22161_Adversarially_Batched_On.pdf（36 页）
- C:/Users/EthanLi/Desktop/paper/52604_Oracle_Efficient_and_Par.pdf（35 页）
- C:/Users/EthanLi/Desktop/paper/40934_An_Exponential_Determini.pdf（57 页）

逐页全文提取记录位于 tmp/pdfs/online_literature_20261018/{22161,52604,40934}.txt。阅读包含正文、参考文献所在页、全部附录与证明；页码以下均指 PDF 页码。报告所述三份附件均为匿名 ICLR 2027 under-review 稿件，不据此声称录用。根任务提供的公开检索信息将第二篇对应到 arXiv:2610.10499，v1 日期 2026-10-07，作者 Sasha Voitovych、Adam Block、Alexander Rakhlin、Abhishek Shetty；本报告的页码、定理和算法分析以 35 页匿名附件为准。另两篇未确认公开对应版本。

我们关注 W0 预训练后在线 RL 继续更新参数，并用有界知识修正库共享形成、合并、替换或保留经验；树结构只是可能的实现。关键指标是未知情境和切换表现，以及持久容量、交互开销、CPU 与 wall time。当前实验为 2 个局部槽加 1 个共享位置，keep / replace / mean / none 规则；每次行动后局部 PPO 继续训练共 12,288 步，每 256 步更新一次，8 个 seed、32 个 prefix、256 个 branch。后续复现比例可能改变作用，但 past-frequency 尚未稳定胜过两个固定端点，历史 gain / lower 目前未显示额外价值。历史来源在样本数上不平衡（旧来源各 2,048 步、新来源各 1,024 步），成熟度仍需控制。以下仅提出下一步可归因的小实验，不表示已经运行或有效。

## 1. Adversarially Batched Online Learning

**书目信息。** PDF 标题页标题为 *Adversarially Batched Online Learning*；匿名 ICLR 2027 under-review 稿，36 页。正文在 PDF 第 1–12 页，目录之后的 Appendix A–I 位于第 13–36 页。

**问题、反馈和比较器。** 论文研究每轮损失仍逐轮产生、但学习者只能在外部给定的批次边界更新决策规则的在线学习。批次开始时学习者承诺一个策略 / 预定动作序列，不能依据批内新反馈调整；这不等于每轮实际执行同一个动作：随机 bandit 的预定序列可以轮转不同臂，而 OCO 在批内保持同一个决策。批次结束时才收到该批反馈。它分别分析固定未知均值的 i.i.d. 随机 bandit、在线凸优化、专家问题和对抗 bandit。随机 bandit 每批只观察所选臂回报，目标是对比最优固定均值臂的期望 pseudo-regret；OCO 批次汇总一阶反馈；专家问题每批获得专家损失，属于 full information；对抗 bandit 只看到被选臂的批次损失。各类界是相对静态最优均值臂 / 点 / 专家的 regret，并非动态比较器或 RL 策略改进保证。随机臂设定中分布固定；对抗损失可以变化，但学习者按给定批次才获得反馈。批次时刻由环境 / 协议提供，算法不自主决定下一次更新时刻或批长。

**关键量与操作。** 定义 1（PDF 第 3 页）用前一批结束时间 t_(j−1) 和当前批长 b_j 定义最大相对延迟 α = max_j b_j / t_(j−1)，并令 t_0 = 1。Algorithm 1（第 7 页，§2）是按轮转方式采样活动臂、逐步淘汰的批处理 successive elimination。若早期反馈块太短，算法把统计量合并累积，直到当前合并块长度至少达到前一块的 2 倍，再作淘汰；它延后比较 / 淘汰来应对既定的稀疏反馈边界，没有选择外部更新日程。

**核心结果。**

- Theorem 1（第 3–4 页）：随机 bandit gap-dependent 期望 regret 上界为 O(min{T, α² log(NT) Σ_(i:Δ_i>0) 1/Δ_i})。
- Theorem 2（第 4 页）：gap-independent 上界为 Õ(min{T, √(αNT)})。Proposition 1（第 4、17–20 页）给出 Ω(α Σ_i 1/Δ_i) 型 lower bound；Theorem 3 给出 Ω(√(αNT)) 下界。
- Theorem 4（第 4 页，证明在第 20–23 页）给出更强 gap-dependent lower bound，随 α 为 α^(3/2) Σ_i 1/Δ_i 量级，显示 gap-dependent 上下界仍有 α 的幂次差距。Proposition 4 的构造在证明中也展示了朴素逐批比较的损失。
- Theorem 5（第 4–5 页；Appendix D–F，第 23–32 页）：对凸 OCO，regret 上界为 O(DG‖b‖₂)；full-information experts 为 O(‖b‖₂√log N)；对抗 bandit 为 O(‖b‖₂√(N log N))，其中 b 为批长向量。Theorem 6 给出相应量级的下界 / 困难实例。这里必须保留反馈差异：专家能看完整专家损失，bandit 只能看被选项反馈。
- Theorem 7 与 Algorithm 2（第 5、8–9 页；Appendix G–I，第 32–36 页）：基于已发生批长平方和的 doubling / 自适应学习率方案，在 batch-oblivious 反馈设置中保持相应界。它能适配已实现的批长尺度或重启学习率，仍不决定批次边界。附录 I 比较朴素 doubling：例子中朴素 Hedge 可有 Ω(T) regret，而自适应方法给出 O(T^(3/4)) 量级。

**对当前系统的含义和边界。** 这篇最有用的是把“反馈不可用多久”显式纳入 regret：长更新延迟本身会放大成本；稀疏反馈时先合并统计再比较，可以推迟过早替换。它没有给出 agent 自主选择批长的算法或保证。当前 12,288 步是 continuation 总预算，PPO 每 256 步更新，并非一个冻结 12,288 步的 rollout / batch，因此不能把 12,288 直接代入 α 或据此判断当前节奏受理论保护。α 应按实际外部批次边界与该批开始前的累计轮数计算；若未来有明确的延迟反馈条件，应单独列出边界再分析。其结果只针对上述静态比较器和在线损失模型，没有 PPO 的参数非凸性、策略分布漂移、部分可观测状态、环境交互成本或固定容量库；定理也不计 wall / CPU 和持久存储。

**小型可归因实验建议（未运行）。** 保持 2 个局部槽、1 个共享位置、12,288 步 continuation 预算和每 256 步的 PPO 更新、种子与分支；只改变共享写入时机：立即提交一批，或暂存新数据，直到累计新数据达到上一次已提交块的 2 倍再提交 / 比较。两个条件保持库合并 / 替换规则相同。在线比较固定环境配置、外生随机数种子和交互预算，但允许不同策略产生不同轨迹。报告未知情境与切换性能、提交次数和每次提交的样本量、额外交互数、CPU / wall time、持久字节。这个对照只检验延迟提交是否在本工作负载中有用；论文的 batch regret 界不能直接推出实验会获益。

## 2. Oracle-Efficient and Parameter-Free Agnostic Smoothed Online Learning

**书目信息。** PDF 标题页标题为 *Oracle-Efficient and Parameter-Free Agnostic Smoothed Online Learning*；匿名 ICLR 2027 under-review 稿，35 页。公开对应版本信息见本报告开头。正文约在 PDF 第 1–13 页；Appendix A–C 为第 14–35 页。

**问题、反馈与分布假设。** 这是在线监督预测：每轮先看到 X_t 并预测，再看到真标签 Y_t。比较器 f 来自固定假设类 F，目标是对固定 f 的 agnostic static regret。每轮条件协变量分布 p_t 可以依赖此前交互，但要求相对于同一个固定而未知的基分布 μ 满足 σ-smooth：密度 p_t / μ ≤ 1/σ。它不是任意非平稳分布保证，更不是策略交互式 RL。反馈是完整标签对 (X_t,Y_t)；对有限候选函数，可以在标签揭示后计算各候选的损失，因此是 full-information supervised feedback，不是 bandit。

**算法操作。** 二元标签的 Gaussian FTPL（§2、式 (2)/(6)，PDF 第 3、6 页）每轮对过去样本的候选损失加入高斯扰动，调用加权 agnostic ERM，选出最小加权损失假设，再预测。Theorem 3.1（第 6–9 页）给出的期望 regret 为 O(d log²(2T/σ) √(T/σ))，即 Õ(d √(T/σ))；算法不需预先知道 T、σ 或 μ。论文指出统计上基于 VC 维的率约为 √(dT/σ)（忽略对数），所以其界在 d 依赖上仍差 √d。调用数是每轮一次、总计 T 次加权 ERM，但每次目标都含历史样本和随机权重。

实值凸、1-Lipschitz 损失的版本（Theorem 3.2，第 6–7 页）每轮取 t 个独立 FTPL 解并平均预测。其界为
inf_(ε≤1/16) O((1 + fat_(cε)(F) + √T ε) √(T/σ) log^(7/2)(2T/(σε)))。
在 fat-shattering 维数受 d log(1/ε) 控制时为 Õ(d √(T/σ))；若 fat 维数为 O(ε^(−α))，论文推得 Õ_α(T^((2α+1)/(2α+2)) / √σ)（第 6–7、30–35 页）。实值版本每轮 t 次 oracle 调用，总量 Σ_(t=1)^T t = T(T+1)/2，而非“每轮一次”。

**Oracle 与存储成本。** 二元版本有 T 次 ERM 调用，实值版本为 Θ(T²) 次；论文的 oracle-call 计数不等同于 wall-clock 快，也不保证固定 memory。二元每次调用需要访问至多 t−1 个历史样本；若朴素地把每个历史目标完整物化，各轮累计会是 O(T²) 样本项。实值版本按轮累计输入项上界可达到 O(T³)。后两个总量是由算法逐轮历史目标推得的朴素实现成本，不是论文对优化后 storage / CPU 的定理。其 oracle 模型把一次加权 ERM 查询计作一次，不按权重向量长度、迭代次数或输出成本收费。

**对当前系统的含义和边界。** 可借鉴的是把候选策略 / 修正规则的历史评分放进扰动式选择器，并明确统计 oracle / 评估成本；不应把参数无关或“一次 oracle 每轮”误读成固定内存、快 wall time。RL 下候选策略会改变后续状态与回报，通常只有被执行策略得到反馈，故缺少 full-information 标签；rollout 回报的分布也未证明相对同一个 μ 满足 σ-smooth。该论文没有证明 Gaussian FTPL 能保证 PPO 继续训练后的回报、未知情境表现或受限共享槽收益。

**小型可归因实验建议（未运行）。** 固定 PPO 检查点和同一组现有候选快照。可先用同一组离线回放数据比较历史累计分数的确定性最佳候选选择与 Gaussian 扰动选择，但只将其视为给定 replay 上的选择器诊断。若比较在线未知情境 / 切换表现，则固定环境配置、外生随机数种子和交互预算，允许候选策略自然产生不同轨迹；记录额外交互、候选数、每轮实际评估数、CPU / wall time、保存的评分历史字节，以及未知情境 / 切换指标。候选集合、槽数与训练更新固定。将其称为选择器启发式对照，不把监督学习 regret 界迁移到 RL。

## 3. An Exponential Deterministic-Randomized Separation in Oracle-Based Online Learning

**书目信息。** PDF 标题页标题为 *An Exponential Deterministic-Randomized Separation in Oracle-Based Online Learning*；匿名 ICLR 2027 under-review 稿，57 页。正文 PDF 第 1–10 页，Appendix A–F 从第 11 页延续至第 57 页。

**问题、反馈与成本单位。** 这是 transductive online learning：T 个互不相同的例子预先给出，存在未知线性顺序和固定阈值目标 c_z(x)=1[x≼z]。每轮预测后揭示目标标签。Oracle 是一致性 oracle，可针对样本返回一个与样本一致的完整标签向量，或返回不可实现符号 ⊥；另一个弱版本只返回 realizable / 不可实现这一位。论文计错数 M 和 oracle 查询数 Q，不以 regret、样本交互次数或运行时间计成本。环境在实验开始时固定顺序和阈值，因此不是非平稳问题；查询提供的反馈可能依赖学习者选定的样本集。一次完整 oracle 调用即使输出许多标签，成本仍按 1 计。

**主要结论。**

- Theorem 3.1（PDF 第 4–6 页；Appendix A）：面对一类固定预先声明的 min-prefix / max-prefix 极端选择规则，确定性学习者最坏情况下满足 M+Q ≥ T−ε（空前缀约定 ε=0，非空 ε=1）。因此确定性 learner 若想用 O(log T) 错误，最坏情形需接近 T 次调用。Corollary 4.1 扩展到对手先选、随后冻结的 oracle 规则。结论依赖选择规则与对手构造，并非任何 oracle 的统一下界。
- Theorem 3.2（第 4、7–8 页；Appendix C，第 29–41 页）：对固定的 minimal-prefix 规则和 oblivious 有限 hard distribution，随机学习者满足 E[M] ≥ ((T+1−ε)·128^(−E[Q])−1)/2。与论文引用的 AHR25 上界合并，得到随机 ERM 查询复杂度和期望错误均为 Θ(log T) 量级。其证明借助 revelation oracle；单次查询可能揭示大量标签，不能把 Q 当成每次只能读一个样本。
- Theorem 3.4（第 5、19–21 页）：oracle 规则影响结果。选用合法的 median 规则时，确定性算法能做到 M,Q ≤ ceil(log₂T)；而 global-median 规则有线性下界 M+Q ≥ T−m，m=floor((k₀+T)/2)。论文的指数随机化分离依赖具体 oracle 选择行为，不是所有合法 ERM 规则都如此。
- Appendix D（第 41–48 页）给出小 T / 固定调用预算下的精确 regime；例如 M*_E(5,3)=3、M*_N(6,4)=3，说明中间区域不能简单用 max(T−Q, floor(log T)) 概括；当 Q≥2(T−1) 时，已知顺序下足够达到对数错误。
- Appendix E（第 49–55 页，Theorem E.1）：若弱 oracle 只回答可实现性一位，确定性策略有 66T / 67(T−1) 型调用上界，而随机及确定性 O(log T) 错误都需要 Θ(T) 量级调用。弱 oracle 信息量变化会消除强 oracle 下的调用优势。

**对当前系统的含义和边界。** 这篇支持的窄结论是：当某种昂贵选择器 / 评估 oracle 允许一次返回结构化的大量信息时，随机化与 oracle 的 tie-breaking 规则可能显著改变查询数。它不支持“有界知识库树越深越好”或“随机替换对 PPO 有指数优势”。论文中的 Q 只记抽象 oracle 查询；查询可携带大样本并返回 T 维标签向量，未计每次扫描量、调用延迟、CPU / wall time、存储，也没有 RL policy 更新。仅有可行性 bit 时调用复杂度完全不同；因此必须先定义当前系统里一次候选评估到底返回什么、读写多少数据、耗时多少。

**小型可归因实验建议（未运行）。** 固定当前候选快照与评估预算，只改变候选分数相同（或预先约定容差内相同）时的 tie-break：固定选最旧、固定选最新、按候选排序的中位项、随机选择。纯离线诊断可以固定回放轨迹，但结论限于给定数据上的选择器行为。在线比较则固定环境配置、外生随机数种子和交互预算，允许不同策略产生不同轨迹；保持槽位、PPO 训练量和提交时刻一致。记录未知情境 / 切换指标、候选评估查询数及每次读取字节、额外交互、CPU / wall time、持久库字节。该实验检验实际库选择语义及成本，不检验论文的阈值 oracle 定理。

## 汇总：可借鉴的决策点与不可迁移假设

| 论文 | 可以形成的具体工作假设 | 保证没有覆盖的部分 |
| --- | --- | --- |
| Adversarially Batched Online Learning | 稀疏反馈时先合并样本，延迟共享库比较 / 淘汰；报告真实 rollout 批长与延迟。 | 不让学习者选择更新时刻；静态 regret / 固定均值 bandit，不是 PPO；成本界不含库容量或 wall / CPU。 |
| Oracle-Efficient and Parameter-Free Agnostic Smoothed Online Learning | 用带扰动的历史评分作为候选修正选择器的对照，并完整计 oracle 调用和历史数据访问。 | 假设全标签反馈与固定未知基分布平滑性；一次 ERM 调用并不代表快或定长 memory。 |
| An Exponential Deterministic-Randomized Separation in Oracle-Based Online Learning | 预先定义一次候选评估的信息输出、tie-breaking 与计费单位；比较随机和确定性选择规则。 | 固定 realizable threshold、特制 oracle；查询按常数收费，可能一次输出大量标签；不是 RL 或存储受限保证。 |

三个理论方向在当前问题上的共同价值是帮助设计有归因的操作对照：何时提交、如何利用历史反馈选候选、以及同分时怎样打破平局。但它们的比较器、反馈模型和成本单位都与在线 RL 更新共享库不同。只有在控制 PPO 更新量、数据成熟度、复现比例、库容量和评估交互的实验中，观察到未知情境 / 切换收益并计入 CPU、wall 与持久字节，才能判断这些启发式是否值得保留。上述建议尚未执行，不能视为有效结论。

## 全文覆盖记录

- 22161_Adversarially_Batched_On.pdf：PDF 第 1–36 页逐页阅读，包括正文、全部 Appendix A–I 的构造、引理与证明。
- 52604_Oracle_Efficient_and_Par.pdf：PDF 第 1–35 页逐页阅读，包括正文、全部 Appendix A–C 的证明与复杂度推导。
- 40934_An_Exponential_Determini.pdf：PDF 第 1–57 页逐页阅读，包括正文、全部 Appendix A–F 的下界、构造和固定预算 / 弱 oracle 结果。
- 三份 PDF 页数与按页文本提取记录相符；图 / 公式可由文本提取辨认，本次没有额外渲染页面。
- 本次只新增本报告；未运行 RL、未运行验证实验、未复制 PDF、未修改其他文件、未 push。
