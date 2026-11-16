# ICLR 2027：持续在线策略学习与有限容量知识组织文献筛查

更新时间：2026-11-10。本文是文献线索和摘要层面的研究分析，不是论文全文审查或本项目实验报告。

## 检索范围与证据

项目问题限定为：预训练初始化后，以真实在线交互持续更新策略参数；在有限持久容量下形成、组织、写入和保留可重用知识。包含与此密切相关的策略保留、复用、行为固化、负迁移和评估诊断；离线 RL、世界模型和机器人策略单列，避免误认为同一设置。

用户提供的第三方公开目录快照及全部摘要分片：构建日期 2026-10-07，43,013 条记录、87 个摘要分片；目录副本的 SHA-256 为 `B1422AC10231B7B8FA1D868996E8FDA8434B7FF4FE5AD295267B2922BA00DC68`。压缩包内目录与单独下载的目录逐字节哈希一致，摘要条数与目录条数一致。快照的[元数据](https://raw.githubusercontent.com/yzc-666/iclr2027-explorer/cursor/openreview-fetch-scripts/docs/data/meta.json)是第三方整理，不能替代 OpenReview 当前状态。

在全部标题、关键词与摘要上进行持续/终身/非平稳、RL、策略、复用、记忆、遗忘、容量等多组文本筛查，并人工阅读重点摘要。一次高精度条件（包含 `continual reinforcement learning`、`lifelong reinforcement learning`、`continual RL` 或 `lifelong RL`，且包含相关机制词）产生 34 条机器命中，其中有无关项；另一组标题和关键词交叉检索产生 117 条候选。两者并非全部相关论文的数量，且关键词会漏掉不用这些措辞的研究。下列归类为人工判断，不构成穷尽保证。所有机制和效果均为作者摘要中的陈述，尚未逐篇复核正文和实验。

快照中的状态是 10 月 7 日状态，不能据此断定当前状态。2026-11-10 通过 OpenReview 官方接口定向核对了 27 条：`dDR1HJnjRK` 和 `rZHYYyftSI` 的 venue 标识为 Desk Rejected Submission；另 25 条为 Conference Submission，涵盖第 1–9 项、第 11–12 项、第 14–18 项、第 20 项、第 22 项、第 24–26 项及第 28–31 项。Submission 不等于录用；清单其余条目的当前状态尚需逐篇核实。[ICLR 2027 官方日程](https://iclr.cc/Conferences/2027/CallForPapers)所列最终决定日期为 2026-12-16。

## 第一梯队：直接相关或存在重要创新交集

1. [SpikeARC: Spike-Guided Routing and Selective Policy Consolidation for Continual Reinforcement Learning](https://openreview.net/forum?id=4R9CT0wc2u) — 受保护的 Stable 路径加 Fast 状态依赖动作修正、行为空间约束和逐步固化；包含 PPO 受控实验。与本项目“基础策略＋快速修正＋长期写入”的表面结构高度相近，是必须通读全文的最重要竞争文献之一。需核对是否为有限完整持久字节预算、淘汰前写回、同组知识继承、访问与评估协议；不能仅据摘要断定完全相同或存在区别。
2. [How to Find and Reuse Policies for Continuous Adaptation in Lifelong Reinforcement Learning](https://openreview.net/forum?id=3AID9zXvtG) — 基于在线状态—动作—奖励经验构造 Wasserstein 任务嵌入，用稀疏权重选择并组合历史策略。对应“旧知识能否被正确选择、何时迁移”的核心问题；多源库的真实存储和计算成本需要核查。
3. [An Expert Knowledge Composition Method for Continual Reinforcement Learning](https://openreview.net/forum?id=R9JlSIYlHe) — 将历史知识表示为专家向量，双层路由，并在容量达到上限时匹配与合并。与容量受限下的组织、复用和生命周期直接重叠；摘要没有建立与本项目相同的端到端字节和交互公平性。
4. [Policy Improvement and Preservation in Continual Reinforcement Learning with Latent Families](https://openreview.net/forum?id=cHdnmrLCJt) — 将潜在环境族识别与共享决策分开，使用存储数据和锚策略判断保护约束，不满足时拆分。尤其值得检查“共享应受保护、拆分应计入完整成本”的形式化边界；其理论依赖明确假设。
5. [Not Forgotten, Not Retrieved: A Same-Checkpoint Decomposition of Failure in Modular Continual RL](https://openreview.net/forum?id=oyt490RUa3) — 用同一检查点的任务感知强制专家与学习路由对比，分离获取失败、保留损失和检索差距；基于模块化 Dreamer。摘要同时指出多技能检查点很少，不能据此外推检索故障频率。对本项目而言，已知组标签条件必须与无标签路由条件区分。
6. [When Policy Representations Fail: Diagnosis and Repair through Behavioral Evidence](https://openreview.net/forum?id=KAE8EmtOzO) — 将低维策略适应失败分解为信息不足、搜索不足、组织不佳或确实缺失方向；以独立成功行为见证，逐级测试修复并在必要时扩容。提醒本项目先证伪固定容量下的绑定、写入与优化问题，再考虑结构增长；独立见证获取成本必须计入。
7. [EFFECT-GUIDED STRUCTURAL GROWTH IN PRETRAINED SKILL SPACES FOR LIFELONG REINFORCEMENT LEARNING](https://openreview.net/forum?id=qJJJfUVB0Q) — 用预训练技能空间的支持失配和保留执行数据验证新增方向，比较固定容量及可比增长对照。相关于未来结构扩容的准入逻辑，但其允许增长，不自动满足项目的固定持久预算。
8. [Decoupled Policy Adaptation for Non-Stationary Reinforcement Learning](https://openreview.net/forum?id=kBfGqc4Ay4) — 高斯策略由共享基础策略和上下文相关残差调制构成，另用 KL 锚定基础策略。它在共享加残差方面构成近邻；需要核对连续上下文和超网络、总容量、是否涉及知识淘汰等差异。
9. [How Much of a Policy Library Must Be Kept in Continual Offline RL?](https://openreview.net/forum?id=iESNwi18TU) — 针对历史策略建立行为兼容有向图，再选择能够覆盖任务的策略集合，并研究何时停止额外验证。与“哪些知识值得留”非常相关；训练是静态数据的离线 RL，且需要计入配对重置验证交互，不能当成同条件在线结果。
10. [Continual Memory: Separating Selection and Representation in Continual Reinforcement Learning](https://openreview.net/forum?id=dDR1HJnjRK) — 固定策略与训练预算，区分回放状态的选择方式和相似性表示对保持与前向迁移的作用。与旧行为证据如何取样直接相关；当前官方 venue 标识为 Desk Rejected Submission，仍可作为未通过该轮筛选的公开研究线索，不作录用论文引用。
11. [NAMI: A Heuristic Forward Transfer Approach for Continual Reinforcement Learning](https://openreview.net/forum?id=17Sh85W6WQ) — 从旧策略中按神经元权重范数选择初始化，强调单网络推理；值得作为简单迁移基线线索，但需要核实旧网络保留和选择时的完整成本。
12. [Remembering How to Act through Greedy Action Consolidation in Continual Reinforcement Learning](https://openreview.net/forum?id=ZY7de6Eki9) — 值函数方法保护历史最优动作相对排序的 margin，而非精确保留所有 Q 值。可启发行为功能目标的设计；其离散动作值方法不能直接等同于连续高斯策略 KL。
13. [Continual Reinforcement Learning with Quasimetric Structure Evolving across Tasks](https://openreview.net/forum?id=cfjhUpBv22) — 学习跨任务非对称可组合的距离表征，以获取可迁移结构。可作为有方向迁移关系的竞争机制，具体在线信息和成本需读全文。
14. [FAULT-AWARE POLICY REUSE VIA SKILL LIBRARIES FOR ADAPTIVE ROBOT LEARNING](https://openreview.net/forum?id=vcDfr1FrNk) — 离线预训练故障索引策略库；已见故障直接调用，未见故障用历史策略初始化在线适应。关注机器人故障变化，与任务族切换不同；离线预训练、识别与库容量均须计入。

## 第二梯队：学习稳定性、强基线与评估

15. [Task diversity produces systematic transfer but inhibits continual reinforcement learning](https://openreview.net/forum?id=4upIAmqZ4o) — Banyan 可控制任务多样性维度；作者观察到更多多样性可增强初始迁移，同时抑制后续持续适应。提示应分开评价初始收益与在线学习斜率，并报告任务流条件。
16. [Bellman Demand and Critic Response: A Spectral View of Continual RL](https://openreview.net/forum?id=7IRG4Qhq0H) — 在 SAC 中分析新任务 Bellman 修正需求与 critic 学习响应错配，并用周期谱投影。提醒 actor 的迁移失败可能来自 critic，而非知识存储本身；方法不能直接作为 PPO 结果。
17. [Decomposed Contrastive Critics for Continual Reinforcement Learning](https://openreview.net/forum?id=gtI9WJv8V0) — 在稀疏奖励目标条件操纵任务中分离共享与任务专用 critic，并使用成功回放和行为监督门控。共享结构、缓冲和 critic 都变化，不能把其收益单独归因于一个机制。
18. [Maintain Your Ranks: Enabling Continual Learning in RL Agents with Growing and Pruning](https://openreview.net/forum?id=67TYXrYPBb) — 通过 critic 神经元增长与剪枝维护有效秩、改善持续学习。与固定容量情景相关但不能仅按最终参数数目忽略生长峰值和优化器状态。
19. [Preserving Plasticity in Continual Robot Learning from Pixels with Stiefel-Constrained Optimization](https://openreview.net/forum?id=yuA7HsZIhg) — 研究约束优化器在像素 PPO 持续控制及机器人行为克隆中的可塑性作用；其观测和预算与小型状态输入控制不同。
20. [Do Not Disturb: When Plasticity Interventions Degrade Offline-to-Online Reinforcement Learning](https://openreview.net/forum?id=ImyrQKmvTo) — 报告离线到在线控制中无条件参数扰动可能破坏预训练策略，并提出诊断门控。提示不要默认重置、扰动或扩容有益；相关结果是作者对特定方法与环境的报告。
21. [Auditing Plasticity Diagnostics with Function-Preserving Rescalings](https://openreview.net/forum?id=rZHYYyftSI) — 指出一些归一化可塑性诊断对功能保持的重缩放不变，而固定 SGD 适应性能可变化；因此单个 rank 或 dormancy 指标不能直接证明可学习能力。当前官方 venue 标识为 Desk Rejected Submission。
22. [Continual Reinforcement Learning with Neuroevolution](https://openreview.net/forum?id=UxsGk4on0e) — 比较进化策略、遗传算法与持续 RL，研究稳定性—可塑性与权重空间邻域；对比须计入整个种群、交互和计算，且设置未必同 PPO。
23. [Space-sampled Value Decay: Forgetting Mechanisms for Non-stationary Reinforcement Learning](https://openreview.net/forum?id=H7As82D8sq) — 针对不重访旧条件的单系统连续漂移，用值衰减遗忘过时信息；与需保留反复重访任务的场景形成边界对照。
24. [First Visits and Revisits: Learning from the Deployment Stream in Model-Based Control](https://openreview.net/forum?id=Jv0qc3b9nq) — 冻结与学习的预测头共同服务模型预测控制，区分首次访问和重访；其世界模型规划并不是在线 PPO，但评估维度相关。
25. [LOOPADAPT: PERSISTENT POLICY ADAPTATION VIA LOOP-TO-SCOPE CERTIFICATION](https://openreview.net/forum?id=a24VEpKCGy) — 从重复失败梯度研究有限参数作用范围内是否存在共同下降方向；验证和梯度估计开销、应用范围需通读正文。

## 第三梯队：机制相近但设定差异明显

26. [Continual Reinforcement Learning via Behavioral Foundation Models](https://openreview.net/forum?id=0yGAv04uSW) — 离线无奖励预训练、在线任务推断与保覆盖缓冲更新。更适合用于比较预训练覆盖与在线更新，不能把无梯度策略切换直接算成在线策略学习。
27. [Adaptive Behavioral Foundation Models: Continual Training from Reward-Free Experience](https://openreview.net/forum?id=p5RzOSUXH2) — 从无奖励在线经验持续微调并处理未知动力学，和项目的奖励驱动 PPO 训练不同。
28. [AWM: Accumulative World Modeling for Continual Reinforcement Learning](https://openreview.net/forum?id=twI4JbUqck) — 共享世界模型、组合适配器、回报约束压缩和无标签路由；关注世界模型知识的累积，而非同一种 actor 写入。
29. [Routing-Aware Masked Decision Transformer for Task-Agnostic Continual Offline Reinforcement Learning](https://openreview.net/forum?id=vb7DK4t9I8) — 离线轨迹、可学习 mask、低秩扩容和任务推断；可参考无标签路由，但不等于在线训练。
30. [EvoLPO: Continual Reinforcement Learning through Policy Evolution](https://openreview.net/forum?id=9Xytb5MZg2) — 在数学和代码等可验证奖励序列中选择历史 checkpoint 和后续恢复操作；对应父策略选择与恢复的联合价值，但属于 LLM RLVR。
31. [Simple Agentic Memory for Generalist Robot Policies](https://openreview.net/forum?id=01f2FYocEa) — 冻结机器人策略、在线维护并检索结构化外部状态；属于推理期交互记忆，不是参数在线更新。
32. [Layer-Selective Utility Gating for Continual Learning](https://openreview.net/forum?id=0OjuBwn9tu) — 比较不同网络层的更新门控，主要包含监督任务和受控 Gridworld；可参考门控位置，不提供直接的项目结论。
33. [Decision Retention Is Not Trajectory Retention: Preserving Cross-Hero Capabilities During Offline-to-Online Game Reinforcement Learning](https://openreview.net/forum?id=7J6T14MQNP) — 在游戏 PPO 的离线到在线适应中，比较旧策略蒸馏与状态上的决策保持；摘要强调决策一致不自动推出轨迹保持。这一限制与项目 H036 的 KL 代理尤其相关；它的游戏设定和数据与项目不同。
34. [When Learned Options Become Infrastructure: Promotion-Induced Maintenance in Control Lifecycles](https://openreview.net/forum?id=kN1zSW9B5D) — 研究选项更新后如何依据依赖图限制可能过期的调用并重新验证；作者指出部分预注册终点未确认。若未来引入版本依赖图，这是维护成本和因果对照方面的邻近工作，并非当前项目已实现功能。
35. [Counterfactual Dynamics-Aware Experience Retrieval for Policy Transfer](https://openreview.net/forum?id=KzXESEbXNO) — 根据相同动作序列下的动力学响应，从固定策略库中选择控制器。研究检索所需试探动作及误选代价，和已知组标签的当前实验应分开。
36. [Dream to Remember: Continual Steering of Generalist Robot Policies with World Models](https://openreview.net/forum?id=WfjGMorGzm) — 在冻结通用机器人策略上训练引导模块，并通过世界模型想象旧任务以抑制遗忘；有些新任务在真实环境中学习。世界模型是额外状态和计算，不是零成本旧知识。
37. [SEES: A Self-Evolving Embodied System via Failure-Guided VLA Policy Adaptation](https://openreview.net/forum?id=MtxF8GBKlZ) — 将相关原子技能归入共享适配器，针对失败构造模拟任务并在线 RL 更新；模拟恢复与 LLM 任务构造带来不同信息和计算假设。
38. [DiPA-WM: Diffusion Policy Adaptation with Continual World Modeling](https://openreview.net/forum?id=9tMHRvCO4I) — 离线到在线场景下同时更新扩散策略和世界模型适配器，采用不同时间尺度缓解表征漂移；并非同一策略参数化。
39. [COE: Mitigating Degradation while Improving Fine-tuning in Offline-to-Online RL](https://openreview.net/forum?id=AcdNDic47i) — 结合保守离线策略和快速在线策略，以离策略评估逐步交接，主要关注微调初期退化与后期学习速度。
40. [Trackable Skill Library Lifecycle Management for Agentic Reinforcement Learning](https://openreview.net/forum?id=eFi8mdUJNV) — 在 LLM agent GRPO 中估计外部技能的边际作用并管理技能入库、保留与退役；技能是语言智能体的外部脚手架，并非本项目策略参数组件。
41. [SkillAge: Lifecycle-Aware Skill Aging and Safe Pruning for Self-Evolving Agents](https://openreview.net/forum?id=uqsxkPPHBa) — 针对语言智能体技能库，依据表现、冗余及上下文成本进行保留决策，并通过隔离与回放检查控制剪枝风险；不等同于连续控制策略库。
42. [Dynamic Skill Lifecycle Management for Agentic Reinforcement Learning](https://openreview.net/forum?id=ZSRMEDG6G8) — 基于去除单项技能的边际作用，管理语言智能体的技能保留、退出和扩展；动态活跃集合不等于硬固定容量。
43. [SAPO: Skill-Augmented Policy Optimization](https://openreview.net/forum?id=JCbLToErpJ) — 以基线与使用技能后的对照轨迹估计候选外部技能贡献，涉及入库、检索排序和剪枝；属于 LLM agent 的技能库而非控制策略组件。
44. [Relative Bellman Constraints for Value Learning in Residual Reinforcement Learning](https://openreview.net/forum?id=1qf1fepoTe) — 冻结预训练机器人策略，以动作残差适应；在 critic 中用配对回放和相对 Bellman 约束监督残差引起的价值差。是残差策略与价值学习方面的重要近邻，摘要未建立有限容量长期淘汰机制。
45. [Fisher-Feature Proximal Policy Optimization](https://openreview.net/forum?id=i0Y9UOlFDz) — 质疑仅凭特征秩诊断 PPO 可塑性，提出经过策略输出头的 Fisher 特征惩罚；摘要报告 MuJoCo 和变化动力学实验。是 PPO 更新稳定性的对照线索。
46. [CRL-VLA: Continual Vision–Language–Action Learning](https://openreview.net/forum?id=g5Wj5C4Dfi) — 连续 VLA 强化学习后训练，结合冻结的 retention critic、可训练 adaptation critic 与策略正则；机器人 VLA 场景和本项目策略规模不同。
47. [The Three Regimes of Offline-to-Online Reinforcement Learning](https://openreview.net/forum?id=rhoSvAPoHA) — 根据离线数据和预训练策略的相对价值区分微调情形，主张稳定性与可塑性要求随先验变化；不直接研究持续模块淘汰。
48. [IA-World: Residual VLA Post-Training with Latent World Model](https://openreview.net/forum?id=9EjgmHopm0) — 冻结 VLA 提出动作计划，潜在世界模型预测后果，再以在线学习的残差策略修正；VLA 与世界模型引入额外模型和训练成本。
49. [Reset Cost Does Not Determine Reset Value: The Role of Post Reset Update Budgets](https://openreview.net/forum?id=b9w0Lt8xtJ) — 在 SAC 中比较重置的即时代价与后续更新预算，指出前者不足以决定长期价值；对于项目中写入后的恢复预算是相邻启示。

## 对项目设计和创新判断的直接影响

- **不宜再把“冻结基础策略＋快速残差＋长期整合”单独当作新颖性。** SpikeARC、DAPHNI 以及相关策略库论文至少在摘要层面形成近邻。必须检查全文、时间和完整设置，再论证明确区别。
- 可以继续检验的细化问题是：在**完整固定持久容量**下，一次局部槽淘汰能否借助已有在线证据，将可复用行为写入同组长期父策略，同时保护已有居民并改善未来重访或迁移；公平计入 buffer、优化器、元数据、验证交互、峰值及算力。这是待验证的项目假设，并非已有贡献。
- H037 等淘汰前写回想法目前是候选设计，不能从文献相似性或代理 KL 推出项目有奖励提升。E161、E162、E163 和 E164 的限制仍然成立。
- 评估要区分能力获得、保留、可检索性、迁移前初始化和迁移后的在线适应；已知任务/组身份及额外验证交互必须显式标注。结构增长和可控条件若作为后续研究方向，须另行确定资源和对照。
- 此清单以摘要为主。优先通读前 1–10 项正文、附录、开源代码及后续论文状态，然后制作逐项 novelty matrix；未检索到的论文、10 月 7 日后的新增内容、不同术语表达与不可访问的修订均可能造成遗漏。
