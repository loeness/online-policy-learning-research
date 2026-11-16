# NeurIPS 2026：持续强化学习和有限知识复用文献分析

检索日期：2026-11-10。项目对照基于 [现有算法与实验总览](PROJECT_ALGORITHM_EXPERIMENTS_2026-11-03.md)及 [V2 架构审计](V2_ARCHITECTURE_AUDIT.md)：预训练 W0，真实在线交互和 PPO 更新，有限 local 槽及共享 parent；近期实验已知 task/group/bout，H026、H033、H036 未获确认。

## 检索与状态边界

检索 NeurIPS 官方 [2026 活动目录](https://neurips.cc/Downloads/2026)、作者论文、作者或机构主页及 OpenReview。官方目录混合 poster、期刊展示、workshop 等，不能把全部条目都称为主会论文；下列逐项注明。官方 poster 页面在本次自动访问中受 robots 限制时，以官方目录与论文作者公开信息共同核验；部分论文的官方最终版本可能仍待公开，分析注明已访问版本。这里是针对项目的相关文献筛查，不宣称穷尽全部相关论文。

## 一、直接相关：持续强化学习与在线更新

### 1. Permanent and Transient Representations for Continual Reinforcement Learning

来源：[作者 arXiv 全文](https://arxiv.org/html/2611.07327)，[摘要与作者的 NeurIPS 2026 录用说明](https://arxiv.org/abs/2611.07327)，[官方 poster](https://neurips.cc/virtual/2026/poster/149172)。阅读主文及方法、实验相关附录。

- **问题与方法：**将 action-value 函数分为长期的参数化永久分量和快速适应的非参数临时分量。临时项用 MinHash、分桶和固定槽位近邻表，桶满时 LRU；永久项周期性学习组合预测，临时项衰减或重置。核心是价值函数的双时间尺度与不同泛化范围，不是 parent/local actor。
- **实验：**图像分类式 MDP 的任务切换隐藏，报告 30 seeds；Craftax-Classic 报告两流及 250M 步实验，后者 10 seeds、1024 并行环境。论文报告更快学习和若干基线优势，但规模、动作空间和预算与项目不同；线性理论不能直接推出深度网络或 PPO 的保证。
- **项目启发：**永久与临时信息需要不同的泛化半径；项目不仅要测 parent 和 local 如何更新，也应检查共享 critic 是否发生不必要的跨情境干扰。哈希表、键、值、回放和检索计算不能视为免费；离散网格观察的 MinHash 假设亦不能直接推广。它是很重要的机制近邻，但不能说已经证明 actor 的淘汰前交接有效。

### 2. Local Guidance, Global Impact: Gaussian-Reshaped Trust Region Unlocks Behavior Transitions

来源：[作者全文](https://arxiv.org/html/2606.03382)，[作者确认 NeurIPS 2026](https://ling-pan.github.io/publications/)，[官方 poster](https://neurips.cc/virtual/2026/poster/155658)。该链接为 arXiv v2，需与最终发表版本区分。

- **方法：**针对非平稳条件下 PPO 的局部更新难以进入新行为模式，作者提出 Gaussian Trust Region Policy Optimization（GTR），用非单调有界高斯形状的约束和近期策略混合锚点，平衡局部更新与较大行为转变。
- **实验限制：**主机器人比较基于三 seeds；相关任务使用 Walker、Dog 连续动作任务，不同场景比较采用任务专用输入层和 actor 输出头；Craftax 长训练达到 10 亿环境步。额外锚点、计算和任务头必须计费，作者的经验结论不证明当前项目 PPO 已出现同类失效。
- **项目启发：**在认定 parent 容量或组织方式不足之前，先诊断 PPO 自身对当前任务是否仍能产生有效更新。将更新规则和 parent 写入分开比较，避免结构与优化器一起改而无法归因。GTR 是候选强基线，不直接解决有限持久容量或保存选择。

### 3. From Anti-Forgetting to Fast Adaptation: Continual Reinforcement Learning with World Models

来源：[官方 NeurIPS poster](https://neurips.cc/virtual/2026/poster/150468)、[OpenReview 页面](https://openreview.net/forum?id=hoG3Hx7DHo)。本次已获得投稿摘要，**全文尚未获得**；作者、具体公式、实验预算与结果尚需全文复核。

摘要中的 HERD 使用模型集成分歧、自适应规划 horizon 和分层高质量 replay，关注持续任务中的保持与快速适应。它与“保存哪些在线证据以及何时重访”相关，但 world model、ensemble、replay 的资源和模型误差影响均不能省略。不能因摘要说性能改善就将其当作项目的实测结论，或假定支持固定完整字节预算。

### 4. Beyond One-Size-Fits-All: Diagnosis-Driven Online Reinforcement Learning with Offline Priors

来源：[作者全文](https://arxiv.org/html/2606.25527)，[作者确认 NeurIPS 2026 Position Paper Track](https://zerlinwang.github.io/)，[官方 poster](https://neurips.cc/virtual/2026/poster/140053)。**这是观点论文，不是提出已验证的通用新算法。**

作者将离线先验在在线 RL 中区分为初始化、约束/参考和辅助经验三类，主张按部署条件诊断其作用；配对实验覆盖 D4RL 上多种任务和离线数据。相同先验可能帮助某些场景，却在另一些场景抑制适应。

对项目，W0、parent、旧行为教师和 evidence 并不是同一个对象：分别可能承担初始化、行为约束或辅助数据。宜分别测每一种来源的收益和负迁移，而不是统一定义为“历史知识”。E161 中 future old-major 与 new-major 下结果相反，是项目自身的条件性观察，不能把本论文观点当作它的因果证明。

### 5. Learning From Failures: Efficient Reinforcement Learning Control with Episodic Memory

来源：[2026-03 作者 arXiv 版本](https://arxiv.org/html/2603.07110)、[实验室 NeurIPS 2026 accepted poster 列表](https://gr.xjtu.edu.cn/zeuslan/zh_CN/zhym/1067919/list/index.htm)、[官方活动目录](https://neurips.cc/Downloads/2026)。**版本提示：**公开 arXiv v1 仅列 Chenyang Miao 一位作者，实验室 NeurIPS 版本列出多位作者；以下方法来自 v1，尚未逐项核对与最终录用稿是否一致。

FEMA 存储短失败轨迹并在接近危险状态时利用相似经验辅助动作选择；作者报告可结合多种 model-free RL，并演示 PPO 机器人场景。它启发“错误经验也可能有价值”，但失败警告、附加检索和约束策略不同于学习可重用 parent；不能默认仅保留成功片段或仅保留失败片段一定更优，也不能把论文的样本效率数字外推到项目。

## 二、机制邻近，但与项目设定不同

### 6. Retrieve-then-Steer: Online Success Memory for Test-Time Adaptation of Generative VLAs

来源：[论文全文](https://arxiv.org/html/2605.10094)、[官方 poster](https://neurips.cc/virtual/2026/poster/154552)。

基础 VLA 冻结，系统积累、检索成功的 observation-action 片段，用可靠性调整动作生成；不更新策略参数。论文在单一持续部署任务的 300 条轨迹中，报告 FIFO 上限 1k/3k/5k 条对应累计成功率 64.0%/68.5%/70.8%，无限内存 71.2%；完整检索生成流程的相对推理时间为 1.10×。它需要额外 progress critic、每任务一条成功参考视频、存储和检索，且条目数不等同实际字节。

它适合作为“冻结底座＋外部记忆”的邻近基线和容量曲线启发；并非在线 PPO 参数更新或跨任务共享 parent。论文同时报告直接 replay 可能比基础策略更差，说明取出的历史行为不能保证在当前状态安全有效。

### 7. UniSteer: Unified Noise Steering for Efficient Human-Guided VLA Adaptation

来源：[作者论文](https://arxiv.org/abs/2605.10821)、[官方 poster](https://neurips.cc/virtual/2026/poster/152531)。冻结 VLA 生成器，同时学习轻量 noise actor、critic；人工纠正通过动作到噪声的逆变换引导更新。论文报告四个真实机器人任务平均成功率由 20% 升至 90%，平均 66 分钟；方法依赖人工纠正及额外示范、计算等资源。

它展示了固定基础模型加可训练修正可以在部署中使用，但 actor 表达、人工信息、任务、模型规模、优化算法均与项目不同；并未证明 parent 组织或受限长期保存有效。

### 8. HIDRA: Hierarchical Dual-Routing Attention for Replay-Free Lifelong Imitation Learning

来源：[作者论文](https://arxiv.org/abs/2610.36374)、[官方 poster](https://neurips.cc/virtual/2026/poster/154222)。冻结共享 backbone，采用指令级检索和上下文细路由，增量添加专家，并用 key-level 正则缓解路由漂移。与 V2 的“先筛来源、再做细粒度调用”有结构关系，但这是连续模仿学习、可扩张专家库，不是固定字节下的在线奖励驱动 PPO。不能把路由准确或无 replay 当作免费持久状态。

### 9. Online Finetuning Decision Transformers with Pure RL Gradients

来源：[作者论文](https://arxiv.org/abs/2601.00167)、[官方 poster](https://neurips.cc/virtual/2026/poster/150024)。文章讨论 Decision Transformer 在线微调中的 hindsight return relabeling 与重要性比率不相容，并使用纯 RL 梯度。它启发我们检查历史数据与当前策略比率/采样语义，但目标是在线更新 DT，不研究有限容量 parent 的保存与回访。不得推断当前项目 PPO 存在相同 bug。

### 10. Continual Robot Learning via Language-Guided Skill Acquisition

来源：[TMLR 正文](https://openreview.net/pdf?id=oYRNxxGN9u)、[NeurIPS 2026 Journal Track 官方名单](https://blog.neurips.cc/2026/10/15/announcing-the-neurips-2026-journal-track/)。这是 **TMLR 2026 发表、NeurIPS 2026 Journal Track 展示**，不是 NeurIPS 主会新录用论文。

LG-SAIL 将语言模型、任务运动规划、在线强化学习与技能库结合，并基于语义检索以初始化新技能。值得借鉴的抽象是把技能发现、检索及在线学习分开；其符号谓词、LLM、额外技能库和机器人任务带来不同信息与资源条件。

### 11. Learning Options for Compositional Motor Control with Adapter Banks

来源：[作者论文](https://arxiv.org/html/2609.17042)、[NeurIPS 2026 官方 poster](https://neurips.cc/virtual/2026/poster/149390)。该工作在共享循环网络上学习残差 adapter 库，以离散代码选择模块，随后冻结网络，通过较高层的选择来组合运动。它是共享核心与可复用残差库的结构近邻；但关注闭环生物力学控制和技能组合，不能等同于项目的持续在线 PPO、有限容量淘汰与回访保持。

### 12. Continually Evolving Skill Knowledge in Vision Language Action Model

来源：[作者论文](https://arxiv.org/abs/2511.18085)、[作者项目页](https://stellarvla.github.io/)、[NeurIPS 2026 官方 poster](https://neurips.cc/virtual/2026/poster/155372)。Stellar VLA 用演化的任务知识及层次化任务—技能表示引导专家路由，作者报告网络参数不随任务增长，并在 LIBERO 和实体机器人做持续学习评估。它在“有限架构内共享任务与技能”的研究问题上是重要近邻，但方法属于持续模仿学习，且“不增加网络参数”不自动保证知识状态、1% replay 及总持久字节不增长；不能视为本项目 PPO 保存机制的公平比较。

## 三、对项目的决策意义

1. **不能把“长期与临时双通路”作为独立创新。** Permanent/Transient value、SpikeARC actor 和相关机器人适配均已形成近邻。项目应证明有限完整状态下，何种内容值得写入、在哪些任务上可用，以及淘汰后的回访价值。
2. **先诊断 PPO/critic 与知识组织的责任。** GTR 和 PT 分别从 actor 更新几何、critic 价值分解说明：新任务适应失败未必由 parent/local 容量造成。作为候选对照时需固定策略表示、训练交互和剩余资源。
3. **评估记忆质量和净收益。** 对每种可能的状态、失败或成功证据，明确写入时可获得性、检索与验证成本、样本与旧任务覆盖；不能把论文的 test-time 非参数 memory 当作 PPO 参数已持续学习的证据。
4. **坚持实际容量而非模块数。** 对储存池、外部数据、teacher、anchor、优化器状态和推理期检索逐项计费；额外模型与人工示范也要单列。一次淘汰前交接的行为拟合或 KL 改善不等于闭环重访收益。
5. **下一问题仍由主控确定。** 可优先围绕同一 checkpoint 分解获得/保存/调用/重访；如要检验 H037，先由算法实现给出确定教师、目标、binding、补偿、资源及对照的冻结合同，再独立审查。新论文不授权自动开始实验。

本次只开展文献检索、论文阅读和本机静态项目材料对照；未运行项目代码、测试、实验或远端任务。
