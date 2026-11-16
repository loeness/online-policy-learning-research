# Citation 与来源索引

引用属于研究档案。先前已保存 BibTeX、书目说明和附件来源，但未统一映射到 H/E；本索引补齐这层导航。

- [references.bib](../references/references.bib)：55 条公开来源的正式书目键与元数据。
- [原书目说明](../references/README.md)：历史核验日期、版本选择和借用边界；正文中的 53 条计数是补入 CPC 前的历史快照。
- [citations.json](citations.json)：全部 H/E 的来源映射、原文定位及缺失状态。
- [CITATION.cff](../CITATION.cff)：本软件/数据档案自身的引用方式；保留原发布信息，没有新建论文、版本发布或 DOI。
- [12 份附件来源清单](../docs/literature/20261018_sources.json)：标题、页数、哈希和公开匹配状态。3 份对应现有 BibTeX，9 份仍未核实公开对应；不复制附件或补造作者/编号。

2026-10-20 的复核以本地已保存来源为主。除补漏 CPC 外，没有逐篇重新核验全部论文或重做文献综述。CPC 的标题、作者、年份和 v2 元数据已核对 [arXiv 主记录](https://arxiv.org/abs/2601.22475v2)；这里只补齐已有设计引用，不新增实验。

软件包版本另由 [requirements.txt](../requirements.txt) 和 [requirements-lock.txt](../requirements-lock.txt) 保存；依赖清单与学术书目是互补来源，不为每个包补造一篇论文。

## 使用含义

2026-10-21 E161 结果后更新：当前 37 个 H、96 个 E；E161 已完成但科学结论 INCONCLUSIVE，
H036 为旧行为保护候选、H037 为暂缓的残差交接备选，二者均无执行 E。
[信息与结构审查](appendix/H033/20261021_INFORMATION_AND_STRUCTURE_REVIEW.md) 定向核对了
R-learner、SPIBB、P&C 和 A-GEM 的公开原始来源；这不是全面新颖性检索。
H033 增补 SPIBB 的 baseline 保留原则，H036 继承蒸馏/保护原则，均不继承原方法的理论保证。

2026-10-21 较早的导航增量把 E161 登记为确认方案，沿用 H026 的候选与引用关系；
该计划随后已执行，当前状态以上方结果后更新为准。2026-10-20 的 95 个历史入口及
书目核验范围保持原义，不能用本次新增记录改写其历史范围。

“提及”或“相关工作”不等于实现；“机制改编”不等于原论文复现；引用不能转移理论保证、实验分数或新颖性。每个 H 的引用入口汇总文献关系，E 的实际方法以其冻结协议和代码为准。源码复用、算法启发、背景文献、拟比较方法与软件工具分别说明。精确节/公式/版本未记载时保持 UNKNOWN，不能靠书目存在来掩盖。

公共软件背景：PPO、Stable-Baselines3、Gymnasium 与 Matplotlib 的书目/说明见下表及原相关工作文档；它们不是新增比较 arm。工作流引用同样不构成科研结果。未登记额外借用，不等于证明方法独创。

## 按 H 找引用与借用说明

| H | 文献关系 / 原文入口 | 使用边界 |
| --- | --- | --- |
| [H001](records/H001.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H002](records/H002.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H003](records/H003.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H004](records/H004.md) | [Hinton2015Distillation](#hinton2015distillation)、[Hu2022LoRA](#hu2022lora)、[Sun2024Wanda](#sun2024wanda) | LoRA、Wanda 与蒸馏用于自定义残差压缩的启发；不能把这些 arm 称为原论文复现。 |
| [H005](records/H005.md) | [BouAmmar2014PGELLA](#bouammar2014pgella)、[He2026CPC](#he2026cpc)、[Qian2025TreeLoRA](#qian2025treelora) | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H006](records/H006.md) | [Balazy2024LoRAXS](#balazy2024loraxs)、[BouAmmar2014PGELLA](#bouammar2014pgella)、[He2026CPC](#he2026cpc)、[Kirkpatrick2017EWC](#kirkpatrick2017ewc)、[Qian2025TreeLoRA](#qian2025treelora)、[Rolnick2019CLEAR](#rolnick2019clear)、[Schwarz2018ProgressCompress](#schwarz2018progresscompress) | E018 的 EWC / 行为 replay 为自定义改编；E019 的小中间矩阵有既有表示先例。PG-ELLA、TreeLoRA、CPC、P&C 属文献关系，不能统称全部已实现。 |
| [H007](records/H007.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H008](records/H008.md) | [Alami2023RestartedBOCPD](#alami2023restartedbocpd)、[Laroche2019SPIBB](#laroche2019spibb)、[Mendez2022ModularRL](#mendez2022modularrl)、[Rusu2015PolicyDistillation](#rusu2015policydistillation)、[Schwarz2018ProgressCompress](#schwarz2018progresscompress)、[Zhang2022DaCoRL](#zhang2022dacorl) | 蒸馏、局部学习/慢整合与动力学分组有部分借鉴；完整方法和其保证未复现。具体使用范围以对应 E 和协议为准。 |
| [H009](records/H009.md) | [Schulman2015TRPO](#schulman2015trpo)、[Schwarz2018ProgressCompress](#schwarz2018progresscompress)、[Zhang2022DaCoRL](#zhang2022dacorl) | P&C / DaCoRL 是机制启发；TRPO 用于说明行为分布条件，不转移理论保证。 |
| [H010](records/H010.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H011](records/H011.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H012](records/H012.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H013](records/H013.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H014](records/H014.md) | [Chaudhry2019AGEM](#chaudhry2019agem)、[Yu2020PCGrad](#yu2020pcgrad)、[Schulman2015TRPO](#schulman2015trpo)、[Schulman2017PPO](#schulman2017ppo) | projected_gate 借用 A-GEM / PCGrad 的参考约束及梯度冲突思想，为自定义适配。 |
| [H015](records/H015.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H016](records/H016.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H017](records/H017.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H018](records/H018.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H019](records/H019.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H020](records/H020.md) | [Li2026ImplicitTargetShift](#li2026implicittargetshift)、[Lu2026OPTSTTPO](#lu2026optsttpo)、[Voitovych2026SmoothedOnline](#voitovych2026smoothedonline) | Target Shift 等用于提出来源剂量/顺序的控制问题，不把其定理当成本项目结论。 |
| [H021](records/H021.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H022](records/H022.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H023](records/H023.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H024](records/H024.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H025](records/H025.md) | 未识别到独立登记的外部机制；共享软件及全局相关工作见下表 | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H026](records/H026.md) | [Rusu2015PolicyDistillation](#rusu2015policydistillation)、[Schwarz2018ProgressCompress](#schwarz2018progresscompress)、[Teh2017Distral](#teh2017distral) | E129/E130 借用 Policy Distillation / Distral 的行为拟合思想，并参考 P&C 的阶段分工；九系数候选为自定义实现。E131/E132 是相关诊断。 |
| [H027](records/H027.md) | [Arumugam2021RateDistortion](#arumugam2021ratedistortion)、[Caccia2023ThreeRL](#caccia2023threerl)、[Cannon2024HOP](#cannon2024hop)、[Hafez2024TAPD](#hafez2024tapd)、[Konidaris2010CST](#konidaris2010cst)、[Laroche2019SPIBB](#laroche2019spibb)、[Li2019LearnToGrow](#li2019learntogrow)、[Mendez2021Compositional](#mendez2021compositional)、[Mendez2022ModularRL](#mendez2022modularrl)、[Qian2025TreeLoRA](#qian2025treelora)、[Schwarz2018ProgressCompress](#schwarz2018progresscompress)、[Sener2018MGDA](#sener2018mgda)、[Uther1998ContinuousUTree](#uther1998continuousutree)、[Wang2022DPMixture](#wang2022dpmixture)、[Yu2020PCGrad](#yu2020pcgrad)、[BouAmmar2014PGELLA](#bouammar2014pgella)、[He2026CPC](#he2026cpc)、[Wang2021Adaptivity](#wang2021adaptivity)、[Yamin2026GeneralistMemory](#yamin2026generalistmemory) | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H028](records/H028.md) | [Arumugam2021RateDistortion](#arumugam2021ratedistortion)、[Caccia2023ThreeRL](#caccia2023threerl)、[Cannon2024HOP](#cannon2024hop)、[Hafez2024TAPD](#hafez2024tapd)、[Konidaris2010CST](#konidaris2010cst)、[Laroche2019SPIBB](#laroche2019spibb)、[Li2019LearnToGrow](#li2019learntogrow)、[Mendez2021Compositional](#mendez2021compositional)、[Mendez2022ModularRL](#mendez2022modularrl)、[Qian2025TreeLoRA](#qian2025treelora)、[Schwarz2018ProgressCompress](#schwarz2018progresscompress)、[Sener2018MGDA](#sener2018mgda)、[Uther1998ContinuousUTree](#uther1998continuousutree)、[Wang2022DPMixture](#wang2022dpmixture)、[Yu2020PCGrad](#yu2020pcgrad)、[BouAmmar2014PGELLA](#bouammar2014pgella)、[He2026CPC](#he2026cpc)、[Wang2021Adaptivity](#wang2021adaptivity)、[Yamin2026GeneralistMemory](#yamin2026generalistmemory) | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H029](records/H029.md) | [Arumugam2021RateDistortion](#arumugam2021ratedistortion)、[Caccia2023ThreeRL](#caccia2023threerl)、[Cannon2024HOP](#cannon2024hop)、[Hafez2024TAPD](#hafez2024tapd)、[Konidaris2010CST](#konidaris2010cst)、[Laroche2019SPIBB](#laroche2019spibb)、[Li2019LearnToGrow](#li2019learntogrow)、[Mendez2021Compositional](#mendez2021compositional)、[Mendez2022ModularRL](#mendez2022modularrl)、[Qian2025TreeLoRA](#qian2025treelora)、[Schwarz2018ProgressCompress](#schwarz2018progresscompress)、[Sener2018MGDA](#sener2018mgda)、[Uther1998ContinuousUTree](#uther1998continuousutree)、[Wang2022DPMixture](#wang2022dpmixture)、[Yu2020PCGrad](#yu2020pcgrad) | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H030](records/H030.md) | [Li2026ImplicitTargetShift](#li2026implicittargetshift)、[Lu2026OPTSTTPO](#lu2026optsttpo)、[Voitovych2026SmoothedOnline](#voitovych2026smoothedonline) | Sparse Forest 仅为待实施局部维护的启发；E118 没有实现该结构机制。 |
| [H031](records/H031.md) | [Arumugam2021RateDistortion](#arumugam2021ratedistortion)、[Caccia2023ThreeRL](#caccia2023threerl)、[Cannon2024HOP](#cannon2024hop)、[Hafez2024TAPD](#hafez2024tapd)、[Konidaris2010CST](#konidaris2010cst)、[Laroche2019SPIBB](#laroche2019spibb)、[Li2019LearnToGrow](#li2019learntogrow)、[Mendez2021Compositional](#mendez2021compositional)、[Mendez2022ModularRL](#mendez2022modularrl)、[Qian2025TreeLoRA](#qian2025treelora)、[Schwarz2018ProgressCompress](#schwarz2018progresscompress)、[Sener2018MGDA](#sener2018mgda)、[Uther1998ContinuousUTree](#uther1998continuousutree)、[Wang2022DPMixture](#wang2022dpmixture)、[Yu2020PCGrad](#yu2020pcgrad)、[BouAmmar2014PGELLA](#bouammar2014pgella)、[He2026CPC](#he2026cpc) | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H032](records/H032.md) | [Arumugam2021RateDistortion](#arumugam2021ratedistortion)、[Caccia2023ThreeRL](#caccia2023threerl)、[Cannon2024HOP](#cannon2024hop)、[Hafez2024TAPD](#hafez2024tapd)、[Konidaris2010CST](#konidaris2010cst)、[Laroche2019SPIBB](#laroche2019spibb)、[Li2019LearnToGrow](#li2019learntogrow)、[Mendez2021Compositional](#mendez2021compositional)、[Mendez2022ModularRL](#mendez2022modularrl)、[Qian2025TreeLoRA](#qian2025treelora)、[Schwarz2018ProgressCompress](#schwarz2018progresscompress)、[Sener2018MGDA](#sener2018mgda)、[Uther1998ContinuousUTree](#uther1998continuousutree)、[Wang2022DPMixture](#wang2022dpmixture)、[Yu2020PCGrad](#yu2020pcgrad)、[Li2026ImplicitTargetShift](#li2026implicittargetshift)、[Lu2026OPTSTTPO](#lu2026optsttpo)、[Voitovych2026SmoothedOnline](#voitovych2026smoothedonline)、[Wang2021Adaptivity](#wang2021adaptivity)、[Yamin2026GeneralistMemory](#yamin2026generalistmemory) | 文献提及仅作为来源上下文；未在现存 H/协议中识别到额外明确的外部机制借用，不据此宣称独创。 |
| [H033](records/H033.md) | [Dudik2011DoublyRobustPolicyEvaluation](#dudik2011doublyrobustpolicyevaluation)、[Kallus2022StatefulOfflineContextual](#kallus2022statefulofflinecontextual)、[Nie2017QuasiOracleHTE](#nie2017quasioraclehte)、[Li2026ImplicitTargetShift](#li2026implicittargetshift)、[Lu2026OPTSTTPO](#lu2026optsttpo) | R-learner、DR 和 stateful contextual reduction 仍是设计模板；E162 为 NO_ADVANCEMENT，H033 保持 INCONCLUSIVE，不能据此称这些方法已在本次完整实现。 |
| [H034](records/H034.md) | [Voitovych2026SmoothedOnline](#voitovych2026smoothedonline)、[Cannon2024HOP](#cannon2024hop)、[Qian2025TreeLoRA](#qian2025treelora)、[Wang2021Adaptivity](#wang2021adaptivity)、[Yamin2026GeneralistMemory](#yamin2026generalistmemory) | 理论问题的文献上下文；未形成新的 RL 定理或容量最优性保证。 |
| [H035](records/H035.md) | [Li2026ImplicitTargetShift](#li2026implicittargetshift)、[Lu2026OPTSTTPO](#lu2026optsttpo)、[Voitovych2026SmoothedOnline](#voitovych2026smoothedonline) | 递归统计与因果顺序的候选启发；公开匹配与附件内容版本需分开。 |
| [H036](records/H036.md) | [Rusu2015PolicyDistillation](#rusu2015policydistillation)、[Schwarz2018ProgressCompress](#schwarz2018progresscompress)、[Rolnick2019CLEAR](#rolnick2019clear)、[Teh2017Distral](#teh2017distral)、[He2026CPC](#he2026cpc)、[Lei2025DMPEL](#lei2025dmpel) | Policy Distillation/P&C/CLEAR 提供行为拟合与旧知识保护的背景；本方法保护 keep 初始化 prior。Distral/CPC 是共享整合背景，DMPEL 仅为条件结构背景。H036 当前仍 INCONCLUSIVE；E163 固定 v1 必要 KL 筛查为 NO_ADVANCEMENT（8/16，低于预设 12/16）。E164 为 COMPLETE_VALID / EXPLORATORY_ONLY：5 fits、10 个评分包、16 个逻辑配对、0 新交互和 0 GPU，未授权后续阶段。所引论文完整方法未复现，也不转移其结果或保证。见 [E164 最终报告](../experiments/exp164/analyses/a001/report.md)、[最终审计](../experiments/exp164/analyses/a003/audit.json)、[主控综合](appendix/H036/COORDINATOR_E164_SYNTHESIS_20261023.md)及[最终验证状态](../experiments/exp164/FINAL_VALIDATION.json)。 |
| [H037](records/H037.md) | 无新增外部机制引用 | 源码启发的同组残差交接备选，暂缓；不冒称外部算法复现。 |

## 公开书目与来源定位

下列元数据是 canonical BibTeX 的派生展示；修改作者/标题/年份应首先修改该 BibTeX。所有来源都纳入索引，未映射到具体 H 的全局背景/工具来源仍保留，不强行分配为已实施机制。

<a id="schulman2017ppo"></a>
### Schulman2017PPO

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)（2017）。 完整作者及正式条目见 [Schulman2017PPO](../references/references.bib)。

来源定位：[README.md:210](../README.md)、[docs/BASELINES_AND_RELATED_WORK.md:13](../docs/BASELINES_AND_RELATED_WORK.md)、[docs/WRITE_INVESTIGATION_METHOD.md:76](../docs/WRITE_INVESTIGATION_METHOD.md)。

<a id="raffin2021sb3"></a>
### Raffin2021SB3

[Stable-Baselines3: Reliable Reinforcement Learning Implementations](https://jmlr.org/papers/v22/20-1364.html)（2021）。 完整作者及正式条目见 [Raffin2021SB3](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:13](../docs/BASELINES_AND_RELATED_WORK.md)。

<a id="towers2024gymnasium"></a>
### Towers2024Gymnasium

[Gymnasium: A Standard Interface for Reinforcement Learning Environments](https://arxiv.org/abs/2407.17032)（2024）。 完整作者及正式条目见 [Towers2024Gymnasium](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:13](../docs/BASELINES_AND_RELATED_WORK.md)。

<a id="karpathy2026autoresearch"></a>
### Karpathy2026Autoresearch

[autoresearch](https://github.com/karpathy/autoresearch)（2026）。 完整作者及正式条目见 [Karpathy2026Autoresearch](../references/references.bib)。

来源定位：[README.md:212](../README.md)、[docs/BASELINES_AND_RELATED_WORK.md:60](../docs/BASELINES_AND_RELATED_WORK.md)。

<a id="wolczyk2021continualworld"></a>
### Wolczyk2021ContinualWorld

[Continual World: A Robotic Benchmark for Continual Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2021/hash/ef8446f35513a8d6aa2308357a268a7e-Abstract.html)（2021）。 完整作者及正式条目见 [Wolczyk2021ContinualWorld](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:36](../docs/BASELINES_AND_RELATED_WORK.md)。

<a id="grigsby2024amago"></a>
### Grigsby2024AMAGO

[AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7204434dcb9383a1454dc1e97e58ea9c-Abstract-Conference.html)（2024）。 完整作者及正式条目见 [Grigsby2024AMAGO](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:34](../docs/BASELINES_AND_RELATED_WORK.md)。

<a id="grigsby2024amago2"></a>
### Grigsby2024AMAGO2

[AMAGO-2: Breaking the Multi-Task Barrier in Meta-Reinforcement Learning with Transformers](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9f40d2612d0b3b6f6c2a77da21b1067f-Abstract-Conference.html)（2024）。 完整作者及正式条目见 [Grigsby2024AMAGO2](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:34](../docs/BASELINES_AND_RELATED_WORK.md)。

<a id="rakelly2019pearl"></a>
### Rakelly2019PEARL

[Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables](https://proceedings.mlr.press/v97/rakelly19a.html)（2019）。 完整作者及正式条目见 [Rakelly2019PEARL](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:34](../docs/BASELINES_AND_RELATED_WORK.md)。

<a id="nair2020awac"></a>
### Nair2020AWAC

[AWAC: Accelerating Online Reinforcement Learning with Offline Datasets](https://arxiv.org/abs/2006.09359)（2020）。 完整作者及正式条目见 [Nair2020AWAC](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:33](../docs/BASELINES_AND_RELATED_WORK.md)、[references/README.md:7](../references/README.md)。

<a id="nakamoto2023calql"></a>
### Nakamoto2023CalQL

[Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c44a04289beaf0a7d968a94066a1d696-Abstract-Conference.html)（2023）。 完整作者及正式条目见 [Nakamoto2023CalQL](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:33](../docs/BASELINES_AND_RELATED_WORK.md)、[references/README.md:8](../references/README.md)。

<a id="ball2023rlpd"></a>
### Ball2023RLPD

[Efficient Online Reinforcement Learning with Offline Data](https://proceedings.mlr.press/v202/ball23a.html)（2023）。 完整作者及正式条目见 [Ball2023RLPD](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:33](../docs/BASELINES_AND_RELATED_WORK.md)、[references/README.md:7](../references/README.md)。

<a id="zhou2025wsrl"></a>
### Zhou2025WSRL

[Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data](https://proceedings.iclr.cc/paper_files/paper/2025/hash/504491292cb71e7681eedfe0e602b72f-Abstract-Conference.html)（2025）。 完整作者及正式条目见 [Zhou2025WSRL](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:33](../docs/BASELINES_AND_RELATED_WORK.md)。

<a id="bouammar2014pgella"></a>
### BouAmmar2014PGELLA

[Online Multi-Task Learning for Policy Gradient Methods](https://proceedings.mlr.press/v32/ammar14.html)（2014）。 完整作者及正式条目见 [BouAmmar2014PGELLA](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:35](../docs/BASELINES_AND_RELATED_WORK.md)、[docs/BASELINES_AND_RELATED_WORK.md:68](../docs/BASELINES_AND_RELATED_WORK.md)、[docs/C1_SUPPLEMENT_CONCLUSIONS.md:42](../docs/C1_SUPPLEMENT_CONCLUSIONS.md)、[phase_c_plan.md:175](../phase_c_plan.md)、[research/records/H006.md:6](records/H006.md)。

<a id="rolnick2019clear"></a>
### Rolnick2019CLEAR

[Experience Replay for Continual Learning](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html)（2019）。 完整作者及正式条目见 [Rolnick2019CLEAR](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:35](../docs/BASELINES_AND_RELATED_WORK.md)、[references/README.md:33](../references/README.md)、[docs/BASELINES_AND_RELATED_WORK.md:46](../docs/BASELINES_AND_RELATED_WORK.md)。

<a id="bossens2024lifetimepolicyreuse"></a>
### Bossens2024LifetimePolicyReuse

[Lifetime Policy Reuse and the Importance of Task Capacity](https://doi.org/10.3233/AIC-230040)（2024）。 完整作者及正式条目见 [Bossens2024LifetimePolicyReuse](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:35](../docs/BASELINES_AND_RELATED_WORK.md)、[references/README.md:9](../references/README.md)。

<a id="cannon2024hop"></a>
### Cannon2024HOP

[Hierarchical Orchestra of Policies](https://arxiv.org/abs/2411.03008)（2024）。 完整作者及正式条目见 [Cannon2024HOP](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:36](../docs/BASELINES_AND_RELATED_WORK.md)、[docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md:16](../docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md)、[references/README.md:10](../references/README.md)、[research_v1_plan.md:140](../research_v1_plan.md)。

<a id="qian2025treelora"></a>
### Qian2025TreeLoRA

[TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree](https://proceedings.mlr.press/v267/qian25b.html)（2025）。 完整作者及正式条目见 [Qian2025TreeLoRA](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:36](../docs/BASELINES_AND_RELATED_WORK.md)、[docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md:15](../docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md)、[phase_c_plan.md:175](../phase_c_plan.md)、[research/records/H006.md:6](records/H006.md)、[research_v1_plan.md:139](../research_v1_plan.md)。

<a id="wang2021adaptivity"></a>
### Wang2021Adaptivity

[Provably Efficient Reinforcement Learning with Linear Function Approximation Under Adaptivity Constraints](https://arxiv.org/abs/2101.02195)（2021）。 完整作者及正式条目见 [Wang2021Adaptivity](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:38](../docs/BASELINES_AND_RELATED_WORK.md)、[research_v1_plan.md:141](../research_v1_plan.md)。

<a id="yamin2026generalistmemory"></a>
### Yamin2026GeneralistMemory

[What Must Generalist Agents Remember?](https://arxiv.org/abs/2606.18746)（2026）。 完整作者及正式条目见 [Yamin2026GeneralistMemory](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:37](../docs/BASELINES_AND_RELATED_WORK.md)、[research_v1_plan.md:142](../research_v1_plan.md)。

<a id="silver2025eraexperience"></a>
### Silver2025EraExperience

[Welcome to the Era of Experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)（2025）。 完整作者及正式条目见 [Silver2025EraExperience](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:58](../docs/BASELINES_AND_RELATED_WORK.md)、[references/README.md:11](../references/README.md)。

<a id="hu2022lora"></a>
### Hu2022LoRA

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)（2022）。 完整作者及正式条目见 [Hu2022LoRA](../references/references.bib)。

来源定位：[compression_alternatives_protocol.md:19](../compression_alternatives_protocol.md)、[docs/BASELINES_AND_RELATED_WORK.md:52](../docs/BASELINES_AND_RELATED_WORK.md)、[docs/BASELINES_AND_RELATED_WORK.md:68](../docs/BASELINES_AND_RELATED_WORK.md)。

<a id="sun2024wanda"></a>
### Sun2024Wanda

[A Simple and Effective Pruning Approach for Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/14c856c7a41297804de4c4890e846b25-Abstract-Conference.html)（2024）。 完整作者及正式条目见 [Sun2024Wanda](../references/references.bib)。

来源定位：[compression_alternatives_protocol.md:19](../compression_alternatives_protocol.md)、[docs/BASELINES_AND_RELATED_WORK.md:52](../docs/BASELINES_AND_RELATED_WORK.md)。

<a id="hinton2015distillation"></a>
### Hinton2015Distillation

[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)（2015）。 完整作者及正式条目见 [Hinton2015Distillation](../references/references.bib)。

来源定位：[compression_alternatives_protocol.md:19](../compression_alternatives_protocol.md)、[docs/BASELINES_AND_RELATED_WORK.md:52](../docs/BASELINES_AND_RELATED_WORK.md)。

<a id="eckart1936lowrank"></a>
### Eckart1936LowRank

[The Approximation of One Matrix by Another of Lower Rank](https://doi.org/10.1007/BF02288367)（1936）。 完整作者及正式条目见 [Eckart1936LowRank](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:52](../docs/BASELINES_AND_RELATED_WORK.md)。

<a id="kassis2026scientificagentskills"></a>
### Kassis2026ScientificAgentSkills

[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065)（2026）。 完整作者及正式条目见 [Kassis2026ScientificAgentSkills](../references/references.bib)。

来源定位：[README.md:213](../README.md)、[docs/BASELINES_AND_RELATED_WORK.md:60](../docs/BASELINES_AND_RELATED_WORK.md)、[references/README.md:12](../references/README.md)。

<a id="hunter2007matplotlib"></a>
### Hunter2007Matplotlib

[Matplotlib: A 2D Graphics Environment](https://doi.org/10.1109/MCSE.2007.55)（2007）。 完整作者及正式条目见 [Hunter2007Matplotlib](../references/references.bib)。

来源定位：[references/README.md:13](../references/README.md)。

<a id="kirkpatrick2017ewc"></a>
### Kirkpatrick2017EWC

[Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)（2017）。 完整作者及正式条目见 [Kirkpatrick2017EWC](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:46](../docs/BASELINES_AND_RELATED_WORK.md)、[references/README.md:27](../references/README.md)。

<a id="schwarz2018progresscompress"></a>
### Schwarz2018ProgressCompress

[Progress \& Compress: A scalable framework for continual learning](https://proceedings.mlr.press/v80/schwarz18a.html)（2018）。 完整作者及正式条目见 [Schwarz2018ProgressCompress](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:46](../docs/BASELINES_AND_RELATED_WORK.md)、[docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md:42](../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md)、[docs/KNOWLEDGE_V2_VALIDATION.md:45](../docs/KNOWLEDGE_V2_VALIDATION.md)、[docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md:19](../docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md)、[docs/RETENTION_CANDIDATE_PROTOCOL.md:75](../docs/RETENTION_CANDIDATE_PROTOCOL.md)、[docs/RETENTION_CANDIDATE_REVIEW.md:107](../docs/RETENTION_CANDIDATE_REVIEW.md)、[references/README.md:29](../references/README.md)、[references/README.md:57](../references/README.md)、[references/README.md:58](../references/README.md)。

<a id="balazy2024loraxs"></a>
### Balazy2024LoRAXS

[LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters](https://arxiv.org/abs/2405.17604)（2024）。 完整作者及正式条目见 [Balazy2024LoRAXS](../references/references.bib)。

来源定位：[docs/BASELINES_AND_RELATED_WORK.md:68](../docs/BASELINES_AND_RELATED_WORK.md)、[docs/C1_SOURCE_BASIS_DERIVATION.md:71](../docs/C1_SOURCE_BASIS_DERIVATION.md)、[docs/C1_SUPPLEMENT_CONCLUSIONS.md:42](../docs/C1_SUPPLEMENT_CONCLUSIONS.md)、[references/README.md:30](../references/README.md)、[research/records/H006.md:6](records/H006.md)。

<a id="konidaris2010cst"></a>
### Konidaris2010CST

[Constructing Skill Trees for Reinforcement Learning Agents from Demonstration Trajectories](https://papers.nips.cc/paper_files/paper/2010/hash/27ed0fb950b856b06e1273989422e7d3-Abstract.html)（2010）。 完整作者及正式条目见 [Konidaris2010CST](../references/references.bib)。

来源定位：[docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md:17](../docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md)。

<a id="uther1998continuousutree"></a>
### Uther1998ContinuousUTree

[Tree Based Discretization for Continuous State Space Reinforcement Learning](https://www.cs.cmu.edu/~mmv/papers/will-aaai98.pdf)（1998）。 完整作者及正式条目见 [Uther1998ContinuousUTree](../references/references.bib)。

来源定位：[docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md:18](../docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md)。

<a id="hafez2024tapd"></a>
### Hafez2024TAPD

[Continual Deep Reinforcement Learning with Task-Agnostic Policy Distillation](https://www.nature.com/articles/s41598-024-80774-8)（2024）。 完整作者及正式条目见 [Hafez2024TAPD](../references/references.bib)。

来源定位：[docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md:20](../docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md)。

<a id="mendez2022modularrl"></a>
### Mendez2022ModularRL

[Modular Lifelong Reinforcement Learning via Neural Composition](https://arxiv.org/abs/2207.00429)（2022）。 完整作者及正式条目见 [Mendez2022ModularRL](../references/references.bib)。

来源定位：[docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md:43](../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md)、[docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md:21](../docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md)。

<a id="wang2022dpmixture"></a>
### Wang2022DPMixture

[A Dirichlet Process Mixture of Robust Task Models for Scalable Lifelong Reinforcement Learning](https://arxiv.org/abs/2205.10787)（2022）。 完整作者及正式条目见 [Wang2022DPMixture](../references/references.bib)。

来源定位：[docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md:22](../docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md)。

<a id="arumugam2021ratedistortion"></a>
### Arumugam2021RateDistortion

[Deciding What to Learn: A Rate-Distortion Approach](https://proceedings.mlr.press/v139/arumugam21a.html)（2021）。 完整作者及正式条目见 [Arumugam2021RateDistortion](../references/references.bib)。

来源定位：[docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md:84](../docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md)。

<a id="caccia2023threerl"></a>
### Caccia2023ThreeRL

[Task-Agnostic Continual Reinforcement Learning: Gaining Insights and Overcoming Challenges](https://arxiv.org/abs/2205.14495)（2023）。 完整作者及正式条目见 [Caccia2023ThreeRL](../references/references.bib)。

来源定位：[docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md:23](../docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md)。

<a id="li2019learntogrow"></a>
### Li2019LearnToGrow

[Learn to Grow: A Continual Structure Learning Framework for Overcoming Catastrophic Forgetting](https://proceedings.mlr.press/v97/li19m.html)（2019）。 完整作者及正式条目见 [Li2019LearnToGrow](../references/references.bib)。

来源定位：[docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md:25](../docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md)。

<a id="mendez2021compositional"></a>
### Mendez2021Compositional

[Lifelong Learning of Compositional Structures](https://arxiv.org/abs/2007.07732)（2021）。 完整作者及正式条目见 [Mendez2021Compositional](../references/references.bib)。

来源定位：[docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md:25](../docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md)。

<a id="laroche2019spibb"></a>
### Laroche2019SPIBB

[Safe Policy Improvement with Baseline Bootstrapping](https://proceedings.mlr.press/v97/laroche19a.html)（2019）。 完整作者及正式条目见 [Laroche2019SPIBB](../references/references.bib)。

来源定位：[docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md:46](../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md)、[docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md:74](../docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md)。

<a id="yu2020pcgrad"></a>
### Yu2020PCGrad

[Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)（2020）。 完整作者及正式条目见 [Yu2020PCGrad](../references/references.bib)。

来源定位：[docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md:41](../docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md)、[docs/WRITE_DECISION_PROTOCOL.md:90](../docs/WRITE_DECISION_PROTOCOL.md)、[docs/WRITE_INVESTIGATION_METHOD.md:78](../docs/WRITE_INVESTIGATION_METHOD.md)、[research/records/H014.md:43](records/H014.md)。

<a id="sener2018mgda"></a>
### Sener2018MGDA

[Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650)（2018）。 完整作者及正式条目见 [Sener2018MGDA](../references/references.bib)。

来源定位：[docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md:41](../docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md)。

<a id="rusu2015policydistillation"></a>
### Rusu2015PolicyDistillation

[Policy Distillation](https://arxiv.org/abs/1511.06295)（2015）。 完整作者及正式条目见 [Rusu2015PolicyDistillation](../references/references.bib)。

来源定位：[docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md:41](../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md)、[docs/RETENTION_CANDIDATE_PROTOCOL.md:75](../docs/RETENTION_CANDIDATE_PROTOCOL.md)、[docs/RETENTION_CANDIDATE_REVIEW.md:107](../docs/RETENTION_CANDIDATE_REVIEW.md)、[references/README.md:53](../references/README.md)、[references/README.md:54](../references/README.md)。

<a id="zhang2022dacorl"></a>
### Zhang2022DaCoRL

[Dynamics-Adaptive Continual Reinforcement Learning via Progressive Contextualization](https://arxiv.org/abs/2209.00347)（2022）。 完整作者及正式条目见 [Zhang2022DaCoRL](../references/references.bib)。

来源定位：[docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md:44](../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md)、[docs/KNOWLEDGE_V2_VALIDATION.md:45](../docs/KNOWLEDGE_V2_VALIDATION.md)。

<a id="alami2023restartedbocpd"></a>
### Alami2023RestartedBOCPD

[Restarted Bayesian Online Change-point Detection for Non-Stationary Markov Decision Processes](https://proceedings.mlr.press/v232/alami23a.html)（2023）。 完整作者及正式条目见 [Alami2023RestartedBOCPD](../references/references.bib)。

来源定位：[docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md:45](../docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md)。

<a id="schulman2015trpo"></a>
### Schulman2015TRPO

[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)（2015）。 完整作者及正式条目见 [Schulman2015TRPO](../references/references.bib)。

来源定位：[docs/KNOWLEDGE_V2_VALIDATION.md:45](../docs/KNOWLEDGE_V2_VALIDATION.md)、[docs/WRITE_INVESTIGATION_METHOD.md:79](../docs/WRITE_INVESTIGATION_METHOD.md)。

<a id="chaudhry2019agem"></a>
### Chaudhry2019AGEM

[Efficient Lifelong Learning with A-GEM](https://arxiv.org/abs/1812.00420)（2019）。 完整作者及正式条目见 [Chaudhry2019AGEM](../references/references.bib)。

来源定位：[docs/WRITE_DECISION_PROTOCOL.md:89](../docs/WRITE_DECISION_PROTOCOL.md)、[docs/WRITE_INVESTIGATION_METHOD.md:77](../docs/WRITE_INVESTIGATION_METHOD.md)、[research/records/H014.md:43](records/H014.md)。

<a id="li2026implicittargetshift"></a>
### Li2026ImplicitTargetShift

[Implicit Target Shift in Online Learning: Characterization and Correction](https://arxiv.org/abs/2605.07886)（2026）。 完整作者及正式条目见 [Li2026ImplicitTargetShift](../references/references.bib)。

来源定位：[docs/literature/20261018_SYNTHESIS.md:19](../docs/literature/20261018_SYNTHESIS.md)、[docs/literature/20261018_decisions.md:11](../docs/literature/20261018_decisions.md)、[docs/literature/20261018_sources.json:60](../docs/literature/20261018_sources.json)、[references/README.md:21](../references/README.md)。

<a id="voitovych2026smoothedonline"></a>
### Voitovych2026SmoothedOnline

[Oracle-Efficient and Parameter-Free Agnostic Smoothed Online Learning](https://arxiv.org/abs/2610.10499)（2026）。 完整作者及正式条目见 [Voitovych2026SmoothedOnline](../references/references.bib)。

来源定位：[docs/literature/20261018_SYNTHESIS.md:19](../docs/literature/20261018_SYNTHESIS.md)、[docs/literature/20261018_online_theory.md:13](../docs/literature/20261018_online_theory.md)、[docs/literature/20261018_sources.json:143](../docs/literature/20261018_sources.json)、[references/README.md:21](../references/README.md)。

<a id="lu2026optsttpo"></a>
### Lu2026OPTSTTPO

[OPTS-TTPO: Enhancing Finite-Sample Policy-Gradient Learning with Tree Search](https://arxiv.org/abs/2609.40035)（2026）。 完整作者及正式条目见 [Lu2026OPTSTTPO](../references/references.bib)。

来源定位：[docs/literature/20261018_SYNTHESIS.md:19](../docs/literature/20261018_SYNTHESIS.md)、[docs/literature/20261018_decisions.md:12](../docs/literature/20261018_decisions.md)、[docs/literature/20261018_sources.json:245](../docs/literature/20261018_sources.json)、[references/README.md:21](../references/README.md)。

<a id="nie2017quasioraclehte"></a>
### Nie2017QuasiOracleHTE

[Quasi-Oracle Estimation of Heterogeneous Treatment Effects](https://arxiv.org/abs/1712.04912)（2017）。 完整作者及正式条目见 [Nie2017QuasiOracleHTE](../references/references.bib)。

来源定位：[docs/literature/20261019_ACTION_CONTRAST_NOTES.md:26](../docs/literature/20261019_ACTION_CONTRAST_NOTES.md)、[references/README.md:45](../references/README.md)。

<a id="dudik2011doublyrobustpolicyevaluation"></a>
### Dudik2011DoublyRobustPolicyEvaluation

[Doubly Robust Policy Evaluation and Learning](https://icml.cc/2011/papers/554_icmlpaper.pdf)（2011）。 完整作者及正式条目见 [Dudik2011DoublyRobustPolicyEvaluation](../references/references.bib)。

来源定位：[docs/literature/20261019_ACTION_CONTRAST_NOTES.md:49](../docs/literature/20261019_ACTION_CONTRAST_NOTES.md)、[references/README.md:46](../references/README.md)。

<a id="kallus2022statefulofflinecontextual"></a>
### Kallus2022StatefulOfflineContextual

[Stateful Offline Contextual Policy Evaluation and Learning](https://proceedings.mlr.press/v151/kallus22a.html)（2022）。 完整作者及正式条目见 [Kallus2022StatefulOfflineContextual](../references/references.bib)。

来源定位：[docs/literature/20261019_ACTION_CONTRAST_NOTES.md:53](../docs/literature/20261019_ACTION_CONTRAST_NOTES.md)、[references/README.md:47](../references/README.md)。

<a id="teh2017distral"></a>
### Teh2017Distral

[Distral: Robust Multitask Reinforcement Learning](https://arxiv.org/abs/1707.04175)（2017）。 完整作者及正式条目见 [Teh2017Distral](../references/references.bib)。

来源定位：[docs/RETENTION_CANDIDATE_PROTOCOL.md:75](../docs/RETENTION_CANDIDATE_PROTOCOL.md)、[docs/RETENTION_CANDIDATE_REVIEW.md:107](../docs/RETENTION_CANDIDATE_REVIEW.md)、[references/README.md:54](../references/README.md)、[references/README.md:55](../references/README.md)。

<a id="he2026cpc"></a>
### He2026CPC

[Continual Policy Consolidation for Lifelong Robot Learning](https://arxiv.org/abs/2601.22475v2)（2026）。 完整作者及正式条目见 [He2026CPC](../references/references.bib)。

来源定位：[research/records/H006.md:6](records/H006.md)、[phase_c_plan.md:175](../phase_c_plan.md)、[references/references.bib:566](../references/references.bib)。

<a id="lei2025dmpel"></a>
### Lei2025DMPEL

[Dynamic Mixture of Progressive Parameter-Efficient Expert Library for Lifelong Robot Learning](https://arxiv.org/abs/2506.05985v3)（2025；arXiv v3 于 2026-05-28 修订）。完整作者及书目元数据见 [Lei2025DMPEL](../references/references.bib)。

来源定位：[research/appendix/H036/DESIGN_AND_VALIDATION_PROPOSAL_20261022.md:42](appendix/H036/DESIGN_AND_VALIDATION_PROPOSAL_20261022.md)。仅作条件性结构背景：增长低秩 expert/router 与系数回放；本轮不采用这些机制，也不把引用当作复现或结论。

## 附件身份与未核实项

| 附件 ID | 标题 | 公共对应 | 阅读笔记 |
| --- | --- | --- | --- |
| 22161 | Adversarially Batched Online Learning | UNVERIFIED（未补造书目） | [docs/literature/20261018_online_theory.md](../docs/literature/20261018_online_theory.md) |
| 49683 | Forward Online Learning Deep Networks | UNVERIFIED（未补造书目） | [docs/literature/20261018_structures.md](../docs/literature/20261018_structures.md) |
| 35029 | Implicit Target Shift in Online Learning: Characterization and Correction | [Li2026ImplicitTargetShift](#li2026implicittargetshift) | [docs/literature/20261018_decisions.md](../docs/literature/20261018_decisions.md) |
| 28205 | Strict Online Learning for Wild Streaming Data | UNVERIFIED（未补造书目） | [docs/literature/20261018_structures.md](../docs/literature/20261018_structures.md) |
| 8985 | Decentralized Online Exp-Concave Optimization | UNVERIFIED（未补造书目） | [docs/literature/20261018_optimization.md](../docs/literature/20261018_optimization.md) |
| 52262 | Decomposed Online Learning for Adaptive LLM Inference in Formal Mathematics | UNVERIFIED（未补造书目） | [docs/literature/20261018_decisions.md](../docs/literature/20261018_decisions.md) |
| 52604 | Oracle-Efficient and Parameter-Free Agnostic Smoothed Online Learning | [Voitovych2026SmoothedOnline](#voitovych2026smoothedonline) | [docs/literature/20261018_online_theory.md](../docs/literature/20261018_online_theory.md) |
| 47394 | NLT-prop: Nonlocal Temporal Credit Assignment for Online Learning in Spiking Neural Networks | UNVERIFIED（未补造书目） | [docs/literature/20261018_optimization.md](../docs/literature/20261018_optimization.md) |
| 3225 | Sparse Forest: Efficient Online Learning and Unlearning Framework for Tree Ensembles | UNVERIFIED（未补造书目） | [docs/literature/20261018_structures.md](../docs/literature/20261018_structures.md) |
| 40934 | An Exponential Deterministic-Randomized Separation in Oracle-Based Online Learning | UNVERIFIED（未补造书目） | [docs/literature/20261018_online_theory.md](../docs/literature/20261018_online_theory.md) |
| 34181 | Online Geometric Optimization via Projection, Hierarchical Discretization, and Convex Relaxation | UNVERIFIED（未补造书目） | [docs/literature/20261018_optimization.md](../docs/literature/20261018_optimization.md) |
| 6540 | OPTS-TTPO: Enhancing Finite-Sample Policy-Gradient Learning with Tree Search | [Lu2026OPTSTTPO](#lu2026optsttpo) | [docs/literature/20261018_decisions.md](../docs/literature/20261018_decisions.md) |

附件的哈希/页码和原位置以原清单为准。未核实的公开对应不意味着没有公开版；匿名审稿页眉也不代表录用。

## 2026-10-22 定向导航增量

本次只定向更新 H036 提案的来源关系、修复 E162 的 H033 导航，并把 E163 标为待执行计划；不是全库文献复核。DMPEL 新增为 arXiv v3 条目，CPC 沿用既有键并固定 v2。H036 对 Policy Distillation、P&C、CLEAR 的借用边界是行为拟合和旧 prior 保护；Distral/CPC 仅提供共享整合背景，DMPEL 仅提供条件结构背景。任何一项引用都不代表完整方法复现或结论保证。

在本节最初登记快照时：原有 95 个实验入口，加上已完成登记的 E161、E162，共 97 个；E163 是额外登记的 Stage 0，状态为 `REGISTERED_PENDING_IMPLEMENTATION_AND_FROZEN_REVIEW/NOT_RUN`，`execution_authorized=false`，尚无资源执行授权；见[协调合同](appendix/H036/COORDINATOR_CONTRACT_20261022.md)与[实验入口](../experiments/exp163/README.md)。总计 98 个入口。E162 为 NO_ADVANCEMENT，H033 为 INCONCLUSIVE；该快照时未新增环境交互。后续状态见下节。

### 2026-10-22 后续定向状态增量

H036 自定义科学 v1 已实现并冻结（21 个源文件、48 个输入、23 项合成检查通过），但尚无实证验证；这不表示复现了所引论文完整方法，也不继承其理论保证。E163 run001 的 CPU 计量预检因复用 SSH shell 的 PID 历史污染，以 `ACCOUNTING_INVALID` 停止。实际 worker、fit 和环境交互均为 0，12/16 门槛未评价；该无效预检不是 NO_ADVANCEMENT，也不是 H036 的阴性结果。repair002 已通过独立静态复审，科学源、输入和门槛未变；尚未 SSH 传输或执行，资源授权仍待用户确认。详见[预检报告](../experiments/exp163/analyses/a001/report.md)、[计量记录](../experiments/exp163/analyses/a001/cost_accounting.json)、[审计记录](../experiments/exp163/analyses/a001/audit.json)、[repair002 说明](../experiments/exp163/preparation/validation/repair002/README.md)及[复审状态](../experiments/exp163/preparation/validation/repair002/reviewed_readiness.json)。本增量只更新 H036/E163 导航，未新增 H/E 或扩大文献复核范围。

## 2026-10-23 定向状态增量

E163 run002 已有效完成固定 v1 的必要 KL 筛查：32 fits、16 prefixes 中 8/16，低于预设 12/16 门槛，筛查结论为 NO_ADVANCEMENT；独立 1,876 项检查零失败。7 个通过项满足 D 非 C0；另一个通过项（seed 26209）的 D 合法返回 C0，不能归因于学到的保护机制。新增环境交互和 GPU 使用均为 0。由此 H036 当前为 INCONCLUSIVE：方法已实现并冻结，但未来 reward 表现未测试。固定 v1 停止；未授权 lambda 搜索、Stage A、新 seed 或新 fit。Policy Distillation/P&C/CLEAR 的借用仍限于行为拟合和保护当前 keep 初始化 prior；Distral/CPC 是共享整合背景，DMPEL 是条件结构背景；不声称完整论文方法复现、继承论文结果或得到确证。

run002 记录 CPU 8.625535 秒、wall 10.011705487 秒；含收尾预留的上界为 CPU 13.625535 秒、wall 20.011705487 秒，这不是含 footer 的精确实耗，运维成本另行记录。run001 的 `ACCOUNTING_INVALID` 和 UNKNOWN 完整成本保留，E163 累计完整成本仍为 UNKNOWN。两次 reservation 均关闭且不可复用；`active_job=null`、`execution_authorized=false`。来源：[run002 报告](../experiments/exp163/analyses/a002/report.md)、[摘要](../experiments/exp163/analyses/a002/summary.json)、[审计报告](../experiments/exp163/analyses/a002/audit.md)、[审计数据](../experiments/exp163/analyses/a002/audit.json)、[计量记录](../experiments/exp163/analyses/a002/cost_accounting.json)和[E163 根预算账本](../experiments/exp163/budget.json)。本条记录 E163 结果快照；后续 E164 状态见下节。

### 2026-10-23 E164 设计与验证路径（执行前快照）

本段保留 E164 执行前的设计快照：当时 E164 为 `NOT_RUN`，无 reservation，`resource_authorized=false`、`execution_authorized=false`；60 CPU 秒、120 wall 秒、64 MiB 和 0 次交互仅为 planned limits。随后算法方完成冻结实现，主控确认可行性并交由消融方独立验证；最终状态见下一节。原受限设计为同轨迹 F/U selector，区分 V/UG/UR，在搜索中执行 closure 接纳筛选，并在 float32 转换后执行新训练 KL 复查；覆盖全部 16 个配对，精确复用 11 个原 D，对 5 个缺失参数进行恢复，新增评分包最多 15 个。不扩展结构、不增加生产 learner 状态。标量 replay 仅是基于已见数据的设计支持，不是预注册实证；该设计阶段本身不代表新恢复或验证已经完成，也不构成新文献/novelty 或生产算法结论。来源：[主控实现接受](appendix/H036/COORDINATOR_ACCEPTANCE_IMPLEMENTATION_20261023.md)、[设计接受记录](appendix/H036/COORDINATOR_DESIGN_ACCEPTANCE_20261023.json)、[下一设计审查](appendix/H036/NEXT_DESIGN_REVIEW_20261023.md)、[E164 实验入口](../experiments/exp164/README.md)和[E164 根预算账本](../experiments/exp164/budget.json)。E163 额度不复用，累计完整成本仍为 UNKNOWN；此导航不新增 ID。

### 2026-10-23 E164 最终状态

E164 已完成，为 `COMPLETE_VALID / EXPLORATORY_ONLY`；5 fits、10 个新评分包、16 个逻辑配对，0 新交互、0 GPU，reservation 已关闭，未授权下一阶段。它是对已见 E161/E163 数据的有限诊断，不是独立新种子确认。UG−V 在全部 16 项上精确为零；UR−UG 的旧 KL 在 5 项改善，`G_new` 在 2 项改善、3 项退化、11 项不变。结果显示受限旧保护与新拟合之间的权衡，没有确证 H036 或支持普遍收益；H036 维持 INCONCLUSIVE。

最终审计由单独保存的 a003 副本完成，包含离线 metadata/runtime mapping 修正；未重跑 campaign。早期 INVALID/INCOMPLETE 审计、核验器、补丁及主控接受链仍按原证据保留，不把修正描述为原冻结核验器一次通过。来源：[E164 最终报告](../experiments/exp164/analyses/a001/report.md)、[最终审计](../experiments/exp164/analyses/a003/audit.json)、[主控综合](appendix/H036/COORDINATOR_E164_SYNTHESIS_20261023.md)和[最终验证状态](../experiments/exp164/FINAL_VALIDATION.json)。不启动新阶段；不新增文献或新颖性结论。

[算法侧本轮分析](appendix/H036/ALGORITHM_E164_ANALYSIS_20261023.md)已获主控验收；验证、主控与算法三项交付齐备，本轮停止。下一轮实验验证等待用户新指令。

## 2026-10-24 H036 / E165 设计与实现登记

E165 已关联 H036，状态为 PLANNED / IMPLEMENTATION_AND_STATIC_REVIEW；当前仅登记设计、实现与静态审查，尚无 SSH、传输或研究运行许可。导航入口：[算法设计](appendix/H036/NEXT_DESIGN_20261024.md)、[主控合同](appendix/H036/COORDINATOR_E165_CONTRACT_20261024.md)、[E165 实验入口](../experiments/exp165/README.md)和[预算账本](../experiments/exp165/budget.json)。H036 仍为 INCONCLUSIVE；E163/E164 结果和引用关系不变，本次不增加文献主张。
