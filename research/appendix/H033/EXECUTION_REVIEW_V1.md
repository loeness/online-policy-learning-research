# H033 v1 · 实验执行审查与准入草案

日期：2026-10-21。负责人：消融实验（1）。本文件不修改方法合同、不占用 E ID、不批准预算、不更新 H 科学状态。候选编号 E162 仅供主控或用户确认；当前未建立 exp162、未运行真实特征提取或交叉拟合。

## 已确定的执行对象

- 研究问题：固定 old-major/new-major 等权未来下，三个操作前特征及固定 ridge 是否产生超越简单规则的留出选择价值。
- 方法唯一来源：[METHOD_V1.md](METHOD_V1.md)，交接：[HANDOFF_V1.md](HANDOFF_V1.md)。不增加特征、改模型、换折、调参或改变动作绑定。
- 冻结源：delivery_v1/source_snapshot；父 Git SHA `81de4afd2d703d17c3953facb78fd04b19cb64a9`，完整源身份 `eed12dc721e9d3a56e6d079cd51ecd4bb29f87c884f362f044d97ada8ba0351c`。dirty/untracked 文件以交付 manifest、snapshot 和 patch 绑定。
- 输入：原 E161 的 16 个 prefix，每个仅读 checkpoint.pt / actor.bin / evidence.npz；标签单独读取 E161 analysis001/analysis.json（SHA256 `813561c90e3219bc1463133c67aead68103849b465c7a72bbf7c20fc798bbaf8`）。全部实际 D: 路径见交付 manifest。
- 只用本地已有镜像，无 SSH、远端写入、GPU、sudo 或新增环境交互；不沿用 E161 剩余预算。只执行一个冻结批次。
- 交付合成检查记录为 28 passed；不把它计作真实科学结果，不重复无关测试。独立代码与逐文件/输入身份审查结果另附。

## 提请接受的执行准入（尚未批准）

建议为这一次离线诊断分配 E162，CPU 累计不超过 300 秒、campaign 墙钟不超过 300 秒（包括真实阶段的启动、失败和一次技术重试），一个进程、一个 Torch/BLAS thread，新增输出不超过 50 MiB。估计实际为数秒至数十秒；该估计不是实测。新增环境交互精确为 0。必要输入只读，不复制 E161 大型档案。

运行顺序为冻结 CLI 的 features → outcomes → crossfit，前一阶段成功并核验后才进入下一阶段。原交付源码逐文件哈希在前后复核；操作前特征和标签分别落盘。输出路径建议 experiments/exp162/runs/run001，各次技术重试独立 run，不覆盖失败证据。若任何 seed 输入无效、完整性不符或预算到限，停止整批；不得删 seed、补值、改特征、换目标或做环境重跑。方法错误必须退回算法实现负责人修订并另交版本。

## 结果前固定的统计提案（尚未批准）

主报告为五种规则各自在 16 个留出 seed 上的选择频率、实际所选 reward/step、相对 keep 增量，以及 crossfit 相对四个简单规则的逐 seed 配对差。四个联合层各 4 个，故全 seed 平均等于联合层等权平均。每个 seed 内先平均两个 replica，再将两个主要未来日程等权；balanced 仅对同一留出决策作次要评分。

建议筛查门槛：crossfit 相对 fixed_keep、fixed_replace、intercept_only、frequency_rule 的四个平均选择价值增量均至少为 **0.001 reward/step**，且每个比较在四个留出折中的均值都为正，才返回“值得由主控设计新 seed 确认”的筛查信号。任何一项未过则返回“本最小诊断不支持追加该选择器复杂度”。这是新诊断的拟定实际价值门槛，不沿用或改写 E161 的确认门槛。

不确定性描述固定为每项比较的全部 16 个 seed 差、正/零/负个数、四个折均值及其范围；不把分支或四折当作独立重复，不给这 16 个交叉拟合得分套用独立样本 t 区间。训练集重叠、复用已见 E161 和仅 16 个独立 prefix，限制了泛化与推断。此处不报告显著性或确认性置信区间；折间一致性只是预定稳定性筛查，不是误差率保证。MSE/相关性即使改善也不用于推进。

本门槛、预算与 E ID 必须在任何真实特征/预测输出出现前获得主控或用户明确接受；实验聊天不自行将其设为既定科学规则。通过也不自动启动确认、完整流、H035 或 H036；只交回证据，由主控作研究决策。

## 产物与可复现性

获准后建立新 E 的 README、预算和 run provenance，绑定交付 Git + source identity + 全部输入哈希。保存 features.json、outcomes.json、crossfit.json、外部阶段计时/命令/日志、只读统计 report.json / report.md、逐 seed CSV，以及交付/输入前后哈希验证。图为 NOT_APPLICABLE：本诊断用完整表格与逐 seed/折数据，不为填目录生成图。

仅写获准 E 目录及本审查文件；交付 source_snapshot、E161 工件、在线算法、其他 H/E 记录保持不变。共享索引/预算的登记与研究结论集成归主控；任何本轮代登记须获得用户明确指派。两份机器本地指令文件与凭据不进入任何源包或科学工件。

## 独立审查完成（真实运行前）

方法合同与实现核对通过；19 个冻结文件、52 项输入引用（其中 48 个操作前文件）的大小和 SHA256 全部匹配。D: 独立 Python 与所需 CPU 依赖可用。旧先验/新绑定、原候选重构、整 seed 留出与无 balanced 泄漏均通过代码审查。额外绑定 E161 c1 配置的 temperature=2 / epsilon=.05 来源哈希。详见 [review.json](execution_v1/review.json) 与 [checks.json](execution_v1/checks.json)。统计汇总器 [report.py](execution_v1/report.py) 以合成算例验证均值、折均值、选择次数及无准入/错误所选值拒绝；没有读取真实特征或生成真实预测。

剩余准入仅为主控或用户明确接受 E 登记、预算及筛查门槛；本审查未代替该决定。
