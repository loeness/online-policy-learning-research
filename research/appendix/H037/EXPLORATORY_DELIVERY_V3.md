# H037 首轮探索：方法与执行合同 v3

状态：本地静态冻结交付，尚非远端检查或运行许可。本文将 v2 静态审查提出的六项阻断修复映射为 v3 的实现要求与待执行检查。六项是审查闭环，不是六个科学消融，也不增加比较臂。精确代码/配置身份以 v3 交付 manifest 为准；独立审查、远端检查和逐阶段准入仍待完成。监督器 p003 尚未准入。

v3 保留 v1 的研究问题、A/B 协议、方法、阈值、种子、交互计划及 learner 包上限；只明确并优先规定异常与容量处理语义。若本文与 v1 的异常、容量或继续运行处理不一致，以本 v3 为准。它不覆盖或改写既有实验记录，不把单 seed 探索提升为确认结论。

## 研究问题与比较范围

在受控导航任务与已知任务组、固定 W0、两个 local 和一个 parent 下，比较自然 LRU（A）与在首次同组 victim 淘汰前尝试一次 H037 交接、未通过普通科学 gate 时退回原 LRU（B）。两臂始终承担相同的按槽 H037 缓存成本。该比较隔离的是是否启用交接，不能称为相对于无该缓存纯 LRU 的净部署比较。无动态树、学习式 Use Gate、额外 parent 或新增实验臂。

固定首要探索指标为首次旧组任务 2 淘汰旧组任务 0 后，随后任务 0 首次回访时 256 步 reward/step 的 B−A。完整报告两臂事件、未触发、缺证据、普通拒绝、错误、容量事件和成本。单 seed 点差仅作探索描述，不构成显著性、确认性或因果结论。

## 固定输入、时序与交互计划

- 输入仅来自最终 PPO rollout 入队后、淘汰前已观测的记录。每槽至多 128 行；样本必须匹配 task、group、slot、generation 和 parent version，并按时间顺序处理。历史 behavior probabilities 不充当当前教师。
- 每槽的时间有序记录取末尾 25% 作操作验收，其余作拟合；不足 victim train 24/验收 8、old train 24/验收 8，或任一驻留槽验收 8 行时不拟合并记录缺证据。128 行时为 96/32；较少时沿用冻结实现中先满足最小训练/验收数的切分规则。训练/验收边界和 parent version 筛选必须由独立断言验证。
- 在 reset 前读取当前绑定的同组 victim 与 parent；每个驻留旧组槽还须有同版本证据。只在副本中把历史观测的路由字段改成目标当前一基槽路由。用淘汰时当前策略重新计算教师，不将历史样本描述为当前策略采集。
- 冻结教师参数为 T=2、epsilon=0.05、四动作均匀混合；教师分别为 victim 的 W0+C0+A_v、old prior 的 W0+C0+0、resident 的 W0+C0+A_r。
- 只用训练切分拟合：`mean KL(victim_teacher || W0+C) + 1.0 * mean KL(old_prior_teacher || W0+C) + 1e-4 * mean((C-C0)^2)`。两组 KL 分别按行取均值，均为前向 KL。操作验收和后续 reward 不用于选参数。
- 从 float32 C0 转 float64 初始化 PyTorch L-BFGS；lr=1，strong-Wolfe，最多 32 次迭代、96 次 closure，梯度/变化容差分别为 1e-9/1e-12。history size 沿用冻结实现；发布 v3 精确包时须由源文件和 SHA 身份绑定其数值，不得凭本文另选。
- 将拟合输出转为 float32 后执行 resident 补偿，再用真实 int8 actor codec 编码/解码。候选与 no-write 参照必须使用同一个实际 codec；no-write 参照先清零 victim local。不得以 float32 替身结果代替实际验收路径。

固定验收门槛：victim 操作验收集上的 `KL(victim_teacher || decoded_no_write_prior) - KL(victim_teacher || decoded_candidate_prior) >= 1e-4`；old prior 的 `KL(old_teacher || decoded_candidate_prior) <= 0.01` nats；每个 resident 槽相对于写前教师的最大单动作概率绝对差 `<= 0.01`。同时记录 no-write old-prior KL 与 resident 漂移。编码前 resident 有效矩阵误差 `<= 1e-6` 且 binding 不变；完整 learner 序列化包 `<= 2,097,152 B`。成功时安装解码结果，清除受影响 resident/shared 优化器状态，保留 critic 和未受影响状态；victim optimizer 在随后正常 reset 清除。上述为冻结探索规则，不代表已有校准或最优性证据。

交互顺序固定如下，计入所有已发生的交互，不把失败交互扣除，也不含未获准的重试：

1. 共同前缀：旧组任务 0、1 各交互 512 步形成 parent；随后旧组任务 0、1 各交互 256 步，共 1,536 步，只执行一次。
2. 从同一共同前缀分叉；A、B 各依次执行旧组任务 2、0、1 和新组任务 0，每个任务 256 步，各 1,024 步。
3. 计划总数为 3,584 次环境交互。分叉前和两臂恢复后必须逐项核对模型、critic、optimizer、缓存、版本、采样/PPO/全局 RNG、模拟器及模拟器 RNG；任何不一致均停，不进入后续交互。模拟器状态属于分叉状态，不能默认在线 learner 免费。

## 事务、拒绝、错误与容量语义

`resolve` 先更新双向映射和 LRU；交接事务从该状态之后开始，不能再次调用 `resolve`。普通科学 gate 拒绝（victim 改善、old prior 或 resident gate 未通过）须记录一次尝试及原因，回滚该候选/交接事务对 actor、optimizer、controller、RNG 等状态的改动，然后按原路径恰好执行一次 LRU reset/分配并继续。该普通拒绝不是运行错误。

以下两类情况均须记录事件、保留失败证据、回滚候选/交接事务，并在 reset 或任何后续环境交互发生前停止本轮：

1. candidate 或 integration 中未预期的执行错误，包括拟合、策略前向、codec、序列化或恢复失败。兼容字段 `accepted: false` 本身不足以表达普通拒绝；执行错误必须具有 `status: error` 和错误类型/消息，传播到 runner 的停止路径。
2. 完整 learner 包超过 2,097,152 B。它是候选包上限事件，不是普通 gate 拒绝；事件可保留 `status: skip`，但必须带 `reason: complete_package_exceeds_budget`，runner 不得继续 reset。

回滚基准是事务开始前（即 `resolve` 已完成后的）状态；停止时保留 `resolve` 已记下的映射/LRU 和事件账本，同时不做 reset/分配或额外环境 step。此时映射已指向新任务而槽仍为旧 victim，是不可恢复的失败现场；不得作为有效 learner checkpoint 或从该现场续跑。现有入口只保留此前共同前缀 checkpoint 和失败记录，无失败现场续跑入口。普通 gate 拒绝则在同一回滚基准上按原 LRU 路径 reset 一次。BaseException 类资源中断交由外层监督器处理并保存部分证据，不能伪装成科学拒绝。

项目外层空间或 inode 准入失败与 learner 包超限分别记账、分别停止：目标项目路径的聚合可用字节、inode、项目用量/限额和预计峰值不满足时，不得开始相应远端传输或阶段；这不是候选 gate，也不能通过 LRU reset 恢复。不得扫描授权路径以外的文件或释放他人资源。空间/inode 准入通过也不等于运行获准。

## v2 六项审查阻断与 v3 闭环证据

| # | v2 阻断 | v3 必须实现或补齐的远端零交互检查 |
|---|---|---|
| 1 | `propose_h037` 将任意普通 `Exception` 吞为 `accepted: false`，可能把拟合、策略前向或 codec 故障伪装成科学拒绝。 | 显式区分 gate rejection 与执行错误；对实际执行错误注入/断言错误状态、事件和候选回滚，且 runner 停止。不得将故障归入拒绝。 |
| 2 | `_prepare_task` 对 `status: error` 记日志后仍 reset 并继续。 | 修正 runner 及原 `test_prepare_task_runs_handoff_once_before_exactly_one_reset` 的预期：错误保留回滚证据；不执行 reset，不发生后续环境交互。普通 gate rejection 仍只走一次原 LRU reset。 |
| 3 | 配置用 int8，但现有 synthetic transaction 走 float32，未验证实际候选与 no-write 比较所用 codec。 | 添加窄范围实际 int8 encode/decode 检查，覆盖候选和同 codec 的 no-write 参照，验证真实验收计算路径。 |
| 4 | 缺少直接断言证明有序 train/holdout 切分不把后续行放入 train，且剔除 stale-parent-version 行。 | 用有序且含 stale version 的固定输入直接断言 train/holdout 行身份、顺序和版本过滤边界。 |
| 5 | 共同前缀恢复合成测试的 H037 evidence cache 为空。 | 在 paired restore 测试中填入非空 H037 cache，检查两臂都恢复相同 rows、identity、ordering 和 version。 |
| 6 | learner 包超限目前被当作 `skip` 继续；未与外层空间/inode 准入失败清楚区分。 | 超 2 MiB fixture 必须产生包超限事件、事务回滚并在 reset/交互前停止；外层空间/inode 准入失败需单独记录为阶段前置条件失败并停止，不运行 candidate。 |

上述检查只能证明对应合成断言；不能证明真实导航模拟器恢复、真实 reward 表现或研究主张。v2 review 点名的检查文件为 `tests/test_h037_initial.py` 和 `tests/test_h037_transactions.py`；v3 实际文件名、节点和内容须以最终冻结包为准。

## 远端检查先决条件与阶段边界

目前没有获准的 v3 远端检查或运行。v2 review 记载用户报告 p002 已上传且校验，但当时远端当前状态和容量未独立观察，v2 文件未报告上传；不得据此推定 v3 文件已在服务器、已被接受或可运行。保留 p002 原物；v3 若获准准备，使用新鲜、中性命名的项目子目录及新 allowlist，不覆盖旧包或历史证据。

任何 CHECKS 开始前，必须具备：

- v3 方法/实现冻结，明确全部待改文件、配置、依赖和测试；生成新的精确 allowlist、manifest 与 SHA/字节身份。两份本机指令文件不得进入 Git、传输、source snapshot 或 archive manifest。
- 算法实现完成六项修复；独立静态审查接受精确版本。审查的源包/配置身份必须与拟传输版本一致。
- 用户/主控对指定项目路径、逐项输入清单和必要传输有明确授权；只在已授权的 `/data/robot/p001` 项目范围内，以 robot 身份操作。不得扩大路径、访问其他账户资料、sudo 或更改共享环境。
- 仅对精确目标路径做聚合容量、inode、项目用量/100 GB 上限及峰值准入；记录估算包括已保留工件、新输出、缓存与临时写入。任一准入失败即停止，不转移、不运行。
- 监督器须冻结并独立审查，覆盖新鲜 supervisor birth 起的启动、导入/预检、测试进程树、输出/临时写入、完整阶段收尾及失败路径；记录解释器、依赖、硬件、线程、资源、输入身份、命令、节点和原始日志。内部限时不能替代外层监督。
- 为该检查阶段另有主控给出的精确 admitted-spec/start directive。监督器 p003 目前仍未准入；本合同、v2 预算建议、既往 p002 上传和静态审查均不构成 p003 或任何远端阶段许可。

第一阶段仅可申请固定 H037 合成 CHECKS，零环境交互。记录实际收集节点、通过/失败/跳过、JUnit、解释器/依赖、监督器与资源计量、输入和包 SHA。检查失败、容量准入失败、错误、资源账目未知或阶段范围未获准时，不启动 RUN。任何后续 RUN 都必须经过新的精确身份审查及单独的阶段运行指令；不得自动重试、自动扩展阶段或依据 CHECKS 通过推断 RUN 已获准。

## 保留的 v2 有限预算建议（非授权）

以下仍只是 v2 的规划建议，未经主控接受和阶段准入，不是可用额度或开跑指令：

- CHECKS：至多 120 aggregate CPU 秒 / 180 受控墙钟秒，0 环境交互。
- RUN：至多 900 aggregate CPU 秒 / 1,200 受控墙钟秒，环境交互上限 3,584。
- AUDIT：至多 120 aggregate CPU 秒 / 180 受控墙钟秒，0 环境交互。
- 建议总计：1,140 CPU 秒 / 1,560 受控墙钟秒；每阶段一次，无自动重试。建议峰值新输出、缓存和临时文件合计不超过 256 MiB；这与每份 learner 包 2 MiB 上限是不同限制。
- 单 worker、单数值线程、0 GPU。SSH、准备和传输单独记账；未测 CPU 记 UNKNOWN。阶段边界、监督器自身与 finalization 必须落入获准的完整预算。

CHECKS 通过也不会自动获准 RUN 或 AUDIT。执行完成并由算法负责人/主控分析后停止；下一阶段和新增预算等待新的明确指令。
