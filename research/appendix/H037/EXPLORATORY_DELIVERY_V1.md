# H037 首轮探索：方法和执行交接 v1

状态：本地静态交付；未经远端合成检查、独立审查或运行准入。本文及配置不构成资源或服务器操作授权。信息经用户手动转达。旧上传包和服务器目录保持原样。

## 问题和分支

受控导航、已知任务与组、固定 W0、两个 local 和一个 parent。A 为自然 LRU；B 在第一次满足条件的同组 victim 淘汰前尝试交接，失败走原 LRU。两臂持有相同的 H037 按槽缓存，其成本均计入；因此该比较隔离是否启用交接，不能称为相对于无该缓存的纯 LRU 的完整部署净成本。没有动态树、学习式 Use Gate、额外 parent 或额外实验矩阵。

先各对旧组任务 0、1 交互 512 步形成 parent，再对旧组任务 0、1 各交互 256 步；这 1536 步仅执行一次。按共同前缀恢复后，两臂各依次执行旧组任务 2、0、1、新组任务 0，每个 256 步，各 1024 步。总计 1536+1024+1024=3584 次环境交互，计入因失败已产生的交互，不含未计量的重试。分叉前及每臂恢复后核对模型、critic、优化器、缓存、版本、采样/PPO/全局 RNG、模拟器与其 RNG；不一致立即停止。模拟器含研究者分叉状态，不默认为在线 learner 免费状态。

第一旧组任务 2 会按 LRU 淘汰旧组任务 0；固定首要探索指标是随后任务 0 第一次回访的 256 步 reward/step 的 B−A。两臂全部事件、未触发、缺证据、拒绝、容量失败、异常及成本均须报告；单 seed 点差不是显著性或确认结果。

## 输入、时序与教师

只使用最终 PPO rollout 入队后、淘汰前已观测的数据。每槽最多128行，以 task、group、slot、generation、parent version 限定，按时间顺序取末尾 25% 作为操作验收，其余拟合；不足 victim train 24/验收8或 old train 24/验收8、每驻留验收8就跳过。128行时为96/32；较少时按实现先保证最小拟合/验收数。数据可来自历史采样策略的状态分布，历史 behavior_probs 不作为当前教师。没有 episode 独立性声明。

在 reset 前读取同组、当前绑定的 victim 和 parent，驻留旧组槽需有同版本证据。将历史观测路由字段在副本中改为对应当前一基槽路由。`T=2`、`epsilon=0.05` 与四动作均匀混合来自冻结 c1 配置和真实策略定义。用淘汰时当前策略对历史观测重新算 victim 的 `W0+C0+A_v`、old prior 的 `W0+C0+0` 和 resident 的 `W0+C0+A_r` 教师；不要把它解释为轨迹在当前策略下采样。

只以 train 拟合 `mean KL(victim_teacher||W0+C)+1.0*mean KL(old_prior_teacher||W0+C)+1e-4*mean((C-C0)^2)`；两组分别行均值，均为前向 KL。以 float64 从 float32 C0 初始化 PyTorch L-BFGS（lr=1，strong-Wolfe，最大32迭代、96 closure、梯度/变化容差1e-9/1e-12，history 见冻结源码）；超限或非有限拒绝。不使用操作验收或后续 reward 选择参数。输出转换为 float32 后进行 resident 补偿和真实 int8 actor 编码解码。

## 验收、事务与恢复

候选和 no-write 参照均使用相同实际 codec：将 victim local 清零后分别编码解码。victim gate 为验收集 `KL(victim_teacher||decoded_no_write_prior) - KL(victim_teacher||decoded_candidate_prior) >= 1e-4`；旧 prior gate 为 `KL(old_teacher||decoded_candidate_prior) <= 0.01` nats；resident gate 为每个驻留槽上，相对写前 resident 教师的最大单动作概率绝对差 `<=0.01`。同时记录 no-write 的旧 prior KL 和 resident 漂移；不是用差值代替这两项绝对门槛。编码前 resident 有效矩阵误差须不超过1e-6，binding 不变，完整 learner 序列化大小不得超过2,097,152 B。成功时安装解码结果并清除受影响 resident 和 shared 的优化器状态，critic 及未受影响状态保留；victim 的优化器在随后正常 reset 清除。此状态处理是本探索固定约定，不是已证实最优方案。

`resolve` 会先更新双向映射及 LRU；交接事务从该时刻开始，而非再次调用 resolve。候选拒绝、容量不符或普通异常恢复候选造成的 actor、optimizer、controller、RNG 等变动，保留一次尝试及原因记录，再按原路径恰好 reset/分配一次。不得跨未完成 rollout；检查分支或事务无法证明等价时停批。所有 BaseException 类资源中断交外层监督器处理并保留部分结果，不当作正常科学拒绝。

阈值是本次指定的探索性工程值，不依据本轮结果选取：1e-4 nats 是非零但很小的 victim KL 改善门槛；0.01 nats 限制旧 prior 的观测集偏移；0.01 是单动作概率的绝对漂移尺度。三者均未依据独立校准证明有用、无风险或对应 reward；若无法触发/通过，应如实报告，不在本轮调整门槛。

## 检查和资源建议

执行前由验证方独立审查具体版本并逐阶段准入。第一阶段仅执行固定 H037 合成检查，零环境交互，检查教师路由与真实概率、时间切分、success、普通拒绝、完整包容量拒绝、异常恢复、优化器和 RNG、两臂共同前缀恢复。最终以远端实际收集的节点与回执为准。测试替身不能作为真实环境正确性或 reward 证据。

预期以受控环境、单线程、项目内临时目录及禁用自动插件的 pytest 执行 `tests/test_h037_initial.py` 和 `tests/test_h037_transactions.py`；必须记录实际收集节点、失败/跳过、JUnit、版本和资源。真实策略概率对照使用随机初始化的合成 FlexibleSharedPolicy，不加载 W0 工件。事务检查使用合成 core 和候选替身；前缀检查的模拟器亦为替身，因此通过不能单独证明真实导航模拟器恢复，正式 RUN 的恢复相等检查及独立审查仍必需。执行责任方应先冻结其 supervisor、解释器、pytest 入口与路径，不能将下面接口示意当作执行指令：

```text
<project-python> -m pytest -q --noconftest -p no:cacheprovider --basetemp=<approved-project-temp> tests/test_h037_initial.py tests/test_h037_transactions.py
<project-python> scripts/run_h037_initial.py --config configs/h037_exploration_v1.json --output <fresh-approved-project-output> --seed 27101
```

两个命令必须分别由已准入的阶段 supervisor 启动；真实目录、环境变量、导入依赖、资源计量和每阶段 start 指令均以验证方准备及主控接受为准。

建议新独立预算：CHECKS 最多120 aggregate CPU秒/180受控墙钟秒；探索 RUN 最多900 aggregate CPU秒/1200受控墙钟秒；结果与账本 AUDIT 最多120 CPU秒/180墙钟秒；合计1140 CPU秒/1560累计受控墙钟秒。各阶段均含进程启动、预检、失败及收尾，一阶段一次，无自动重试。RUN 内部 CPU/墙钟上限840/1110秒，给外层监督器预留60 CPU秒/90墙钟秒；CHECKS/AUDIT 各预留20 CPU秒/30墙钟秒收尾。每阶段单 worker、单数值线程、0 GPU；CHECKS/AUDIT 各0交互，RUN总交互最多3584。整个新轮输出+缓存+临时峰值建议256 MiB；独立的 learner完整持久包每份上限2 MiB，不把前者当成后者。2 MiB 大于历史轮次的262144 B，因此本次不得声称与该历史容量等额；A/B 两臂仍使用相同新上限。SSH、传输和运维耗时另记，未测 CPU 为 UNKNOWN。以上是保守规划建议，非耗时实测和非资源授权；容量、inode、存储、清单及项目 100 GB 限额需由责任方准入。需要能限制和记录树内子进程、时间、交互及路径内空间的 supervisor；内部 rlimit 不覆盖完整 campaign。

研究运行只由消融实验在指定受控远端、主控接受预算及精确运行指令后执行。合成检查失败则停止，不启动交互；候选拒绝/缺证据是可报告探索结果，但错误及无效计量不能冒充科学结果。远端保留原始记录与失败工件，完成算法和主控分析后停止。

## 待主控／独立审查确定

接受或调整上面的未校准阈值、单 seed 27101、2 MiB learner cap 和分阶段预算；确认监督器、远端新路径、输入白名单、空间峰值、依赖身份和测量定义。结果前冻结实际配置和全部文件 SHA。正式确认或泛化结论需要另一个独立协议，本交付不启动或预设下一轮。
