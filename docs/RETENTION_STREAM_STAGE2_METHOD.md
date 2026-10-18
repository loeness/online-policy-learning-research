# 自然淘汰后的共享写入：本轮实际算法

这是受控的预训练后在线参数学习实验。局部情境编号和切换提示已知；共享家族关系不提供。没有实现未知边界发现、深树、父模块替换、世界模型或一般持续学习保证。

## 1. 持久结构

基础 actor W0 冻结。固定输出基和特征基上，每个局部槽有三个 3×3 矩阵 A，两个共享模块各有一个 3×3 矩阵 C。当前有效修正是：

    M[i,b] = A[i,b] + C[group(i)]
    logits = logits_W0 + H M[i,b(s)] P phi_W0(s).

未分组时 C 为零。band 是已给定的 x 位置特征分区，不是在线发现的树层级。

低容量使用 3 局部 + 2 共享，对照为 4 个独立局部；宽容量使用 6+2，对照为 7 个独立局部。局部槽按相同 LRU 规则重用，清除局部参数、动量和近期历史，并递增 generation。共享模块和跨槽证据可继续保留。

每个槽保存最近 256 个可见事件的有界数值窗口。独立的跨槽 archive 最多保存 8 个原型、每个原型 24 行 reservoir 证据。原型由观测到的动作响应和奖励统计匹配，没有真值 family 标签。当前 actor 使用 int8 编解码；窗口、证据、critic、优化器和 pending 候选也计入保存包。

## 2. 每 256 步的顺序

1. 按真实行为分布记录本批 obs/action/reward、完整动作概率、所选动作 log-probability 和 episode ID；计算 GAE。
2. 从近期窗口得到 profile，分组后以补偿方式绑定共享模块，保持已有有效策略。
3. 使用这批数据检验上一批提出并冻结的候选。若新 profile 弃权，不把保留的 acting group 当成新的分组证据。
4. 全部新行进入局部 PPO；direct_shared 同时通过 PPO 更新共享参数。
5. 更新固定大小的 archive。安装已批准的旧候选，补偿所有受影响的驻留局部槽。
6. 对尚未使用共享先验的新局部 generation，按预先检验结果至多加一次共享修正。编码并重新部署 int8 actor；所有方法按同一频率清除 actor Adam 动量，保留 critic 动量。
7. 每两个 rollout，若同组至少两个仍驻留的局部槽已学习 256 步，提出其有效修正的平均值。最多保留一个 pending 候选，1024 步内等不到可用检验则过期。

先验诊断在局部 PPO 前做，实际注入在 PPO 后做，因此被注入的最新策略不是原诊断中的策略。记录实际概率变化，但不将该诊断写成单调改进保证。识别需要真实交互，因此每次新槽的前 256 步不具备自动共享冷启动保证。

## 3. 删除局部后的写入检验

候选 C* 是当前多个来源 M 的简单平均，跨三个 band 共同拟合一个 3×3 公共矩阵。gate 比较仅使用 C* 与仅使用旧 C 的策略，删除了局部修正，试图衡量可留存内容的增益。

在保存了行为策略 μ 的证据上，使用：

    mean[(clip(pi_candidate(a|s)/mu(a|s))
          - clip(pi_old_parent(a|s)/mu(a|s))) * advantage_mu].

计算实际包含 temperature/epsilon 的动作概率。支持检查包括样本数、episode 数、重要性采样 ESS 和行为 KL。误差惩罚使用 row-weighted 均值的 centered episode cluster sums，允许 episode 长度不同。它是经验代理，不是长期回报估计或可靠安全下界。

retention_gate 要求至少两个历史原型有支持、平均经验 lower score 为正、每个原型下降不超过阈值、所有受保护历史可评估，且新的同组 rollout 通过检查。历史行可能参与过候选形成；只有后到的 fresh rollout 在时间上未用于形成该候选。多个原型也可能来自同一情境，不能把其个数当独立任务验证数。

## 4. 共享安装与复用不同

共享写入时对全部驻留成员执行：

    A[i,b] <- A[i,b] + C_old - C_new.

这样编码前 M 不变。写入本身没有立即提升驻留策略，其作用应在局部淘汰后的后续复用中检验。实际 int8 部署会有量化误差，逐 rollout 记录系数误差及当前行为观测上的概率/KL 漂移。

新局部先从 W0 学一批数据，再选择性加入一个 C；这是行为改变。mean/direct_shared 只要求 profile 匹配，mean_reuse_gate/retention_gate 还检查共享先验收益。后两臂具有相同的使用门控，二者的差别才直接对应新增的写入门控。

## 5. 对照与结论范围

五臂是 independent、direct_shared、mean、mean_reuse_gate、retention_gate。独立臂局部学习率 .004，共享臂局部 .003、direct_shared 的共享 .001；因此 independent 与 retention_gate 的差别不只包含写入规则。不能仅凭二者回报差归因于新增机制。

所有方法使用同一外生任务日程、基础权重、每个 seed 相同的采样均匀数和完整在线数据。状态轨迹仍会随动作不同而不同。每种方法实际执行的 PPO minibatch 数单独记录。

低档 actor cap 140 B，宽档 236 B；完整 learner 保存包统一不超过 320 KiB。这是 actor 配额与保存包上限的组合，不是固定总字节下最优的资源再分配比较。基线还应在后续得到更多槽或回放的公平预算选择。完整包大小也不代表峰值 RAM、推理延迟或学习加速。

结果、冻结协议和统计报告分别见 [本轮结论](RETENTION_STREAM_STAGE2_REVIEW.md)、[运行协议](RETENTION_STREAM_STAGE2_PROTOCOL.md) 和 [逐种子报告](../outputs/retention_stream_stage2_report_v2/report.md)。此前提取实验见 [Stage 1](CONSOLIDATION_RETENTION_STAGE1_REVIEW.md)。
