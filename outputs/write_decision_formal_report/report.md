# Write-decision cohort 分析

输入 cohort：`D:\MovedFromC\ChatGPT\online\outputs\write_decision_formal`。阶段：`formal`；seed：1009, 1013, 1019。

统计单位是 seed。episode、transition 和 gate 决策只用于计算每个 seed 的指标，不作为独立重复。当前 seed 数量只支持描述性比较；不报告显著性或因果结论。

负奖励任务的 `goal_reached` 不计入正向 success。`after_source` 的 A2+ 是 A2 训练之前的 cold-transfer 观察；final A2+ 是完成其后续 exposure 后的观察。

## 条件均值（每 seed 先汇总）

| 条件 | n seeds | Online reward / train step | 正向训练 goal rate | Final 正向 eval goal | Cold A2+ goal | CPU s | Wall s | Interactions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| independent_fast | 3 | 0.0148 | 0.3443 | 0.3750 | 0.1250 | 36.18 | 36.84 | 50767 |
| dynamics_shared | 3 | 0.0152 | 0.3811 | 0.5694 | 0.1250 | 36.76 | 37.54 | 50531 |
| reward_shared | 3 | 0.0191 | 0.4424 | 0.5694 | 0.1250 | 36.91 | 37.67 | 50534 |
| trial_gate_relaxed | 3 | 0.0188 | 0.4237 | 0.5139 | 0.1250 | 37.15 | 37.78 | 50626 |
| projected_gate | 3 | 0.0153 | 0.3541 | 0.4722 | 0.1250 | 36.29 | 36.95 | 50668 |

缺少条件：trial_gate, local_promote, trial_rollback。该输入不是完整的八条件 cohort。

## Paired 差值

下表给出相同 seed 内的 candidate − reference 均值；括号中列出 seed 数。数值是描述性差异，不代表统计显著或已识别共享收益。完整逐 seed 差值见 `analysis.json`。

| 条件 | 对照 | Online reward / step Δ | 正向训练 goal Δ | Final 正向 eval return Δ | Cold A2+ return Δ | n paired seeds |
|---|---|---:|---:|---:|---:|---:|
| independent_fast | independent_fast | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3 |
| independent_fast | reward_shared | -0.0044 | -0.0981 | -0.7138 | 0.0000 | 3 |
| dynamics_shared | independent_fast | 0.0004 | 0.0368 | 0.6731 | 0.0000 | 3 |
| dynamics_shared | reward_shared | -0.0040 | -0.0613 | -0.0407 | 0.0000 | 3 |
| reward_shared | independent_fast | 0.0044 | 0.0981 | 0.7138 | 0.0000 | 3 |
| reward_shared | reward_shared | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3 |
| trial_gate_relaxed | independent_fast | 0.0040 | 0.0794 | 0.4606 | 0.0000 | 3 |
| trial_gate_relaxed | reward_shared | -0.0004 | -0.0187 | -0.2532 | 0.0000 | 3 |
| projected_gate | independent_fast | 0.0005 | 0.0098 | 0.2535 | 0.0000 | 3 |
| projected_gate | reward_shared | -0.0038 | -0.0883 | -0.4603 | 0.0000 | 3 |

## Gate、ESS 与 policy drift

| 条件 | Proposal decisions | Gate attempts | Gate accepts | Gate rejects | Main reject reasons | Cold ESS 均值 | Cold behavior KL 均值 | Old aggregate ESS 均值 | Old aggregate max KL 均值 | Decision CPU s |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| independent_fast | 0.0 | 0.0 | 0.0 | 0.0 | — | — | — | — | — | 0.3125 |
| dynamics_shared | 0.0 | 0.0 | 0.0 | 0.0 | — | — | — | — | — | 0.3021 |
| reward_shared | 0.0 | 0.0 | 0.0 | 0.0 | — | — | — | — | — | 0.2708 |
| trial_gate_relaxed | 40.0 | 40.0 | 22.3 | 17.7 | cold_gain_below_threshold:33, cold_support_failed:20 | 206.4568 | 0.0977 | 24.8311 | 0.0055 | 0.4427 |
| projected_gate | 40.0 | 39.7 | 2.7 | 37.0 | cold_support_failed:92, cold_gain_below_threshold:11, old_slot_unsupported:7 | 185.8110 | 0.1302 | 11.7384 | 0.0039 | 0.5573 |

## Group coverage

`shadow_group` 表示证据分组器的分配；`group` 表示该条件实际部署的分组。group ID 只是局部分组编号，不是任务或 reward 标签。Coverage 按每个 seed 的 assignment 数与比例汇总。

| 条件 | Assignment rows / seed | Shadow assigned / seed | Shadow coverage | Deployed assigned / seed | Deployed coverage | Assignment reasons |
|---|---:|---:|---:|---:|---:|---|
| independent_fast | 6.0 | 6.0 | 1.0000 | 0.0 | 0.0000 | matched:12, first_group:3, opposite_reward_group:3 |
| dynamics_shared | 6.0 | 6.0 | 1.0000 | 6.0 | 1.0000 | matched:15, first_group:3 |
| reward_shared | 6.0 | 6.0 | 1.0000 | 6.0 | 1.0000 | matched:12, first_group:3, opposite_reward_group:3 |
| trial_gate_relaxed | 6.0 | 6.0 | 1.0000 | 6.0 | 1.0000 | matched:12, first_group:3, opposite_reward_group:3 |
| projected_gate | 6.0 | 6.0 | 1.0000 | 6.0 | 1.0000 | matched:12, first_group:3, opposite_reward_group:3 |

Reward identification is not an oracle: reward coefficients are inferred from the finite transition/reward evidence available in each profile. A profile with no observed terminal transitions may have weak or zero terminal-feature estimates after ridge regularization, which can look like `reward_mismatch` against a better-supported profile. That reason does not establish task or reward incompatibility. The measured group coverage is limited to these probe assignments; unresolved abstentions and mismatches mean routing coverage is not shown to be solved.

Gate accepts/rejects 是当前规则基于 cold-parent 与有限旧 anchor surrogate 的输出。没有独立 counterfactual 时，共享是否真实有益为 undetermined；因此 false accept 和 false reject 目前都不可测。Surrogate ESS/KL、优势估计及 episode-cluster SE 不是共享收益真值或安全保证。

## 资源与物理容量

| 条件 | Train steps / seed | Fit / Holdout | Probe steps / seed | Eval steps / seed | Actor peak B | Full checkpoint .pt peak B | W0 + checkpoint package peak B | CPU s | Eval CPU s | Wall s | Decision CPU s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| independent_fast | 40960 | 30720 / 10240 | 1536 | 8271 | 666 | 106477 | 161954 | 36.18 | 3.91 | 36.84 | 0.3125 |
| dynamics_shared | 40960 | 30720 / 10240 | 1536 | 8035 | 738 | 106712 | 162189 | 36.76 | 3.90 | 37.54 | 0.3021 |
| reward_shared | 40960 | 30720 / 10240 | 1536 | 8038 | 738 | 106669 | 162146 | 36.91 | 3.67 | 37.67 | 0.2708 |
| trial_gate_relaxed | 40960 | 30720 / 10240 | 1536 | 8130 | 738 | 106733 | 162210 | 37.15 | 3.86 | 37.78 | 0.4427 |
| projected_gate | 40960 | 30720 / 10240 | 1536 | 8172 | 738 | 106755 | 162232 | 36.29 | 3.90 | 36.95 | 0.5573 |

## 数据账与审计

Dataset SHA256：`493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2`。

| Seed | 条件 | 审计通过 | 失败项 |
|---:|---|---|---|
| 1009 | independent_fast | 是 | — |
| 1009 | dynamics_shared | 是 | — |
| 1009 | reward_shared | 是 | — |
| 1009 | trial_gate_relaxed | 是 | — |
| 1009 | projected_gate | 是 | — |
| 1013 | independent_fast | 是 | — |
| 1013 | dynamics_shared | 是 | — |
| 1013 | reward_shared | 是 | — |
| 1013 | trial_gate_relaxed | 是 | — |
| 1013 | projected_gate | 是 | — |
| 1019 | independent_fast | 是 | — |
| 1019 | dynamics_shared | 是 | — |
| 1019 | reward_shared | 是 | — |
| 1019 | trial_gate_relaxed | 是 | — |
| 1019 | projected_gate | 是 | — |

计费复核从原始 `steps.csv`、`probes.csv` 和 `evaluations.csv` 计数，并核对 summary 中的 training/probe/evaluation/total interactions、fit/holdout、decision CPU 及每个 checkpoint 与 actor sidecar 的磁盘字节。预算检查将 actor sidecar 与 actor cap 比较，并将 `full_persistent_bytes`（W0 foundation 加序列化状态）与 package cap 比较。来源快照与 W0/dataset manifest hash 会逐项验证。

## 解释边界

这些结果是本 cohort 上的描述性比较。两个或三个 seed 不足以支撑显著性表述。Whole-run arm 的对照可以比较平均结果，但单次 gate 的 false accept/reject 仍需要独立、预先定义的 counterfactual 结果才能判断。不得据此称该方法为 SOTA，也不能据此宣称已证明创新性。

## 图表

- online_outcomes: `online_outcomes.png` / `online_outcomes.pdf`
- evaluation_goal_rates: `evaluation_goal_rates.png` / `evaluation_goal_rates.pdf`
- costs: `costs.png` / `costs.pdf`
- capacity: `capacity.png` / `capacity.pdf`
