# Write-decision cohort 分析

输入 cohort：`D:\MovedFromC\ChatGPT\online\outputs\write_decision_supported_validation`。阶段：`validation`；seed：859, 863。

统计单位是 seed。episode、transition 和 gate 决策只用于计算每个 seed 的指标，不作为独立重复。当前 seed 数量只支持描述性比较；不报告显著性或因果结论。

负奖励任务的 `goal_reached` 不计入正向 success。`after_source` 的 A2+ 是 A2 训练之前的 cold-transfer 观察；final A2+ 是完成其后续 exposure 后的观察。

## 条件均值（每 seed 先汇总）

| 条件 | n seeds | Online reward / train step | 正向训练 goal rate | Final 正向 eval goal | Cold A2+ goal | CPU s | Wall s | Interactions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| independent_fast | 2 | 0.0045 | 0.1438 | 0.2500 | 0.1875 | 19.42 | 19.89 | 30368 |
| dynamics_shared | 2 | 0.0045 | 0.1599 | 0.3125 | 0.1875 | 19.77 | 20.28 | 30320 |
| reward_shared | 2 | 0.0063 | 0.1640 | 0.4375 | 0.1875 | 21.50 | 22.07 | 30274 |
| trial_gate | 2 | 0.0054 | 0.1596 | 0.3125 | 0.1875 | 22.09 | 22.55 | 30316 |
| trial_gate_relaxed | 2 | 0.0062 | 0.1732 | 0.3542 | 0.1875 | 22.95 | 23.37 | 30234 |
| projected_gate | 2 | 0.0057 | 0.1700 | 0.3125 | 0.1875 | 23.27 | 23.98 | 30302 |
| local_promote | 2 | 0.0049 | 0.1498 | 0.2500 | 0.1875 | 23.34 | 23.94 | 30385 |
| trial_rollback | 2 | 0.0055 | 0.1709 | 0.2917 | 0.1875 | 23.10 | 23.71 | 30322 |

## Paired 差值

下表给出相同 seed 内的 candidate − reference 均值；括号中列出 seed 数。数值是描述性差异，不代表统计显著或已识别共享收益。完整逐 seed 差值见 `analysis.json`。

| 条件 | 对照 | Online reward / step Δ | 正向训练 goal Δ | Final 正向 eval return Δ | Cold A2+ return Δ | n paired seeds |
|---|---|---:|---:|---:|---:|---:|
| independent_fast | independent_fast | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2 |
| independent_fast | reward_shared | -0.0018 | -0.0202 | -0.6392 | 0.0000 | 2 |
| dynamics_shared | independent_fast | -0.0000 | 0.0161 | 0.2815 | 0.0000 | 2 |
| dynamics_shared | reward_shared | -0.0019 | -0.0041 | -0.3577 | 0.0000 | 2 |
| reward_shared | independent_fast | 0.0018 | 0.0202 | 0.6392 | 0.0000 | 2 |
| reward_shared | reward_shared | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2 |
| trial_gate | independent_fast | 0.0009 | 0.0158 | 0.3529 | 0.0000 | 2 |
| trial_gate | reward_shared | -0.0009 | -0.0044 | -0.2862 | 0.0000 | 2 |
| trial_gate_relaxed | independent_fast | 0.0017 | 0.0294 | 0.4346 | 0.0000 | 2 |
| trial_gate_relaxed | reward_shared | -0.0001 | 0.0092 | -0.2046 | 0.0000 | 2 |
| projected_gate | independent_fast | 0.0012 | 0.0261 | 0.3642 | 0.0000 | 2 |
| projected_gate | reward_shared | -0.0006 | 0.0060 | -0.2750 | 0.0000 | 2 |
| local_promote | independent_fast | 0.0004 | 0.0060 | 0.0183 | 0.0000 | 2 |
| local_promote | reward_shared | -0.0015 | -0.0142 | -0.6208 | 0.0000 | 2 |
| trial_rollback | independent_fast | 0.0010 | 0.0271 | 0.2369 | 0.0000 | 2 |
| trial_rollback | reward_shared | -0.0008 | 0.0069 | -0.4023 | 0.0000 | 2 |

## Gate、ESS 与 policy drift

| 条件 | Proposal decisions | Gate attempts | Gate accepts | Gate rejects | Main reject reasons | Cold ESS 均值 | Cold behavior KL 均值 | Old aggregate ESS 均值 | Old aggregate max KL 均值 | Decision CPU s |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| independent_fast | 0.0 | 0.0 | 0.0 | 0.0 | — | — | — | — | — | 0.1250 |
| dynamics_shared | 0.0 | 0.0 | 0.0 | 0.0 | — | — | — | — | — | 0.1250 |
| reward_shared | 0.0 | 0.0 | 0.0 | 0.0 | — | — | — | — | — | 0.1719 |
| trial_gate | 20.0 | 20.0 | 6.0 | 14.0 | cold_support_failed:20, cold_gain_below_threshold:5, old_slot_unsupported:3 | 215.1158 | 0.0680 | 29.2634 | 0.0032 | 0.2188 |
| trial_gate_relaxed | 20.0 | 20.0 | 17.5 | 2.5 | cold_gain_below_threshold:4, cold_support_failed:1 | 221.5274 | 0.0546 | 27.2938 | 0.0056 | 0.2969 |
| projected_gate | 20.0 | 20.0 | 6.5 | 13.5 | cold_support_failed:18, cold_gain_below_threshold:5, old_slot_unsupported:4 | 216.7303 | 0.0635 | 15.7374 | 0.0025 | 0.3516 |
| local_promote | 20.0 | 20.0 | 2.5 | 17.5 | cold_support_failed:20, cold_gain_below_threshold:12, old_slot_unsupported:2 | 212.7303 | 0.0665 | 23.2305 | 0.0108 | 0.2109 |
| trial_rollback | 20.0 | 0.0 | 0.0 | 0.0 | trial_rollback_control:40 | — | — | — | — | 0.1797 |

## Group coverage

`shadow_group` 表示证据分组器的分配；`group` 表示该条件实际部署的分组。group ID 只是局部分组编号，不是任务或 reward 标签。Coverage 按每个 seed 的 assignment 数与比例汇总。

| 条件 | Assignment rows / seed | Shadow assigned / seed | Shadow coverage | Deployed assigned / seed | Deployed coverage | Assignment reasons |
|---|---:|---:|---:|---:|---:|---|
| independent_fast | 6.0 | 6.0 | 1.0000 | 0.0 | 0.0000 | matched:8, first_group:2, opposite_reward_group:2 |
| dynamics_shared | 6.0 | 6.0 | 1.0000 | 6.0 | 1.0000 | matched:10, first_group:2 |
| reward_shared | 6.0 | 6.0 | 1.0000 | 6.0 | 1.0000 | matched:8, first_group:2, opposite_reward_group:2 |
| trial_gate | 6.0 | 6.0 | 1.0000 | 6.0 | 1.0000 | matched:8, first_group:2, opposite_reward_group:2 |
| trial_gate_relaxed | 6.0 | 6.0 | 1.0000 | 6.0 | 1.0000 | matched:8, first_group:2, opposite_reward_group:2 |
| projected_gate | 6.0 | 6.0 | 1.0000 | 6.0 | 1.0000 | matched:8, first_group:2, opposite_reward_group:2 |
| local_promote | 6.0 | 6.0 | 1.0000 | 6.0 | 1.0000 | matched:8, first_group:2, opposite_reward_group:2 |
| trial_rollback | 6.0 | 6.0 | 1.0000 | 6.0 | 1.0000 | matched:8, first_group:2, opposite_reward_group:2 |

Reward identification is not an oracle: reward coefficients are inferred from the finite transition/reward evidence available in each profile. A profile with no observed terminal transitions may have weak or zero terminal-feature estimates after ridge regularization, which can look like `reward_mismatch` against a better-supported profile. That reason does not establish task or reward incompatibility. The measured group coverage is limited to these probe assignments; unresolved abstentions and mismatches mean routing coverage is not shown to be solved.

Gate accepts/rejects 是当前规则基于 cold-parent 与有限旧 anchor surrogate 的输出。没有独立 counterfactual 时，共享是否真实有益为 undetermined；因此 false accept 和 false reject 目前都不可测。Surrogate ESS/KL、优势估计及 episode-cluster SE 不是共享收益真值或安全保证。

## 资源与物理容量

| 条件 | Train steps / seed | Fit / Holdout | Probe steps / seed | Eval steps / seed | Actor peak B | Full checkpoint .pt peak B | W0 + checkpoint package peak B | CPU s | Eval CPU s | Wall s | Decision CPU s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| independent_fast | 20480 | 15360 / 5120 | 1536 | 8352 | 666 | 106552 | 162029 | 19.42 | 3.83 | 19.89 | 0.1250 |
| dynamics_shared | 20480 | 15360 / 5120 | 1536 | 8304 | 738 | 106776 | 162253 | 19.77 | 3.80 | 20.28 | 0.1250 |
| reward_shared | 20480 | 15360 / 5120 | 1536 | 8258 | 738 | 106808 | 162285 | 21.50 | 3.89 | 22.07 | 0.1719 |
| trial_gate | 20480 | 15360 / 5120 | 1536 | 8300 | 738 | 106744 | 162221 | 22.09 | 4.19 | 22.55 | 0.2188 |
| trial_gate_relaxed | 20480 | 15360 / 5120 | 1536 | 8218 | 738 | 106776 | 162253 | 22.95 | 4.48 | 23.37 | 0.2969 |
| projected_gate | 20480 | 15360 / 5120 | 1536 | 8286 | 738 | 106744 | 162221 | 23.27 | 4.40 | 23.98 | 0.3516 |
| local_promote | 20480 | 15360 / 5120 | 1536 | 8369 | 738 | 106776 | 162253 | 23.34 | 4.91 | 23.94 | 0.2109 |
| trial_rollback | 20480 | 15360 / 5120 | 1536 | 8306 | 738 | 106744 | 162221 | 23.10 | 4.44 | 23.71 | 0.1797 |

## 数据账与审计

Dataset SHA256：`493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2`。

| Seed | 条件 | 审计通过 | 失败项 |
|---:|---|---|---|
| 859 | independent_fast | 是 | — |
| 859 | dynamics_shared | 是 | — |
| 859 | reward_shared | 是 | — |
| 859 | trial_gate | 是 | — |
| 859 | trial_gate_relaxed | 是 | — |
| 859 | projected_gate | 是 | — |
| 859 | local_promote | 是 | — |
| 859 | trial_rollback | 是 | — |
| 863 | independent_fast | 是 | — |
| 863 | dynamics_shared | 是 | — |
| 863 | reward_shared | 是 | — |
| 863 | trial_gate | 是 | — |
| 863 | trial_gate_relaxed | 是 | — |
| 863 | projected_gate | 是 | — |
| 863 | local_promote | 是 | — |
| 863 | trial_rollback | 是 | — |

计费复核从原始 `steps.csv`、`probes.csv` 和 `evaluations.csv` 计数，并核对 summary 中的 training/probe/evaluation/total interactions、fit/holdout、decision CPU 及每个 checkpoint 与 actor sidecar 的磁盘字节。预算检查将 actor sidecar 与 actor cap 比较，并将 `full_persistent_bytes`（W0 foundation 加序列化状态）与 package cap 比较。来源快照与 W0/dataset manifest hash 会逐项验证。

## 解释边界

这些结果是本 cohort 上的描述性比较。两个或三个 seed 不足以支撑显著性表述。Whole-run arm 的对照可以比较平均结果，但单次 gate 的 false accept/reject 仍需要独立、预先定义的 counterfactual 结果才能判断。不得据此称该方法为 SOTA，也不能据此宣称已证明创新性。

## 图表

- online_outcomes: `online_outcomes.png` / `online_outcomes.pdf`
- evaluation_goal_rates: `evaluation_goal_rates.png` / `evaluation_goal_rates.pdf`
- costs: `costs.png` / `costs.pdf`
- capacity: `capacity.png` / `capacity.pdf`
