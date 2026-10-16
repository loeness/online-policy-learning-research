# 共享参数巩固保留探针：Stage 1

## 结果摘要

- cohort 状态：`completed`；分析角色：`exploratory`。
- 随机单位是 seed（3 个），不是 episode 或 transition。每个 seed 内先对 8 个 episode 求均值，再跨 seed 报告均值和最小值至最大值。
- 表中配对差值按同一 seed 计算，均为描述统计；不做显著性检验或因果主张。
- `random` 的共享矩阵 Frobenius 范数匹配 `weighted`，不匹配 `mean`。
- 方法含义：`local_only`：删除全部局部修正，仅剩 W0 + 零共享；这是本 assay 的基线；`mean`：有效参数的简单平均；`weighted`：按来源等权归一化的锚点最小二乘；`random`：随机共享矩阵；Frobenius 范数匹配 weighted。

| 目标 | 方法 | signed return：跨 seed 均值 [最小, 最大] | goal rate：跨 seed 均值 [最小, 最大] | positive success：跨 seed 均值 [最小, 最大]（A0−仅为定义性零值） |
|---|---|---:|---:|---:|
| A0+ | `local_only` | -0.800 [-1.139, -0.327] | 0.125 [0.000, 0.250] | 0.125 [0.000, 0.250] |
| A0+ | `mean` | -0.320 [-0.358, -0.280] | 0.167 [0.125, 0.250] | 0.167 [0.125, 0.250] |
| A0+ | `weighted` | -0.478 [-0.590, -0.372] | 0.208 [0.125, 0.250] | 0.208 [0.125, 0.250] |
| A0+ | `random` | -1.024 [-1.389, -0.826] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| A2+ | `local_only` | -0.568 [-2.014, 0.593] | 0.208 [0.000, 0.375] | 0.208 [0.000, 0.375] |
| A2+ | `mean` | -0.010 [-0.486, 0.391] | 0.375 [0.250, 0.500] | 0.375 [0.250, 0.500] |
| A2+ | `weighted` | -0.253 [-1.034, 0.243] | 0.250 [0.125, 0.375] | 0.250 [0.125, 0.375] |
| A2+ | `random` | -0.606 [-1.951, 1.207] | 0.208 [0.000, 0.500] | 0.208 [0.000, 0.500] |
| B2+ | `local_only` | -0.850 [-1.857, 0.513] | 0.167 [0.000, 0.375] | 0.167 [0.000, 0.375] |
| B2+ | `mean` | -1.737 [-2.326, -0.964] | 0.042 [0.000, 0.125] | 0.042 [0.000, 0.125] |
| B2+ | `weighted` | -1.595 [-2.264, -0.850] | 0.042 [0.000, 0.125] | 0.042 [0.000, 0.125] |
| B2+ | `random` | -1.207 [-2.076, 0.064] | 0.083 [0.000, 0.250] | 0.083 [0.000, 0.250] |
| A0- | `local_only` | 1.362 [0.839, 1.982] | 0.042 [0.000, 0.125] | 0.000 [0.000, 0.000] |
| A0- | `mean` | 0.478 [-0.070, 1.339] | 0.208 [0.125, 0.250] | 0.000 [0.000, 0.000] |
| A0- | `weighted` | 0.650 [0.160, 1.183] | 0.208 [0.125, 0.250] | 0.000 [0.000, 0.000] |
| A0- | `random` | 1.151 [0.770, 1.358] | 0.042 [0.000, 0.125] | 0.000 [0.000, 0.000] |

## 逐 seed 对照与目标解释

下表直接比较 Mean、Evidence weighted 各自相对 W0 only 的 seed 配对 signed-return 差值；“方向一致”表示两种共享方法在该 seed 都高于、都低于或都等于基线。

| 目标与 assay 角色 | Seed | Mean − W0 only | Evidence weighted − W0 only | 两方法方向一致 |
|---|---:|---:|---:|---|
| A0+（保留） | 1009 | +0.8150 | +0.7662 | 是 |
| A0+（保留） | 1013 | +0.0475 | -0.1438 | 否 |
| A0+（保留） | 1019 | +0.5775 | +0.3450 | 是 |
| A2+（探索迁移） | 1009 | +1.5275 | +0.9800 | 是 |
| A2+（探索迁移） | 1013 | +0.6725 | +0.5238 | 是 |
| A2+（探索迁移） | 1019 | -0.5288 | -0.5600 | 是 |
| B2+（错误复用诊断） | 1009 | -1.4763 | -1.3625 | 是 |
| B2+（错误复用诊断） | 1013 | -0.4687 | -0.4062 | 是 |
| B2+（错误复用诊断） | 1019 | -0.7150 | -0.4650 | 是 |
| A0-（负目标错误复用诊断（仅看 signed return）） | 1009 | -0.9087 | -0.6787 | 是 |
| A0-（负目标错误复用诊断（仅看 signed return）） | 1013 | -0.6437 | -0.8000 | 是 |
| A0-（负目标错误复用诊断（仅看 signed return）） | 1019 | -1.0975 | -0.6575 | 是 |

- **A0+ 保留：** Mean 相对 W0 only 在 3/3 seeds 为正；Evidence weighted 在 2/3 seeds 为正，负向 seed 为 1013。Evidence weighted 相对 Mean 在 A0+ 的 3/3 seeds signed return 较低。
- **A2+ 探索迁移：** Mean 与 Evidence weighted 相对 W0 only 在 3/3 seeds 方向一致；两者都为正的 seed 是 1009, 1013，都为负的是 1019。Evidence weighted 相对 Mean 在 3/3 seeds 较低。A2+ 是 exploratory validation，不是独立测试。
- **B2+ 错误复用诊断：** Mean 与 Evidence weighted 相对 W0 only 在 3/3 seeds 都为负。该目标呈现固定 parent 的错误复用结果，不参与一般算法选择。
- **A0− 错误复用诊断：** 这是负目标；goal 不是成功。positive_success_rate 在 A0− 没有解释意义，主要看 signed return；Mean 与 Evidence weighted 相对 W0 only 在 3/3 seeds 都为负。
- 在正奖励目标 A0+、A2+，Evidence weighted − Mean 的 signed-return 差值在 6/6 个 target-seed 配对中为负（A0+ 3/3、A2+ 3/3）。这些逐目标结果中 Mean 均高于 Evidence weighted；这是小样本描述，不作显著性主张。
- B2+ 与 A0-（负目标）是刻意设置的错误复用目标。把它们与正目标合成一个跨目标平均会把保留/迁移和诊断性反例混在一起，因此**跨目标平均不用于选择算法**。

## 同 seed 配对差值（完整数据）

下表保留每个目标、每种对比、每项指标的全部 seed 差值和跨 seed 均值；差值定义为左侧方法减右侧方法。A0− 的 positive-success 数值为任务定义下的零值，不代表成功度量。

### A0+ （保留）

| 对比 | 指标 | seed 1009 | seed 1013 | seed 1019 | 跨 seed 均值 |
|---|---|---:|---:|---:|---:|
| `mean_minus_local_only` | `mean_signed_return` | +0.815 | +0.047 | +0.577 | +0.480 |
| `mean_minus_local_only` | `mean_goal_rate` | +0.125 | +0.000 | +0.000 | +0.042 |
| `mean_minus_local_only` | `positive_success_rate` | +0.125 | +0.000 | +0.000 | +0.042 |
| `weighted_minus_local_only` | `mean_signed_return` | +0.766 | -0.144 | +0.345 | +0.322 |
| `weighted_minus_local_only` | `mean_goal_rate` | +0.125 | +0.000 | +0.125 | +0.083 |
| `weighted_minus_local_only` | `positive_success_rate` | +0.125 | +0.000 | +0.125 | +0.083 |
| `random_minus_local_only` | `mean_signed_return` | -0.250 | -0.499 | +0.078 | -0.224 |
| `random_minus_local_only` | `mean_goal_rate` | +0.000 | -0.250 | -0.125 | -0.125 |
| `random_minus_local_only` | `positive_success_rate` | +0.000 | -0.250 | -0.125 | -0.125 |
| `weighted_minus_random` | `mean_signed_return` | +1.016 | +0.355 | +0.267 | +0.546 |
| `weighted_minus_random` | `mean_goal_rate` | +0.125 | +0.250 | +0.250 | +0.208 |
| `weighted_minus_random` | `positive_success_rate` | +0.125 | +0.250 | +0.250 | +0.208 |
| `weighted_minus_mean` | `mean_signed_return` | -0.049 | -0.191 | -0.232 | -0.157 |
| `weighted_minus_mean` | `mean_goal_rate` | +0.000 | +0.000 | +0.125 | +0.042 |
| `weighted_minus_mean` | `positive_success_rate` | +0.000 | +0.000 | +0.125 | +0.042 |

### A2+ （探索迁移）

| 对比 | 指标 | seed 1009 | seed 1013 | seed 1019 | 跨 seed 均值 |
|---|---|---:|---:|---:|---:|
| `mean_minus_local_only` | `mean_signed_return` | +1.528 | +0.673 | -0.529 | +0.557 |
| `mean_minus_local_only` | `mean_goal_rate` | +0.375 | +0.250 | -0.125 | +0.167 |
| `mean_minus_local_only` | `positive_success_rate` | +0.375 | +0.250 | -0.125 | +0.167 |
| `weighted_minus_local_only` | `mean_signed_return` | +0.980 | +0.524 | -0.560 | +0.315 |
| `weighted_minus_local_only` | `mean_goal_rate` | +0.125 | +0.125 | -0.125 | +0.042 |
| `weighted_minus_local_only` | `positive_success_rate` | +0.125 | +0.125 | -0.125 | +0.042 |
| `random_minus_local_only` | `mean_signed_return` | +0.063 | +1.489 | -1.667 | -0.039 |
| `random_minus_local_only` | `mean_goal_rate` | +0.000 | +0.250 | -0.250 | +0.000 |
| `random_minus_local_only` | `positive_success_rate` | +0.000 | +0.250 | -0.250 | +0.000 |
| `weighted_minus_random` | `mean_signed_return` | +0.917 | -0.965 | +1.107 | +0.353 |
| `weighted_minus_random` | `mean_goal_rate` | +0.125 | -0.125 | +0.125 | +0.042 |
| `weighted_minus_random` | `positive_success_rate` | +0.125 | -0.125 | +0.125 | +0.042 |
| `weighted_minus_mean` | `mean_signed_return` | -0.548 | -0.149 | -0.031 | -0.242 |
| `weighted_minus_mean` | `mean_goal_rate` | -0.250 | -0.125 | +0.000 | -0.125 |
| `weighted_minus_mean` | `positive_success_rate` | -0.250 | -0.125 | +0.000 | -0.125 |

### B2+ （错误复用诊断）

| 对比 | 指标 | seed 1009 | seed 1013 | seed 1019 | 跨 seed 均值 |
|---|---|---:|---:|---:|---:|
| `mean_minus_local_only` | `mean_signed_return` | -1.476 | -0.469 | -0.715 | -0.887 |
| `mean_minus_local_only` | `mean_goal_rate` | -0.250 | +0.000 | -0.125 | -0.125 |
| `mean_minus_local_only` | `positive_success_rate` | -0.250 | +0.000 | -0.125 | -0.125 |
| `weighted_minus_local_only` | `mean_signed_return` | -1.362 | -0.406 | -0.465 | -0.745 |
| `weighted_minus_local_only` | `mean_goal_rate` | -0.250 | +0.000 | -0.125 | -0.125 |
| `weighted_minus_local_only` | `positive_success_rate` | -0.250 | +0.000 | -0.125 | -0.125 |
| `random_minus_local_only` | `mean_signed_return` | -0.449 | -0.219 | -0.403 | -0.357 |
| `random_minus_local_only` | `mean_goal_rate` | -0.125 | +0.000 | -0.125 | -0.083 |
| `random_minus_local_only` | `positive_success_rate` | -0.125 | +0.000 | -0.125 | -0.083 |
| `weighted_minus_random` | `mean_signed_return` | -0.914 | -0.187 | -0.062 | -0.388 |
| `weighted_minus_random` | `mean_goal_rate` | -0.125 | +0.000 | +0.000 | -0.042 |
| `weighted_minus_random` | `positive_success_rate` | -0.125 | +0.000 | +0.000 | -0.042 |
| `weighted_minus_mean` | `mean_signed_return` | +0.114 | +0.063 | +0.250 | +0.142 |
| `weighted_minus_mean` | `mean_goal_rate` | +0.000 | +0.000 | +0.000 | +0.000 |
| `weighted_minus_mean` | `positive_success_rate` | +0.000 | +0.000 | +0.000 | +0.000 |

### A0- （负目标；主要看 signed return）

| 对比 | 指标 | seed 1009 | seed 1013 | seed 1019 | 跨 seed 均值 |
|---|---|---:|---:|---:|---:|
| `mean_minus_local_only` | `mean_signed_return` | -0.909 | -0.644 | -1.098 | -0.883 |
| `mean_minus_local_only` | `mean_goal_rate` | +0.125 | +0.125 | +0.250 | +0.167 |
| `mean_minus_local_only` | `positive_success_rate` | +0.000 | +0.000 | +0.000 | +0.000 |
| `weighted_minus_local_only` | `mean_signed_return` | -0.679 | -0.800 | -0.657 | -0.712 |
| `weighted_minus_local_only` | `mean_goal_rate` | +0.125 | +0.125 | +0.250 | +0.167 |
| `weighted_minus_local_only` | `positive_success_rate` | +0.000 | +0.000 | +0.000 | +0.000 |
| `random_minus_local_only` | `mean_signed_return` | -0.069 | -0.625 | +0.062 | -0.210 |
| `random_minus_local_only` | `mean_goal_rate` | +0.000 | +0.000 | +0.000 | +0.000 |
| `random_minus_local_only` | `positive_success_rate` | +0.000 | +0.000 | +0.000 | +0.000 |
| `weighted_minus_random` | `mean_signed_return` | -0.610 | -0.175 | -0.720 | -0.502 |
| `weighted_minus_random` | `mean_goal_rate` | +0.125 | +0.125 | +0.250 | +0.167 |
| `weighted_minus_random` | `positive_success_rate` | +0.000 | +0.000 | +0.000 | +0.000 |
| `weighted_minus_mean` | `mean_signed_return` | +0.230 | -0.156 | +0.440 | +0.171 |
| `weighted_minus_mean` | `mean_goal_rate` | +0.000 | +0.000 | +0.000 | +0.000 |
| `weighted_minus_mean` | `positive_success_rate` | +0.000 | +0.000 | +0.000 | +0.000 |

### 附属描述：跨目标算术平均（不用于选择算法）

将全部四个目标 × 三个 seed 配对后，weighted − Mean signed-return 算术平均为 -0.022，weighted − Random 为 +0.003。这些数字仅供完整性记录；B2+、A0− 是刻意设置的错误复用诊断，不能用这个合并值判定哪种方法总体更优。Random 的矩阵范数匹配 weighted，不匹配 Mean。


## 审计、成本与容量

summary audit：**通过**；purge audit 失败分支：0 / 12；eval 冻结失败分支：0；source unchanged 失败分支：0。
source audit 条目：3。最大 invariance 误差：`{"int8_logit_max_error":0.11130046844482422,"int8_mean_KL":2.3047230988367104e-05,"raw_logit_max_error":1.9073486328125e-06,"raw_probability_max_error":2.384185791015625e-07}`。

| Seed | 方法 | purge audit | purge checks |
|---:|---|---|---|
| 1009 | `local_only` | 通过 | `{"all_actor_gradients_disabled":true,"all_anchor_buffers_empty":true,"all_profiles_unseen":true,"critic_is_fresh_W0":true,"every_local_zero":true,"fixed_fresh_route":true,"group_statistics_empty":true,"movement_profiles_zero":true,"no_online_rollout_or_history_payload":true,"online_counts_zero":true,"optimizer_empty":true,"other_parents_zero":true,"reward_profiles_zero":true}` |
| 1009 | `mean` | 通过 | `{"all_actor_gradients_disabled":true,"all_anchor_buffers_empty":true,"all_profiles_unseen":true,"critic_is_fresh_W0":true,"every_local_zero":true,"fixed_fresh_route":true,"group_statistics_empty":true,"movement_profiles_zero":true,"no_online_rollout_or_history_payload":true,"online_counts_zero":true,"optimizer_empty":true,"other_parents_zero":true,"reward_profiles_zero":true}` |
| 1009 | `weighted` | 通过 | `{"all_actor_gradients_disabled":true,"all_anchor_buffers_empty":true,"all_profiles_unseen":true,"critic_is_fresh_W0":true,"every_local_zero":true,"fixed_fresh_route":true,"group_statistics_empty":true,"movement_profiles_zero":true,"no_online_rollout_or_history_payload":true,"online_counts_zero":true,"optimizer_empty":true,"other_parents_zero":true,"reward_profiles_zero":true}` |
| 1009 | `random` | 通过 | `{"all_actor_gradients_disabled":true,"all_anchor_buffers_empty":true,"all_profiles_unseen":true,"critic_is_fresh_W0":true,"every_local_zero":true,"fixed_fresh_route":true,"group_statistics_empty":true,"movement_profiles_zero":true,"no_online_rollout_or_history_payload":true,"online_counts_zero":true,"optimizer_empty":true,"other_parents_zero":true,"reward_profiles_zero":true}` |
| 1013 | `local_only` | 通过 | `{"all_actor_gradients_disabled":true,"all_anchor_buffers_empty":true,"all_profiles_unseen":true,"critic_is_fresh_W0":true,"every_local_zero":true,"fixed_fresh_route":true,"group_statistics_empty":true,"movement_profiles_zero":true,"no_online_rollout_or_history_payload":true,"online_counts_zero":true,"optimizer_empty":true,"other_parents_zero":true,"reward_profiles_zero":true}` |
| 1013 | `mean` | 通过 | `{"all_actor_gradients_disabled":true,"all_anchor_buffers_empty":true,"all_profiles_unseen":true,"critic_is_fresh_W0":true,"every_local_zero":true,"fixed_fresh_route":true,"group_statistics_empty":true,"movement_profiles_zero":true,"no_online_rollout_or_history_payload":true,"online_counts_zero":true,"optimizer_empty":true,"other_parents_zero":true,"reward_profiles_zero":true}` |
| 1013 | `weighted` | 通过 | `{"all_actor_gradients_disabled":true,"all_anchor_buffers_empty":true,"all_profiles_unseen":true,"critic_is_fresh_W0":true,"every_local_zero":true,"fixed_fresh_route":true,"group_statistics_empty":true,"movement_profiles_zero":true,"no_online_rollout_or_history_payload":true,"online_counts_zero":true,"optimizer_empty":true,"other_parents_zero":true,"reward_profiles_zero":true}` |
| 1013 | `random` | 通过 | `{"all_actor_gradients_disabled":true,"all_anchor_buffers_empty":true,"all_profiles_unseen":true,"critic_is_fresh_W0":true,"every_local_zero":true,"fixed_fresh_route":true,"group_statistics_empty":true,"movement_profiles_zero":true,"no_online_rollout_or_history_payload":true,"online_counts_zero":true,"optimizer_empty":true,"other_parents_zero":true,"reward_profiles_zero":true}` |
| 1019 | `local_only` | 通过 | `{"all_actor_gradients_disabled":true,"all_anchor_buffers_empty":true,"all_profiles_unseen":true,"critic_is_fresh_W0":true,"every_local_zero":true,"fixed_fresh_route":true,"group_statistics_empty":true,"movement_profiles_zero":true,"no_online_rollout_or_history_payload":true,"online_counts_zero":true,"optimizer_empty":true,"other_parents_zero":true,"reward_profiles_zero":true}` |
| 1019 | `mean` | 通过 | `{"all_actor_gradients_disabled":true,"all_anchor_buffers_empty":true,"all_profiles_unseen":true,"critic_is_fresh_W0":true,"every_local_zero":true,"fixed_fresh_route":true,"group_statistics_empty":true,"movement_profiles_zero":true,"no_online_rollout_or_history_payload":true,"online_counts_zero":true,"optimizer_empty":true,"other_parents_zero":true,"reward_profiles_zero":true}` |
| 1019 | `weighted` | 通过 | `{"all_actor_gradients_disabled":true,"all_anchor_buffers_empty":true,"all_profiles_unseen":true,"critic_is_fresh_W0":true,"every_local_zero":true,"fixed_fresh_route":true,"group_statistics_empty":true,"movement_profiles_zero":true,"no_online_rollout_or_history_payload":true,"online_counts_zero":true,"optimizer_empty":true,"other_parents_zero":true,"reward_profiles_zero":true}` |
| 1019 | `random` | 通过 | `{"all_actor_gradients_disabled":true,"all_anchor_buffers_empty":true,"all_profiles_unseen":true,"critic_is_fresh_W0":true,"every_local_zero":true,"fixed_fresh_route":true,"group_statistics_empty":true,"movement_profiles_zero":true,"no_online_rollout_or_history_payload":true,"online_counts_zero":true,"optimizer_empty":true,"other_parents_zero":true,"reward_profiles_zero":true}` |

source audit entries:
- `{"foundation_sha256":"e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222","seed":1009,"shared_max_abs":0.0,"source":"outputs\\write_decision_causal_control\\seed_1009\\trial_rollback\\checkpoints\\checkpoint_step_00032768.pt","source_anchor_evidence":[{"episodes":18,"generation":1,"rows":24,"slot":1,"task":"A0+"},{"episodes":17,"generation":1,"rows":24,"slot":0,"task":"A1+"}],"source_group":0,"source_sha256":"e70439ddc0f40fb140ffced134cbfd4c62504e4b8b4a3f5bce29a60099b2569f","source_slots":[1,0],"steps_by_task":{"A0+":8192,"A0-":8192,"A1+":8192,"A1-":8192,"A2+":0,"A2-":0},"unseen_target_training_steps":0}`
- `{"foundation_sha256":"e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222","seed":1013,"shared_max_abs":0.0,"source":"outputs\\write_decision_causal_control\\seed_1013\\trial_rollback\\checkpoints\\checkpoint_step_00032768.pt","source_anchor_evidence":[{"episodes":19,"generation":1,"rows":24,"slot":5,"task":"A0+"},{"episodes":21,"generation":1,"rows":24,"slot":3,"task":"A1+"}],"source_group":0,"source_sha256":"1f4287a6a0ded76959df5fa46b476e1e14e09b7a9ffa706fc98b6c68a41016b1","source_slots":[5,3],"steps_by_task":{"A0+":8192,"A0-":8192,"A1+":8192,"A1-":8192,"A2+":0,"A2-":0},"unseen_target_training_steps":0}`
- `{"foundation_sha256":"e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222","seed":1019,"shared_max_abs":0.0,"source":"outputs\\write_decision_causal_control\\seed_1019\\trial_rollback\\checkpoints\\checkpoint_step_00032768.pt","source_anchor_evidence":[{"episodes":18,"generation":1,"rows":24,"slot":4,"task":"A0+"},{"episodes":16,"generation":1,"rows":24,"slot":3,"task":"A1+"}],"source_group":0,"source_sha256":"32abff3779a042e40a5b03db028f928ec397985019aa0c831fff597b6236f0f6","source_slots":[4,3],"steps_by_task":{"A0+":8192,"A0-":8192,"A1+":8192,"A1-":8192,"A2+":0,"A2-":0},"unseen_target_training_steps":0}`

复用 source training steps：98304；新增 training steps：0；新增 evaluation steps：22746；新增总交互：22746。evaluation observer clone 的成本计入 cohort；本报告不启动新的 PPO/training。

下表保留每个分支实际记录的字节数。

| Seed | 方法 | Actor bytes | Checkpoint bytes | Foundation bytes | Full persistent bytes | Package budget bytes |
|---:|---|---:|---:|---:|---:|---:|
| 1009 | `local_only` | 230 | 58193 | 55477 | 113670 | 262144 |
| 1009 | `mean` | 230 | 58193 | 55477 | 113670 | 262144 |
| 1009 | `weighted` | 230 | 58193 | 55477 | 113670 | 262144 |
| 1009 | `random` | 230 | 58193 | 55477 | 113670 | 262144 |
| 1013 | `local_only` | 230 | 58193 | 55477 | 113670 | 262144 |
| 1013 | `mean` | 230 | 58193 | 55477 | 113670 | 262144 |
| 1013 | `weighted` | 230 | 58193 | 55477 | 113670 | 262144 |
| 1013 | `random` | 230 | 58193 | 55477 | 113670 | 262144 |
| 1019 | `local_only` | 230 | 58193 | 55477 | 113670 | 262144 |
| 1019 | `mean` | 230 | 58193 | 55477 | 113670 | 262144 |
| 1019 | `weighted` | 230 | 58193 | 55477 | 113670 | 262144 |
| 1019 | `random` | 230 | 58193 | 55477 | 113670 | 262144 |

## 解释边界

- 这是 3 seeds × 每个目标每 seed 8 个 episode 的小样本 exploratory 结果；seed 是聚合与配对单位。
- Source checkpoint 看过 A0/A1。A2 source 未见过，但 A2+ 用作 exploratory validation，因此结果不是 selection 后的独立测试。
- 四个目标/方法都被强制使用同一个存储 parent 和新鲜执行路径。B2+ 与 A0- 用于检验错误复用，不证明自动路由或安全性。
- prototype full-shared 6+2 int8 与 W0 only 同为 230 B，且使用相同容器；这是机制消融，不是容量优越证明。实际各分支字节数见上表。
- 本报告给出逐 seed 结果、审计与实测成本；不据此作正式新颖性或显著性结论。
