# Retention Stream Stage 2 Diagnostics

## Scope and accounting

The corrected main cohort completed all 30 branches: 3 seeds × 2 parameter budgets × 5 methods. Each branch consumed and submitted 24,576 raw steps to PPO, for 737,280 raw interactions and no evaluation interactions. The cohort summary reports no failed branches.

The 1,024-step pilot v2 completed 5 low-budget branches at 12,288 steps each (61,440 raw interactions, no evaluation interactions). Pilot v1 used 512-step bouts and consumed 30,720 raw interactions; it proposed writes but accepted none. Pilot v1 is retained in the accounting but excluded from the interpretation below because it predates the corrected run and used the shorter pilot schedule. Total interaction accounting across main and both pilots is 829,440 raw steps, with no new evaluation steps.

## Main cohort outcomes

Reward per raw step is the observed training-stream reward, not a held-out evaluation. Values below are the mean over the 3 deployment seeds; the seed-level values are included to show the limited sample.

| Budget | Method | Seed values | Mean |
|---|---|---:|---:|
| Low | independent | -0.011618, -0.013032, -0.008078 | -0.010910 |
| Low | direct_shared | -0.002046, 0.000396, -0.003236 | -0.001629 |
| Low | mean | -0.004742, -0.006969, -0.005423 | -0.005711 |
| Low | mean_reuse_gate | -0.010214, -0.010286, -0.007356 | -0.009285 |
| Low | retention_gate | -0.013999, -0.015372, -0.006105 | -0.011825 |
| Wide | independent | -0.008678, -0.003460, -0.005372 | -0.005837 |
| Wide | direct_shared | 0.006051, 0.014983, 0.002908 | 0.007981 |
| Wide | mean | -0.002137, -0.001608, -0.001354 | -0.001700 |
| Wide | mean_reuse_gate | -0.005566, -0.004742, -0.002666 | -0.004325 |
| Wide | retention_gate | -0.002158, -0.006664, -0.001242 | -0.003355 |

Natural local eviction occurred only in the low-budget layout. The table below pools recovery metrics over recurrences explicitly marked as previously evicted; each arm has 12 such recurrences for independent and 17 for each shared arm. These are descriptive per-arm sets, not matched tasks or equally weighted seed means. Use the [matched report](../outputs/retention_stream_stage2_report_v2/report.md) for method comparisons. The wide layout had none.

| Low-budget method | Evicted recurrences | First 256 steps | First 512 steps | First 1,024 steps |
|---|---:|---:|---:|---:|
| independent | 12 | -0.017712 | -0.021496 | -0.017671 |
| direct_shared | 17 | -0.016673 | -0.007941 | -0.001249 |
| mean | 17 | -0.024715 | -0.014059 | -0.009722 |
| mean_reuse_gate | 17 | -0.021613 | -0.020723 | -0.014590 |
| retention_gate | 17 | -0.021383 | -0.018253 | -0.016084 |

### Prior-use ablation

`mean_reuse_gate` leaves the write decision unconditional, matching `mean`; it gates only additive prior use. Proposal and accepted-write counts match between these arms at every seed and budget. In the low budget, each arm proposed 24, 30, and 24 candidates by seed and accepted 17, 25, and 19. In the wide budget, each proposed 26, 32, and 30 and accepted 17, 26, and 23.

The observed reward-per-step difference (`mean_reuse_gate` minus `mean`) was negative for every seed: -0.005472, -0.003317, and -0.001933 in the low budget (mean -0.003574), and -0.003429, -0.003134, and -0.001312 in the wide budget (mean -0.002625). The filtered arm also performed fewer prior injections: 21 vs. 29 in low and 7 vs. 12 in wide. This is negative evidence for the current prior-use rule's net utility in these observed streams. Three deployment seeds do not establish its expected effect over other task distributions or final held-out policy performance.

| Budget | Method | Prior attempts | Applied | Pre-update score lower ≥ 0 |
|---|---|---:|---:|---:|
| Low | mean | 34 | 29 | 19 |
| Low | mean_reuse_gate | 56 | 21 | 25 |
| Low | retention_gate | 10 | 6 | 6 |
| Low | direct_shared | 35 | 35 | 20 |
| Wide | mean | 12 | 12 | 4 |
| Wide | mean_reuse_gate | 20 | 7 | 7 |
| Wide | retention_gate | 1 | 1 | 1 |
| Wide | direct_shared | 18 | 18 | 10 |

In low-budget `mean_reuse_gate`, 25 of 56 prior scores cleared the lower-score threshold, but 4 were invalidated because an accepted write changed the parent version in the same rollout; 21 injections were applied. In wide, all 7 threshold-passing priors were applied. `mean` and `direct_shared` apply priors regardless of the score, subject to the parent-version check. Direct-shared prior events use a pre-PPO shared-matrix snapshot; its `group_versions` counter tracks consolidation writes, not shared PPO gradients, so those version labels do not certify that the gradient-updated parent stayed numerically unchanged.

The accepted injection's measured mean and maximum action-probability changes on the current rollout were:

| Budget | Method | Mean max-probability change | Largest max-probability change | Mean KL | Largest KL |
|---|---|---:|---:|---:|---:|
| Low | mean | 0.284599 | 0.529541 | 0.139176 | 0.464032 |
| Low | mean_reuse_gate | 0.232626 | 0.372071 | 0.071728 | 0.235180 |
| Low | retention_gate | 0.266291 | 0.309745 | 0.105235 | 0.164023 |
| Low | direct_shared | 0.237349 | 0.472495 | 0.099055 | 0.281433 |
| Wide | mean | 0.199772 | 0.377006 | 0.045472 | 0.136496 |
| Wide | mean_reuse_gate | 0.179174 | 0.305653 | 0.033638 | 0.058487 |
| Wide | retention_gate | 0.213971 | 0.213971 | 0.065377 | 0.065377 |
| Wide | direct_shared | 0.121562 | 0.296790 | 0.018794 | 0.084814 |

These are on-rollout policy-distribution changes caused by an applied additive matrix, not realized reward effects.

## Retention write gate

The retention gate tested 61 low-budget candidates and 66 wide-budget candidates. It accepted 4 and 3, respectively. All 127 fresh scores were supported; all 456 historical prototype scores were supported. The 120 rejections were all `mean_lower_not_above_threshold`; none failed because evidence support was unknown. `mean` and `mean_reuse_gate` accepted all tested writes by design, so this comparison tests the additional retention rule rather than a write rule shared with those arms.

| Budget | Historical scores | Mean gain | Mean cluster SE | Mean lower score | Lower-score range | Accepted tests |
|---|---:|---:|---:|---:|---:|---:|
| Low | 216 | 0.002309 | 0.020958 | -0.018648 | -0.108289 to 0.040379 | 4 / 61 |
| Wide | 240 | -0.001232 | 0.035382 | -0.036614 | -0.290313 to 0.062303 | 3 / 66 |

The accepted tests had historical mean lower scores close to zero: 0.007261, 0.001663, 0.000172, and 0.000272 in low, and 0.001663, 0.000671, and 0.000035 in wide. They cleared the configured per-profile loss limit as well as the mean threshold. These are empirical, clipped, behavior-weighted surrogate diagnostics with episode-cluster errors; they are not safety bounds or strict off-policy estimates.

Relative to `mean_reuse_gate`, `retention_gate` changes the write-decision rule while keeping the prior-use rule fixed. It reduced accepted writes from 61 to 4 in low and 66 to 3 in wide, and subsequent prior injections from 21 to 6 and 7 to 1. Changed parent availability and trajectories are downstream effects of this treatment, not separate randomized changes. Thus the comparison estimates the total algorithmic effect of adding the write gate in these streams; it does not isolate a direct effect holding future parent availability and trajectories fixed. Reward per raw step was lower by 0.002540 on average in low, but higher by 0.000970 in wide. This inconsistent direction does not support stable superiority of the new rule.

## Prototype and group provenance

To examine dependence among historical scores, the six main `retention_gate` runs were replayed from their logged transitions. Reconstructed archive arrays and source-slot/generation metadata exactly matched each final checkpoint. At each write-test event, the replay captured the archive state before the current rollout was observed.

| Budget | Gate tests | Tests with overlapping source generations across prototypes | Mean prototypes scored | Mean unique source generations |
|---|---:|---:|---:|---:|
| Low | 61 | 40 | 3.541 | 3.098 |
| Wide | 66 | 50 | 3.636 | 2.530 |
| Total | 127 | 90 | 3.591 | 2.803 |

Across the 127 tests, there were 172 prototype pairs whose reservoirs included rows from the same `(source_slot, source_generation)`. Every one of the 7 accepted tests had at least one such overlapping pair. The gate counts numeric prototypes, not independent task contexts: repeated windows from one local generation can form multiple movement-conflict cells and receive separate weight in the prototype mean. This overlap does not imply that the reservoirs contain identical rows, but it does rule out treating prototype scores as independent context replications.

The archive remained within its 8-cell, 24-row-per-cell bounds. For the retention arm, low-budget updates comprised 243 compatible matches, 21 LRU cell replacements, 21 first/new cell creations, and 3 insufficient-support cells; wide comprised 251 compatible matches, 13 LRU replacements, 21 first/new cell creations, and 3 insufficient-support cells. These cell replacements are distinct from local-slot eviction; archive evidence survived local-generation changes.

Fresh group assignments were stable for most rollouts. Across 288 updates per budget, the retention arm had 2 abstaining assignments in low and 3 in wide, with 4 and 6 assignment changes within a resident slot generation. The archive and group assignment use visible profile evidence; task names here are used only to interpret observer-side provenance logs.

## Critic and codec diagnostics

Each branch recorded pre-update critic return MSE and current-rollout probability drift after int8 actor roundtrip. The critic is shared, ignores route identity, and its learned state is not a separate treatment.

| Budget | Method | Mean critic MSE | Max critic MSE | Mean codec max-probability drift | Max codec drift |
|---|---|---:|---:|---:|---:|
| Low | independent | 0.286646 | 1.065571 | 0.0013519 | 0.0051816 |
| Low | direct_shared | 0.348472 | 1.341625 | 0.0014266 | 0.0044139 |
| Low | mean | 0.312366 | 1.237834 | 0.0010159 | 0.0045740 |
| Low | mean_reuse_gate | 0.289801 | 1.257069 | 0.0011201 | 0.0045442 |
| Low | retention_gate | 0.269292 | 1.406459 | 0.0009885 | 0.0030984 |
| Wide | independent | 0.303799 | 1.372604 | 0.0019521 | 0.0064245 |
| Wide | direct_shared | 0.373760 | 2.203283 | 0.0016820 | 0.0057035 |
| Wide | mean | 0.321868 | 1.124935 | 0.0013399 | 0.0043896 |
| Wide | mean_reuse_gate | 0.309962 | 1.202956 | 0.0013555 | 0.0041562 |
| Wide | retention_gate | 0.313642 | 1.482237 | 0.0014968 | 0.0060896 |

These are diagnostics, not a reward attribution. The largest recorded per-rollout codec probability drift was 0.006425. Checkpoints restored the critic and bounded learner state, but the run audit is a final deserialization check, not a mid-stream simulator-resume test.

## Interpretation limits

- The main cohort has 3 seeds per budget and no fresh evaluation interactions. The observed stream rewards and recurrence windows are training outcomes under the online policy.
- The historical archive rows may have helped train source locals and form the candidates. The later fresh rollout is held out from candidate formation, but historical prototype scores are not independent generalization evidence.
- Several accepted write tests counted multiple prototypes containing rows from the same local context generation. The 7 acceptances therefore do not represent 7 independent context replications.
- Mean versus mean-reuse isolates the configured prior-use rule; candidate values and state trajectories can subsequently diverge. Retention versus mean-reuse isolates the configured write rule. Parent availability and later reuse are mediators of that rule's total effect, so neither comparison fixes every downstream policy state. The small seed count limits generalization of the observed treatment differences.
- `direct_shared` trains its shared parameters through PPO and also applies an ungated pre-PPO shared-matrix snapshot as an additive prior. Its reported group version does not track those gradient updates.
- The critic is shared and route-agnostic. Critic MSE, surrogate lower scores, codec drift, and write decisions do not establish true task harm or benefit.
