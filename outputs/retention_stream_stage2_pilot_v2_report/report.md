# Retention stream exploratory analysis

Analysis role: `exploratory`. Statistical unit: seed. Each seed is summarized before cross-seed means, ranges, or paired differences are calculated. Declared budgets are reported separately: low.

The full-stream outcomes are total reward and reward per raw step. Recurrence recovery is classified within each seed and is counted as eviction recovery only when the task was naturally evicted before it returned. Resident revisits and revisits with zero evictions remain separate categories. Tasks and bouts are not independent replicates.

Profile evidence comes from normal rollouts; there is no independent probe, and initial group identification uses a rollout. The report does not assume an early cold-start gain. The primary recurrence comparisons are the later 512- and 1,024-step windows.

Cohort audit passed: `True`. Branches: 5; seeds: 1; cohort raw steps: 61440; evaluation steps: 0; total interactions: 61440; CPU s: 70.88; wall s: 45.60.

## Full-stream outcomes by budget

### Low budget

| method | seeds | total reward mean [range] | reward / raw step mean [range] |
|---|---:|---:|---:|
| independent | 1 | -197.6400 [-197.6400, -197.6400] | -0.0161 [-0.0161, -0.0161] |
| direct_shared | 1 | -194.1400 [-194.1400, -194.1400] | -0.0158 [-0.0158, -0.0158] |
| mean | 1 | -196.6400 [-196.6400, -196.6400] | -0.0160 [-0.0160, -0.0160] |
| mean_reuse_gate | 1 | -174.6400 [-174.6400, -174.6400] | -0.0142 [-0.0142, -0.0142] |
| retention_gate | 1 | -188.8900 [-188.8900, -188.8900] | -0.0154 [-0.0154, -0.0154] |

#### Low budget paired differences vs independent

| method | metric | paired seeds | per-seed differences | mean difference [range] |
|---|---|---|---|---:|
| independent | total_reward | 18997 | 18997: 0.0000 | 0.0000 [0.0000, 0.0000] |
| independent | reward_per_raw_step | 18997 | 18997: 0.0000 | 0.0000 [0.0000, 0.0000] |
| direct_shared | total_reward | 18997 | 18997: 3.5000 | 3.5000 [3.5000, 3.5000] |
| direct_shared | reward_per_raw_step | 18997 | 18997: 0.0003 | 0.0003 [0.0003, 0.0003] |
| mean | total_reward | 18997 | 18997: 1.0000 | 1.0000 [1.0000, 1.0000] |
| mean | reward_per_raw_step | 18997 | 18997: 0.0001 | 0.0001 [0.0001, 0.0001] |
| mean_reuse_gate | total_reward | 18997 | 18997: 23.0000 | 23.0000 [23.0000, 23.0000] |
| mean_reuse_gate | reward_per_raw_step | 18997 | 18997: 0.0019 | 0.0019 [0.0019, 0.0019] |
| retention_gate | total_reward | 18997 | 18997: 8.7500 | 8.7500 [8.7500, 8.7500] |
| retention_gate | reward_per_raw_step | 18997 | 18997: 0.0007 | 0.0007 [0.0007, 0.0007] |

## Recurrence categories

### Low budget

Each task is averaged inside its seed first. The reported cross-seed mean and range use those seed-level averages.

| method | category | seed count | task count | first 256 mean [range] | recovery 512 mean [range] | recovery 1,024 mean [range] | first goal step mean [range] (n, missing) | first gradient step mean [range] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| independent | evicted_recovery | 1 | 4 | -0.0134 [-0.0134, -0.0134] | -0.0096 [-0.0096, -0.0096] | -0.0143 [-0.0143, -0.0143] | 256.8 [256.8, 256.8] (n=1, missing=0) | 256.0 [256.0, 256.0] |
| independent | resident_revisit | 1 | 2 | -0.0188 [-0.0188, -0.0188] | -0.0119 [-0.0119, -0.0119] | -0.0084 [-0.0084, -0.0084] | 231.0 [231.0, 231.0] (n=1, missing=0) | 256.0 [256.0, 256.0] |
| independent | no_eviction_revisit | 0 | 0 | n/a | n/a | n/a | n/a (n=0, missing=0) | n/a |
| direct_shared | evicted_recovery | 1 | 4 | -0.0202 [-0.0202, -0.0202] | -0.0119 [-0.0119, -0.0119] | -0.0150 [-0.0150, -0.0150] | 252.0 [252.0, 252.0] (n=1, missing=0) | 256.0 [256.0, 256.0] |
| direct_shared | resident_revisit | 1 | 2 | -0.0114 [-0.0114, -0.0114] | -0.0058 [-0.0058, -0.0058] | -0.0012 [-0.0012, -0.0012] | 245.0 [245.0, 245.0] (n=1, missing=0) | 256.0 [256.0, 256.0] |
| direct_shared | no_eviction_revisit | 0 | 0 | n/a | n/a | n/a | n/a (n=0, missing=0) | n/a |
| mean | evicted_recovery | 1 | 4 | -0.0183 [-0.0183, -0.0183] | -0.0178 [-0.0178, -0.0178] | -0.0141 [-0.0141, -0.0141] | 360.3 [360.3, 360.3] (n=1, missing=0) | 256.0 [256.0, 256.0] |
| mean | resident_revisit | 1 | 2 | -0.0217 [-0.0217, -0.0217] | -0.0168 [-0.0168, -0.0168] | -0.0053 [-0.0053, -0.0053] | 720.5 [720.5, 720.5] (n=1, missing=0) | 256.0 [256.0, 256.0] |
| mean | no_eviction_revisit | 0 | 0 | n/a | n/a | n/a | n/a (n=0, missing=0) | n/a |
| mean_reuse_gate | evicted_recovery | 1 | 4 | -0.0185 [-0.0185, -0.0185] | -0.0127 [-0.0127, -0.0127] | -0.0143 [-0.0143, -0.0143] | 482.8 [482.8, 482.8] (n=1, missing=0) | 256.0 [256.0, 256.0] |
| mean_reuse_gate | resident_revisit | 1 | 2 | -0.0110 [-0.0110, -0.0110] | -0.0068 [-0.0068, -0.0068] | 0.0016 [0.0016, 0.0016] | 263.0 [263.0, 263.0] (n=1, missing=0) | 256.0 [256.0, 256.0] |
| mean_reuse_gate | no_eviction_revisit | 0 | 0 | n/a | n/a | n/a | n/a (n=0, missing=0) | n/a |
| retention_gate | evicted_recovery | 1 | 4 | -0.0173 [-0.0173, -0.0173] | -0.0124 [-0.0124, -0.0124] | -0.0144 [-0.0144, -0.0144] | 587.0 [587.0, 587.0] (n=1, missing=0) | 256.0 [256.0, 256.0] |
| retention_gate | resident_revisit | 1 | 2 | -0.0188 [-0.0188, -0.0188] | -0.0061 [-0.0061, -0.0061] | -0.0041 [-0.0041, -0.0041] | 231.0 [231.0, 231.0] (n=1, missing=0) | 256.0 [256.0, 256.0] |
| retention_gate | no_eviction_revisit | 0 | 0 | n/a | n/a | n/a | n/a (n=0, missing=0) | n/a |

#### Low budget paired eviction-recovery differences vs independent

Only the same seed and task/bout observed as previously evicted in both arms is included. Task/bout differences are averaged within seed before cross-seed summaries.

| method | recovery window | matched seeds | matched task/bout observations | per-seed count and mean difference | mean per-seed difference [range] |
|---|---:|---:|---:|---|---:|
| independent | 512 | 1 | 4 | 18997: n=4, 0.0000 | 0.0000 [0.0000, 0.0000] |
| independent | 1,024 | 1 | 4 | 18997: n=4, 0.0000 | 0.0000 [0.0000, 0.0000] |
| direct_shared | 512 | 1 | 4 | 18997: n=4, -0.0023 | -0.0023 [-0.0023, -0.0023] |
| direct_shared | 1,024 | 1 | 4 | 18997: n=4, -0.0008 | -0.0008 [-0.0008, -0.0008] |
| mean | 512 | 1 | 4 | 18997: n=4, -0.0082 | -0.0082 [-0.0082, -0.0082] |
| mean | 1,024 | 1 | 4 | 18997: n=4, 0.0001 | 0.0001 [0.0001, 0.0001] |
| mean_reuse_gate | 512 | 1 | 4 | 18997: n=4, -0.0031 | -0.0031 [-0.0031, -0.0031] |
| mean_reuse_gate | 1,024 | 1 | 4 | 18997: n=4, 0.0000 | 0.0000 [0.0000, 0.0000] |
| retention_gate | 512 | 1 | 4 | 18997: n=4, -0.0028 | -0.0028 [-0.0028, -0.0028] |
| retention_gate | 1,024 | 1 | 4 | 18997: n=4, -0.0001 | -0.0001 [-0.0001, -0.0001] |

#### Low budget all matched task/bout recurrence differences vs independent

Supplementary comparison of the same task/bout in both arms regardless of eviction status. This includes visits retained in one arm and evicted in the other.

| method | metric | matched seeds | matched task/bout observations | per-seed count and mean difference | mean per-seed difference [range] |
|---|---|---:|---:|---|---:|
| independent | first256_reward_per_step | 1 | 6 | 18997: n=6, 0.0000 | 0.0000 [0.0000, 0.0000] |
| independent | recovery512_reward_per_step | 1 | 6 | 18997: n=6, 0.0000 | 0.0000 [0.0000, 0.0000] |
| independent | recovery1024_reward_per_step | 1 | 6 | 18997: n=6, 0.0000 | 0.0000 [0.0000, 0.0000] |
| independent | first_goal_step | 1 | 6 | 18997: n=6, 0.0000 | 0.0000 [0.0000, 0.0000] |
| independent | first_gradient_step | 1 | 6 | 18997: n=6, 0.0000 | 0.0000 [0.0000, 0.0000] |
| direct_shared | first256_reward_per_step | 1 | 6 | 18997: n=6, -0.0021 | -0.0021 [-0.0021, -0.0021] |
| direct_shared | recovery512_reward_per_step | 1 | 6 | 18997: n=6, 0.0005 | 0.0005 [0.0005, 0.0005] |
| direct_shared | recovery1024_reward_per_step | 1 | 6 | 18997: n=6, 0.0019 | 0.0019 [0.0019, 0.0019] |
| direct_shared | first_goal_step | 1 | 6 | 18997: n=6, 1.5000 | 1.5000 [1.5000, 1.5000] |
| direct_shared | first_gradient_step | 1 | 6 | 18997: n=6, 0.0000 | 0.0000 [0.0000, 0.0000] |
| mean | first256_reward_per_step | 1 | 6 | 18997: n=6, -0.0042 | -0.0042 [-0.0042, -0.0042] |
| mean | recovery512_reward_per_step | 1 | 6 | 18997: n=6, -0.0071 | -0.0071 [-0.0071, -0.0071] |
| mean | recovery1024_reward_per_step | 1 | 6 | 18997: n=6, 0.0011 | 0.0011 [0.0011, 0.0011] |
| mean | first_goal_step | 1 | 5 | 18997: n=5, 266.6000 | 266.6000 [266.6000, 266.6000] |
| mean | first_gradient_step | 1 | 6 | 18997: n=6, 0.0000 | 0.0000 [0.0000, 0.0000] |
| mean_reuse_gate | first256_reward_per_step | 1 | 6 | 18997: n=6, -0.0008 | -0.0008 [-0.0008, -0.0008] |
| mean_reuse_gate | recovery512_reward_per_step | 1 | 6 | 18997: n=6, -0.0003 | -0.0003 [-0.0003, -0.0003] |
| mean_reuse_gate | recovery1024_reward_per_step | 1 | 6 | 18997: n=6, 0.0033 | 0.0033 [0.0033, 0.0033] |
| mean_reuse_gate | first_goal_step | 1 | 6 | 18997: n=6, 161.3333 | 161.3333 [161.3333, 161.3333] |
| mean_reuse_gate | first_gradient_step | 1 | 6 | 18997: n=6, 0.0000 | 0.0000 [0.0000, 0.0000] |
| retention_gate | first256_reward_per_step | 1 | 6 | 18997: n=6, -0.0026 | -0.0026 [-0.0026, -0.0026] |
| retention_gate | recovery512_reward_per_step | 1 | 6 | 18997: n=6, 0.0001 | 0.0001 [0.0001, 0.0001] |
| retention_gate | recovery1024_reward_per_step | 1 | 6 | 18997: n=6, 0.0013 | 0.0013 [0.0013, 0.0013] |
| retention_gate | first_goal_step | 1 | 6 | 18997: n=6, 220.1667 | 220.1667 [220.1667, 220.1667] |
| retention_gate | first_gradient_step | 1 | 6 | 18997: n=6, 0.0000 | 0.0000 [0.0000, 0.0000] |

## Capacity, storage, and operation counts

| budget | method | seeds | actor bytes max | full physical bytes max | package budget bytes | evidence bytes max | evictions | rollouts | minibatch updates | proposals / accepts | prior bindings | CPU s | wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low | independent | 1 | 140 | 223577 | 327680 | 21640 | 6.0 | 48.0 | 768.0 | 0.0 / 0.0 | 0.0 | 14.27 | 14.55 |
| low | direct_shared | 1 | 134 | 209171 | 327680 | 21640 | 7.0 | 48.0 | 768.0 | 0.0 / 0.0 | 10.0 | 14.56 | 14.89 |
| low | mean | 1 | 134 | 209747 | 327680 | 21640 | 7.0 | 48.0 | 768.0 | 11.0 / 5.0 | 7.0 | 14.41 | 14.81 |
| low | mean_reuse_gate | 1 | 134 | 209683 | 327680 | 21640 | 7.0 | 48.0 | 768.0 | 11.0 / 5.0 | 5.0 | 14.31 | 14.88 |
| low | retention_gate | 1 | 134 | 209427 | 327680 | 21640 | 7.0 | 48.0 | 768.0 | 11.0 / 0.0 | 0.0 | 13.33 | 13.73 |

## Per-seed records

All negative values and task-level recurrence observations, including missing first-goal outcomes, are retained in `analysis.json`. The table below preserves each seed's full-stream reward rate.

| budget | method | seed | total reward | reward / raw step | evictions |
|---|---|---:|---:|---:|---:|
| low | direct_shared | 18997 | -194.1400 | -0.0158 | 7 |
| low | independent | 18997 | -197.6400 | -0.0161 | 6 |
| low | mean | 18997 | -196.6400 | -0.0160 | 7 |
| low | mean_reuse_gate | 18997 | -174.6400 | -0.0142 | 7 |
| low | retention_gate | 18997 | -188.8900 | -0.0154 | 7 |

## Figure

![Full-stream reward and natural-eviction recovery](figure.png)

This is an exploratory report. Seed-paired differences are descriptive; tasks, bouts, episodes, and transitions are not treated as independent seeds.

## Software

The visualization workflow follows the Matplotlib procedural guidance in Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, arXiv:2609.00065 (current arXiv version v2, revised 2 September 2026). [arXiv record](https://arxiv.org/abs/2609.00065).
