# Selected-Feedback Retention Stream Report

Calibration status: **complete_exploratory_cohort**.

## Calibration

Final 32 of 64 selected labels: mean linear − intercept prequential MSE = **-3.0619875e-06**; descriptive seed-bootstrap 95% CI [-4.3653208e-06, -1.4695639e-06] (n = 6 seeds). Negative favors linear on observed selected labels.
Relations are averaged within seed before the overall seed bootstrap; relation-specific results are also in the JSON report.

Only the selected action's pre-action prediction is scored against `feedback.utility_target`. Unchosen actions have no observed label and are not scored.

Comparison status: **complete_exploratory_cohort**.

## Repeated-retention comparison

Whole-stream return/step includes warmup and all outcome blocks. Postwarmup return/step uses outcome blocks only. The primary contrast is learned_linear − learned_intercept within seed and relation; its four-seed intervals are descriptive. The lower confidence bound is checked separately against zero and the frozen practical threshold.

| Relation | Metric | Mean learned_linear − learned_intercept | 95% seed-bootstrap CI | Positive / negative / tie seeds | Lower CI > 0 / +practical threshold |
|---|---|---:|---:|---:|---|
| dynamics | whole_stream | 0.00020996094 | [-0.00015625, 0.00040852865] | 3 / 1 / 0 of 4 | no / no |
| dynamics | postwarmup | 0.00021870931 | [-0.00016276042, 0.00042555067] | 3 / 1 / 0 of 4 | no / no |
| reward_conflict | whole_stream | 4.6386719e-05 | [-0.0003499349, 0.00056233724] | 2 / 2 / 0 of 4 | no / no |
| reward_conflict | postwarmup | 4.8319499e-05 | [-0.00036451552, 0.00058576796] | 2 / 2 / 0 of 4 | no / no |

The practical threshold is +0.002 reward per step; JSON also records the point estimate minus each threshold.

Learned_linear and learned_intercept share the first 16 uniform warmup decisions and then use the frozen epsilon-greedy schedule. Random stays uniform. Deterministic keep/replace/mean/frequency/lower controls have different exploration costs.

All methods versus every control are included in `analysis.json`; stream and seed-level values are in `per_seed.csv`.

## Cost, retention, and scope

### calibration

| Quantity | Total / maximum |
|---|---:|
| Actual raw step rows | 1216512 |
| Reported environment steps | 1216512 |
| PPO rollouts | 4752 |
| Minibatch updates | 76032 |
| Campaign elapsed wall seconds | 472.916 |
| Campaign total CPU seconds | 1376.031 |
| Campaign child CPU seconds | 1371.266 |
| Campaign orchestrator CPU seconds | 4.766 |
| Sum of per-stream CPU seconds | 1371.266 |
| Sum of per-stream wall seconds (not elapsed) | 1395.055 |
| CPU seconds charged to utility | 1310.328 |
| Model update CPU seconds | 0.109 |
| Parent writes | 586 |
| Retention operations | 768 |
| Workers actually used | 3 |
| Peak package bytes | 240386 / cap 262144 |
| Maximum model state bytes | 1732 |

### comparison

| Quantity | Total / maximum |
|---|---:|
| Actual raw step rows | 6144000 |
| Reported environment steps | 6144000 |
| PPO rollouts | 24000 |
| Minibatch updates | 384000 |
| Campaign elapsed wall seconds | 1059.754 |
| Campaign total CPU seconds | 5936.000 |
| Campaign child CPU seconds | 5912.984 |
| Campaign orchestrator CPU seconds | 23.016 |
| Sum of per-stream CPU seconds | 5912.984 |
| Sum of per-stream wall seconds (not elapsed) | 5966.654 |
| CPU seconds charged to utility | 5605.016 |
| Model update CPU seconds | 0.281 |
| Parent writes | 1847 |
| Retention operations | 3072 |
| Workers actually used | 6 |
| Peak package bytes | 240322 / cap 262144 |
| Maximum model state bytes | 1732 |

Comparison recurrence metrics are factual natural-recurrence first-256 reward rates, including after prior natural eviction. They are reported with bout counts and are not causal recovery loss estimates.

Campaign elapsed and total CPU values are shown only when the root summary logs `cohort_costs`; legacy per-stream wall totals are labeled as sums and are not campaign elapsed time.

This is selected-feedback utility prediction and a four-seed exploratory policy comparison. It does not observe unchosen-action returns, identify per-write causal benefit, or establish novelty of the ridge/discounting component.

Bootstrap settings: 10000 resamples, seed 20261019; intervals are descriptive. Report generation itself runs no RL and adds no environment steps.
