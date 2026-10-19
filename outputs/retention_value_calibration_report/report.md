# Selected-Feedback Retention Stream Report

Calibration status: **complete_exploratory_cohort**.

## Calibration

Final 32 of 64 selected labels: mean linear − intercept prequential MSE = **-3.0619875e-06**; descriptive seed-bootstrap 95% CI [-4.3653208e-06, -1.4695639e-06] (n = 6 seeds). Negative favors linear on observed selected labels.
Relations are averaged within seed before the overall seed bootstrap; relation-specific results are also in the JSON report.

Only the selected action's pre-action prediction is scored against `feedback.utility_target`. Unchosen actions have no observed label and are not scored.

Comparison input: not supplied.

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

Comparison recurrence metrics are factual natural-recurrence first-256 reward rates, including after prior natural eviction. They are reported with bout counts and are not causal recovery loss estimates.

Campaign elapsed and total CPU values are shown only when the root summary logs `cohort_costs`; legacy per-stream wall totals are labeled as sums and are not campaign elapsed time.

This is selected-feedback utility prediction and a four-seed exploratory policy comparison. It does not observe unchosen-action returns, identify per-write causal benefit, or establish novelty of the ridge/discounting component.

Bootstrap settings: 10000 resamples, seed 20261019; intervals are descriptive. Report generation itself runs no RL and adds no environment steps.
