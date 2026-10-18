# Frozen-policy score calibration

This is the single-seed pilot pipeline diagnostic. It is not main-cohort evidence, and the predeclared favorable heuristic is not evaluated here.

This diagnostic tests whether the frozen retention-surplus score predicts actual benefit from inserting the candidate into a frozen policy at a later natural recurrence.

The target is the conditional 128-step frozen-policy gain. It does not estimate adaptation, final return, the complete write benefit, or long-term online effect.

Scores are frozen for all `write_tests` in the first 6 bouts at steps up to 12,288, before any recurrence. The primary outcome weights unique source contexts in the scored archive equally. Prototype-reservoir weighting is secondary. Each candidate is evaluated from four observed recurrence states at offsets 0, 64, 128, and 192, with 8 paired runs of 128 raw steps per policy arm and common action uniforms and reset-ordinal seeds.

Outcome labels use pointwise 95% Student t intervals with 7 degrees of freedom over the 8 paired replicate aggregates. Epsilon is 0: beneficial means the lower endpoint is above zero; harmful means the upper endpoint is below zero; all other cases are ambiguous. These labels are conditional on the observed contexts and recurrence anchors; the intervals are not simultaneous across candidates. No p-values or confidence intervals across seeds are reported. This is the single-seed pilot pipeline diagnostic. It is not main-cohort evidence, and the predeclared favorable heuristic is not evaluated here.

Ambiguous outcomes are retained and counted separately. `original_mean_lower > 0` is a conservative approval gate; a non-positive lower score means not approved, not a prediction of harm. The report counts approvals with MC-labeled harm and non-approvals with MC-labeled benefit (opportunity loss). Writer acceptance is reported as a separate gate classification. Precision and recall are descriptive and marked not estimable when their denominator is zero. AUROC is descriptive over resolved beneficial/harmful outcomes only.

## Seed-level results

| Budget | Seed | Candidates | Excluded | Primary rho | Partial rho by group | Prototype-weighted rho | Score approved but MC harmful | Score not approved but MC beneficial | Score precision / recall | Ambiguous | Gate accepted but MC harmful | Gate not accepted but MC beneficial | Gate precision / recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| low | 19701 | 3 | 0 | not estimable | not estimable | not estimable | 0 | 1 | 1.000 / 0.500 | 1 | 0 | 1 | 1.000 / 0.500 |

## Seed summaries

Means and ranges below summarize the 1 declared seed-level values separately within each budget. They are descriptive, with no across-seed interval or p-value.

| Budget | Primary rho mean (range) | Estimable seeds | Partial rho mean (range) | Prototype rho mean (range) | Positive primary rho seeds | Heuristic checks all met |
| --- | --- | ---: | --- | --- | ---: | --- |
| low | not estimable (not estimable) | 0/1 | not estimable (not estimable) | not estimable (not estimable) | 0/1 | None |

## Predictor diagnostics

These descriptive Spearman correlations compare each recorded predictor with the equal-context mean gain within each seed and budget. Missing predictor values are excluded for that predictor and reported in `analysis.json`. No threshold is fitted or tuned.

| Budget | Seed | original_mean_lower | original_mean_gain | codec_mean_lower | fresh_lower | negative_mean_stderr | time_step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 19701 | not estimable | not estimable | not estimable | not estimable | not estimable | not estimable |

## Exploratory favorable heuristic

Exploratory heuristic: both budgets must have all 3 seed cells estimable, mean primary rho >= 0.3, at least 2 of 3 positive seed rhos, and mean partial rho controlling group > 0.

The main-cohort exploratory heuristic is not evaluated on the pilot cohort.

## Cohort audit and cost

Validated 1 completed seed-by-budget branches, 3 evaluated candidates, 0 excluded tests, and 3 captured write tests. Cohort audit passed: **True**.

Interaction cost: 12288 learner steps + 57344 observer steps = 69632 total new interactions. CPU time: 45.047 s; wall time: 51.889 s.

The figure shows candidate-level primary scores and conditional gains. Zero lines mark the score and outcome sign boundaries. It does not show candidate-level confidence intervals; the inferential labels use the 8 paired replicate aggregates described above.

Plotting used Matplotlib (Kassis et al., 2026, *Scientific Agent Skills*, arXiv:2609.00065v2; https://arxiv.org/abs/2609.00065).
