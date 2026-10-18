# Frozen-policy score calibration

This diagnostic tests whether the frozen retention-surplus score predicts actual benefit from inserting the candidate into a frozen policy at a later natural recurrence.

The target is the conditional 128-step frozen-policy gain. It does not estimate adaptation, final return, the complete write benefit, or long-term online effect.

Scores are frozen for all `write_tests` in the first 6 bouts at steps up to 12,288, before any recurrence. The primary outcome weights unique source contexts in the scored archive equally. Prototype-reservoir weighting is secondary. Each candidate is evaluated from four observed recurrence states at offsets 0, 64, 128, and 192, with 8 paired runs of 128 raw steps per policy arm and common action uniforms and reset-ordinal seeds.

Outcome labels use pointwise 95% Student t intervals with 7 degrees of freedom over the 8 paired replicate aggregates. Epsilon is 0: beneficial means the lower endpoint is above zero; harmful means the upper endpoint is below zero; all other cases are ambiguous. These labels are conditional on the observed contexts and recurrence anchors; the intervals are not simultaneous across candidates. No p-values or confidence intervals across seeds are reported. The 3 seeds within each budget are the statistical units; candidates and the 6 seed-by-budget cells are not treated as independent seeds.

Ambiguous outcomes are retained and counted separately. `original_mean_lower > 0` is a conservative approval gate; a non-positive lower score means not approved, not a prediction of harm. The report counts approvals with MC-labeled harm and non-approvals with MC-labeled benefit (opportunity loss). Writer acceptance is reported as a separate gate classification. Precision and recall are descriptive and marked not estimable when their denominator is zero. AUROC is descriptive over resolved beneficial/harmful outcomes only.

## Seed-level results

| Budget | Seed | Candidates | Excluded | Primary rho | Partial rho by group | Prototype-weighted rho | Score approved but MC harmful | Score not approved but MC beneficial | Score precision / recall | Ambiguous | Gate accepted but MC harmful | Gate not accepted but MC beneficial | Gate precision / recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| low | 19703 | 9 | 0 | -0.867 | -0.779 | -0.867 | 0 | 9 | not estimable / 0.000 | 0 | 0 | 9 | not estimable / 0.000 |
| low | 19717 | 8 | 0 | 0.786 | 0.788 | 0.786 | 0 | 1 | 1.000 / 0.667 | 5 | 0 | 1 | 1.000 / 0.667 |
| low | 19727 | 8 | 0 | -0.048 | 0.843 | -0.238 | 0 | 3 | 1.000 / 0.250 | 4 | 0 | 3 | 1.000 / 0.250 |
| wide | 19703 | 8 | 0 | -0.833 | -0.300 | -0.762 | 0 | 8 | not estimable / 0.000 | 0 | 0 | 8 | not estimable / 0.000 |
| wide | 19717 | 7 | 0 | 0.321 | 0.307 | 0.321 | 0 | 1 | 1.000 / 0.750 | 3 | 0 | 1 | 1.000 / 0.750 |
| wide | 19727 | 7 | 0 | -0.464 | 0.464 | -0.250 | 0 | 5 | 1.000 / 0.286 | 0 | 0 | 5 | 1.000 / 0.286 |

## Seed summaries

Means and ranges below summarize the 3 declared seed-level values separately within each budget. They are descriptive, with no across-seed interval or p-value.

| Budget | Primary rho mean (range) | Estimable seeds | Partial rho mean (range) | Prototype rho mean (range) | Positive primary rho seeds | Heuristic checks all met |
| --- | --- | ---: | --- | --- | ---: | --- |
| low | -0.043 ([-0.867, 0.786]) | 3/3 | 0.284 ([-0.779, 0.843]) | -0.106 ([-0.867, 0.786]) | 1/3 | False |
| wide | -0.325 ([-0.833, 0.321]) | 3/3 | 0.157 ([-0.300, 0.464]) | -0.230 ([-0.762, 0.321]) | 1/3 | False |

## Predictor diagnostics

These descriptive Spearman correlations compare each recorded predictor with the equal-context mean gain within each seed and budget. Missing predictor values are excluded for that predictor and reported in `analysis.json`. No threshold is fitted or tuned.

| Budget | Seed | original_mean_lower | original_mean_gain | codec_mean_lower | fresh_lower | negative_mean_stderr | time_step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 19703 | -0.867 | -0.467 | -0.867 | -0.083 | -0.617 | -0.133 |
| low | 19717 | 0.786 | 0.857 | 0.786 | 0.905 | -0.738 | -0.381 |
| low | 19727 | -0.048 | 0.833 | -0.048 | -0.738 | -0.667 | -0.571 |
| wide | 19703 | -0.833 | -0.762 | -0.833 | -0.095 | -0.714 | -0.452 |
| wide | 19717 | 0.321 | 0.857 | 0.321 | 0.536 | -0.929 | -0.321 |
| wide | 19727 | -0.464 | 0.464 | -0.464 | -0.536 | -0.893 | -0.286 |

## Exploratory favorable heuristic

Exploratory heuristic: both budgets must have all 3 seed cells estimable, mean primary rho >= 0.3, at least 2 of 3 positive seed rhos, and mean partial rho controlling group > 0.

Met for both budgets: **False**. This is an exploratory screening rule, not a significance test or confirmatory claim.

## Cohort audit and cost

Validated 6 completed seed-by-budget branches, 47 evaluated candidates, 0 excluded tests, and 47 captured write tests. Cohort audit passed: **True**.

Interaction cost: 147456 learner steps + 892928 observer steps = 1040384 total new interactions. CPU time: 535.719 s; wall time: 283.846 s.

The figure shows candidate-level primary scores and conditional gains. Zero lines mark the score and outcome sign boundaries. It does not show candidate-level confidence intervals; the inferential labels use the 8 paired replicate aggregates described above.

Plotting software: Matplotlib, cited as J. D. Hunter, “Matplotlib: A 2D Graphics Environment,” *Computing in Science & Engineering* 9(3), 90–95 (2007). [doi:10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55). See [Matplotlib’s citation guidance](https://matplotlib.org/stable/project/citing.html).

Procedural plotting guidance: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
