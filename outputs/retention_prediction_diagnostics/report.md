# Retention prediction diagnostics

This is an offline, descriptive analysis of existing selected-feedback logs. It ran zero new environment steps.

## Inputs and scope

- Calibration streams: 12; decisions: 768; tail labels scored per stream: 32. 
- Calibration relations: dynamics, reward_conflict.
- Original ridge replay mismatches at tolerance 1e-9: 0 of 6144 logged action predictions.
- Maximum absolute replay error: 0; utility target reconstruction error: 0.
- Calibration propensity rows at exactly 0.25 within 1e-12: 768 / 768.
- Comparison coverage rows: 80; complete logs: 80. Comparison labels were not used for prediction scoring.

## Prediction scores

Per-stream MSE for every target and feature set is in `stream_metrics.csv`; relation and pooled seed summaries are in `relation_aggregates.csv`. The scores are prequential and use only the selected action's observed label. At each decision the model predicts first, then the selected feedback updates its state. No unchosen action outcomes are imputed.

The comparison intervals in `seed_bootstrap_differences.csv` resample the six calibration seed IDs and show descriptive paired MSE differences (model variant minus full6). They are not efficacy tests.

Mean seed utility MSE by feature set. Every seed has both relations and 32 scored labels per stream, so relation and seed contributions are equally sized (MSE columns are multiplied by 10⁵; differences and intervals by 10⁶):

| Feature set | Dynamics | Reward conflict | All seeds, both relations | Difference vs full6 [95% seed bootstrap interval] |
| --- | ---: | ---: | ---: | ---: |
| drop_candidate_minus_incumbent_recent_group_frequency | 4.421 | 5.560 | 4.991 | -1.080 [-1.516, -0.644] |
| drop_normalized_log_candidate_minus_parent_source_fit | 4.496 | 5.596 | 5.046 | -0.526 [-0.724, -0.355] |
| reward_ewma_only | 4.602 | 5.545 | 5.073 | -0.252 [-1.114, 0.723] |
| full6 | 4.538 | 5.659 | 5.099 | 0.000 [0.000, 0.000] |
| drop_decisions_since_parent_write_over_16 | 4.544 | 5.656 | 5.100 | 0.013 [-0.011, 0.037] |
| drop_recent_eviction_fraction8 | 4.842 | 5.638 | 5.240 | 1.412 [0.801, 1.966] |
| time_age_only | 4.391 | 6.410 | 5.400 | 3.017 [1.397, 4.337] |
| intercept | 4.401 | 6.409 | 5.405 | 3.062 [1.488, 4.365] |
| drop_candidate_minus_incumbent_recovery_ewma_over_0_02 | 4.458 | 6.526 | 5.492 | 3.933 [3.087, 4.780] |

The lowest pooled utility MSE among these nine feature sets was `drop_candidate_minus_incumbent_recent_group_frequency` at 4.9905768e-05; full6 was 5.0985801e-05. Its descriptive paired seed-bootstrap difference was -1.08e-06 [-1.52e-06, -6.44e-07]. This is a predictive comparison within these calibration trajectories, not evidence of policy benefit.

## Reward and CPU residuals

`residual_decomposition.csv` reports the reward-rate and CPU-rate residual covariance and verifies the identity for utility residuals. Here utility is reward rate minus `utility_cpu_s / block_steps` because the frozen CPU penalty is 1.

For full6 pooled over all 384 held-out labels, utility MSE was 5.0985801e-05; the reward/CPU residual covariance was -1.041e-07, and the MSE identity error was 0.

## Full6 versus intercept error components

The table gives `intercept component MSE − full6 component MSE`, so positive utility improvement means full6 has lower utility MSE. Utility error is reward error minus CPU-rate error; its MSE decomposes exactly into reward MSE, CPU MSE, and the residual cross term `−2 × mean(reward residual × CPU residual)`. Percentages divide each component's improvement by total utility MSE improvement; they may be negative or exceed 100% when components offset.

Per-seed rows are in units of 10⁻⁶ MSE; the exact component MSEs and deltas are in `utility_mse_component_comparison.csv`.

| Relation | Seed | Utility improvement | Reward MSE contribution | CPU MSE contribution | Residual cross-term contribution | Reward % | CPU % | Cross-term % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamics | 23101 | 0.434382 | 0.239648 | 0.011820 | 0.182914 | 55.17% | 2.72% | 42.11% |
| dynamics | 23102 | 2.131561 | 1.740570 | 0.011569 | 0.379422 | 81.66% | 0.54% | 17.80% |
| dynamics | 23103 | -2.686679 | -3.151171 | 0.010534 | 0.453958 | 117.29% | -0.39% | -16.90% |
| dynamics | 23104 | -2.268938 | -2.560873 | 0.011787 | 0.280148 | 112.87% | -0.52% | -12.35% |
| dynamics | 23105 | 0.912169 | 0.560242 | 0.009777 | 0.342150 | 61.42% | 1.07% | 37.51% |
| dynamics | 23107 | -6.769920 | -6.940475 | 0.007803 | 0.162752 | 102.52% | -0.12% | -2.40% |
| reward_conflict | 23101 | 9.888023 | 10.123278 | -0.018039 | -0.217217 | 102.38% | -0.18% | -2.20% |
| reward_conflict | 23102 | 2.327574 | 2.136210 | -0.016802 | 0.208166 | 91.78% | -0.72% | 8.94% |
| reward_conflict | 23103 | 1.676539 | 2.040985 | -0.007269 | -0.357177 | 121.74% | -0.43% | -21.30% |
| reward_conflict | 23104 | 11.035527 | 11.027793 | 0.002813 | 0.004921 | 99.93% | 0.03% | 0.04% |
| reward_conflict | 23105 | 7.190857 | 7.163946 | 0.001423 | 0.025488 | 99.63% | 0.02% | 0.35% |
| reward_conflict | 23107 | 12.872755 | 12.985265 | 0.003753 | -0.116263 | 100.87% | 0.03% | -0.90% |

Across both relations and all six seeds, full6 reduced utility MSE by 3.06199e-06; 96.25% of that reduction came from reward-rate MSE, 0.08% from CPU-rate MSE, and 3.67% from the residual cross term.
For dynamics, full6 utility MSE improvement was -1.37457e-06; reward, CPU, and residual cross-term contributions were -1.68534e-06, 1.05485e-08, and 3.00224e-07.
For reward_conflict, full6 utility MSE improvement was 7.49855e-06; reward, CPU, and residual cross-term contributions were 7.57958e-06, -5.68682e-09, and -7.5347e-08.

These component differences describe prediction errors on selected feedback. The replayed action-specific predictors do not separate common state difficulty from action-specific contrast, and this analysis makes no action-ranking claim.

## Action support

`action_counts.csv` reports selected action counts, discounted observation mass and arithmetic importance-weight ESS for each deterministic one-step fixed-action weighting. Discounted mass is a weighted count, not ESS. ESS describes observed action support at the logged histories; it does not estimate a multi-step or whole-policy value. No whole-policy off-policy evaluation is reported.

| Action | Selected rows | Pooled one-step ESS | Mean discounted observation mass per stream |
| --- | ---: | ---: | ---: |
| keep | 182 | 182 | 6.612 |
| replace | 171 | 171 | 6.657 |
| mean | 226 | 226 | 8.239 |
| none | 189 | 189 | 7.080 |

## Interpretation

These estimates describe prediction of selected feedback along uniform-random calibration trajectories. They do not establish causal benefit, action ranking, or performance under a different policy. The comparison directory is included only for stream and log coverage.

## Output files

- `analysis.json`: complete machine-readable summary and audit details.
- `input_hashes.json`: SHA-256 hashes of the calibration and comparison summaries, frozen configs, and selected-feedback logs.
- `prequential_predictions.csv`: every scored observed-action prediction and target.
- `stream_metrics.csv`, `relation_aggregates.csv`, `seed_bootstrap_differences.csv`, `residual_decomposition.csv`.
- `action_counts.csv`, `calibration_propensity.csv`, `comparison_coverage.csv`.
- `utility_mse_component_comparison.csv`: full6 versus intercept MSE attribution by relation and seed.
- `code_snapshot/diagnose_retention_prediction.py`: executed source snapshot.

## Reproduction

Run these commands from the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -m pytest tests\test_diagnose_retention_prediction.py -q
.venv\Scripts\python.exe scripts\diagnose_retention_prediction.py
```
