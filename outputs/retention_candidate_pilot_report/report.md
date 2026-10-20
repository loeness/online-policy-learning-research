# Retention candidate report

Phase: **pilot**. Role: **exploratory_candidate_content_and_state_coverage**. This is an exploratory fixed-candidate, one-write diagnostic.

The reporter reads saved summaries and hashed ledgers only. It makes no environment calls, does not adapt candidates or analysis choices from reward outcomes, and does not send results back to a learner.

## Design and audit

- Prefixes: 2; branches: 10; actual environment steps: 98,304.
- Common prefix steps: 36,864; branch continuation steps: 61,440.
- Recounted training: 384 rollouts and 6,144 minibatch updates.
- SHA-256 checked raw CSV ledgers: 36; complete grid, restore, pairing, reward, cost, and capacity checks passed: **True**.
- Actor container: 89 bytes; complete package cap: 262,144 bytes.
- Early and late are joint state-coverage conditions. They do not identify a causal effect of age, fit volume, or maturity alone.
- This one-seed pilot has no uncertainty interval. The main report uses six deployment seeds, averaging paired futures and replicas within seed before old-family-stratified bootstrap.
- All five arms, including keep, remain in the report. A lower fit KL by itself is not evidence of improved return.

## Primary reward contrasts at 6,144 steps

Every arm contrast is paired within prefix, future regime, and replicate. Future regimes and replicates are averaged within seed before the descriptive bootstrap.

| relation | state | contrast | seeds | reward / step difference | 95% descriptive interval |
|---|---|---|---:|---:|---|
| dynamics | early | replace_behavioral-minus-replace_mean | 1 | 0.000041 | not estimated |
| dynamics | early | replace_behavioral-minus-keep | 1 | 0.002482 | not estimated |
| dynamics | early | replace_mean-minus-keep | 1 | 0.002441 | not estimated |
| dynamics | early | merge_behavioral-minus-merge_mean | 1 | 0.001261 | not estimated |
| dynamics | early | merge_behavioral-minus-keep | 1 | 0.002889 | not estimated |
| dynamics | early | merge_mean-minus-keep | 1 | 0.001628 | not estimated |
| dynamics | late | replace_behavioral-minus-replace_mean | 1 | 0.002197 | not estimated |
| dynamics | late | replace_behavioral-minus-keep | 1 | -0.007690 | not estimated |
| dynamics | late | replace_mean-minus-keep | 1 | -0.009888 | not estimated |
| dynamics | late | merge_behavioral-minus-merge_mean | 1 | 0.004964 | not estimated |
| dynamics | late | merge_behavioral-minus-keep | 1 | -0.005127 | not estimated |
| dynamics | late | merge_mean-minus-keep | 1 | -0.010091 | not estimated |

## Future-scoped contrasts

These estimates keep the future regime fixed and average the 1 configured replica within seed.

| relation | state | future | contrast | reward / step difference | 95% descriptive interval | seeds |
|---|---|---|---|---:|---|---:|
| dynamics | early | old_major | replace_behavioral-minus-replace_mean | 0.000041 | not estimated | 1 |
| dynamics | early | old_major | replace_behavioral-minus-keep | 0.002482 | not estimated | 1 |
| dynamics | early | old_major | replace_mean-minus-keep | 0.002441 | not estimated | 1 |
| dynamics | early | old_major | merge_behavioral-minus-merge_mean | 0.001261 | not estimated | 1 |
| dynamics | early | old_major | merge_behavioral-minus-keep | 0.002889 | not estimated | 1 |
| dynamics | early | old_major | merge_mean-minus-keep | 0.001628 | not estimated | 1 |
| dynamics | early | old_major | candidate_by_operation_interaction | 0.001221 | not estimated | 1 |
| dynamics | late | old_major | replace_behavioral-minus-replace_mean | 0.002197 | not estimated | 1 |
| dynamics | late | old_major | replace_behavioral-minus-keep | -0.007690 | not estimated | 1 |
| dynamics | late | old_major | replace_mean-minus-keep | -0.009888 | not estimated | 1 |
| dynamics | late | old_major | merge_behavioral-minus-merge_mean | 0.004964 | not estimated | 1 |
| dynamics | late | old_major | merge_behavioral-minus-keep | -0.005127 | not estimated | 1 |
| dynamics | late | old_major | merge_mean-minus-keep | -0.010091 | not estimated | 1 |
| dynamics | late | old_major | candidate_by_operation_interaction | 0.002767 | not estimated | 1 |

## Candidate by operation interaction

The interaction is (merge behavioral − merge mean) − (replace behavioral − replace mean). It is computed within each seed after averaging paired futures and replicas.

| relation | state | horizon | interaction / step | 95% descriptive interval | seeds |
|---|---|---:|---:|---|---:|
| dynamics | early | 6,144 | 0.001221 | not estimated | 1 |
| dynamics | late | 6,144 | 0.002767 | not estimated | 1 |

## First-256-step recovery

Recovery is reported as mean return per 256-step bout, converted to reward per step. The allocated-only view includes bouts that allocated a slot; the all-bouts view includes every future bout. The same within-seed pairing is used.

| population | relation | state | contrast | reward / step difference | 95% descriptive interval | seeds |
|---|---|---|---|---:|---|---:|
| allocated_only | dynamics | early | replace_behavioral-minus-replace_mean | -0.000000 | not estimated | 1 |
| allocated_only | dynamics | early | replace_behavioral-minus-keep | 0.001831 | not estimated | 1 |
| allocated_only | dynamics | early | replace_mean-minus-keep | 0.001831 | not estimated | 1 |
| allocated_only | dynamics | early | merge_behavioral-minus-merge_mean | 0.006592 | not estimated | 1 |
| allocated_only | dynamics | early | merge_behavioral-minus-keep | 0.005005 | not estimated | 1 |
| allocated_only | dynamics | early | merge_mean-minus-keep | -0.001587 | not estimated | 1 |
| allocated_only | dynamics | early | candidate_by_operation_interaction | 0.006592 | not estimated | 1 |
| allocated_only | dynamics | late | replace_behavioral-minus-replace_mean | 0.002319 | not estimated | 1 |
| allocated_only | dynamics | late | replace_behavioral-minus-keep | -0.004761 | not estimated | 1 |
| allocated_only | dynamics | late | replace_mean-minus-keep | -0.007080 | not estimated | 1 |
| allocated_only | dynamics | late | merge_behavioral-minus-merge_mean | 0.007202 | not estimated | 1 |
| allocated_only | dynamics | late | merge_behavioral-minus-keep | -0.003296 | not estimated | 1 |
| allocated_only | dynamics | late | merge_mean-minus-keep | -0.010498 | not estimated | 1 |
| allocated_only | dynamics | late | candidate_by_operation_interaction | 0.004883 | not estimated | 1 |
| all_bouts | dynamics | early | replace_behavioral-minus-replace_mean | 0.000163 | not estimated | 1 |
| all_bouts | dynamics | early | replace_behavioral-minus-keep | 0.006104 | not estimated | 1 |
| all_bouts | dynamics | early | replace_mean-minus-keep | 0.005941 | not estimated | 1 |
| all_bouts | dynamics | early | merge_behavioral-minus-merge_mean | 0.003499 | not estimated | 1 |
| all_bouts | dynamics | early | merge_behavioral-minus-keep | 0.008057 | not estimated | 1 |
| all_bouts | dynamics | early | merge_mean-minus-keep | 0.004557 | not estimated | 1 |
| all_bouts | dynamics | early | candidate_by_operation_interaction | 0.003337 | not estimated | 1 |
| all_bouts | dynamics | late | replace_behavioral-minus-replace_mean | 0.001953 | not estimated | 1 |
| all_bouts | dynamics | late | replace_behavioral-minus-keep | -0.008301 | not estimated | 1 |
| all_bouts | dynamics | late | replace_mean-minus-keep | -0.010254 | not estimated | 1 |
| all_bouts | dynamics | late | merge_behavioral-minus-merge_mean | 0.011393 | not estimated | 1 |
| all_bouts | dynamics | late | merge_behavioral-minus-keep | -0.000732 | not estimated | 1 |
| all_bouts | dynamics | late | merge_mean-minus-keep | -0.012126 | not estimated | 1 |
| all_bouts | dynamics | late | candidate_by_operation_interaction | 0.009440 | not estimated | 1 |

## Candidate fit and codec diagnostics

The behavioral candidate was fit once per prefix from the bounded source buffer. Training KL drove candidate selection; holdout KL is diagnostic only. Branch-specific KL after the action is listed separately from the fit-split KL.

| relation | state | seed | train KL initial | train KL fitted | holdout KL initial | holdout KL fitted | band counts owner 0 / 1 | mean norm | behavioral norm | fit CPU s | buffer bytes |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| dynamics | early | 25189 | 0.028927 | 0.026162 | 0.015208 | 0.007702 | `[[5,24,99],[55,31,42]]` | 0.093267 | 0.864266 | 0.031250 | 2,046 |
| dynamics | late | 25189 | 0.274054 | 0.137002 | 0.292448 | 0.185254 | `[[95,16,17],[66,54,8]]` | 0.395324 | 12.956333 | 0.031250 | 2,088 |

Branch-level before/after int8-codec source-teacher KL, owner-specific counts, action norms, and behavior drift are preserved in `analysis.json`.

## Actual costs and capacity

The common prefix cost was 20.719 CPU seconds including fit; the candidate fit component was 0.062 seconds and is not added twice. Branch setup, action install, continuation, and serialization used 35.812 CPU seconds. Unique measured prefix plus branch time was 56.531 seconds.

| cost component | CPU seconds |
|---|---:|
| Prefix including fit | 20.719 |
| Candidate fit component of prefix | 0.062 |
| Prefix excluding fit estimate | 20.656 |
| Branch continuations | 35.812 |
| Unique prefix plus branch | 56.531 |

Every checked prefix and branch package used the 89-byte actor container and stayed within the 262,144-byte complete-package cap. Source-buffer, transient-candidate, package, and branch CPU records are retained in `analysis.json`.
Evidence sidecars used 4,134 bytes in prefixes, 20,670 bytes in final branch packages, and 20,670 bytes in post-action packages. Those measured copies are included in the corresponding full-package caps.

## Figure

![Retention candidate report](figure.png)

## Interpretation limits

This is a known-identity, fixed-candidate, one-write comparison. It does not test a deployed selector, repeated learned organizer, unknown-context discovery, tree or DAG comparisons, or C2. The late state changes online history and state coverage jointly. Lower fitted KL does not establish a reward gain. These results do not support claims of SOTA performance or organizer success.

## Provenance

Summary SHA-256: 43e1a7e7c820858abc850bd80356d5a14ad4fa1927c4dc4ca01276d97c073054  
Frozen config canonical SHA-256: 21e80571f4ffecc50ed9499c7c69ea3317322b7e7667df58ed892878696be2fe  
Source manifest canonical SHA-256: 1d9887a3ac68750a2fc2f29b8b3cd1503ca75e855c58bb2dda1f4540c29c3ab7  
Runner SHA-256: fe6e3a711cebb85ff811c3641cbc2d8da71762c14c599d569a04c10b4302b33e  
Behavioral fit module SHA-256: 5c57f2913e33e8fd7553550c0c98eeff2a000c4ce67193d5d7535e842daf1d71  
Reporter SHA-256: 1bf93ec45a314e5b31a77e2d01e8d43ce69c404ba004f35d03fce95b95c517b5

## External audit history

Audit files are preserved by SHA-256, including superseded records. Their declared statuses and raw counts are shown verbatim; the report's own ledger checks are recorded separately.

| file | declared status | cases / branches | raw steps | superseded | SHA-256 |
|---|---|---:|---:|---|---|
| `retention_candidate_audit/pilot.json` | passed | 2 / 10 | 98,304 | yes | `f33654d99f1c8b5e82dcc3cf13981954a849dffcac4621d234ee0eef20152827` |

This earlier audit output is preserved as written. The parent reviewer reported auditor contract/replay defects and required a repaired follow-up; this marks an audit-tool revision, not an experiment failure.
| `retention_candidate_audit/pilot_v2.json` | passed | 2 / 10 | 98,304 | no | `5903613d79b4aa4693e5e6e5eae655fc8c40481dae7dee2be28784b9a52eb8a3` |

## Software

Static figures use Matplotlib's object-oriented interface and export PNG, PDF, and SVG. The report preserves the exact reporter source beside its hash.
