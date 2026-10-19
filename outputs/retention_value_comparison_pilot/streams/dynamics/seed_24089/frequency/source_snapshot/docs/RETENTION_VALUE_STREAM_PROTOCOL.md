# Selected feedback and repeated retention: frozen exploratory protocol

Frozen 2026-10-19, before these streams are run. This follows the independent lower confirmation but uses separate seeds and no branch outcomes from that confirmation.

## Questions and decision gates

1. Can bounded history features predict the factual outcome of a selected storage action better than an action-specific discounted intercept? Calibration compares prequential squared error on the final 32 of 64 decisions. Negative linear-minus-intercept error is desirable. This does not identify individual causal write benefits.
2. Does using those features improve a whole repeated-retention policy over the intercept learner and the simple controls? The comparison uses fresh streams and freezes the model, schedule, features and candidate rules before calibration outcomes. It is an exploratory four-seed test, not a high-power confirmation.
3. Does any increment survive natural eviction and recurrence, under a complete persistent-package cap? Report whole-stream return, first-256-step recovery after natural eviction, interaction/update/CPU/wall costs, and actual serialized bytes. A storage selector advantage requires evidence beyond a fixed heuristic; merely outperforming independent learning does not establish it.

The primary comparison is learned_linear minus learned_intercept, separately for dynamics and reward conflict, using seed-cluster paired differences. Also report every method against keep, replace, mean, frequency, generalized lower, direct shared PPO, and independent local learning. Include all seeds, ties and failed runs. Use 10,000 seed bootstrap resamples, seed 20261019; label four-seed intervals descriptive. Report zero and the earlier +0.002 reward/step practical threshold separately. Do not select a successful environment retrospectively and call it confirmation. Structural tree/DAG expansion remains gated on a stable algorithmic increment.

## Common learner and candidate lifecycle

Use the existing pretrained foundation, PPO, two 27-coefficient local slots, and one 9-coefficient shared parent. The actor container is 89 bytes and the complete learner package cap is 262144 bytes. All task IDs, logical groups and bout boundaries are supplied equally. The state stream, episode trajectory and optimizer continue across decisions. Task switches do not reset the environment.

Warm up group 0 variants 0 and 1 for 1024 steps each; install their mean effective policy as the initial parent, bound to group 0 with compensated local residuals. Then train group 1 variants 0 and 1 for 512 steps each. Total warmup: 3072 real steps per stream. Each later candidate is formed only from two current resident generations, same group, different variants, with at least 512 actually fitted steps each. Average effective parameters, including the parent if bound. There is one transient candidate, no persistent candidate archive.

The operations are keep (current parent and current binding), replace (candidate bound to its source group), mean (equal mean of incumbent and candidate, bound to both groups), and none (zero and unbound). Compensate resident residuals to preserve effective policies before quantization. Measure post-codec drift. Discard rejected candidate coefficients after the decision; retain only bounded provenance, features and selected feedback.

After each operation, run three 512-step bouts without another storage operation. The first visits the minority group, variant `decision % 3`; the last two visit the dominant group with variants `decision % 3` and `(decision+1) % 3`. Dominance switches every eight blocks; `seed % 2` sets the first dominant group. The schedule is external experimental machinery and is not supplied to the selector. Its observations include past known group IDs only. The last two bouts become the next candidate sources, so there are no extra uncharged candidate-fitting trajectories. A fixed block has 1536 real steps. Its complete selected outcome is the only new value-model target; unselected labels are missing.

## Bounded selected-feedback model

Both models have four action-specific discounted ridge estimates, ridge 1, discount 0.97. The linear model has six pre-action features:

- Bias.
- Candidate-group frequency minus incumbent-covered frequency in the last eight bouts (wildcard covers both; none covers neither).
- Candidate minus incumbent group mean first-256 reward EWMA, divided by .02 and clipped to [-2,2]; missing evidence is zero. EWMA alpha .25 uses past natural bouts only.
- Difference of log1p candidate mean source fitted steps and parent source fitted steps, divided by log8193 and clipped to [-1,1].
- Parent age in decisions divided by 16 and clipped to [0,2].
- Fraction of the last eight bouts that caused an eviction.

The intercept model sees only bias. Both receive exactly the same selected targets, never an unchosen branch. All methods may maintain both shadow models to match the bounded bookkeeping; report actual state and compute. Models start anew in each deployment stream: no transfer of fitted statistics between seeds or from calibration to comparison.

The utility target is block reward per step minus measured decision-and-block CPU seconds per step (`lambda_cpu=1`). Measure features, inference, selection, installation and block execution before constructing this target. This target excludes its own subsequent fitting/serialization time to avoid circularity; report those costs separately in total measured CPU/wall. Windows process CPU has finite quantization. Do not interpret small timing differences as stable acceleration.

The fixed state includes six-task visit/first-return/eviction statistics, eight recent groups and eviction indicators, parent age/source-fit metadata, two ridge models, selector RNG and at most one pending decision. It belongs in every serialized learner package. Raw experiment traces and externally controlled simulator state are evidence, not memory accessible to the agent; distinguish them from deployment learner memory. No model leverage diagnostic is called a calibrated safety confidence bound.

## Stages and controls

Calibration: six unused seeds `[23101,23102,23103,23104,23105,23107]`, selected only by mapping (three old-family A, three B), two relations, 64 decisions, uniform random storage actions with propensity 1/4. Twelve streams, 1,216,512 real steps including warmup. Score selected predictions logged before each update; compare linear and intercept squared error on the same selected labels. Inspect action support and cost; no adaptive retuning from this result is part of this protocol.

Comparison: four further unused seeds `[24101,24107,24113,24117]` (two old-family A, two B), two relations, 48 decisions, ten methods. Eighty streams, 6,144,000 real steps including warmup. Learned methods use uniform exploration for the first 16 decisions, then epsilon .1 greedy; these two have matched exploration schedules. Random remains uniform. Keep/replace/mean/frequency/lower are deterministic controls, so their exploration costs differ and must be reported. The generalized lower rule compares candidate-bound versus incumbent-bound coefficients separately in each group, weighted by the last-eight group frequencies, and replaces only if all required surrogate lower estimates are valid and the weighted value is positive. It is a repeated-operation extension, not identical to the original one-write lower rule being independently confirmed.

Direct shared PPO keeps a wildcard shared parent and trains it online at .003 together with local residuals; it does not perform discrete retention operations. Independent keeps the parent zero and permanently unbound, training only the two local LRU slots. It reserves the same actor container, so it is a controlled same-layout baseline rather than a claim to the strongest possible allocation of all bytes. The complete-package cap is common; report actual rather than padded sizes.

Pilot: calibration one seed/one relation/four decisions, 9216 steps. Comparison one seed/one relation/four decisions/all ten methods, 92160 steps. Pilots validate plumbing and accounting only. Preserve failures and all consumed costs. No source fitting is shortened below 512 steps.

## Audit and scope

Snapshot the runner, called policy/environment/optimizer sources, config, and this protocol before running. Keep local instruction files and temporary material excluded. Audit chronological pre-action features, action propensity, single pending decision, non-overlapping horizons, selected-only update counts, fixed stats shapes, checkpoint cap, raw row totals, episode continuity and all real PPO updates. At minimum save initial, every eight-decision and final packages, and calculate the package size at each decision even when not retaining all checkpoint copies.

Recurrence recovery is a factual whole-policy measure: a previously seen task returns after its prior local generation was evicted. Its first-256 outcome is not a counterfactual recovery loss. Per-write beneficial/harmful classification and safety guarantees remain unproven. A feature predictor can fail while the broader online storage research question remains open; identify whether failure arises from feedback noise, missing observability, action support, horizon, model or rule.
