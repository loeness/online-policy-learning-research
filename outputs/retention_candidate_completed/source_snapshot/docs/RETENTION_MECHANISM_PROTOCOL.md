# Next validation: prediction decomposition and action/horizon diagnosis

Frozen before new RL runs on 2026-10-19. This is the first two stages of the next validation plan. It is exploratory diagnosis, not a new successful selector claim. Existing raw data, protocols, snapshots and negative results remain unchanged.

## Stage 0: existing selected-feedback logs

Use `outputs/retention_value_calibration_main`: six seeds, two relations, 64 uniformly randomized decisions per stream, propensity 1/4. Replay chronological predict-before-update discounted ridge models, ridge 1 and discount .97. Score the final 32 labels per stream. Compare intercept and six-feature models on (a) reward/step, (b) measured decision-and-block CPU/step and (c) the original reward-minus-CPU utility. Retain the original joint-target result. Separate model residual errors and their covariance when decomposing squared utility error.

Inspect leave-one-nonbias-feature-out, recovery-only and age-only models, plus action counts, propensities and discounted weights. These are exploratory analyses of already inspected data. They cannot serve as independent confirmation of a newly selected feature set. Sufficient-statistic discount weight is not importance-weight effective sample size. Comparison logs may supply descriptive action coverage but cannot become four-action outcome labels. Any one-block off-policy estimate is restricted to the logging-state distribution and its stated horizon, not a complete changed deployment policy.

No new environment calls occur in Stage 0. Save exact input/code hashes and complete per-stream/per-seed results. Bootstrap at seed level; do not pool labels as independent trials.

## Stage 1 questions

1. Does the available four-operation set produce reproducible action differences when stored incumbent and candidate source exposure is matched?
2. Do useful action differences appear after the current 1536-step target, at 6144 or 12288 steps?
3. Does choosing an operation using one stochastic continuation retain an advantage when evaluated on another continuation from the same prefix?

These questions are about available action differences, horizon and stochastic consistency. Four actions around one constructed candidate do not constitute a global candidate-quality ceiling. They do not prove that a deployable past-only selector can identify the best action. A positive result guides the next candidate/selector experiment; a weak result redirects attention to candidate construction, representation or decision-space design rather than automatically increasing selector complexity.

## Learner and prefix

Use the unchanged W0, actor/source basis and common PPO harness. There are two local 27-coefficient slots, one 9-coefficient shared parent, an 89-byte int8 actor container and a 262144-byte complete learner package ceiling. Local/critic learning rates are .003 and shared PPO rate is zero. Local PPO and critic updates use all actual rows in 256-step rollouts. The inherited actor-Adam clearing after codec round trips remains; critic state and update counters continue. This is the common controlled harness, not a new external PPO reproduction.

For each seed, relation and history, fit group-0 variants 0 and 1 for 1024 steps each. Form the incumbent mean effective parent, bind it to group 0 with resident compensation, encode and clear actor moments as in the existing harness. Then execute the existing eight-bout history template, with the first six bouts lasting 512 steps each and the last two group-1 source bouts lasting 1024 steps each. Prefix total: 7168. The stored incumbent sources and current candidate sources therefore each have 1024 fitted steps. Equal exposure is not an equal-quality guarantee; the old/new source roles, age, inheritance and control content remain measured factors.

Here, matched source exposure means the fitted rows in the source matrices **at the time those matrices are stored or proposed**. The incumbent's sources are captured immediately after bootstrap; later history updates to those tasks do not rewrite that frozen parent. It does not mean equal total lifetime interactions with each source task. The tail candidate must come from fresh resident generations with 1024 fitted rows each after natural LRU eviction; verify this rather than infer it from bout length. Record both source-time generation fit and later per-task cumulative exposure. If this condition fails, the matched-source claim fails and must be corrected before the main run.

The current candidate is the mean of the two resident effective tensors over both tasks and all three bands. These are six constituent 3x3 matrices from **two tasks**, not six separately stored policies. Retain one transient candidate only. Verify task/slot/generation/source-fit provenance. Capture the full learner and simulator as the common prefix; snapshots and source matrices used for observer auditing are inaccessible to a deployed selector.

## Actions, future and stochastic replication

Use generic current operations: keep preserves the current binding; replace binds the candidate to its source group; mean averages into a wildcard parent; none clears/unbinds the parent. Measure effective-coefficient and current-resident behavior preservation before codec and drift after codec. Keep must avoid an additional install/quantization. Other operations use compensated installation. No further shared writes occur in a branch. Parent coefficients remain fixed while local PPO, critic learning and natural LRU eviction continue.

Each future contains eight blocks. A block has three 512-step bouts: minority group variant `k % 3`, dominant group variant `k % 3`, dominant group variant `(k+1) % 3`. The dominant group is 0 in old-major and 1 in new-major throughout that diagnostic future. This yields 12288 steps and deliberately measures a single intervention under a fixed subsequent storage rule. It is not the whole-policy value of repeated adaptive decisions. Episode state is continuous across task switches; resets occur only at environment episode termination/truncation.

For every prefix and future, run two stochastic replicas (0 and 1), all four actions. First restore and verify the exact common-prefix learner/simulator semantic hash. Then apply a documented deterministic future-seed transformation: `40000000 + 1000 * original_seed + replica`. Set learner episode-reset seed source and action/behavior, Python, NumPy and Torch RNGs to the declared streams, without resetting the current physical environment/episode. Record pre-transform restore and post-transform hashes and RNG metadata. All actions within a replica share the future stochastic seed; replicas use distinct seeds conditional on the same prefix. They are not independent deployment seeds. Shuffle action execution order with a separate observer RNG derived from the fixed execution-order seed and case identity, never from reward.

The navigation simulator draws randomness from its private `env.np_random` generator. Global NumPy/Python/Torch and agent generator reseeding must leave that private state and the current episode unchanged, verified by before/after simulator hashes. Later ordinary episode resets use the transformed learner seed; no extra reset is inserted. Action divergence can subsequently change reset timing, so shared random streams do not imply identical realized trajectories.

Measure cumulative outcomes at 1536, 6144 and 12288 steps within this one continuation. These horizons share data and cannot be charged or counted as independent runs. Milestones align to rollout boundaries. Record reward/step, installation CPU, cumulative simulation CPU and utility separately; primary scientific comparisons use reward/step. Setup, installation, simulation and serialization timers must not overlap. Snapshot/restore costs and total campaign CPU/wall remain separately reported. Parallel campaign timing is the actual scheduling result, not an isolated acceleration benchmark.

## Grid, resources and accounting

Main seeds: 25101, 25102, 25103, 25104, 25107, 25108. Selection uses only the existing deterministic family map and parity: three old-family A, three B; three even, three odd. No rewards are used. Main has two relations, two histories, two futures, two replicas and four actions: 24 prefixes and 384 branches. Prefix cost is 172032 steps; branch cost 4718592; total **4890624** actual interactions. With 256 rows per rollout this is 19104 rollouts; actual minibatch updates are recorded from the optimizer.

Pilot: seed 25089, dynamics, old-major history and future, two replicas, four actions. Four future blocks yield nested horizons 1536 and 6144. Prefix 7168 plus branches 49152 gives **56320** actual interactions. The pilot checks restoration, actions, times, source exposure, capacity, hashes and ledger consistency only. Do not change algorithm/settings based on pilot reward rankings. Preserve every failed attempt and its counted physical interactions; use new output directories for reruns.

CPU is the installed Torch backend. Limit to two case workers and one Torch thread each; use fewer if live available memory requires it. Each case executes its branches serially. Exact worker count and parent/child CPU are logged. Simulator and observer archive bytes are external to the learner, not a free candidate bank. All serialized deployment-state bytes, W0 and actor sidecar count toward the common package ceiling.

## Analysis, gates and next steps

Primary horizon is 12288. Report all horizons, relations, histories, futures, actions, replicas, ties and failures. Estimate replace-minus-keep, mean-minus-keep, none-minus-keep, and the future-mixture interaction of replace-minus-keep. Average histories/replicas within seed before descriptive seed bootstrap, preserving old-family strata (three A/three B), 10000 resamples, seed 2026101902. Six seeds are exploratory; do not infer precision from raw transition count.

For cross-replica action diagnosis, select the best operation on replica 0 and evaluate that choice on replica 1, then swap. Compare with keep and with fixed replace (chosen from earlier results before these runs), and report argmax agreement and margins. Ties prefer keep, then declared action order. This choice sees conditional future outcomes as an observer and is not a deployment policy. Independent stochastic evaluation reduces noisy-max optimism but does not create a global oracle bound or an independent-prefix sample. If a naive same-sample maximum is displayed, label it optimistic and secondary.

The prior +.002 practical reference applies to **reward/step**, not automatically to CPU-adjusted utility or MSE. Direction, effect magnitude and intervals are all reported. An interval spanning useful and negligible effects is inconclusive, not proof of no action value. If reproducible action space is weak under these specified candidates, prioritize candidate/representation diagnostics; if differences are present, next isolate relative-action prediction and horizon. Only then compare candidate generation versus organizer in a small factorial and confirm a selected-only repeated policy on unused seeds. Current average candidate is retained as a baseline. Candidate distillation/extra fitting and new statistics must be budgeted, and independent confirmation outcomes cannot train or tune them.

No dynamic tree/DAG, hidden-task inference, safety guarantee, whole-stream off-policy estimate or improved organizer is claimed from Stage 0/1. Any later cohort requires its own frozen configuration, held-out seeds, cost ledger and evidence gate. Root will record the gate decision after these results; new methods are not silently fitted from observer branches.

## Artifacts and independent audit

Whitelist snapshots include called sources, config and this protocol; machine-local instructions and temporary material remain excluded. Preserve prefix, post-action and final learner/simulator states, actor sidecars, raw steps/episodes/rollouts, bout records, source-fit provenance, explicit RNG transformations and action order. Audit raw row/reward counts and horizon sums, PPO updates, parent invariance, restores, episode continuity, actor/checkpoint consistency, source hashes and physical package bytes. Reporting and audits make zero environment calls. Original data are never rewritten to satisfy an audit.
