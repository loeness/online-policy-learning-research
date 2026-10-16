# Write Decision Stream Protocol

## Purpose and scope

This runner follows the frozen Step 1 `selection.json` and compares online adaptation over a continuous stream of 12 reward-aware tasks. The current environments use the existing 9 by 9 C1 observation and action basis. `navigation` preserves the C1 transition generator; `detour` adds the fixed wall and opening described in `WRITE_DECISION_ENVIRONMENTS.md`. Detour is a cross-generator pressure test, not a standard benchmark.

The frozen config records a 12-task `base_sequence` and the full 24-bout `sequence`: the ordered tasks followed by their reverse. Each bout contains 2,048 training interactions. The seeds are 1031, 1033, and 1039. Task changes call `env.set_task` while preserving the current position, goal, and episode. The runner resets the environment only after a natural termination or truncation. An episode can therefore cross bout boundaries and contain steps observed under multiple tasks.

## Conditions and selection

The runner reads the two proposed conditions from a Step 1 `write_decision_selection_v1` file, then includes `reward_shared` and `independent_fast`. It also includes `independent6` as a same-capacity alias control, plus `independent7_lru`, `independent7_random`, and `independent7_lfu`. `--conditions selected` expands to the two selected Step 1 conditions. The default run evaluates the selected pair, both required baselines, and all three seven-slot eviction variants.

Shared-write policies use at most six local slots and two reward-aware shared groups by default. Independent controls use six or seven local slots with no shared groups. Every actor commit is scored and installed after an int8 encode/decode round trip. The shared six-slot actor and independent seven-slot actor each have a 236-byte payload ceiling; the independent six-slot actor is 204 bytes and remains under the same 236-byte cap. Every saved learner checkpoint is checked against the combined package budget:

`foundation file bytes + serialized complete learner checkpoint bytes <= 262,144`.

The checkpoint records the actual foundation size and hash. RSS is reported separately from serialized capacity.

## Known-cache phase

The known-cache harness uses the scheduled task name only to look up a bounded resident slot. It is an LRU cache by default. An independent seven-slot condition can use LRU, random, or LFU eviction. Cache contents and evictions are recorded in checkpoints and observer ledgers. The group assigner receives only numeric movement counts and reward regression sufficient statistics; it never receives task names.

The first allocation of a cache slot triggers 128 paid uniform actions on the continuous main environment. Those interactions count as probes and are additional to the bout's 2,048 training steps. The probe profile is committed at most once for a slot generation. If the numeric evidence is insufficient, the runner retries with a new cumulative profile only after additional evidence arrives; a failed profile is never committed repeatedly. Later group changes are deferred to a cycle boundary so the cycle baseline records the group that is actually installed.

## Unknown-boundary phase

The unknown phase calls `WriteDecisionRouter.route(obs)` before every action and `observe(obs, action, reward, next_obs, terminated)` after every transition. It never passes the scheduled task name, `TaskSpec`, family, variant, preference, or a switch notification into the router, grouper, policy, PPO update, or checkpoint routing state. Observer labels appear only in separate CSV/summary fields.

When the router returns an unknown route, behavior probabilities are the mixture `0.5 * base_policy + 0.5 * uniform`. The ledger stores the actual mixed probabilities and their action log probability. Unknown-route rows are excluded from actor PPO through the negative route id; critic values can still be fitted. Unknown mode has no paid uniform probe.

Router allocation, eviction, or route change first flushes the old homogeneous rollout and cycle. Natural episode ends do not clear router history or force a flush; episodes are not task boundaries. A partial fit with at least 32 rows can be fitted; shorter partial rollouts are dropped and counted. A partial cycle without fresh holdout evidence skips its gate and shared promotion, retaining the effective update as a local correction. A boundary-triggering transition is logged but excluded from actor PPO. Only after the flush does the runner reset the allocated local actor, clear its local Adam state and anchors, and increment its generation. A router profile may persist across a local-head reset, but old gradients and anchors cannot cross generations.

Each uninterrupted 1,024-step cycle uses three 256-step PPO fit rollouts followed by one 256-step holdout rollout. A cycle belongs to one current slot and generation. The Step 1 finish-cycle hook handles the selected decision rules. Unknown-only cycles may update the critic while leaving actor rows masked.

## Group assignment and limits

Profiles are learned from visible transitions and signed rewards. Reward comparison is support-aware: coefficients are compared only on features with evidence from both profiles. The two shared groups are not preassigned by family. The assigner can abstain when evidence is insufficient, ambiguous, opposed to existing groups, or would exceed the two-group limit. The fixed protocol does not recycle a full group. Independent controls have zero shared groups and skip group assignment. A slot with prior local PPO updates preserves its effective logits when it joins a shared group. A never-fitted slot can inherit the group prior, recorded as `cold_prior_binding`; this is a heuristic transfer and does not establish that the prior is beneficial. `summary.json` reports both binding types, group abstention counts, the number of max-group abstentions, aggregate unassigned-step frequency, and per-observer-task grouped/unassigned coverage. It also reports unassigned steps and task coverage after the shared groups fill. These are post-hoc observer measurements and do not affect assignment.

The router's current reward feature is Manhattan-distance improvement, a constant, and termination. That model is diagnostic for the known C1 navigation interface and can be misspecified by detour's wall-aware shaping reward. No general POMDP or standard-benchmark claim is made.

## Endpoint evaluation and artifacts

Final evaluation clones the live policy and cache/router state. It runs four complete episodes per target across six or twelve targets. Known-cache evaluation uses a cloned ordinary lookup. Unknown evaluation clears only the clone's recent router history at evaluation start, then interacts normally to identify routes across episodes. Evaluation never updates the live policy, optimizer, router, or group state. Return is the signed reward sum. A goal counts as positive success only for a positive-preference task; negative-reward goals are never counted as success. In the training episode ledger, `episode_task_observers` lists every scheduled task that contributed a step, and `mixed_task_episode` flags episodes spanning a task change. Mixed-task episodes have no single signed-success label and are excluded from positive-goal episode denominators; reward and step totals remain attributed to each interaction's observer task.

Each run snapshots the runner, Step 1 helpers, environment/evidence/policy/router sources, configuration, protocol, and selection file with hashes. It writes `steps.csv` for train/probe/evaluation interactions, `episodes.csv`, `routing.csv` for unknown router events, `profile_assignments.csv`, `decisions.jsonl`, `evaluations.csv`, `capacity.json`, checkpoints, and `summary.json`. The capacity ledger includes actor bytes, foundation bytes, full serialized checkpoint bytes, bounded array bytes, optimizer tensor bytes, and process RSS.

Run with:

```powershell
.venv\Scripts\python.exe scripts\run_write_stream.py --phase known --selection outputs\write_decision\selection.json --output outputs\write_stream_known
.venv\Scripts\python.exe scripts\run_write_stream.py --phase unknown --selection outputs\write_decision\selection.json --output outputs\write_stream_unknown
```

`--workers` accepts one or two. `--conditions`, `--seeds`, `--bout-steps`, `--domain navigation|detour`, and `--local-slots 6|7` support bounded pilot runs. Existing nonempty outputs are preserved and cause an error.

`--foundation PATH` selects another W0 checkpoint while keeping the base dataset. The resolved path and hash are recorded in the frozen configuration, source manifest, and checkpoint; restore uses that saved foundation unless an explicit base configuration is supplied.
