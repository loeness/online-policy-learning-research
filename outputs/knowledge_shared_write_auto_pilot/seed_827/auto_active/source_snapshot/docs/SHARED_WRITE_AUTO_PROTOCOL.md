# S1b: Known routes with family-free shared grouping

This stage tests whether bounded shared corrections can be assigned from short, agent-visible transition probes when route slots are known but family membership is not supplied to the learner. It follows S1a's active shared-write mechanism. It does not test discovery of unknown contexts, route boundaries, slot eviction, capacity replacement, or ambiguity-driven exploration.

## Conditions

Each seed and condition uses a fresh policy copied from the same W0 foundation and the same randomized context-to-slot permutation.

| Condition | Shared correction | Group assignment | After boot |
| --- | --- | --- | --- |
| `independent` | None | Every slot remains local | Local and critic PPO updates continue |
| `oracle_active` | Two groups | The context family selects a group | Shared, local, and critic PPO updates continue |
| `auto_active` | Two groups | A transition-profile grouper receives only the context's probe profile | Shared, local, and critic PPO updates continue |
| `auto_frozen` | Two groups | Same profile-only rule as `auto_active` | Shared matrices freeze after boot; local and critic PPO updates continue |

Only `oracle_active` uses the context family label to select a group. The automatic grouper is called with a numeric profile only. It does not receive the context name, route slot, or family label. An unresolved automatic assignment remains local. Each context is assigned at most once; this stage does not reassign a slot later.

## Probe and training schedule

The six known context routes are assigned a per-seed random permutation of the six policy slots. On the first encounter with each context, the runner collects exactly 512 transitions using uniform random actions. Probes make no policy, critic, or optimizer updates. The raw observation, action, reward, next observation, termination flags, and one-step interaction cost are recorded in `probe_steps.csv`. The probe cost is counted separately from PPO training and evaluation interactions.

Probe action and reset random streams are deterministic for a seed and context, so all four conditions receive the same probe plan. Profiles count visible one-cell movement by observed x band, commanded action, and resulting direction. The profile contains no rewards, context metadata, or target-policy outputs. A context's profile is submitted to the automatic grouper once, after its 512 probe steps.

For the pilot, seed 827 runs four seen bouts (`A0`, `B0`, `A1`, `B1`) at 2,048 training steps each, followed by 2,048 steps each on `A2` and `B2`. For the formal study, seeds 677, 691, and 709 run `A0`, `B0`, `A1`, `B1`, `A0`, `B0`, `A1`, `B1` at 4,096 training steps each, then `A2` and `B2` at 4,096 each.

The first two source bouts are `A0` and `B0`. After those bouts, and before the boot evaluation, the runner collects the 512-step `A2` and `B2` probes and makes their initial group decisions. Their boot behavior is therefore probe-mediated cold transfer: it uses visible target-context transitions but receives no target-context policy gradient before boot. `A1` and `B1` are probed immediately before their first training bouts. Each context is probed exactly once per run, for 3,072 total probe steps.

The `auto_active` and `auto_frozen` runs share the same initialization, probes, grouping decisions, training data, and optimizer settings through boot. The runner records a pre-boot audit comparing their boot actor payloads, critic parameters, assignments, and pre-boot action/reward sequence. The freeze is applied only after saving and evaluating the boot stage.

## Grouping rule

The grouper uses the fixed two-group defaults in `configs/shared_write_auto.json`: at least four overlapping supported action cells, agreement of at least 0.6 for a match, and agreement no greater than 0.3 to create a second group. A cell is supported after two observations when one movement direction accounts for at least 80 percent of observations. If evidence is insufficient, ambiguous, or at capacity, the slot remains ungrouped and uses its local correction.

For automatic conditions, the grouper's cumulative aggregate is saved after every decision. Decision logs include the assigned group, reason, overlap and agreement scores, and profile hash. Group assignment rows also retain an audit-only family label so the report can calculate pairwise co-clustering accuracy; that audit label is not passed to the grouper.

## Learning and evaluation

PPO reuses the S1a implementation and settings: 256-step rollouts, three fit rollouts and one holdout rollout in every 1,024 **training** steps, with the same GAE, clipping, entropy, and critic behavior. Probe steps do not count toward PPO rollout or holdout budgets. There is no replay. Local, shared, and critic learning rates are 0.003, 0.001, and 0.003. The independent arm uses the same local and critic rates as the automatic arms.

The runner evaluates all six contexts at boot (after two seen bouts), seen (after the full seen sequence), and adapted (after heldout training). Within each seed, every stage uses the same per-context evaluation seed plan. Formal evaluation uses 16 episodes for `A2` and `B2`, and 8 for each earlier context; pilot evaluation uses 8 and 4, respectively. Evaluation is read-only and does not change global random streams.

The actor is encoded as float32 with a 1,024-byte cap. The saved checkpoint plus W0 foundation must fit within 256 KiB. Checkpoints include local and shared matrices, slot assignments, critic and optimizer state, grouping aggregates, probe profile metadata, slot map, counts, RNG state, and the W0 hash. Each run snapshots the exact runner, S1a helpers/configuration, policy, storage codec, grouper, evaluator, S1b config, and this protocol.

## Interpretation

This is a controlled routing study: the experiment harness supplies the route slot for every context. The automatic condition removes family membership from group assignment but does not discover routes or context boundaries. Heldout results measure probe-mediated cold transfer followed by online adaptation, not zero-experience transfer. Report all seeds, conditions, evaluation stages, training/probe/evaluation interaction counts, grouping scores, pairwise assignment accuracy, and resource costs. Do not describe the automatic grouping result as general context discovery.
