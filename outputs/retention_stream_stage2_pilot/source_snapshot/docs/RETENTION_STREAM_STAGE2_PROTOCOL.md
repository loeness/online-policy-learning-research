# Natural eviction and retention writes: exploratory Stage 2

Configuration is frozen before new interaction runs. This is a controlled,
known-local-index mechanism study. The learner has a bounded context-to-local
cache but receives no true family labels for grouping or archive assignment.
Unknown boundaries, parent replacement, new structures and independent
confirmation are subsequent stages, not results of this protocol.

## Layout and schedule

Use the existing six positive navigation tasks. For each deployment seed,
permute family roles and variants. The first two variants of one family arrive
consecutively, then two of the other family, then both third variants. A seeded
permutation of the six contexts follows. Each bout lasts exactly 2048 learner
raw steps, regardless of goals, grouping, decisions or PPO update counts.
The same external sequence is used by every method and budget at that seed.
Context changes do not reset the live environment. Known boundaries terminate
GAE credit propagation for the old homogeneous segment; actual environment
termination and the learning-boundary flag are separately logged.

Low actor cap 140 B: shared 3 local + 2 parent (134 B), independent 4 local
(140 B). Wide cap 236 B: shared 6 + 2 (230 B), independent 7 (236 B).
All use LRU local eviction. Full learner package cap is 256 KiB, including
W0, checkpoint, generated actor sidecar, archive and pending candidate state.
This pairs parameter budgets; it does not show optimal allocation of one
unrestricted total-state budget. Equal full-state allocation/replay remains
required before an overall storage-efficiency claim.

## Methods and learning

- independent: complete data, local learning rate .004, no parent.
- direct_shared: complete data, local .003/shared .001, reward-aware groups.
- mean: local .003, parent frozen between mean consolidation writes.
- mean_reuse_gate: same writes as mean, conservative prior-use gate.
- retention_gate: same prior-use gate, plus post-local-deletion write gate.

Every 256-row homogeneous rollout is first finalized for stored advantages,
then used to test a candidate frozen before that rollout. All rows subsequently
enter PPO. No uniform probe or permanently withheld fourth rollout is used.
The shared parent is frozen in the three consolidation arms; direct_shared
learns its parent through normal PPO gradients.

Numeric movement/reward profiles are formed from the current 256 visible
transitions. Group binding preserves the already learned effective matrix.
An insufficient-support new window retains an earlier assignment for the
same still-resident known context generation; a reused slot starts unbound.
An abstaining fresh profile is not tagged as same-group archived evidence,
even when its resident acting assignment is retained.
For a newly allocated local generation, an available nonzero parent may be
added once after a complete local PPO update, for at most the first 1024 fitted
steps. This is an explicit additive prior injection, not an invariant
reparameterization. mean/direct_shared accept context-compatible priors;
mean_reuse_gate/retention_gate also test the frozen pre-update additive prior
against the current local policy on fresh data. The local PPO change occurs
after that diagnostic, so the surrogate is not a monotonic improvement
guarantee. Prior events and pre/post-injection changes are logged. Detection
requires real interaction, so initial pre-gradient recovery can be identical
across arms; no zero-shot recovery improvement is presumed.

The additional mean_reuse_gate arm isolates write decisions from prior-use
selection. All comparisons retain the simpler baselines even if they win.
The bounded 256-row PPO working buffer is empty at checkpoint boundaries;
the package accounting is persistent learner state, not peak process RAM.

## Consolidation and evidence

At most one pending 3x3 candidate is retained. It is the mean effective
correction of at least two live, sufficiently fitted locals assigned by visible
evidence to the same parent. It is tested on a later rollout. Expiry and parent
version mismatch cancel it. No parent-capacity replacement is allowed yet.

The bounded archive has 8 numeric prototypes and 24 reservoir rows each.
Prototype matching/replacement uses visible movement and reward evidence,
never task/family labels. Profile versions clear stale associations when an
archive cell is replaced; local eviction alone leaves archive evidence intact.
All methods allocate the same bounded archive, while its extra cost and
unused baseline capacity are explicitly reported rather than hidden.

The retention write gate compares candidate-only and old-parent-only policies
(all locals removed) on compatible archived prototypes, and requires fresh
same-group evidence. At least two supported historical profiles, positive
mean empirical lower score and bounded per-profile loss are required. Unknown
historical scores block a protected write rather than count as zero benefit.
The empirical error penalty uses episode blocks, not independent timesteps;
it is not a statistical safety bound or reliable long-horizon OPE estimate.
The row-weighted score uses centered cluster sums for unequal episode sizes.
Historical rows may have helped train the source policies and form candidates;
only the later fresh rollout is temporally held out from candidate formation.
Historical scores are retention diagnostics, not independent generalization tests.

Approved mean writes compensate every resident of the affected parent, so
their pre-codec effective matrices remain unchanged. Exact matrix invariance
and deployed-probability drift after int8 roundtrip are measured. All methods
clear actor optimizer moments on the same rollout cadence; critic treatment
is identical. Slot reuse clears local parameters/moments/profile and advances
the generation. Parent, archive and candidate state are serialized.
Pre-update critic return MSE and advantage moments are recorded by generation.
The existing shared critic ignores route identity; no context-conditioned critic
or unbiased advantage claim is made.

## Metrics, costs and progression

Primary metrics: complete learner-stream reward per raw step, and 512/1024
step recovery reward on actual naturally evicted recurrences. Record the first
256 pre-update window, first goal (with censoring), local evictions, actual
PPO rows/updates, proposals/accepts/prior bindings, process CPU/wall and full
physical bytes. Wide-budget recurrences without eviction are reported
separately, not mislabeled as recovery after forgetting.

The pilot uses a separate seed, 512 steps per bout, low budget and all five
arms (30720 raw steps total). It checks plumbing and state boundaries; do not
tune the method from its ranking. Main exploration is 3 seeds x 2 budgets x
5 arms x 24576 = 737280 new raw interactions, zero free probes/evaluations.
Unit tests are separate software validation, not experimental samples.
Original data, source hashes, failures, pilot and all arms are preserved.

Run parent replacement, unknown-boundary extension or new-seed confirmation
only after interpreting this stage. A negative result is reported and used to
identify a concrete mechanism failure, not silently replaced by a winning
configuration. No existing experiment is rewritten.
