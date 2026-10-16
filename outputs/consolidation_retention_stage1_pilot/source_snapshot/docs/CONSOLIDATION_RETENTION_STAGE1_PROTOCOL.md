# Consolidation, deletion and frozen reuse: exploratory Stage 1

Frozen before new interaction-generating runs. This stage tests whether bounded
online local corrections can leave useful knowledge in a persistent parent
after all source local state is removed. It does not tune PPO or test an
automatic organization/routing algorithm.

## Sources and candidates

Reuse seeds 1009/1013/1019 of the paid trial_rollback cohort at 32,768 training
steps. Each has A0+/A1+ trained for 8,192 steps, zero shared matrices, and no
A2+ training/profile. Each eligible source supplies at most 24 saved anchor
observations. Negative-objective source slots cannot enter extraction.

Candidates are zero/no extraction, coefficient mean over two sources and
three bands, equal-source activation-weighted ridge least squares, and a
seeded random matrix whose Frobenius norm matches the weighted candidate.
The weighted objective fits source correction outputs at bounded stored
features. Its relative ridge is 0.001, with the mean as prior for unsupported
directions. Advantage signs and target rewards are not proposal weights.

For an absolute common matrix C, compensate all destination-group members
and source slots so their pre-codec effective local-plus-parent matrices stay
unchanged. Observer forward passes across the public navigation coordinate
grid test raw logit and deployed probability preservation. These forward
passes provide no rewards or candidate-selection data. A pre-codec maximum
raw-logit deviation above 1e-5 stops the experiment. Int8 error is recorded
separately, not treated as exact algebraic preservation.

## Strong deletion boundary

Do not reuse the source learner as the target learner. Construct a fresh W0
agent and transfer only the extracted C into the chosen parent. All local
matrices are zero, not only the two source locals. The critic and RNG are fresh;
optimizer state, anchors, local profiles and group statistics are empty.
Metadata task names are fresh fixed protocol labels, not learned task memories.
Save a complete bounded checkpoint, restore it, verify the purge, and evaluate
that restored actor. No source observations, parameters, critic, moments or
history other than the retained common actor correction enter this capsule.
The offline W0 backbone/source basis is common to all arms.

All targets use one fresh local slot assigned to the same stored source
parent. A0+ measures exact recurrence; A2+ is an untrained related variant;
B2+ and A0- deliberately test wrong reuse. Their forced route is not evidence
that a deployment router can select or reject that parent. There is no target
PPO, probing, memory update or reward-based candidate selection during rollout.

## Costs, pairing and interpretation

Eight complete episodes per target/arm/source, capped by the 64-step
environment horizon. Reset seeds and per-step uniform action draws are paired
across candidate arms. All simulator steps are paid observer evaluation, never
free evidence for a learner. Three sources × four arms × four targets × eight
episodes × 64 steps bounds the new interaction cost by 24,576. Reused source
training is listed separately and is not charged again as newly executed PPO.

Statistical unit is source seed; episodes and branches are not independent
training seeds. A2+ is unseen during source learning but is used in exploratory
candidate evaluation, not independent method confirmation. Mean/random
comparison is not norm matched; weighted/random is. All arms use the same
230-byte actor container for mechanism ablation, so this test cannot by itself
establish capacity superiority over an independently allocated flat learner.
Full W0-plus-checkpoint physical bytes are recorded and capped at 256 KiB.

Preserve every outcome. If retention succeeds but related transfer does not,
report retention only. Prefer mean if weighting has no incremental evidence.
Any further online closed loop, tighter-capacity comparison or eight-seed
confirmation is a new stage with its own frozen configuration.
