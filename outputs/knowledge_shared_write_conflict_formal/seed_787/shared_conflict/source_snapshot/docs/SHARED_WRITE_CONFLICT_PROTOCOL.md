# Same-dynamics reward-conflict diagnostic

## Question and scope

This C1 diagnostic asks whether nearly identical observed action dynamics are
enough to justify writing a shared actor correction when the two routes have
opposing reward preferences. The plus and minus routes both use C1 A0
and have exactly the same observations, action-to-movement map, termination,
and episode limit. The minus route returns the negation of the C1 reward.
Reward mode and route name are absent from the observation. The actual signed
reward is supplied to PPO and recorded.

This test does not discover unknown route boundaries, test capacity eviction,
implement a policy-benefit gate, or establish a safety guarantee. The shared
group is chosen automatically from transition profiles only. The experiment
does not presume that shared learning will fail; results, including null and
positive results, are retained.

All deployment seeds load the same configured W0 foundation file and hash;
they are not separate pretraining runs or independent W0 initializations.

## Conditions and policy

Each condition starts from the same per-seed W0 and zero corrections:

| Condition | Actor state |
| --- | --- |
| independent_conflict | Six local 3 × 3 × 3 corrections; no shared matrices |
| shared_conflict | Six local corrections plus two 3 × 3 shared matrices; slots 0 and 1 use group 0 |
| frozen_conflict | Same shared representation; group 0 freezes after the plus bout while local learning continues |

All conditions count the same six local rows. Shared conditions retain both
shared matrices in storage; group 1 remains zero and unused. Actor payloads
use float32 with a common 1,024-byte cap. The W0 file plus each full checkpoint
must fit within 256 KiB. No int8 quantization or actor Adam reset is used.
The source foundation, critic, optimizer, group assignments, probe profiles,
grouping state, RNG states, counts, and source/config hashes are included in
agent-state checkpoints for evaluation/restoration. These are not exact
trajectory-continuation checkpoints.

The three conditions use fresh on-policy PPO data and the same local, shared,
and critic rates as S1: 0.003, 0.001, and 0.003. Each 1,024 training steps
uses three 256-step fit rollouts and one 256-step holdout rollout. There is no
PPO replay. The frozen arm is identical to the active arm through plus;
only then is its shared parameter frozen. Local parameters continue training.

## Probe and schedule

The fixed sequence is plus followed by minus. Each mode receives exactly
512 uniform-action probe steps before training. Probes make no actor, critic,
or optimizer update. All conditions and both modes reuse the same probe reset
and action seeds. Raw observations, actions, signed rewards, next observations,
termination flags, and step costs are saved. The profile builder uses only
observation, action, and next observation; rewards and route/mode labels are
not passed to DynamicsGrouper.

The plus profile is submitted first and the minus profile second. The shared
conditions require the grouper to assign both to group 0; the decision log
retains profile hashes, overlap/agreement scores, and reasons. A failed or
unresolved assignment stops that run rather than silently switching to an
oracle assignment.

Formal seeds are 773, 787, and 797, with 8,192 training steps per mode. The
pilot seed is 853, with 2,048 steps per mode. Initial, after-plus, and
after-minus evaluations run both modes for 16 complete episodes per mode
(pilot: 8). For a given seed and mode, reset/action episode seeds are identical
across all conditions and stages. Evaluation uses local random generators and
does not change PPO, Torch, or training-action RNG state.

Probe, training, and evaluation interactions are reported separately and as
one total. CPU time, wall time, peak RSS, per-checkpoint actor bytes, full
checkpoint bytes, and W0-plus-checkpoint bytes are recorded. Training logs
include every transition and policy probability; evaluation logs retain each
episode's actions and signed return. Wall and CPU timing start before policy
construction and include initialization, source snapshots, probes, training,
evaluation, and checkpoint writes. RSS sampling starts after policy
construction and covers source snapshots, probes, training, evaluation, and
checkpoint writes; it is a sampled process peak rather than a process-start
measurement.

## Interpretation

The primary comparison is shared active versus independent on the real signed
reward in minus, alongside retention on the original plus objective from
after-plus to after-minus. The frozen condition isolates continued shared
writing from retrieval of the shared correction already learned on plus.
Report per-seed paired differences; the seeds and episodes are not treated as
independent training replications for a formal significance claim.

For plus, goal completion is reported as success. For minus, the goal
completion flag is only environment termination metadata; it is not a measure
of task success. Compare its actual reversed reward return instead.

A matched transition profile is evidence about observed action dynamics only.
It is not evidence that a shared policy correction improves both reward
objectives. This experiment measures that distinction in this fixed C1 pair;
it does not establish a general rule for route identification, safe sharing,
or unknown environments.
