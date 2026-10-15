# Bounded shared-write cache capacity diagnostic

## Scope

This is the next controlled step after the shared-write mechanism check. It
asks whether a persistent family correction plus a six-slot local cache helps
when twelve known contexts revisit a fixed, deterministic schedule. Context
IDs and family membership are deliberately visible to the outer cache manager
for this diagnostic. The learner receives only the existing four-value
observation, reward, and a route slot. This experiment does not test discovery
of unknown contexts, automatic organization, learned replacement, or exhaustive
global-capacity limits.

The 12 contexts are family A/B × three band variants × two local modes. The
first mode reproduces the existing C1 action map. The second keeps the same
family permutation, then swaps the alternate family-specific adjacent pair in
the selected band: A swaps (1, 2), B swaps (2, 3). Names such as `A_v1_m0`
encode family, band variant, and local mode. All maps are deterministic on the
same 9 × 9 task and retain the C1 reset, four-value observation, reward, info,
and 64-step episode interface.

## Fixed comparison

Each condition gets six local `3 × 3 × 3` corrections and the same shared
critic. `shared_cache` and `frozen_shared_cache` also get two global `3 × 3`
parents, one for each known family. On a cache miss, the outer manager selects
the first free slot or evicts the least recently used resident (slot index
breaks ties), zeros that slot's local correction, clears only that local
optimizer state, advances its generation, and leaves the shared parents
intact. Family assignment is given only to the two shared conditions. There
is no unbounded seen-context history in the cache: persisted bookkeeping is
limited to current resident keys, six LRU clocks, six generations, and one
clock counter.

The conditions are:

- `independent_cache`: local cache only, with no shared actor parameters.
- `independent_fast_cache`: the same independent cache with local learning
  rate 0.004, controlling for a larger effective update rate.
- `shared_cache`: local cache plus family parents trained throughout.
- `frozen_shared_cache`: same initialization and training through the first
  six mode-0 contexts, then frozen parents while local learning continues.

The ordered sequence is the six mode-0 contexts, the six mode-1 contexts,
then each six-context block once more. The pilot uses seed 839 and 1,024
training steps per bout. The formal schedule uses seeds 727, 743, and 757 with
4,096 steps per bout: 24 bouts and 98,304 training steps per condition/seed.
Each 1,024-step cycle uses 3 fresh PPO fit rollouts and 1 fresh holdout
rollout, each 256 steps. The fit/holdout split is restarted at each bout.
Local, shared, and critic learning rates are 0.003, 0.001, and 0.003;
the independent-fast control alone uses local rate 0.004. There are four
conditions and 1,179,648 total formal training interactions.

## Storage and evaluation

Actor state is encoded and decoded at every 1,024-step boundary. All
conditions use the same exact INT8 actor cap of 230 bytes. Six independent
local rows use 204 bytes; two shared parents use the remaining 26 bytes. A
seventh independent row would require 236 bytes under this codec, so the
shared arm is not a compression win. Actor Adam state is reset after each
quantize/install event in every condition; critic Adam state is retained.
The checkpoint plus its exact W0 foundation is measured against 256 KiB.

The current context is evaluated before and after every bout with paired
complete episodes (4 in the pilot, 8 in formal), without gradients. A context
uses the same episode reset/action plan at each visit and in every condition.
At the endpoint, all twelve contexts are evaluated from separate clones of
the identical final checkpoint and cache state. A resident target retains its
local correction; a nonresident target incurs an ordinary cold miss and uses
the assigned family parent plus a zeroed local slot, if shared parameters are
available. Endpoint cache changes happen only on those clones. Training and
evaluation steps are reported separately.

Every 1,024-step checkpoint contains the encoded actor, critic, optimizer,
resident keys, LRU clocks, generations, RNG states, condition, and W0 hash.
The run stores exact source snapshots and hashes, config, raw training steps,
episodes, evaluation episodes, cache events, and per-cycle byte measurements.
Checkpoints are stage-end snapshots; no in-flight environment or partial PPO
rollout is represented.

## Interpretation boundary

This diagnostic measures bounded reuse with oracle context IDs and family
groups. It does not establish an unknown-context router, automatic family
discovery, a learned eviction rule, or performance at global budget
exhaustion. Shared results must be interpreted with the stated 26-byte larger
actor than the six-local independent representation and with evaluation
interactions reported separately from the equal training interaction budget.
Formal launch is gated on review of the preceding S1 result.
