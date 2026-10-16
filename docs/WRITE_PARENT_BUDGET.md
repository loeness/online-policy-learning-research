# Parent group budget reuse

`online_rl.write_parent_budget` is a bounded helper for the two shared parent
matrices in `FlexibleSharedPolicy`. It does not change policy or grouper
classes and it does not use task names, observer families, evaluation returns,
or learned utility estimates.

The caller passes the current slot's numeric profile after the normal
`RewardAwareGroups.assign(..., commit=False)` attempt. The helper repeats that
noncommitting attempt and proceeds only when the grouper reports
`max_groups_reached` (the current capacity-full result; `capacity_full` is
accepted as an equivalent reason for compatible grouper implementations).
Insufficient evidence, ambiguous evidence, and ordinary group matches do not
trigger recycling.

## Reuse rule

The helper reads slot membership from `core.slot_groups` and profile rows by
integer slot index. It first prefers a parent with no resident policy slots.
If both parents have residents, it considers the least recently used parent
and requires the current slot plus another distinct, seen, ungrouped resident
slot to have compatible evidence. Compatibility is checked in a fresh
`RewardAwareGroups` instance in reward mode, so both movement dynamics and
reward regression must match. The current slot cannot serve as its own second
profile. If no pair passes, the call leaves the policy, grouper, profiles, and
budget counters unchanged.
Generation `-1` is accepted as the caller's vacant-slot sentinel; stale profile
rows for vacant slots cannot count as the required partner.

For a parent with residents, every slot using that parent is detached with
`preserve_effective=True`, which adds its shared matrix to its local matrix.
The selected shared matrix and its fixed grouper statistics are then cleared.
Statistics are cleared through a copied `state_dict()` and
`RewardAwareGroups.from_state_dict()`, preserving the grouper's fixed array
sizes and configuration. The current profile and, when required, one partner
profile are committed exactly once to the replacement grouper. Their local
matrices remain unchanged as they join the new group because its shared matrix
starts at zero.

`ParentBudgetManager` keeps three fixed two-element `uint64` arrays:
`last_used`, `touch_count`, and `version`. Call `touch(group, step)` when a
parent is used. Successful reuse increments that group's version and records a
touch at `step`; choosing between candidates uses least-recently-used step,
then group index as a stable tie-breaker. `state_dict()` and
`from_state_dict()` preserve this bounded state.

## Caller responsibilities

Finish the current PPO cycle and reset actor optimizer state **before** calling
`try_recycle`. The helper does not manage optimizer state or rollouts. On
success, use `affected_slots`, `affected_generations`, `removed_slots`,
`added_slots`, `cleared_shadow_slots`, and `committed_profile_slots` to update
caller-owned caches and seen markers. Individual profile arrays and slot
generations are retained by the helper. `seen_updates` reports the existing
per-slot profile-seen value for each affected slot; the helper does not clear
it because the individual profile evidence remains valid.

Parent version is separate from slot generation. The result returns both the
reused parent's previous and new version. Tag future anchor rows with
the parent version active when they were collected, and filter anchors by
parent identity/version when forming a later group comparison. Old rows remain
valid observations of their saved effective policies, but rows from an older
parent version must not be mixed into evidence for a new parent version. The
helper does not increment slot generations or rewrite anchor banks.

The result returns the replacement grouper, the current slot's assignment,
the reused group and its new version, and the affected slots. It is an LRU
capacity-recovery rule with an evidence-compatibility check, not an
utility-optimal organization algorithm. It performs no training or evaluation.
