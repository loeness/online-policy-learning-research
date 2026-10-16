# Write Stream Parent Recycling Protocol

## Scope

This is a separate wrapper cohort for the write stream. Run it with
`scripts/run_write_recycle.py`; the existing `scripts/run_write_stream.py`
entry point keeps its fixed two-parent behavior. The wrapper snapshots its own
source, `write_parent_budget.py`, `write_cumulative_evidence.py`, and this
protocol. Its frozen and per-run configuration records both
`cumulative_profiles: true` and `parent_recycle: true` by default.

In the known-cache phase, assignment uses a fixed-size cumulative sufficient
statistic for the resident slot rather than only its recent probe window. It
accumulates each visible transition without retaining transition history or
task labels, and clears the statistic when the local slot is allocated to a
new generation. The unknown phase continues to use the router's numeric
profile. Set `cumulative_profiles: false` in the configured stream JSON to
disable cumulative statistics for a separate run.

The feature handles one narrow capacity case: a shared write policy has two
occupied parent groups and an otherwise eligible numeric profile cannot be
assigned because both groups are full. It does not search task names, observer
labels, returns, or evaluation results, and it does not estimate which group
organization is best. Reuse is a bounded capacity policy, not an automatic
organization or utility-optimality claim.

## When a parent can be reused

The wrapper first runs the existing evidence grouper. A recycle attempt is
possible only after it returns `max_groups_reached` or `capacity_full`, and
only when the current PPO cycle is empty. The helper uses the fixed-size
movement and reward profiles already held by the stream. If one parent has no
resident local slot, it can be reused directly. Otherwise, the current profile
and at least one other resident ungrouped slot must pass a temporary
reward-aware compatibility assignment. The current slot cannot serve as its
own partner.

When both parents have resident members, the manager selects the least-recently
used parent, with group index as a deterministic tie-break. It detaches that
parent's resident members while compensating their local matrices, clears the
selected parent's group statistics and shared matrix, then assigns the current
profile and compatible partner. The new parent starts with a zero shared
matrix. Profile arrays remain fixed-size; the manager stores only two-element
`uint64` arrays for last-use steps, touch counts, and parent versions (48
array bytes total).

The helper preserves the pre-recycle logits for every affected resident before
actor serialization. After a successful recycle, the wrapper clears actor
optimizer state and applies the stream's configured actor encode/decode
roundtrip. Critic state remains intact. No gradient update occurs during
recycling. Anchors for affected slots are cleared so future comparisons cannot
mix evidence from different parent versions. Detached slots may be considered
again by ordinary profile assignment at a later cycle boundary.

## Accounting and recovery

Each routed action touches the assigned shared group at most once per raw
interaction, including paid probes. Assignment rows retain the stream's
existing columns and put the recycle outcome, affected slots, and parent
versions in the `scores` JSON field. Run counters include successful recycle,
detached-slot, and joined-slot totals.

Use `--no-recycle` to run the same cumulative known-slot evidence and fixed
parent-capacity behavior. This provides a matched evidence comparison: the
only intended policy difference is whether a capacity-full assignment may
reuse a parent. In this mode the fixed-size manager is still saved, but its
touch, version, and recycle state does not change.

Checkpoints include the bounded manager and optional cumulative evidence
states. The recorded capacity is recomputed from the final serialized
checkpoint bytes and the foundation file:

`foundation file bytes + serialized complete learner checkpoint bytes <= 262,144`.

The separate actor payload remains under its configured actor-byte cap. A
checkpoint without manager state is still accepted by the wrapper's restore
path and starts with a fresh manager, which keeps older stream checkpoints
readable. Checkpoints restore learner state, not the live environment episode
or its position.

## Running

Use the same CLI and selection file as the fixed stream runner, but invoke the
wrapper entry point:

```powershell
.venv\Scripts\python.exe scripts\run_write_recycle.py --phase known --selection outputs\write_decision\selection.json --output outputs\write_recycle_known
.venv\Scripts\python.exe scripts\run_write_recycle.py --phase unknown --selection outputs\write_decision\selection.json --output outputs\write_recycle_unknown
.venv\Scripts\python.exe scripts\run_write_recycle.py --phase known --selection outputs\write_decision\selection.json --output outputs\write_fixed_parent_cumulative --no-recycle
```

The wrapper preserves the stream runner's one- or two-worker limit and its
existing controls for conditions, seeds, bout length, domain, local slots,
evaluation targets, and foundation selection. It does not launch training
unless its CLI is explicitly invoked.
