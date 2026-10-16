# Write Stream Interaction-Budget Baseline

## Purpose

This post-selection wrapper adds a stronger independent G0 control for the
write-stream comparison. It runs only `independent7_lru`, using the stream's
existing frozen W0, local learning rate of 0.004, router, seven-slot LRU cache,
int8 actor budget of 236 bytes, and full persistent package cap of 262,144
bytes. The wrapper does not change the stream or policy implementation.

The comparison matches training-plus-probe interactions. It does not match
PPO updates or compute: the baseline spends every interaction on training and
fits the fourth 256-row rollout in each 1,024-row cycle. Each run reports its
rollout count, minibatch updates, fit and holdout rows, decision CPU time, and
total process CPU time. Evaluation interactions are recorded separately from
the training-plus-probe budget.

## Interaction budgets

The known phase uses 2,144 training interactions per bout and 24 bouts, for
51,456 training interactions with zero paid probes. This matches the shared
six-slot stream's budget of 2,048 training interactions per bout plus 128
uniform-probe interactions for each of 18 slot allocations:

`2,144 × 24 = 2,048 × 24 + 128 × 18 = 51,456`.

The unknown phase uses 2,048 training interactions per bout and 24 bouts, for
49,152 training interactions with zero paid probes, matching the unknown
stream's training-plus-probe budget. These values are the defaults; a
`--bout-steps` override is recorded as a pilot budget.

## Training and accounting

The config and each run snapshot declare `budget_baseline: true`,
`probe_steps: 0`, and `all_rows_fit: true`. The stock agent constructor uses a
temporary 128-row evidence window internally because a zero-capacity window is
invalid; the externally recorded paid-probe budget remains zero. Generation
resets retain that internal window capacity.

The wrapper marks every fresh nonempty rollout as `fit` before passing it to
the stock PPO implementation. The fourth rollout is trained once from its
current behavior data, with its recorded behavior log probabilities. There is
no replay of old rollouts. G0 keeps its ordinary direct local-policy commit,
actor optimizer reset, and int8 serialization roundtrip every 1,024 rows.
Existing partial-rollout handling at route boundaries remains in force, so a
forced fragment shorter than the stock 32-row minimum is counted as dropped.
Fit, dropped, rollout, minibatch-update, and CPU totals are reported.

The wrapper's source and this protocol are included in each source snapshot;
the frozen cohort config and per-run config retain the comparison flags. The
selection file is required by the common cohort runner, but the wrapper forces
the cohort condition to `independent7_lru` regardless of which candidates the
selection lists. This is a post-selection baseline addition, not method tuning
or a state-of-the-art reproduction.

## Running

```powershell
.venv\Scripts\python.exe scripts\run_write_budget_baseline.py --phase known --selection outputs\write_decision\selection.json --output outputs\write_budget_baseline_known --workers 1
.venv\Scripts\python.exe scripts\run_write_budget_baseline.py --phase unknown --selection outputs\write_decision\selection.json --output outputs\write_budget_baseline_unknown --workers 1
```

The default seeds are 1031, 1033, and 1039. Use `--workers 2` for two
concurrent cohort workers, `--seeds` to select seeds, or `--bout-steps` for a
pilot. Existing nonempty output folders are preserved by the common stream
runner.
