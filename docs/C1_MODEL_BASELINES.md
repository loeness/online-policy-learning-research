# C1 Model Based Baselines

These controllers give the frozen source actor three history-only ways to
adapt its command distribution to C1. The actor still describes the physical
direction it wants. A controller estimates what each command currently does,
then returns probabilities over commands. `C1ModelController` and the source
response loader are in `src/online_rl/c1_strong_models.py`.

## Common information and initialization

Every controller receives only the normalized observation, the command sent,
and the resulting observation. The optional `reward` argument is ignored.
Controller state contains no context name, context ID, schedule position,
switch time, or call to `mapping_for_state`. Positions are rounded to the
visible 9 by 9 grid, and the current x coordinate selects one of three bands.
Observed movement is treated as physical direction in up, right, down, left
order.

By default, `load_source_prior()` reads recorded source transitions from
`data/pretrain_source.npz` and counts each moving response by command and
direction. Stationary moves at walls are omitted because their direction is
censored. The base actor's probabilities stay in physical-direction space;
the estimated command response model maps them to the C1 command space. The
count model begins with one pseudo-observation per command, distributed by
the learned source response frequencies. Assignment models begin with 0.999
prior probability on the source-learned permutation and spread the remaining
0.001 uniformly over the other 23 permutations. This keeps the initial policy
close to the source policy while allowing any of the 24 bijections to be
learned from observed transitions.

Pass a 4 by 4 `source_prior` matrix to reproduce a run with an explicit prior.
Rows are commands, columns are observed physical directions. The controller
normalizes each nonempty row, so callers can pass counts or probabilities.
`action_probabilities(obs, base_probs)` returns a finite normalized vector of
four command probabilities.

## Controller variants

| Kind | Model | Memory and adaptation |
| --- | --- | --- |
| `window_counts` | Separate recent categorical direction counts for each band and command. Each command row is estimated independently; the four rows do not have to form a permutation. | Retains at most `window` transitions. Blocked observations stay in the event window but add no direction count. |
| `window_assignment` | Bayesian posterior over all 24 command-to-direction permutations, separately for each band. A predicted direction is clipped at the visible grid boundary before likelihood evaluation, so a blocked transition can constrain the posterior. | Recomputes from at most `window` transitions. Evidence from an older context naturally leaves as events leave the window. |
| `adaptive_assignment` | The same 24-permutation posterior with recency-weighted evidence. It flags a transition whose probability is below `change_threshold` after the band posterior is confident, then drops older evidence for that band and learns from the new response. | Recent evidence is discounted by `adaptive_decay`; a detected change clears only that band's working evidence. It mixes in up to `exploration_rate` probability over commands according to expected information gain while assignment entropy remains high. |

The permutation models encode a structural assumption: within a band,
commands select physical directions bijectively. The assignment prior is
derived from source data, and evidence can move its probability to any of the
other permutations. These models do not use a hidden C1 mapping table. The
independent count model makes a weaker assumption about row independence but
can represent noisy or non-bijective command responses.

## Accounting

These are statistical response models, not policy-gradient training. Their
`parameter_update_count` is always zero; `statistical_update_count` records
observed transitions. A runner should report gradient updates separately and
should apply its common external exploration rule once to the returned
command probabilities. `memory_summary()` reports controller-owned arrays,
Python event storage, and the bounded event limit. `features()` returns eight
bounded diagnostics: four command response confidences followed by four
normalized response entropies. `snapshot()`, `from_snapshot()`, and
`clear_history()` preserve or discard only the controller's online evidence;
the source prior remains available after a clear.

The source counts and visible grid geometry are assumptions shared by the
baselines. `window_assignment` can be slow to identify an untried command;
`adaptive_assignment` uses small information-gain exploration to reduce that
ambiguity. Neither makes a reward-based claim about an unobserved context, and
the confidence value describes response-model certainty rather than policy
quality. These C1 models do not address C2.
