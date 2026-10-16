# Reward-aware write-decision environments

`online_rl.write_decision_envs` supplies a controlled environment set for
shared-write diagnostics. It reuses this repository's `C1Context`,
`C1NavigationEnv`, and `mapping_for_state` implementation from
`src/online_rl/c1_envs.py`; it introduces no new runtime dependency and does
not alter the existing C1 sources or experiment outputs.

## Tasks and hidden preferences

`TaskSpec(family, variant, preference)` accepts families `A` and `B`, variants
`0`, `1`, and `2`, and preferences `+1` or `-1`. The module provides all 12
combinations through `TASK_SPECS` and `task_specs()`. `task_name()` and
`task_from_name()` convert the canonical runner schedule labels `A0+`, `A0-`,
through `B2+`, `B2-` to and from those specs. Names are for an external runner's
schedule only; `make_env()` takes a `TaskSpec` directly.

The context, variant, and preference are not included in observations or
`info`. Both domains return the C1 four-coordinate observation and four-action
space. The `info` mapping contains only `goal_reached`, `episode_length`, and
`distance`; `goal_reached` describes the terminal geometry and is not inferred
from reward sign. `observer_counters()` remains an evaluator-only helper and is
not copied into `info`.

The reward returned at each step is `preference * base_reward`. Thus, for a
fixed domain, C1 context, reset seed, and action sequence, preference `+1` and
`-1` produce the same state and termination trajectory with opposite rewards.
`set_task(spec)` changes the hidden C1 context and preference immediately,
including during an episode, without changing its current position, goal,
episode length, or observer counters. This permits online boundaries to occur
inside an episode without exposing a task marker to a policy.

## Domains

`make_env(spec, domain="navigation")` uses the existing C1 reset, action
mapping, termination, and reward formula:

`base_reward = 0.25 * (previous Manhattan distance - current Manhattan distance) - 0.03 + 1.0 on reaching the goal`.

`make_env(spec, domain="detour")` keeps the same 9 by 9 observation and action
interfaces and the same `mapping_for_state(context, x)` action mapping. It adds
a fixed vertical wall at `x = 4` with the single opening `(4, 4)`. Actions that
would enter a wall cell leave the state unchanged while still consuming one
step. The two C1 reset endpoints lie on opposite outer edges, so they are
always outside the wall and connected through the opening. The shortest-path
distance used by `info` and potential shaping is computed on this fixed free
cell graph. The base reward is:

`-0.03 + 0.10 * (previous shortest-path distance - current shortest-path distance) + 1.0 on reaching the goal`.

The detour environment is a controlled cross-generator pressure case for
testing whether an online method handles changed transition geometry and
reward density. It is not a standard benchmark and does not support a
state-of-the-art claim. The fixed wall, single opening, known C1 state and
action interface, and task schedule make this a deliberately bounded
diagnostic rather than a general navigation result.

## Minimal use

```python
from online_rl.write_decision_envs import make_env, task_from_name

env = make_env(task_from_name("A1-"), domain="detour")
observation, info = env.reset(seed=17)
observation, reward, terminated, truncated, info = env.step(0)
```
