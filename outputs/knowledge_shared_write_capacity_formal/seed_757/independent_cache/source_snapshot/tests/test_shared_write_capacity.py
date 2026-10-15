"""Controlled map and Gymnasium-interface tests for the capacity diagnostic."""

from __future__ import annotations

import sys
import copy
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from online_rl.c1_envs import C1Context, C1NavigationEnv, C1_CONTEXTS, mapping_for_state  # noqa: E402
from online_rl.shared_write_capacity_env import (  # noqa: E402
    CAPACITY_CONTEXTS,
    CapacityContext,
    SharedWriteCapacityEnv,
    mapping_for_capacity_state,
)


def _profile(spec: CapacityContext) -> tuple[tuple[int, ...], ...]:
    return tuple(mapping_for_capacity_state(spec, x) for x in range(9))


def test_registry_has_twelve_distinct_transition_profiles() -> None:
    assert len(CAPACITY_CONTEXTS) == 12
    assert set(CAPACITY_CONTEXTS) == {
        f"{family}_v{variant}_m{mode}"
        for family in ("A", "B")
        for variant in range(3)
        for mode in range(2)
    }
    assert len({_profile(spec) for spec in CAPACITY_CONTEXTS.values()}) == 12


@pytest.mark.parametrize("name,base_name", [
    (f"{family}_v{variant}_m0", f"{family}{variant}")
    for family in ("A", "B")
    for variant in range(3)
])
def test_mode_zero_matches_original_c1_mapping(name: str, base_name: str) -> None:
    spec = CAPACITY_CONTEXTS[name]
    original = C1_CONTEXTS[base_name]
    assert all(
        mapping_for_capacity_state(spec, x) == mapping_for_state(original, x)
        for x in range(9)
    )


@pytest.mark.parametrize("family,alternate_pair", [("A", (1, 2)), ("B", (2, 3))])
def test_mode_one_swaps_alternate_pair_after_family_permutation(
    family: str, alternate_pair: tuple[int, int]
) -> None:
    left, right = alternate_pair
    for variant in range(3):
        mode0 = CAPACITY_CONTEXTS[f"{family}_v{variant}_m0"]
        mode1 = CAPACITY_CONTEXTS[f"{family}_v{variant}_m1"]
        band = set(range(variant * 3, variant * 3 + 3))
        for x in range(9):
            if x not in band:
                assert mapping_for_capacity_state(mode0, x) == mapping_for_capacity_state(mode1, x)
                continue
            # Mode one replaces the mode-zero local swap. It does not compose
            # two local swaps; start from the family's map outside the band.
            base = (1, 0, 2, 3) if family == "A" else (3, 1, 2, 0)
            expected = list(base)
            expected = [right if value == left else left if value == right else value for value in expected]
            assert mapping_for_capacity_state(mode1, x) == tuple(expected)


@pytest.mark.parametrize("name", list(CAPACITY_CONTEXTS))
def test_environment_preserves_c1_observation_reward_info_and_episode_contract(name: str) -> None:
    context = CAPACITY_CONTEXTS[name]
    capacity_env = SharedWriteCapacityEnv(context)
    base_spec = C1Context(context.family, context.variant)
    base_env = C1NavigationEnv(base_spec)
    try:
        assert capacity_env.observation_space.shape == base_env.observation_space.shape == (4,)
        assert capacity_env.action_space.n == base_env.action_space.n == 4
        assert capacity_env.grid_size == base_env.grid_size == 9
        assert capacity_env.max_steps == base_env.max_steps == 64

        observation, info = capacity_env.reset(seed=177)
        assert observation.shape == (4,)
        assert observation.dtype == np.float32
        assert set(info) == {"success", "episode_length", "distance"}
        assert not ("family" in info or "variant" in info or "local_mode" in info)

        # Mode zero must replay the exact original environment stream. Mode one
        # shares its reset and can differ only after a state in the active band.
        base_observation, base_info = base_env.reset(seed=177)
        assert np.array_equal(observation, base_observation)
        assert info == base_info
        actions = [0, 1, 2, 3] * 16
        for action in actions:
            result = capacity_env.step(action)
            assert len(result) == 5
            next_observation, reward, terminated, truncated, info = result
            assert next_observation.shape == (4,)
            assert next_observation.dtype == np.float32
            assert np.isfinite(reward)
            assert isinstance(terminated, bool) and isinstance(truncated, bool)
            assert set(info) == {"success", "episode_length", "distance"}
            if terminated or truncated:
                break
    finally:
        capacity_env.close()
        base_env.close()


@pytest.mark.parametrize("name", [f"{family}_v{variant}_m0" for family in ("A", "B") for variant in range(3)])
def test_mode_zero_replays_original_step_stream(name: str) -> None:
    context = CAPACITY_CONTEXTS[name]
    base = C1_CONTEXTS[f"{context.family}{context.variant}"]
    capacity_env, original_env = SharedWriteCapacityEnv(context), C1NavigationEnv(base)
    try:
        left, _ = capacity_env.reset(seed=991)
        right, _ = original_env.reset(seed=991)
        assert np.array_equal(left, right)
        for action in (0, 3, 2, 1) * 12:
            a = capacity_env.step(action)
            b = original_env.step(action)
            assert np.array_equal(a[0], b[0])
            assert a[1:] == b[1:]
            if a[2] or a[3]:
                break
    finally:
        capacity_env.close()
        original_env.close()


def _agent(condition="shared_cache", seed=823):
    import torch
    from scripts.run_shared_write_capacity import make_capacity_agent, CONFIG_PATH
    torch.set_num_threads(1)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    agent, base = make_capacity_agent(condition, seed, config)
    return agent, base, config


def test_full_schedule_eviction_is_bounded_and_preserves_shared_parameters():
    from scripts.run_shared_write_capacity import CapacityCache
    import torch
    agent, _, config = _agent()
    with torch.no_grad():
        for p in agent.core.shared_parameters():
            p.fill_(0.25)
    parents = agent.core.read_shared_matrix().copy()
    previous_local = agent.core._nodes["1"].A
    agent.optimizer.state[previous_local] = {"sentinel": True}
    events = []
    for name in config["bout_sequence"]:
        event = agent.cache.ensure_resident(agent.core, agent.condition, name, optimizer=agent.optimizer)
        events.append(event)
        assert len(agent.cache.resident_by_context) <= 6
        assert np.count_nonzero(agent.core.read_local_matrix()[event["slot"]]) == 0
        np.testing.assert_array_equal(agent.core.read_shared_matrix(), parents)
        assert CapacityCache.from_state_dict(agent.cache.state_dict()).state_dict() == agent.cache.state_dict()
    assert all(e["action"] == "miss" for e in events)
    assert sum(e["evicted_context"] is not None for e in events) == 18
    assert [e["slot"] for e in events] == list(range(6)) * 4
    assert previous_local not in agent.optimizer.state
    assert agent.cache.generations == [4] * 6
    broken = agent.cache.state_dict()
    broken["resident_by_context"]["not_a_context"] = 0
    with pytest.raises(ValueError):
        CapacityCache.from_state_dict(broken)


def test_fast_independent_control_and_quantization_reset_scope():
    from scripts.run_shared_write_capacity import _quantize_and_install
    agent, _, config = _agent("independent_fast_cache")
    assert agent.core.shared_groups == 0
    assert next(g["lr"] for g in agent.optimizer.param_groups if g["name"] == "local") == 0.004
    local = agent.core._nodes["1"].A
    critic = next(g["params"][0] for g in agent.optimizer.param_groups if g["name"] == "critic")
    agent.optimizer.state[local] = {"sentinel": True}
    agent.optimizer.state[critic] = {"sentinel": True}
    record = _quantize_and_install(agent, config)
    assert record["actor_bytes"] == 204
    assert local not in agent.optimizer.state
    assert critic in agent.optimizer.state
    assert record["actor_optimizer_states_reset"] == 1


def test_capacity_checkpoint_and_paired_evaluation_preserve_policy_and_cache(tmp_path):
    import random
    import torch
    from scripts.run_shared_write_capacity import (
        _quantize_and_install, _evaluation_slot, evaluate_capacity_context,
        save_checkpoint, restore_capacity_run,
    )
    agent, base, config = _agent()
    for name in config["contexts"][:6]:
        event = agent.cache.ensure_resident(agent.core, agent.condition, name)
        agent.slot_generation[event["slot"]] = event["generation"]
    with torch.no_grad():
        agent.core._nodes["1"].A.normal_(0, .1)
        for p in agent.core.shared_parameters():
            p.normal_(0, .1)
    _quantize_and_install(agent, config)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    saved = save_checkpoint(agent, tmp_path, "final", config, base)
    assert saved["actor_bytes"] == 230
    assert saved["whole_package_bytes"] <= config["whole_package_cap_bytes"]
    restored, _, _ = restore_capacity_run(tmp_path)
    assert restored.cache.state_dict() == agent.cache.state_dict()
    np.testing.assert_array_equal(restored.core.read_local_matrix(), agent.core.read_local_matrix())
    np.testing.assert_array_equal(restored.core.read_shared_matrix(), agent.core.read_shared_matrix())
    assert restored.action_rng.bit_generator.state == agent.action_rng.bit_generator.state
    cache_before = copy.deepcopy(agent.cache.state_dict())
    state_before = {k:v.clone() for k,v in agent.core.state_dict().items()}
    torch_before, python_before = torch.get_rng_state().clone(), random.getstate()
    numpy_before = np.random.get_state()
    left_core, _, left_slot, event = _evaluation_slot(agent, config["contexts"][6])
    assert event["action"] == "miss"
    left = evaluate_capacity_context(left_core, config["contexts"][6], left_slot, seed=177, episodes=2)
    right = evaluate_capacity_context(left_core, config["contexts"][6], left_slot, seed=177, episodes=2)
    assert left == right
    assert agent.cache.state_dict() == cache_before
    assert all(torch.equal(v, agent.core.state_dict()[k]) for k,v in state_before.items())
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
