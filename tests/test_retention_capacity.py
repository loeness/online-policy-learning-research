from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from online_rl.policy import OBS_DIM, TreePolicy  # noqa: E402
from online_rl.write_decision_policy import encode_actor  # noqa: E402
from scripts import run_write_stream as stream  # noqa: E402


@pytest.mark.parametrize("capacity", range(1, 9))
def test_task_cache_evicts_and_resumes_at_each_supported_capacity(capacity: int) -> None:
    cache = stream.BoundedTaskCache(capacity=capacity, eviction="lru", seed=17)
    for task in stream.TASKS[:capacity]:
        cache.resolve(task)

    cache.resolve(stream.TASKS[0])
    inserted = cache.resolve(stream.TASKS[capacity])
    expected_evicted_slot = 0 if capacity == 1 else 1
    assert inserted == {
        "slot": expected_evicted_slot,
        "allocated": True,
        "evicted_slot": expected_evicted_slot,
        "evicted_task": stream.TASKS[expected_evicted_slot],
    }
    assert len(cache.task_to_slot) == len(cache.slot_to_task) == capacity

    restored = stream.BoundedTaskCache.from_state_dict(cache.state_dict())
    assert restored.task_to_slot == cache.task_to_slot
    assert restored.slot_to_task == cache.slot_to_task
    np.testing.assert_array_equal(restored.last_used, cache.last_used)
    np.testing.assert_array_equal(restored.frequency, cache.frequency)

    next_task = stream.TASKS[capacity + 1]
    assert restored.resolve(next_task) == cache.resolve(next_task)
    assert restored.task_to_slot == cache.task_to_slot
    assert restored.slot_to_task == cache.slot_to_task


def _reference_policy() -> TreePolicy:
    return TreePolicy(
        observation_space=spaces.Box(
            low=np.asarray([-2.0] * (OBS_DIM - 1) + [0.0], dtype=np.float32),
            high=np.asarray([2.0] * (OBS_DIM - 1) + [8.0], dtype=np.float32),
            dtype=np.float32,
        ),
        action_space=spaces.Discrete(4),
        lr_schedule=lambda _: 1e-2,
    )


@pytest.mark.parametrize(
    ("condition", "slots", "groups", "actor_bytes"),
    [
        ("reward_shared", 3, 2, 134),
        ("independent_fast", 4, 0, 140),
        ("reward_shared", 6, 2, 230),
        ("independent7_lru", 7, 0, 236),
    ],
)
def test_stream_factory_builds_shared_and_flat_capacity_layouts(
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
    slots: int,
    groups: int,
    actor_bytes: int,
) -> None:
    class EmptyEnvironment:
        @staticmethod
        def close() -> None:
            pass

    monkeypatch.setattr(
        stream, "_foundation",
        lambda *_args: SimpleNamespace(policy=_reference_policy(), env=EmptyEnvironment()),
    )
    monkeypatch.setattr(stream, "seed_everything", lambda *_args: None)
    config = {
        **stream._load_config(),
        "local_slots": slots,
        "actor_budget_bytes": actor_bytes,
    }

    agent = stream._new_agent(condition, 21, config, {})

    assert agent.slots == slots
    assert agent.core.shared_groups == groups
    local, shared, assignments = (
        agent.core.read_local_matrix(),
        agent.core.read_shared_matrix(),
        agent.core.slot_groups.detach().cpu().numpy().astype(np.int8),
    )
    encoded = encode_actor(local, shared, assignments, codec="int8", budget_bytes=actor_bytes)
    assert encoded.nbytes == actor_bytes == 12 + 32 * slots + 13 * groups


@pytest.mark.parametrize(
    ("slots", "groups", "actor_bytes"),
    [(3, 2, 134), (4, 0, 140), (6, 2, 230), (7, 0, 236)],
)
def test_int8_actor_codec_enforces_exact_layout_budget(
    slots: int, groups: int, actor_bytes: int,
) -> None:
    local = np.zeros((slots, 27), dtype=np.float32)
    shared = np.zeros((groups, 9), dtype=np.float32)
    assignments = np.full(slots, -1, dtype=np.int8)

    encoded = encode_actor(local, shared, assignments, codec="int8", budget_bytes=actor_bytes)

    assert encoded.nbytes == actor_bytes
    with pytest.raises(ValueError, match="exceeding budget"):
        encode_actor(local, shared, assignments, codec="int8", budget_bytes=actor_bytes - 1)


def test_slot_layout_preflight_uses_the_supplied_actor_budget() -> None:
    stream._validate_local_slot_conditions(["reward_shared"], 6, actor_budget_bytes=230)
    stream._validate_local_slot_conditions(["independent_fast"], 7, actor_budget_bytes=236)
    with pytest.raises(ValueError, match="actor_budget_bytes=229"):
        stream._validate_local_slot_conditions(["reward_shared"], 6, actor_budget_bytes=229)
    with pytest.raises(ValueError, match="independent G0"):
        stream._validate_local_slot_conditions(["reward_shared"], 7, actor_budget_bytes=236)
