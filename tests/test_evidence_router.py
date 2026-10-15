import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.evidence_router import EvidenceRouter


_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def _transition(event):
    band, action, direction = event
    x = (1, 4, 7)[band]
    y = 4
    dx, dy = _DELTAS[direction]
    obs = np.asarray((x / 8, y / 8, 0.0, 1.0), dtype=np.float32)
    nxt = np.asarray(((x + dx) / 8, (y + dy) / 8, 0.0, 1.0), dtype=np.float32)
    return obs, action, nxt


def _feed(router, events):
    results = []
    for event in events:
        obs, action, nxt = _transition(event)
        results.append(router.observe(obs, action, 0.0, nxt))
    return results


def _profile(direction_a=0, direction_b=1):
    return [
        (0, 0, direction_a),
        (1, 1, direction_b),
        (0, 0, direction_a),
        (1, 1, direction_b),
    ]


def test_profile_learning_is_independent_of_row_order():
    first = EvidenceRouter(window=16)
    second = EvidenceRouter(window=16)
    events = _profile()
    _feed(first, events)
    _feed(second, [events[1], events[0], events[3], events[2]])

    assert first.allocations == second.allocations == 1
    np.testing.assert_array_equal(first._counts, second._counts)
    for observation in (_transition((0, 0, 0))[0], _transition((1, 1, 1))[0]):
        assert first.route(observation) == second.route(observation)


def test_partial_profile_match_learns_a_previously_unseen_cell():
    router = EvidenceRouter(window=16)
    _feed(router, _profile())
    slot = router.active_slots[0]
    assert router._cell_total(slot, 2, 3) == 0

    _feed(router, [(2, 3, 2), (2, 3, 2)])

    assert router.allocations == 1
    assert router._cell_total(slot, 2, 3) == 2
    routed = router.route(_transition((2, 3, 2))[0])
    assert routed["slot"] == slot
    assert routed["reason"] in {"supported", "partial_profile_match"}


def test_novel_cells_without_common_evidence_do_not_allocate():
    router = EvidenceRouter(window=16)
    _feed(router, _profile())
    router.reset_history()

    _feed(router, [(2, 3, 2), (2, 3, 2), (0, 2, 3), (0, 2, 3)])

    assert router.allocations == 1
    assert len(router.active_slots) == 1
    result = router.route(_transition((2, 3, 2))[0])
    assert result["slot"] == -1
    assert result["reason"] == "unseen_cells_only"


def test_supported_contradiction_allocates_and_returning_profile_is_reused():
    router = EvidenceRouter(window=16)
    a = _profile(0, 1)
    b = _profile(2, 3)
    _feed(router, a)
    allocated = _feed(router, b)
    assert any(row["allocated_slot"] == 1 for row in allocated)
    assert router.allocations == 2

    reused = _feed(router, a)
    assert router.allocations == 2
    assert router.switches >= 1
    assert not any(row["allocated_slot"] is not None for row in reused)
    assert router.route(_transition(a[-1])[0])["slot"] == 0


def test_lru_replacement_keeps_the_profile_bank_within_capacity():
    router = EvidenceRouter(max_contexts=2, window=16)
    _feed(router, _profile(0, 1))
    first_slot = router.active_slots[0]
    router.route(_transition((0, 0, 0))[0])  # Refresh the first profile.
    _feed(router, _profile(2, 3))
    replaced = _feed(router, _profile(1, 2))

    assert len(router.active_slots) == 2
    assert router.allocations == 3
    assert router.evictions == 1
    allocation = next(row for row in replaced if row["allocated_slot"] is not None)
    assert allocation["evicted_slot"] == first_slot
    assert router._cell_total(first_slot, 0, 0) >= 2
    assert int(np.argmax(router._counts[first_slot, 0, 0])) == 1


def test_snapshot_round_trip_preserves_routing_and_bounded_state():
    router = EvidenceRouter(window=16)
    _feed(router, _profile())
    _feed(router, [(2, 3, 2), (2, 3, 2)])
    restored = EvidenceRouter.from_state_dict(json.loads(json.dumps(router.state_dict())))

    assert restored.state_dict() == router.state_dict()
    obs, action, nxt = _transition((0, 0, 0))
    assert restored.observe(obs, action, 0.0, nxt) == router.observe(obs, action, 0.0, nxt)
    assert restored.state_dict() == router.state_dict()
    assert restored.nbytes()["total"] < 20_000


def test_stationary_wall_transitions_are_censored():
    router = EvidenceRouter()
    obs = np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float32)
    result = router.observe(obs, 0, 0.0, obs)
    assert result["censored"] is True
    assert router._censored == 1
    assert not router.active_slots
