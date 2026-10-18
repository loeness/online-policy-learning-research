import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.write_decision_evidence import EvidenceProfile
from online_rl.retention_evidence import (
    RetentionEvidenceArchive,
    profiles_compatible,
)


def _profile(*, offset=0, reward=(0.5, 0.7), support=3):
    counts = np.zeros((3, 4, 4), dtype=np.uint16)
    for band in range(3):
        for action in range(4):
            counts[band, action, (action + offset) % 4] = support

    # Independent movement and intercept support, with termination absent.
    xtx = np.diag(np.asarray([4.0, 4.0, 0.0], dtype=np.float32)).astype(np.float32)
    xty = np.asarray([4.0 * reward[0], 4.0 * reward[1], 0.0], dtype=np.float32)
    return EvidenceProfile(counts, xtx, xty, np.float32(8.0))


def _sparse_profile():
    profile = _profile()
    counts = np.zeros_like(profile.movement_counts)
    counts[0, 0, 0] = 3
    return EvidenceProfile(counts, profile.reward_xtx.copy(), profile.reward_xty.copy(), profile.reward_n)


def _rows(start, count, *, slot=0, generation=0):
    return [
        {
            "obs": np.full(13, index / 100.0, dtype=np.float32),
            "action": index % 4,
            "log_prob": float(np.log(0.25)),
            "probabilities": np.full(4, 0.25, dtype=np.float32),
            "advantage": 1000.0 + index,
            "returns": 2000.0 + index,
            "episode_id": index // 3,
            "done": index % 3 == 2,
            "slot": slot,
            "generation": generation,
            "task_name": "ignored-label",
        }
        for index in range(start, start + count)
    ]


def test_profiles_compatible_requires_supported_movement_and_reward_fit():
    first = _profile()
    same = _profile()
    opposite_reward = _profile(reward=(-0.5, -0.7))

    compatible = profiles_compatible(first, same)
    assert compatible["compatible"] is True
    assert compatible["overlap"] == 12
    assert compatible["agreement"] == 1.0
    assert compatible["reward_comparable"] is True

    conflict = profiles_compatible(first, opposite_reward)
    assert conflict["compatible"] is False
    assert conflict["reward_cosine"] < 0

    insufficient = profiles_compatible(first, _sparse_profile())
    assert insufficient["compatible"] is False
    assert insufficient["overlap"] == 1


def test_reward_unidentifiable_or_untried_actions_abstain():
    first = _profile()
    no_reward_support = EvidenceProfile(
        first.movement_counts.copy(),
        np.zeros((3, 3), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.float32(8.0),
    )

    result = profiles_compatible(first, no_reward_support)
    assert result["compatible"] is False
    assert result["reward_comparable"] is False


def test_rows_survive_source_slot_generation_changes_and_keep_finalized_values():
    archive = RetentionEvidenceArchive(max_prototypes=2, per_prototype=8, seed=9)
    first = _rows(0, 1, slot=1, generation=3)[0]
    first["advantage"] = 7.25
    result1 = archive.observe(_profile(), [first], step=1)
    second = _rows(1, 1, slot=1, generation=4)[0]
    second["advantage"] = -13.5
    result2 = archive.observe(_profile(), [second], step=2)

    assert result1["matched"] is False
    assert result2["matched"] is True
    rows = archive.arrays(result1["prototype"])
    assert rows["source_generation"].tolist() == [3, 4]
    assert rows["source_slot"].tolist() == [1, 1]
    assert rows["advantage"].tolist() == pytest.approx([7.25, -13.5])
    assert rows["behavior_probs"].shape == (2, 4)
    assert rows["obs"].shape == (2, 13)


def test_lru_replacement_reuses_cell_with_a_new_version():
    archive = RetentionEvidenceArchive(max_prototypes=2, per_prototype=4, seed=0)
    a = _profile(offset=0)
    b = _profile(offset=1)
    c = _profile(offset=2)

    first = archive.observe(a, [], step=1)
    second = archive.observe(b, [], step=2)
    refreshed = archive.observe(_profile(offset=0), [], step=3)
    replacement = archive.observe(c, [], step=4)

    assert refreshed["matched"] is True
    assert refreshed["prototype"] == first["prototype"]
    assert replacement["matched"] is False
    assert replacement["evicted"] is True
    assert replacement["prototype"] == second["prototype"]
    assert replacement["version"] == second["version"] + 1
    assert replacement["reason"].startswith("lru_replaced_")
    assert archive.active_indices() == (0, 1)
    assert np.array_equal(archive.profile(replacement["prototype"]).movement_counts, c.movement_counts)


def test_uniform_reservoir_rng_roundtrip_is_deterministic():
    archive = RetentionEvidenceArchive(max_prototypes=1, per_prototype=5, seed=101)
    profile = _profile()
    archive.observe(profile, _rows(0, 12), step=1)
    restored = RetentionEvidenceArchive.from_state_dict(archive.state_dict())

    incoming = _rows(12, 20)
    archive.observe(profile, incoming, step=2)
    restored.observe(profile, incoming, step=2)
    first = archive.arrays(0)
    second = restored.arrays(0)
    assert first.keys() == second.keys()
    for name in first:
        assert np.array_equal(first[name], second[name])
    assert archive.state_dict()["seen"].tolist() == [32]


def test_fixed_storage_and_label_free_observe_signature():
    archive = RetentionEvidenceArchive(max_prototypes=3, per_prototype=7, seed=2)
    expected_nbytes = archive.nbytes
    archive.observe(_profile(), _rows(0, 30), step=1)
    assert archive.nbytes == expected_nbytes
    assert archive.state_dict()["nbytes"] == expected_nbytes

    parameters = inspect.signature(RetentionEvidenceArchive.observe).parameters
    assert tuple(parameters) == ("self", "profile", "rows", "step")
    assert "task_name" not in parameters
    assert "family" not in parameters


def test_transition_objects_without_episode_id_are_accepted():
    class Transition:
        def __init__(self, step, done):
            self.obs = np.zeros(13, dtype=np.float32)
            self.action = 0
            self.log_prob = float(np.log(0.25))
            self.probabilities = np.full(4, 0.25, dtype=np.float32)
            self.advantage = 2.0
            self.returns = 3.0
            self.done = done
            self.slot = 4
            self.generation = 7
            self.step = step

    archive = RetentionEvidenceArchive(max_prototypes=1, per_prototype=4)
    archive.observe(_profile(), [Transition(1, False), Transition(2, True), Transition(3, False)], step=1)
    assert archive.arrays(0)["episode_id"].tolist() == [-1, -1, -2]
