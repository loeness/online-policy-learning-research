import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl.write_decision_gate import AnchorBank, evaluate_change, project_delta  # noqa: E402


def _anchor_row(index, *, generation=0, episode_id=None, advantage=1.0):
    return {
        "obs": np.full(13, index, dtype=np.float32),
        "action": 0,
        "log_prob": float(np.log(0.5)),
        "probabilities": np.asarray([0.5, 0.5, 0.0, 0.0], dtype=np.float32),
        "advantage": advantage,
        "returns": float(index + 0.25),
        "generation": generation,
        "episode_id": index if episode_id is None else episode_id,
    }


def test_anchor_bank_generation_change_clears_slot_evidence():
    bank = AnchorBank(slots=2, per_slot=4, seed=7)
    bank.add(0, 0, [_anchor_row(1, episode_id=10), _anchor_row(2, episode_id=10)])

    old = bank.arrays(0, 0)
    assert old["obs"].shape == (2, 13)
    np.testing.assert_array_equal(old["episode_id"], [10, 10])

    bank.add(0, 1, [_anchor_row(3, generation=1, episode_id=11)])
    assert bank.arrays(0, 0)["obs"].shape == (0, 13)
    current = bank.arrays(0, 1)
    assert current["obs"].shape == (1, 13)
    assert np.all(current["generation"] == 1)
    assert current["obs"][0, 0] == 3


def test_anchor_bank_roundtrip_preserves_buffers_rng_and_fixed_memory():
    bank = AnchorBank(slots=2, per_slot=3, seed=42)
    bank.add(1, 4, [_anchor_row(i, generation=4, episode_id=i // 2) for i in range(12)])
    restored = AnchorBank.from_state_dict(bank.state_dict())

    assert restored.nbytes == bank.nbytes
    for key, value in bank.arrays(1, 4).items():
        np.testing.assert_array_equal(restored.arrays(1, 4)[key], value)

    next_row = [_anchor_row(99, generation=4, episode_id=99)]
    bank.add(1, 4, next_row)
    restored.add(1, 4, next_row)
    for key, value in bank.arrays(1, 4).items():
        np.testing.assert_array_equal(restored.arrays(1, 4)[key], value)


def test_anchor_bank_rejects_rows_from_a_different_generation():
    bank = AnchorBank(slots=1, per_slot=2, seed=1)
    with pytest.raises(ValueError, match="does not match"):
        bank.add(0, 3, [_anchor_row(1, generation=2)])


def test_evaluate_change_uses_behavior_logprob_clips_ratio_and_reports_ess():
    incumbent = np.tile([0.5, 0.5, 0.0, 0.0], (4, 1))
    candidate = np.asarray([
        [0.9, 0.1, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0],
    ])
    result = evaluate_change(
        incumbent,
        candidate,
        actions=np.zeros(4, dtype=np.int64),
        behavior_logp=np.full(4, np.log(0.5)),
        behavior_probs=incumbent,
        advantages=np.asarray([1.0, 1.0, -1.0, -1.0]),
        episode_ids=np.asarray([7, 7, 8, 8]),
        min_ess=2.0,
        kl_limit=0.6,
    )

    assert result["mean_delta"] == pytest.approx(0.2)
    assert result["episode_se"] == pytest.approx(0.0)
    assert result["ess"] == pytest.approx(16.0 / 6.56)
    assert result["weight_clip_fraction"] == pytest.approx(1.0)
    assert result["mean_kl"] == pytest.approx(result["behavior_kl"])
    assert result["support_ok"] is True


def test_evaluate_change_uses_supplied_sampling_logprob_as_ratio_denominator():
    incumbent = np.tile([0.25, 0.75, 0.0, 0.0], (2, 1))
    candidate = np.tile([0.5, 0.5, 0.0, 0.0], (2, 1))
    # The stored behavior distribution is intentionally rounded/inconsistent;
    # the exact sampling log probability is authoritative for the IS ratio.
    result = evaluate_change(
        incumbent,
        candidate,
        actions=np.zeros(2, dtype=np.int64),
        behavior_logp=np.full(2, np.log(0.25)),
        behavior_probs=np.tile([0.5, 0.5, 0.0, 0.0], (2, 1)),
        advantages=np.ones(2),
        episode_ids=np.asarray([1, 2]),
        min_ess=1.0,
        kl_limit=1.0,
    )
    # Candidate ratio is 0.5 / 0.25 = 2, clipped to 1.2; incumbent ratio is 1.
    assert result["mean_delta"] == pytest.approx(0.2)


def test_evaluate_change_clusters_standard_error_by_episode():
    incumbent = np.tile([0.5, 0.5, 0.0, 0.0], (4, 1))
    candidate = np.tile([0.6, 0.4, 0.0, 0.0], (4, 1))
    common = dict(
        incumbent_probs=incumbent,
        candidate_probs=candidate,
        actions=np.zeros(4, dtype=np.int64),
        behavior_logp=np.full(4, np.log(0.5)),
        behavior_probs=incumbent,
        advantages=np.asarray([1.0, 1.0, -1.0, -1.0]),
        min_ess=0.0,
        kl_limit=2.0,
    )
    one_episode = evaluate_change(**common, episode_ids=np.asarray([1, 1, 1, 1]))
    two_episodes = evaluate_change(**common, episode_ids=np.asarray([1, 1, 2, 2]))

    assert np.isinf(one_episode["episode_se"])
    assert one_episode["support_ok"] is False
    assert two_episodes["episode_se"] > 0


def test_project_delta_preserves_ascent_direction_when_resolving_conflict():
    delta = np.zeros(9)
    delta[:2] = [-2.0, 3.0]
    references = np.zeros((1, 9))
    references[0, 0] = 1.0

    result = project_delta(delta, references, tolerance=0.0, passes=3)

    assert result.satisfied is True
    assert result.delta[0] == pytest.approx(0.0)
    assert result.delta[1] == pytest.approx(3.0)
    assert result.delta[1] > 0  # Projection does not reverse the proposal.


def test_project_delta_reports_unresolved_collision_after_too_few_passes():
    delta = np.zeros(9)
    delta[:2] = [-1.0, -1.0]
    references = np.zeros((2, 9))
    references[0, 0] = 1.0
    references[1, :2] = [-1.0, 1.0]

    partial = project_delta(delta, references, tolerance=0.0, passes=1)
    converged = project_delta(delta, references, tolerance=0.0, passes=100)

    assert partial.satisfied is False
    assert partial.max_violation > 0
    assert converged.satisfied is True
    assert np.all(references @ converged.delta >= -1e-10)
