import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl.retention_consolidation import (  # noqa: E402
    common_mean,
    compensated_install,
    decide_retention_write,
    surrogate_gain,
)


def _evidence(rows=12, *, advantages=None, block_ids=None, behavior=None, logp=None):
    if advantages is None:
        advantages = np.ones(rows, dtype=np.float64)
    if block_ids is None:
        block_ids = np.arange(rows, dtype=np.int64) // 4
    if behavior is None:
        behavior = np.tile([0.5, 0.5, 0.0, 0.0], (rows, 1))
    if logp is None:
        logp = np.full(rows, np.log(0.5), dtype=np.float64)
    return {
        "action": np.zeros(rows, dtype=np.int64),
        "behavior_probs": np.asarray(behavior, dtype=np.float64),
        "behavior_log_prob": np.asarray(logp, dtype=np.float64),
        "advantage": np.asarray(advantages, dtype=np.float64),
        "episode_id": np.asarray(block_ids, dtype=np.int64),
    }


def test_common_mean_uniformly_averages_effective_bands_and_does_not_wrap_group_minus_one():
    local = np.zeros((3, 27), dtype=np.float64)
    shared = np.zeros((2, 9), dtype=np.float64)
    groups = np.asarray([-1, 1, -1], dtype=np.int64)
    local[0] = 1.0
    local[1] = 2.0
    local[2] = 3.0
    shared[1] = 10.0

    result = common_mean(local, shared, groups, source_slots=[0, 0, 2])

    np.testing.assert_allclose(result, np.full((3, 3), 2.0))
    with pytest.raises(ValueError, match="at least two"):
        common_mean(local, shared, groups, source_slots=[0, 0])


def test_compensated_install_preserves_destination_residents_joiners_other_group_and_unbound_slots():
    rng = np.random.default_rng(82)
    local = rng.normal(size=(5, 27)).astype(np.float64)
    shared = rng.normal(size=(2, 9)).astype(np.float64)
    groups = np.asarray([0, 0, 1, -1, -1], dtype=np.int64)
    common = rng.normal(size=(3, 3))
    local_before = local.copy()
    shared_before = shared.copy()
    groups_before = groups.copy()

    old_effective = local.reshape(5, 3, 3, 3).copy()
    assigned = groups >= 0
    old_effective[assigned] += shared[groups[assigned]].reshape(-1, 1, 3, 3)

    new_local, new_shared, new_groups, diagnostics = compensated_install(
        local, shared, groups, group=0, common=common, join_slots=[2, 3]
    )

    new_effective = new_local.reshape(5, 3, 3, 3).copy()
    new_assigned = new_groups >= 0
    new_effective[new_assigned] += new_shared[new_groups[new_assigned]].reshape(-1, 1, 3, 3)
    np.testing.assert_allclose(new_effective[[0, 1, 2, 3]], old_effective[[0, 1, 2, 3]], atol=1e-14)
    np.testing.assert_array_equal(new_groups, [0, 0, 0, 0, -1])
    np.testing.assert_array_equal(new_shared[0].reshape(3, 3), common)
    np.testing.assert_array_equal(new_shared[1], shared_before[1])
    np.testing.assert_array_equal(new_local[4], local_before[4])
    assert diagnostics["affected_slots"] == (0, 1, 2, 3)
    assert diagnostics["max_effective_preservation_error"] < 1e-14
    np.testing.assert_array_equal(local, local_before)
    np.testing.assert_array_equal(shared, shared_before)
    np.testing.assert_array_equal(groups, groups_before)


def test_empty_install_does_not_bind_unrelated_unassigned_slot_or_wrap_minus_one():
    local = np.arange(54, dtype=np.float64).reshape(2, 27)
    shared = np.ones((2, 9), dtype=np.float64)
    groups = np.asarray([-1, 1], dtype=np.int64)
    common = np.full((3, 3), 7.0)

    new_local, new_shared, new_groups, diagnostics = compensated_install(
        local, shared, groups, group=0, common=common
    )

    np.testing.assert_array_equal(new_local, local)
    np.testing.assert_array_equal(new_groups, groups)
    np.testing.assert_array_equal(new_shared[0].reshape(3, 3), common)
    np.testing.assert_array_equal(new_shared[1], shared[1])
    assert diagnostics["affected_slots"] == ()


def test_surrogate_gain_preserves_advantage_sign_and_uses_stored_behavior_log_probability():
    rows = 12
    behavior = np.tile([0.9, 0.1, 0.0, 0.0], (rows, 1))
    evidence = _evidence(
        rows,
        behavior=behavior,
        logp=np.full(rows, np.log(0.5)),  # Deliberately differs from behavior_probs[action].
    )
    baseline = np.tile([0.5, 0.5, 0.0, 0.0], (rows, 1))
    candidate = np.tile([0.6, 0.4, 0.0, 0.0], (rows, 1))

    positive = surrogate_gain(evidence, candidate, baseline, max_mean_kl=1.0)
    negative = surrogate_gain(
        _evidence(
            rows,
            advantages=-np.ones(rows),
            behavior=behavior,
            logp=np.full(rows, np.log(0.5)),
        ),
        candidate,
        baseline,
        max_mean_kl=1.0,
    )

    assert positive["supported"] is True
    assert positive["gain"] == pytest.approx(0.2)
    assert positive["kl"]["candidate"] == pytest.approx(
        0.9 * np.log(0.9 / 0.6) + 0.1 * np.log(0.1 / 0.4)
    )
    assert negative["supported"] is True
    assert negative["gain"] == pytest.approx(-0.2)


def test_surrogate_gain_clusters_correlated_rows_for_uncertainty():
    rows = 12
    advantages = np.repeat([1.0, -1.0, 1.0], 4)
    evidence = _evidence(rows, advantages=advantages, block_ids=np.repeat([10, 11, 12], 4))
    baseline = np.tile([0.5, 0.5, 0.0, 0.0], (rows, 1))
    candidate = np.tile([0.6, 0.4, 0.0, 0.0], (rows, 1))

    result = surrogate_gain(evidence, candidate, baseline, max_mean_kl=1.0)

    # The three block means are [0.2, -0.2, 0.2], despite twelve rows.
    expected_gain = 0.2 / 3.0
    expected_stderr = np.std([0.2, -0.2, 0.2], ddof=1) / np.sqrt(3)
    assert result["supported"] is True
    assert result["blocks"] == 3
    assert result["gain"] == pytest.approx(expected_gain)
    assert result["stderr"] == pytest.approx(expected_stderr)
    assert result["lower"] == pytest.approx(expected_gain - expected_stderr)


def test_surrogate_gain_uses_row_weighted_cluster_se_for_unequal_episode_sizes():
    rows = 12
    advantages = np.concatenate((np.ones(2), -np.ones(3), np.ones(7)))
    block_ids = np.concatenate((np.full(2, 10), np.full(3, 11), np.full(7, 12)))
    evidence = _evidence(rows, advantages=advantages, block_ids=block_ids)
    baseline = np.tile([0.5, 0.5, 0.0, 0.0], (rows, 1))
    candidate = np.tile([0.6, 0.4, 0.0, 0.0], (rows, 1))

    result = surrogate_gain(evidence, candidate, baseline, max_mean_kl=1.0)

    # Row gains are +0.2 for two rows, -0.2 for three, and +0.2 for seven.
    # The point estimate remains row weighted; the SE clusters centered row
    # deltas and applies the finite-sample block correction.
    expected_gain = 0.1
    centered_sums = np.asarray([0.2, -0.9, 0.7])
    expected_variance = (3.0 / 2.0) * np.dot(centered_sums, centered_sums) / (rows * rows)
    assert result["supported"] is True
    assert result["blocks"] == 3
    assert result["gain"] == pytest.approx(expected_gain)
    assert result["stderr"] == pytest.approx(np.sqrt(expected_variance))
    assert result["lower"] == pytest.approx(expected_gain - np.sqrt(expected_variance))


def test_surrogate_gain_returns_unknown_when_rows_or_behavior_coverage_are_insufficient():
    rows = 8
    behavior = np.tile([0.5, 0.5, 0.0, 0.0], (rows, 1))
    evidence = _evidence(rows, behavior=behavior)
    baseline = np.tile([0.5, 0.5, 0.0, 0.0], (rows, 1))
    unsupported_candidate = np.tile([1.0, 0.0, 0.0, 0.0], (rows, 1))

    result = surrogate_gain(evidence, unsupported_candidate, baseline)
    missing = surrogate_gain({}, baseline, baseline)

    assert result["supported"] is False
    assert result["status"] == "unknown"
    assert result["gain"] is None
    assert result["lower"] is None
    assert "insufficient_rows" in result["reason"]
    assert "candidate_lacks_behavior_coverage" in result["reason"]
    assert missing["status"] == "unknown"
    assert missing["gain"] is None
    assert missing["lower"] is None


def test_surrogate_gain_checks_candidate_and_baseline_ess_and_coverage_separately():
    rows = 12
    behavior = np.tile([0.99, 0.01, 0.0, 0.0], (rows, 1))
    behavior[0] = [0.01, 0.99, 0.0, 0.0]
    evidence = _evidence(
        rows,
        behavior=behavior,
        logp=np.log(behavior[:, 0]),
    )
    baseline = behavior.copy()
    candidate = behavior.copy()
    candidate[0] = [0.99, 0.01, 0.0, 0.0]
    candidate[1:] = [0.01, 0.99, 0.0, 0.0]

    low_ess = surrogate_gain(evidence, candidate, baseline, max_mean_kl=10.0)

    assert low_ess["supported"] is False
    assert low_ess["ess_ratio"]["candidate"] < 0.2
    assert low_ess["ess_ratio"]["baseline"] == pytest.approx(1.0)
    assert low_ess["coverage"] == {"candidate": True, "baseline": True}
    assert "candidate_low_ess" in low_ess["reason"]
    assert "baseline_low_ess" not in low_ess["reason"]

    uniform_behavior = np.tile([0.5, 0.5, 0.0, 0.0], (rows, 1))
    coverage_evidence = _evidence(rows, behavior=uniform_behavior)
    covered_candidate = uniform_behavior.copy()
    uncovered_baseline = np.tile([1.0, 0.0, 0.0, 0.0], (rows, 1))
    coverage = surrogate_gain(
        coverage_evidence,
        covered_candidate,
        uncovered_baseline,
        max_mean_kl=10.0,
    )

    assert coverage["coverage"] == {"candidate": True, "baseline": False}
    assert "baseline_lacks_behavior_coverage" in coverage["reason"]
    assert "candidate_lacks_behavior_coverage" not in coverage["reason"]


def test_decide_retention_write_requires_multiple_supported_profiles_and_safe_lowers():
    assert decide_retention_write([{"supported": True, "lower": 0.2}])["accepted"] is False
    assert decide_retention_write(
        [
            {"supported": True, "lower": 0.2},
            {"supported": True, "lower": -0.06},
        ],
        max_profile_loss=0.05,
    )["reason"] == "supported_profile_loss_too_large"
    accepted = decide_retention_write(
        [
            {"supported": True, "lower": 0.1},
            {"supported": True, "lower": 0.2},
            {"supported": False, "lower": None},
        ]
    )
    assert accepted["accepted"] is True
    assert accepted["supported_profiles"] == 2
    assert accepted["mean_lower"] == pytest.approx(0.15)
