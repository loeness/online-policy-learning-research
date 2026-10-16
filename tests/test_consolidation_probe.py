import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl.consolidation_probe import extract_common, recompose_shared


def test_weighted_recovers_common_when_every_effective_matrix_matches():
    common = np.array(
        [[0.4, -0.3, 0.2], [0.1, 0.8, -0.6], [-0.7, 0.5, 0.9]],
        dtype=np.float64,
    )
    effective = np.broadcast_to(common, (3, 3, 3, 3)).copy()
    psi = [np.array([[1.0, 0.2, -0.5], [0.3, 1.0, 0.7]]) for _ in range(3)]
    bands = [np.array([0, 2], dtype=np.int64) for _ in range(3)]

    result, diagnostics = extract_common(
        "weighted", effective, anchor_psi=psi, anchor_bands=bands
    )

    np.testing.assert_allclose(result, common, atol=2e-7)
    assert diagnostics["source_count"] == 3
    assert diagnostics["row_count"] == 6
    assert len(diagnostics["gram_eigenvalues"]) == 3


def test_weighted_uses_mean_prior_for_unsupported_feature_directions():
    rng = np.random.default_rng(21)
    effective = rng.normal(size=(2, 3, 3, 3))
    prior = effective.mean(axis=(0, 1))
    psi = [np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]) for _ in range(2)]
    bands = [np.array([0, 2], dtype=np.int64) for _ in range(2)]

    result, _ = extract_common("weighted", effective, anchor_psi=psi, anchor_bands=bands)

    np.testing.assert_allclose(result[:, 1:], prior[:, 1:], atol=2e-7)


def test_weighted_gives_each_context_equal_weight_despite_row_count_difference():
    effective = np.zeros((2, 3, 3, 3), dtype=np.float64)
    effective[1] = 100.0
    psi = [
        np.array([[1.0, 0.0, 0.0]]),
        np.tile(np.array([[1.0, 0.0, 0.0]]), (9, 1)),
    ]
    bands = [np.array([0], dtype=np.int64), np.zeros(9, dtype=np.int64)]

    result, diagnostics = extract_common("weighted", effective, anchor_psi=psi, anchor_bands=bands)

    np.testing.assert_allclose(result[:, 0], 50.0, atol=2e-5)
    assert diagnostics["row_counts"] == (1, 9)


def test_random_matches_weighted_norm_and_is_seeded():
    effective = np.arange(2 * 3 * 3 * 3, dtype=np.float64).reshape(2, 3, 3, 3) / 7.0
    psi = [np.eye(3), np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])]
    bands = [np.array([0, 1, 2]), np.array([2, 0])]
    weighted, _ = extract_common("weighted", effective, anchor_psi=psi, anchor_bands=bands)

    first, diagnostics = extract_common(
        "random", effective, anchor_psi=psi, anchor_bands=bands, seed=81
    )
    second, _ = extract_common(
        "random", effective, anchor_psi=psi, anchor_bands=bands, seed=81
    )

    np.testing.assert_array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(np.linalg.norm(weighted), rel=1e-6)
    assert diagnostics["reference_norm"] == pytest.approx(np.linalg.norm(weighted), rel=1e-12)
    assert diagnostics["seed"] == 81


def test_recompose_preserves_destination_and_moving_source_effective_matrices():
    rng = np.random.default_rng(41)
    local = rng.normal(size=(4, 27)).astype(np.float64)
    shared = rng.normal(size=(2, 9)).astype(np.float64)
    groups = np.array([0, 1, 1, 0], dtype=np.int64)
    common = rng.normal(size=(3, 3))
    old_local = local.copy()
    old_shared = shared.copy()
    old_groups = groups.copy()
    old_effective = local.reshape(4, 3, 3, 3) + shared[groups].reshape(4, 1, 3, 3)

    new_local, new_shared, new_groups, diagnostics = recompose_shared(
        local, shared, groups, source_slots=[0], group=1, common=common
    )

    new_effective = new_local.reshape(4, 3, 3, 3) + new_shared[new_groups].reshape(4, 1, 3, 3)
    np.testing.assert_allclose(new_effective[[0, 1, 2]], old_effective[[0, 1, 2]], atol=1e-14)
    np.testing.assert_array_equal(new_groups, np.array([1, 1, 1, 0]))
    np.testing.assert_array_equal(new_shared[1].reshape(3, 3), common)
    np.testing.assert_allclose(new_local[3], old_local[3], atol=0)
    np.testing.assert_allclose(new_shared[0], old_shared[0], atol=0)
    assert diagnostics["affected_slots"] == (0, 1, 2)
    assert diagnostics["max_effective_preservation_error"] < 1e-14
    np.testing.assert_array_equal(local, old_local)
    np.testing.assert_array_equal(shared, old_shared)
    np.testing.assert_array_equal(groups, old_groups)


def test_recompose_preserves_effective_matrices_when_source_already_in_destination():
    local = np.arange(54, dtype=np.float64).reshape(2, 27) / 10.0
    shared = np.arange(9, dtype=np.float64).reshape(1, 9) / 5.0
    groups = np.zeros(2, dtype=np.int64)
    old_effective = local.reshape(2, 3, 3, 3) + shared[groups].reshape(2, 1, 3, 3)

    new_local, new_shared, new_groups, diagnostics = recompose_shared(
        local, shared, groups, source_slots=[1], group=0, common=np.ones((3, 3))
    )

    new_effective = new_local.reshape(2, 3, 3, 3) + new_shared[new_groups].reshape(2, 1, 3, 3)
    np.testing.assert_allclose(new_effective, old_effective, atol=1e-14)
    assert diagnostics["affected_slots"] == (0, 1)


def test_recompose_moves_unassigned_source_and_preserves_unrelated_unassigned_slot():
    local = np.arange(81, dtype=np.float64).reshape(3, 27) / 13.0
    shared = np.arange(18, dtype=np.float64).reshape(2, 9) / 7.0
    groups = np.array([-1, 1, -1], dtype=np.int64)
    common = np.array([[0.3, -0.2, 0.1], [0.5, 0.7, -0.4], [0.6, -0.8, 0.9]])
    old_effective = local.reshape(3, 3, 3, 3).copy()
    old_effective[groups >= 0] += shared[groups[groups >= 0]].reshape(-1, 1, 3, 3)
    old_local = local.copy()
    old_shared = shared.copy()

    new_local, new_shared, new_groups, diagnostics = recompose_shared(
        local, shared, groups, source_slots=[0], group=1, common=common
    )

    new_effective = new_local.reshape(3, 3, 3, 3).copy()
    new_effective[new_groups >= 0] += new_shared[new_groups[new_groups >= 0]].reshape(-1, 1, 3, 3)
    np.testing.assert_allclose(new_effective[[0, 1]], old_effective[[0, 1]], atol=1e-14)
    np.testing.assert_array_equal(new_groups, np.array([1, 1, -1]))
    np.testing.assert_array_equal(new_local[2], old_local[2])
    np.testing.assert_array_equal(new_shared[0], old_shared[0])
    assert diagnostics["affected_slots"] == (0, 1)


@pytest.mark.parametrize(
    "effective,psi,bands",
    [
        (np.full((1, 3, 3, 3), np.nan), [np.ones((1, 3))], [np.array([0])]),
        (np.zeros((1, 3, 3, 3)), [np.array([[np.nan, 0.0, 0.0]])], [np.array([0])]),
        (np.zeros((1, 3, 3, 3)), [np.ones((1, 3))], [np.array([3])]),
        (np.zeros((1, 3, 3, 3)), [np.empty((0, 3))], [np.empty((0,), dtype=np.int64)]),
    ],
)
def test_weighted_rejects_invalid_or_nonfinite_anchor_evidence(effective, psi, bands):
    with pytest.raises(ValueError):
        extract_common("weighted", effective, anchor_psi=psi, anchor_bands=bands)


def test_recompose_rejects_invalid_groups_and_slots():
    local = np.zeros((2, 27), dtype=np.float64)
    shared = np.zeros((2, 9), dtype=np.float64)

    with pytest.raises(ValueError):
        recompose_shared(local, shared, np.array([0, 2]), source_slots=[0], group=1, common=np.zeros((3, 3)))
    with pytest.raises(ValueError):
        recompose_shared(local, shared, np.array([0, 1]), source_slots=[1, 1], group=0, common=np.zeros((3, 3)))
