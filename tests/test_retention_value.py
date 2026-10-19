import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl.retention_value import RetentionValueModel  # noqa: E402


class _FixedRng:
    def __init__(self, random_value, action_index=None):
        self.random_value = random_value
        self.action_index = action_index

    def random(self):
        return self.random_value

    def integers(self, high):
        assert self.action_index is not None
        assert 0 <= self.action_index < high
        return self.action_index


def test_update_uses_only_selected_outcome_and_discounts_all_action_evidence():
    model = RetentionValueModel(2, ridge=2.0, discount=0.5, lambda_cpu=4.0)
    first_features = np.asarray([1.0, 2.0])
    first_utility = model.update("keep", first_features, 3.0, 1.0, 2)
    first_features[:] = -99.0  # Inputs are consumed by value and never retained.

    assert first_utility == pytest.approx(1.0)
    np.testing.assert_array_equal(model.A[0], 2.0 * np.eye(2) + np.outer([1.0, 2.0], [1.0, 2.0]))
    np.testing.assert_array_equal(model.b[0], [1.0, 2.0])

    second_features = np.asarray([2.0, 0.0])
    second_utility = model.update("mean", second_features, 2.0, 0.0, 1)

    assert second_utility == pytest.approx(2.0)
    np.testing.assert_allclose(model.A[0], 2.0 * np.eye(2) + 0.5 * np.outer([1.0, 2.0], [1.0, 2.0]))
    np.testing.assert_allclose(model.b[0], [0.5, 1.0])
    np.testing.assert_array_equal(model.A[1], 2.0 * np.eye(2))
    np.testing.assert_array_equal(model.b[1], [0.0, 0.0])
    np.testing.assert_array_equal(model.A[2], 2.0 * np.eye(2) + np.outer([2.0, 0.0], [2.0, 0.0]))
    np.testing.assert_array_equal(model.b[2], [4.0, 0.0])
    np.testing.assert_array_equal(model.A[3], 2.0 * np.eye(2))
    np.testing.assert_array_equal(model.b[3], [0.0, 0.0])
    np.testing.assert_array_equal(model.counts, [1, 0, 1, 0])
    np.testing.assert_allclose(model.discounted_counts, [0.5, 0.0, 1.0, 0.0])


def test_prediction_ties_use_configured_default_and_uncertainty_is_diagnostic():
    model = RetentionValueModel(2, default_action="mean")

    assert model.predict([0.5, 1.0]) == {
        "keep": 0.0,
        "replace": 0.0,
        "mean": 0.0,
        "none": 0.0,
    }
    choice = model.choose([0.5, 1.0])
    assert choice.selected == "mean"
    assert choice.action == "mean"
    assert choice.propensity == 1.0
    assert choice.explored is False

    means, leverage = model.predict_with_uncertainty([0.5, 1.0])
    assert means == choice.predicted_values
    expected = (0.5**2 + 1.0**2) / model.ridge
    assert leverage == {action: pytest.approx(expected) for action in model.ACTIONS}


def test_epsilon_choice_reports_marginal_propensity_for_both_mixture_branches():
    model = RetentionValueModel(1)
    model.update("replace", [1.0], 4.0, 0.0, 1)

    random_action = model.choose([1.0], epsilon=0.2, rng=_FixedRng(0.1, action_index=0))
    assert random_action.selected == "keep"
    assert random_action.predicted_values["replace"] > random_action.predicted_values["keep"]
    assert random_action.propensity == pytest.approx(0.05)
    assert random_action.explored is True

    greedy_action = model.choose([1.0], epsilon=0.2, rng=_FixedRng(0.9))
    assert greedy_action.selected == "replace"
    assert greedy_action.propensity == pytest.approx(0.85)
    assert greedy_action.explored is False

    random_greedy = model.choose([1.0], epsilon=0.2, rng=_FixedRng(0.1, action_index=1))
    assert random_greedy.selected == "replace"
    assert random_greedy.propensity == pytest.approx(0.85)
    assert random_greedy.explored is True


def test_state_round_trip_is_exact_independent_and_fixed_size():
    model = RetentionValueModel(3, ridge=0.25, discount=0.9, lambda_cpu=1.5)
    model.update("none", [1.0, -2.0, 0.5], 0.75, 2.0, 4)
    expected_bytes = 4 * 3 * 3 * 8 + 4 * 3 * 8 + 4 * 8 + 4 * 8
    snapshot = model.state_dict()
    restored = RetentionValueModel.from_state_dict(snapshot)

    assert model.numeric_nbytes() == expected_bytes
    assert restored.numeric_nbytes() == expected_bytes
    for key in ("A", "b", "counts", "discounted_counts"):
        np.testing.assert_array_equal(getattr(model, key), getattr(restored, key))
    assert restored.predict([0.4, 0.2, -0.1]) == model.predict([0.4, 0.2, -0.1])

    snapshot["A"][:] = 0.0
    snapshot["b"][:] = 0.0
    snapshot["counts"][:] = 99
    assert model.numeric_nbytes() == expected_bytes
    assert np.linalg.cholesky(model.A[0]) is not None
    assert restored.counts[3] == 1


def test_constructor_update_and_restore_validate_inputs():
    with pytest.raises(ValueError, match="n_features"):
        RetentionValueModel(0)
    with pytest.raises(ValueError, match="ridge"):
        RetentionValueModel(1, ridge=0.0)
    with pytest.raises(ValueError, match="discount"):
        RetentionValueModel(1, discount=1.01)
    with pytest.raises(ValueError, match="discount"):
        RetentionValueModel(1, discount=np.nan)

    model = RetentionValueModel(2)
    with pytest.raises(ValueError, match="shape"):
        model.predict([1.0])
    with pytest.raises(ValueError, match="finite"):
        model.predict([1.0, np.inf])
    with pytest.raises(ValueError, match="external NumPy Generator"):
        model.choose([1.0, 0.0], epsilon=0.1)
    with pytest.raises(ValueError, match="environment_steps"):
        model.update("keep", [1.0, 0.0], 1.0, 0.0, 0)
    with pytest.raises(ValueError, match="cpu_s"):
        model.update("keep", [1.0, 0.0], 1.0, -1.0, 1)

    bad_state = model.state_dict()
    bad_state["A"] = np.zeros((4, 1, 1))
    with pytest.raises(ValueError, match="A must have shape"):
        RetentionValueModel.from_state_dict(bad_state)
