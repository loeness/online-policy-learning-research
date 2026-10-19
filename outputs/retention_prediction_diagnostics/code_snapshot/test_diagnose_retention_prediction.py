from __future__ import annotations

from scripts.diagnose_retention_prediction import (
    _discounted_action_mass,
    _feature_specs,
    _target_values,
    prequential_predictions,
)


FEATURES = (
    "bias",
    "candidate_minus_incumbent_recent_group_frequency",
    "candidate_minus_incumbent_recovery_ewma_over_0_02",
    "normalized_log_candidate_minus_parent_source_fit",
    "decisions_since_parent_write_over_16",
    "recent_eviction_fraction8",
)


def test_feature_variants_cover_intercept_full_features_leave_one_out_and_baselines():
    specs = dict(_feature_specs(FEATURES))

    assert len(specs) == 9
    assert specs["intercept"] == (0,)
    assert specs["full6"] == tuple(range(6))
    assert specs["drop_candidate_minus_incumbent_recent_group_frequency"] == (0, 2, 3, 4, 5)
    assert specs["reward_ewma_only"] == (0, 2)
    assert specs["time_age_only"] == (0, 4)
    assert all(len(indices) == 5 for name, indices in specs.items() if name.startswith("drop_"))


def test_prequential_prediction_uses_current_label_only_after_scoring_it():
    features = [[1.0], [1.0], [1.0], [1.0]]
    actions = ["keep"] * 4
    update_flags = [True] * 4
    original = prequential_predictions(features, actions, [2.0, 4.0, 8.0, 16.0], update_flags, (0,), 1)
    changed_current_tail_label = prequential_predictions(
        features, actions, [2.0, 400.0, 8.0, 16.0], update_flags, (0,), 1
    )

    # With ridge 1, the first update leaves a Gram value of 2 and target vector of 2.
    assert original[0]["decision_index"] == 1
    assert original[0]["prediction"] == 1.0
    assert changed_current_tail_label[0]["prediction"] == original[0]["prediction"]
    # The changed label enters only after decision 1 is scored and affects decision 2.
    assert changed_current_tail_label[1]["prediction"] != original[1]["prediction"]


def test_target_decomposition_uses_cpu_per_block_step():
    values = _target_values(
        [{"reward_rate": 0.03, "utility_cpu_s": 1.5, "block_steps": 100}]
    )

    assert values["reward_rate"] == [0.03]
    assert values["cpu_rate"] == [0.015]
    assert values["utility"] == [0.015]


def test_discounted_observation_mass_decays_once_per_selected_feedback_update():
    mass = _discounted_action_mass(["keep", "replace", "keep"], [True, True, False])

    assert mass["keep"] == 0.97
    assert mass["replace"] == 1.0
    assert mass["mean"] == 0.0
    assert mass["none"] == 0.0
