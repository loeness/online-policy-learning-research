import copy
import json

import pytest

from online_rl.h033_diagnostic import FEATURE_NAMES, crossfit_diagnostic, make_folds, make_targets, validate_config


SEEDS = list(range(100, 116))


def config():
    return {
        "format": "h033_diagnostic_config_v1",
        "seeds": SEEDS.copy(),
        "feature_names": list(FEATURE_NAMES),
        "primary_future_regimes": ["old_major", "new_major"],
        "secondary_future_regime": "balanced",
        "replicas": [0, 1],
        "horizon": 6144,
        "folds": 4,
        "ridge_alpha": 1.0,
        "scale_floor": 1e-12,
        "replace_threshold": 0.0,
    }


def feature_packet():
    rows = []
    for position, seed in enumerate(SEEDS):
        stratum = position // 4
        rows.append(
            {
                "seed": seed,
                "old_family": "A" if stratum < 2 else "B",
                "history_regime": "old_major" if stratum in (0, 2) else "new_major",
                "features": {
                    "recent_frequency_contrast": float(position - 7),
                    "new_fit_gain": float((position % 5) - 2),
                    "old_prior_loss": float((position * 3) % 7),
                },
                "provenance": {"source": "synthetic"},
            }
        )
    return {"format": "h033_features_v1", "rows": rows, "metadata": {"source": "synthetic"}}


def outcome_packet(delta_fn=None, balanced_delta_fn=None):
    if delta_fn is None:
        delta_fn = lambda seed: float(((seed - SEEDS[0]) % 5) - 2)
    if balanced_delta_fn is None:
        balanced_delta_fn = lambda seed: delta_fn(seed) + 4.0
    rows = []
    for seed in SEEDS:
        for future in ("old_major", "new_major", "balanced"):
            for replica in (0, 1):
                keep = float(10 + (seed - SEEDS[0]) * 0.25 + replica * 0.1)
                delta = balanced_delta_fn(seed) if future == "balanced" else delta_fn(seed)
                for arm in ("keep", "replace_behavioral"):
                    rows.append(
                        {
                            "seed": seed,
                            "future_regime": future,
                            "replica": replica,
                            "arm": arm,
                            "horizon": 6144,
                            "reward_per_step": keep if arm == "keep" else keep + delta,
                        }
                    )
    return {"format": "h033_outcomes_v1", "rows": rows, "metadata": {"source": "synthetic"}}


def fold_for(result, seed):
    return next(fold for fold in result["folds"] if seed in fold["test_seeds"])


def test_folds_hold_out_every_seed_once_with_four_strata_per_fold():
    folds = make_folds(feature_packet()["rows"], config())
    assert len(folds) == 4
    assert all(len(test_seeds) == 4 for test_seeds in folds)
    assert sorted(seed for test_seeds in folds for seed in test_seeds) == SEEDS
    assert len({seed for test_seeds in folds for seed in test_seeds}) == 16
    for test_seeds in folds:
        assert len(set(test_seeds)) == 4
        for family, history in (("A", "old_major"), ("A", "new_major"), ("B", "old_major"), ("B", "new_major")):
            matching = [
                row
                for row in feature_packet()["rows"]
                if row["seed"] in test_seeds
                and row["old_family"] == family
                and row["history_regime"] == history
            ]
            assert len(matching) == 1

    result = crossfit_diagnostic(feature_packet(), outcome_packet(), config())
    assert all(len(fold["train_seeds"]) == 12 for fold in result["folds"])
    for fold in result["folds"]:
        assert set(fold["train_seeds"]).isdisjoint(fold["test_seeds"])
        assert set(fold["train_seeds"]) | set(fold["test_seeds"]) == set(SEEDS)


def test_primary_target_is_equal_weighted_and_excludes_balanced():
    packet = outcome_packet(
        delta_fn=lambda seed: 2.0,
        balanced_delta_fn=lambda seed: 2000.0,
    )
    unequal_deltas = {("old_major", 0): -3.0, ("old_major", 1): 1.0,
                      ("new_major", 0): 5.0, ("new_major", 1): 9.0}
    for row in packet["rows"]:
        if row["arm"] == "replace_behavioral" and row["future_regime"] != "balanced":
            keep = 10 + (row["seed"] - SEEDS[0]) * 0.25 + row["replica"] * 0.1
            row["reward_per_step"] = keep + unequal_deltas[(row["future_regime"], row["replica"])]
    target = make_targets(packet, config())[SEEDS[0]]
    assert target["delta"] == pytest.approx(3.0)
    assert target["balanced_delta"] == pytest.approx(2000.0)
    assert target["keep"] == pytest.approx(10.05)
    assert target["replace"] == pytest.approx(13.05)


def test_heldout_outcome_change_does_not_change_its_fold_fit_or_prediction():
    features = feature_packet()
    outcomes = outcome_packet()
    seed = SEEDS[0]
    before = crossfit_diagnostic(features, outcomes, config())
    changed = copy.deepcopy(outcomes)
    for row in changed["rows"]:
        if row["seed"] == seed and row["future_regime"] in ("old_major", "new_major"):
            if row["arm"] == "replace_behavioral":
                row["reward_per_step"] += 100.0
    after = crossfit_diagnostic(features, changed, config())

    before_fold = fold_for(before, seed)
    after_fold = fold_for(after, seed)
    for key in ("train_seeds", "test_seeds", "mean", "scale", "beta", "intercept"):
        assert before_fold[key] == after_fold[key]
    before_row = next(row for row in before["rows"] if row["seed"] == seed)
    after_row = next(row for row in after["rows"] if row["seed"] == seed)
    assert before_row["predicted_delta"] == after_row["predicted_delta"]
    assert before_row["target_delta"] != after_row["target_delta"]


def test_balanced_outcomes_cannot_change_any_fit_prediction_or_decision():
    features = feature_packet()
    outcomes = outcome_packet()
    before = crossfit_diagnostic(features, outcomes, config())
    changed = copy.deepcopy(outcomes)
    for row in changed["rows"]:
        if row["future_regime"] == "balanced" and row["arm"] == "replace_behavioral":
            row["reward_per_step"] -= 1000.0
    after = crossfit_diagnostic(features, changed, config())
    assert before["folds"] == after["folds"]
    for left, right in zip(before["rows"], after["rows"]):
        for key in ("predicted_delta", "target_delta", "decisions", "values_primary"):
            assert left[key] == right[key]
        assert left["values_balanced"]["fixed_replace"] != right["values_balanced"]["fixed_replace"]


def test_heldout_feature_change_does_not_change_its_fold_scaler_or_coefficients():
    features = feature_packet()
    outcomes = outcome_packet()
    seed = SEEDS[0]
    before = crossfit_diagnostic(features, outcomes, config())
    changed = copy.deepcopy(features)
    row = next(row for row in changed["rows"] if row["seed"] == seed)
    row["features"]["recent_frequency_contrast"] = 100000.0
    row["features"]["new_fit_gain"] = -50000.0
    after = crossfit_diagnostic(changed, outcomes, config())

    before_fold = fold_for(before, seed)
    after_fold = fold_for(after, seed)
    for key in ("train_seeds", "test_seeds", "mean", "scale", "beta", "intercept"):
        assert before_fold[key] == after_fold[key]
    before_row = next(row for row in before["rows"] if row["seed"] == seed)
    after_row = next(row for row in after["rows"] if row["seed"] == seed)
    assert before_row["predicted_delta"] != after_row["predicted_delta"]


def test_constant_feature_column_uses_scale_one():
    features = feature_packet()
    for row in features["rows"]:
        row["features"]["new_fit_gain"] = 7.0
    result = crossfit_diagnostic(features, outcome_packet(), config())
    for fold in result["folds"]:
        assert fold["mean"][1] == 7.0
        assert fold["scale"][1] == 1.0
        assert fold["beta"][1] == 0.0


def test_zero_prediction_ties_choose_keep():
    features = feature_packet()
    outcomes = outcome_packet(delta_fn=lambda seed: 0.0, balanced_delta_fn=lambda seed: 0.0)
    result = crossfit_diagnostic(features, outcomes, config())
    assert all(row["predicted_delta"] == 0.0 for row in result["rows"])
    assert all(row["decisions"]["crossfit"] == "keep" for row in result["rows"])


def test_selected_values_and_gains_are_paired_against_each_seed_keep_value():
    targets = make_targets(outcome_packet(), config())
    result = crossfit_diagnostic(feature_packet(), outcome_packet(), config())
    assert result["format"] == "h033_crossfit_v1"
    assert result["scope"] == "exploratory_reuse_not_confirmation"
    assert set(result["aggregates"]) == {
        "crossfit",
        "fixed_keep",
        "fixed_replace",
        "intercept_only",
        "frequency_rule",
    }
    for row in result["rows"]:
        target = targets[row["seed"]]
        keep = target["keep"]
        for method, decision in row["decisions"].items():
            expected_primary = row["values_primary"][method]
            assert expected_primary == pytest.approx(target["keep"] if decision == "keep" else target["replace"])
            assert row["primary_gain_vs_keep"][method] == pytest.approx(expected_primary - keep)
        assert row["values_balanced"]["fixed_keep"] < row["values_balanced"]["fixed_replace"]

    json.dumps(result, allow_nan=False)
    for method, summary in result["aggregates"].items():
        values = [row["values_primary"][method] for row in result["rows"]]
        gains = [row["primary_gain_vs_keep"][method] for row in result["rows"]]
        assert summary["mean_selected_primary"] == pytest.approx(sum(values) / len(values))
        assert summary["mean_primary_gain_vs_keep"] == pytest.approx(sum(gains) / len(gains))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda packet: packet["rows"].pop(),
        lambda packet: packet["rows"].__setitem__(15, copy.deepcopy(packet["rows"][0])),
        lambda packet: packet["rows"][0]["features"].__setitem__("recent_frequency_contrast", float("nan")),
        lambda packet: packet["rows"][0]["features"].__setitem__("unknown", 1.0),
        lambda packet: packet["rows"][0].__setitem__("old_family", "C"),
    ],
)
def test_invalid_feature_packets_are_rejected(mutate):
    packet = feature_packet()
    mutate(packet)
    with pytest.raises(ValueError):
        crossfit_diagnostic(packet, outcome_packet(), config())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda packet: packet["rows"].pop(),
        lambda packet: packet["rows"].__setitem__(-1, copy.deepcopy(packet["rows"][0])),
        lambda packet: packet["rows"][0].__setitem__("reward_per_step", float("nan")),
        lambda packet: packet["rows"][0].__setitem__("horizon", 6143),
        lambda packet: packet["rows"][0].__setitem__("arm", "other"),
    ],
)
def test_invalid_outcome_packets_are_rejected(mutate):
    packet = outcome_packet()
    mutate(packet)
    with pytest.raises(ValueError):
        make_targets(packet, config())


def test_config_rejects_boolean_seed_and_unknown_feature():
    invalid_seed = config()
    invalid_seed["seeds"][0] = True
    with pytest.raises(ValueError):
        validate_config(invalid_seed)
    invalid_feature = config()
    invalid_feature["feature_names"][0] = "future_feature"
    with pytest.raises(ValueError):
        validate_config(invalid_feature)

