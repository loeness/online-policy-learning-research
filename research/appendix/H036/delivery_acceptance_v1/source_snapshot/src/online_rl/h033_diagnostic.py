"""Minimal offline action-ranking diagnostic for H033.

The module consumes the frozen ``h033_features_v1`` and
``h033_outcomes_v1`` packets. It fits a four-fold, train-only standardized
ridge predictor for the primary (old-major/new-major) outcome delta. The
balanced outcome is held out from fitting and only used to score the same
decisions. This is exploratory reuse, not confirmation.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import numpy as np


FEATURE_NAMES = (
    "recent_frequency_contrast",
    "new_fit_gain",
    "old_prior_loss",
)

_CONFIG_KEYS = {
    "format",
    "seeds",
    "feature_names",
    "primary_future_regimes",
    "secondary_future_regime",
    "replicas",
    "horizon",
    "folds",
    "ridge_alpha",
    "scale_floor",
    "replace_threshold",
}
_FEATURE_PACKET_KEYS = {"format", "rows", "metadata"}
_FEATURE_ROW_KEYS = {"seed", "old_family", "history_regime", "features", "provenance"}
_OUTCOME_PACKET_KEYS = {"format", "rows", "metadata"}
_OUTCOME_ROW_KEYS = {
    "seed",
    "future_regime",
    "replica",
    "arm",
    "horizon",
    "reward_per_step",
}
_FUTURE_REGIMES = ("old_major", "new_major", "balanced")
_PRIMARY_REGIMES = ("old_major", "new_major")
_ARMS = ("keep", "replace_behavioral")
_STRATA = (("A", "old_major"), ("A", "new_major"), ("B", "old_major"), ("B", "new_major"))
_METHODS = ("crossfit", "fixed_keep", "fixed_replace", "intercept_only", "frequency_rule")


def _object(value: Any, label: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ValueError(f"{label} keys mismatch (missing={missing}, extra={extra})")
    return value


def _is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _integer(value: Any, label: str) -> int:
    if _is_bool(value) or not isinstance(value, (Integral, np.integer)):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _finite_number(value: Any, label: str) -> float:
    if _is_bool(value) or not isinstance(value, (Real, np.integer, np.floating)):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _check_fixed_number(config: dict[str, Any], key: str, expected: float) -> float:
    value = _finite_number(config[key], f"config.{key}")
    if value != expected:
        raise ValueError(f"config.{key} must equal {expected}")
    return value


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the canonical H033 v1 configuration."""
    config = _object(config, "config", _CONFIG_KEYS)
    if config["format"] != "h033_diagnostic_config_v1":
        raise ValueError("config.format must be 'h033_diagnostic_config_v1'")

    seeds = _list(config["seeds"], "config.seeds")
    if len(seeds) != 16:
        raise ValueError("config.seeds must contain exactly 16 seed IDs")
    seed_ids = [_integer(seed, "config seed") for seed in seeds]
    if len(set(seed_ids)) != 16:
        raise ValueError("config.seeds must not contain duplicates")

    feature_names = _list(config["feature_names"], "config.feature_names")
    if feature_names != list(FEATURE_NAMES):
        raise ValueError(f"config.feature_names must equal {list(FEATURE_NAMES)}")
    if config["primary_future_regimes"] != list(_PRIMARY_REGIMES):
        raise ValueError("config.primary_future_regimes must be ['old_major', 'new_major']")
    if config["secondary_future_regime"] != "balanced":
        raise ValueError("config.secondary_future_regime must be 'balanced'")

    replicas = _list(config["replicas"], "config.replicas")
    if len(replicas) != 2 or [_integer(replica, "config replica") for replica in replicas] != [0, 1]:
        raise ValueError("config.replicas must be [0, 1]")
    if _integer(config["horizon"], "config.horizon") != 6144:
        raise ValueError("config.horizon must equal 6144")
    if _integer(config["folds"], "config.folds") != 4:
        raise ValueError("config.folds must equal 4")

    ridge_alpha = _check_fixed_number(config, "ridge_alpha", 1.0)
    scale_floor = _check_fixed_number(config, "scale_floor", 1e-12)
    replace_threshold = _check_fixed_number(config, "replace_threshold", 0.0)

    return {
        "format": "h033_diagnostic_config_v1",
        "seeds": seed_ids,
        "feature_names": list(FEATURE_NAMES),
        "primary_future_regimes": list(_PRIMARY_REGIMES),
        "secondary_future_regime": "balanced",
        "replicas": [0, 1],
        "horizon": 6144,
        "folds": 4,
        "ridge_alpha": ridge_alpha,
        "scale_floor": scale_floor,
        "replace_threshold": replace_threshold,
    }


def _validate_feature_rows(rows: Any, config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = _list(rows, "feature rows")
    if len(rows) != 16:
        raise ValueError("feature rows must contain exactly one row for each configured seed")
    configured_seeds = set(config["seeds"])
    by_seed: dict[int, dict[str, Any]] = {}
    strata_counts = {stratum: 0 for stratum in _STRATA}

    for index, raw_row in enumerate(rows):
        row = _object(raw_row, f"feature rows[{index}]", _FEATURE_ROW_KEYS)
        seed = _integer(row["seed"], f"feature rows[{index}].seed")
        if seed not in configured_seeds:
            raise ValueError(f"feature row has unlisted seed {seed}")
        if seed in by_seed:
            raise ValueError(f"duplicate feature row for seed {seed}")

        family = row["old_family"]
        history = row["history_regime"]
        if family not in ("A", "B"):
            raise ValueError(f"feature rows[{index}].old_family must be 'A' or 'B'")
        if history not in ("old_major", "new_major"):
            raise ValueError(f"feature rows[{index}].history_regime is invalid")

        raw_features = row["features"]
        if not isinstance(raw_features, dict) or set(raw_features) != set(FEATURE_NAMES):
            raise ValueError(f"feature rows[{index}].features must contain exactly {list(FEATURE_NAMES)}")
        features = {
            name: _finite_number(raw_features[name], f"feature rows[{index}].features.{name}")
            for name in FEATURE_NAMES
        }
        if not isinstance(row["provenance"], dict):
            raise ValueError(f"feature rows[{index}].provenance must be an object")

        strata = (family, history)
        strata_counts[strata] += 1
        by_seed[seed] = {
            "seed": seed,
            "old_family": family,
            "history_regime": history,
            "features": features,
            "provenance": row["provenance"],
        }

    missing = sorted(configured_seeds - set(by_seed))
    if missing:
        raise ValueError(f"feature rows are missing configured seeds {missing}")
    if any(count != 4 for count in strata_counts.values()):
        raise ValueError(f"each of the four feature strata must contain exactly 4 seeds; got {strata_counts}")
    return by_seed


def _validate_feature_packet(packet: Any, config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    packet = _object(packet, "feature packet", _FEATURE_PACKET_KEYS)
    if packet["format"] != "h033_features_v1":
        raise ValueError("feature packet format must be 'h033_features_v1'")
    if not isinstance(packet["metadata"], dict):
        raise ValueError("feature packet metadata must be an object")
    return _validate_feature_rows(packet["rows"], config)


def _validate_outcome_packet(packet: Any, config: dict[str, Any]) -> dict[tuple[int, str, int, str], float]:
    packet = _object(packet, "outcome packet", _OUTCOME_PACKET_KEYS)
    if packet["format"] != "h033_outcomes_v1":
        raise ValueError("outcome packet format must be 'h033_outcomes_v1'")
    if not isinstance(packet["metadata"], dict):
        raise ValueError("outcome packet metadata must be an object")

    rows = _list(packet["rows"], "outcome packet rows")
    expected_count = 16 * len(_FUTURE_REGIMES) * 2 * len(_ARMS)
    if len(rows) != expected_count:
        raise ValueError(f"outcome packet must contain exactly {expected_count} rows")
    configured_seeds = set(config["seeds"])
    by_cell: dict[tuple[int, str, int, str], float] = {}

    for index, raw_row in enumerate(rows):
        row = _object(raw_row, f"outcome rows[{index}]", _OUTCOME_ROW_KEYS)
        seed = _integer(row["seed"], f"outcome rows[{index}].seed")
        if seed not in configured_seeds:
            raise ValueError(f"outcome row has unlisted seed {seed}")
        future = row["future_regime"]
        if future not in _FUTURE_REGIMES:
            raise ValueError(f"outcome rows[{index}].future_regime is invalid")
        replica = _integer(row["replica"], f"outcome rows[{index}].replica")
        if replica not in (0, 1):
            raise ValueError(f"outcome rows[{index}].replica must be 0 or 1")
        arm = row["arm"]
        if arm not in _ARMS:
            raise ValueError(f"outcome rows[{index}].arm is invalid")
        horizon = _integer(row["horizon"], f"outcome rows[{index}].horizon")
        if horizon != 6144:
            raise ValueError(f"outcome rows[{index}].horizon must equal 6144")
        reward = _finite_number(row["reward_per_step"], f"outcome rows[{index}].reward_per_step")
        cell = (seed, future, replica, arm)
        if cell in by_cell:
            raise ValueError(f"duplicate outcome cell {cell}")
        by_cell[cell] = reward

    expected = {
        (seed, future, replica, arm)
        for seed in config["seeds"]
        for future in _FUTURE_REGIMES
        for replica in (0, 1)
        for arm in _ARMS
    }
    missing = expected - set(by_cell)
    extra = set(by_cell) - expected
    if missing or extra:
        raise ValueError(f"outcome packet cells mismatch (missing={len(missing)}, extra={len(extra)})")
    return by_cell


def _finite_mean(values: list[float], label: str) -> float:
    if not values:
        raise ValueError(f"cannot average an empty set for {label}")
    magnitude = max(abs(value) for value in values)
    if magnitude == 0.0:
        return 0.0
    scaled_mean = math.fsum(value / magnitude for value in values) / len(values)
    result = magnitude * scaled_mean
    if not math.isfinite(result):
        raise ValueError(f"non-finite mean for {label}")
    return result


def _difference(left: float, right: float, label: str) -> float:
    result = left - right
    if not math.isfinite(result):
        raise ValueError(f"non-finite difference for {label}")
    return result


def _targets_from_cells(cells: dict[tuple[int, str, int, str], float], config: dict[str, Any]) -> dict[int, dict[str, float]]:
    targets: dict[int, dict[str, float]] = {}
    for seed in config["seeds"]:
        primary: dict[str, float] = {}
        balanced: dict[str, float] = {}
        for arm in _ARMS:
            primary[arm] = _finite_mean(
                [
                    cells[(seed, future, replica, arm)]
                    for future in _PRIMARY_REGIMES
                    for replica in (0, 1)
                ],
                f"seed {seed} primary {arm}",
            )
            balanced[arm] = _finite_mean(
                [cells[(seed, "balanced", replica, arm)] for replica in (0, 1)],
                f"seed {seed} balanced {arm}",
            )
        targets[seed] = {
            "keep": primary["keep"],
            "replace": primary["replace_behavioral"],
            "delta": _difference(primary["replace_behavioral"], primary["keep"], f"seed {seed} primary delta"),
            "balanced_keep": balanced["keep"],
            "balanced_replace": balanced["replace_behavioral"],
            "balanced_delta": _difference(
                balanced["replace_behavioral"], balanced["keep"], f"seed {seed} balanced delta"
            ),
        }
    return targets


def make_targets(outcomes: dict[str, Any], config: dict[str, Any]) -> dict[int, dict[str, float]]:
    """Return equal-weight per-seed primary and balanced arm values."""
    canonical = validate_config(config)
    cells = _validate_outcome_packet(outcomes, canonical)
    return _targets_from_cells(cells, canonical)


def make_folds(feature_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[list[int]]:
    """Assign one seed from each of four strata to each deterministic fold."""
    canonical = validate_config(config)
    rows = _validate_feature_rows(feature_rows, canonical)
    folds: list[list[int]] = [[] for _ in range(4)]
    for stratum in _STRATA:
        stratum_seeds = sorted(
            seed
            for seed, row in rows.items()
            if (row["old_family"], row["history_regime"]) == stratum
        )
        if len(stratum_seeds) != 4:
            raise ValueError(f"stratum {stratum} must contain exactly 4 seeds")
        for fold_index, seed in enumerate(stratum_seeds):
            folds[fold_index].append(seed)
    return [sorted(fold) for fold in folds]


def _column_mean_std(values: np.ndarray, label: str) -> tuple[float, float]:
    magnitude = float(np.max(np.abs(values)))
    if not math.isfinite(magnitude):
        raise ValueError(f"non-finite training values for {label}")
    if magnitude == 0.0:
        return 0.0, 0.0
    scaled = values / magnitude
    scaled_mean = math.fsum(float(value) for value in scaled) / len(values)
    mean = magnitude * scaled_mean
    scaled_variance = math.fsum(float(value - scaled_mean) ** 2 for value in scaled) / len(values)
    std = magnitude * math.sqrt(scaled_variance)
    if not math.isfinite(mean) or not math.isfinite(std):
        raise ValueError(f"non-finite training statistics for {label}")
    return mean, std


def _standardize_row(features: dict[str, float], names: list[str], mean: list[float], scale: list[float]) -> np.ndarray:
    standardized: list[float] = []
    for index, name in enumerate(names):
        value = features[name]
        center = mean[index]
        magnitude = max(abs(value), abs(center))
        if magnitude == 0.0:
            standardized.append(0.0)
            continue
        scaled_difference = value / magnitude - center / magnitude
        result = scaled_difference * (magnitude / scale[index])
        if not math.isfinite(result):
            raise ValueError(f"non-finite standardized feature {name}")
        standardized.append(result)
    return np.asarray(standardized, dtype=np.float64)


def _fit_fold(
    train_seeds: list[int],
    feature_by_seed: dict[int, dict[str, Any]],
    targets: dict[int, dict[str, float]],
    config: dict[str, Any],
) -> tuple[list[float], list[float], list[float], float]:
    names = config["feature_names"]
    x_train = np.asarray(
        [[feature_by_seed[seed]["features"][name] for name in names] for seed in train_seeds],
        dtype=np.float64,
    )
    y_train = np.asarray([targets[seed]["delta"] for seed in train_seeds], dtype=np.float64)
    mean: list[float] = []
    raw_std: list[float] = []
    for column_index, name in enumerate(names):
        column_mean, column_std = _column_mean_std(x_train[:, column_index], name)
        mean.append(column_mean)
        raw_std.append(column_std)
    scale = [1.0 if std < config["scale_floor"] else std for std in raw_std]
    z_train = np.vstack(
        [_standardize_row(dict(zip(names, row.tolist())), names, mean, scale) for row in x_train]
    )
    intercept = _finite_mean([float(value) for value in y_train], "training primary delta")
    centered_y = y_train - intercept
    if not np.all(np.isfinite(centered_y)) or not np.all(np.isfinite(z_train)):
        raise ValueError("non-finite centered training data")
    gram = z_train.T @ z_train + np.eye(len(names), dtype=np.float64) * config["ridge_alpha"]
    right = z_train.T @ centered_y
    try:
        beta_array = np.linalg.solve(gram, right)
    except np.linalg.LinAlgError as exc:
        raise ValueError("ridge solve failed") from exc
    beta = [float(value) for value in beta_array]
    if not all(math.isfinite(value) for value in [*mean, *scale, *beta, intercept]):
        raise ValueError("non-finite fitted ridge parameters")
    return mean, scale, beta, intercept


def _selected_value(target: dict[str, float], decision: str, balanced: bool = False) -> float:
    if decision not in ("keep", "replace"):
        raise ValueError(f"unknown decision {decision}")
    if balanced:
        return target["balanced_keep" if decision == "keep" else "balanced_replace"]
    return target["keep" if decision == "keep" else "replace"]


def crossfit_diagnostic(
    feature_packet: dict[str, Any], outcome_packet: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Run the frozen cross-fit diagnostic without balanced-outcome leakage."""
    canonical = validate_config(config)
    feature_by_seed = _validate_feature_packet(feature_packet, canonical)
    outcome_cells = _validate_outcome_packet(outcome_packet, canonical)
    targets = _targets_from_cells(outcome_cells, canonical)
    test_folds = make_folds(feature_packet["rows"], canonical)

    model_by_fold: list[dict[str, Any]] = []
    fold_for_seed: dict[int, int] = {}
    for fold_index, test_seeds in enumerate(test_folds):
        train_seeds = sorted(set(canonical["seeds"]) - set(test_seeds))
        mean, scale, beta, intercept = _fit_fold(train_seeds, feature_by_seed, targets, canonical)
        model_by_fold.append(
            {
                "fold": fold_index,
                "train_seeds": train_seeds,
                "test_seeds": test_seeds,
                "mean": mean,
                "scale": scale,
                "beta": beta,
                "intercept": intercept,
            }
        )
        for seed in test_seeds:
            fold_for_seed[seed] = fold_index

    rows: list[dict[str, Any]] = []
    for seed in sorted(canonical["seeds"]):
        fold_index = fold_for_seed[seed]
        model = model_by_fold[fold_index]
        feature_values = feature_by_seed[seed]["features"]
        z_test = _standardize_row(feature_values, canonical["feature_names"], model["mean"], model["scale"])
        predicted_delta = model["intercept"] + float(np.dot(z_test, np.asarray(model["beta"], dtype=np.float64)))
        if not math.isfinite(predicted_delta):
            raise ValueError(f"non-finite prediction for seed {seed}")

        decisions = {
            "crossfit": "replace" if predicted_delta > canonical["replace_threshold"] else "keep",
            "fixed_keep": "keep",
            "fixed_replace": "replace",
            "intercept_only": "replace" if model["intercept"] > canonical["replace_threshold"] else "keep",
            "frequency_rule": "replace"
            if feature_values["recent_frequency_contrast"] > canonical["replace_threshold"]
            else "keep",
        }
        target = targets[seed]
        values_primary = {method: _selected_value(target, decision) for method, decision in decisions.items()}
        values_balanced = {
            method: _selected_value(target, decision, balanced=True) for method, decision in decisions.items()
        }
        gains = {method: _difference(value, target["keep"], f"seed {seed} {method} primary gain") for method, value in values_primary.items()}
        rows.append(
            {
                "seed": seed,
                "fold": fold_index,
                "features": {name: feature_values[name] for name in canonical["feature_names"]},
                "predicted_delta": float(predicted_delta),
                "target_delta": target["delta"],
                "decisions": decisions,
                "values_primary": values_primary,
                "values_balanced": values_balanced,
                "primary_gain_vs_keep": gains,
            }
        )

    aggregates: dict[str, dict[str, float]] = {}
    for method in _METHODS:
        aggregates[method] = {
            "mean_selected_primary": _finite_mean(
                [row["values_primary"][method] for row in rows], f"{method} mean selected primary"
            ),
            "mean_primary_gain_vs_keep": _finite_mean(
                [row["primary_gain_vs_keep"][method] for row in rows], f"{method} mean primary gain"
            ),
            "mean_selected_balanced": _finite_mean(
                [row["values_balanced"][method] for row in rows], f"{method} mean selected balanced"
            ),
            "mean_balanced_gain_vs_keep": _finite_mean(
                [
                    _difference(
                        row["values_balanced"][method],
                        targets[row["seed"]]["balanced_keep"],
                        f"seed {row['seed']} {method} balanced gain",
                    )
                    for row in rows
                ],
                f"{method} mean balanced gain",
            ),
        }

    result = {
        "format": "h033_crossfit_v1",
        "scope": "exploratory_reuse_not_confirmation",
        "feature_names": list(canonical["feature_names"]),
        "folds": model_by_fold,
        "rows": rows,
        "aggregates": aggregates,
    }
    _assert_finite_tree(result)
    return result


def _assert_finite_tree(value: Any, path: str = "result") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{path}[{index}]")

