"""Offline prequential diagnostics for selected-feedback retention value logs.

This module reads existing logs only. It never constructs or steps an
environment, and all target errors are scored for the action whose feedback
was actually observed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from online_rl.retention_value import RetentionValueModel


SCRIPT_VERSION = "retention_prediction_diagnostics_v1"
EXPECTED_RIDGE = 1.0
EXPECTED_DISCOUNT = 0.97
EXPECTED_LAMBDA_CPU = 1.0
DEFAULT_TAIL_DECISIONS = 32
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_261_019
FEATURE_REWARD_EWMA = "candidate_minus_incumbent_recovery_ewma_over_0_02"
FEATURE_TIME_AGE = "decisions_since_parent_write_over_16"
TARGETS = ("reward_rate", "cpu_rate", "utility")
ACTIONS = RetentionValueModel.ACTIONS


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric data")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite numeric data") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite numeric data")
    return number


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _jsonl_row_count(path: Path) -> tuple[int | None, str | None]:
    if not path.is_file():
        return None, "missing"
    count = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    return count, f"line {line_number} is not an object"
                count += 1
    except (OSError, json.JSONDecodeError) as exc:
        return count, f"{type(exc).__name__}: {exc}"
    return count, None


def _path_from_summary(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _feature_specs(feature_names: Sequence[str]) -> list[tuple[str, tuple[int, ...]]]:
    names = tuple(feature_names)
    if len(names) != 6:
        raise ValueError(f"expected six logged features, found {len(names)}")
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    if "bias" not in names:
        raise ValueError("feature list has no bias feature")
    bias = names.index("bias")
    all_features = tuple(range(len(names)))
    specs: list[tuple[str, tuple[int, ...]]] = [
        ("intercept", (bias,)),
        ("full6", all_features),
    ]
    for index, name in enumerate(names):
        if index == bias:
            continue
        specs.append((f"drop_{name}", tuple(i for i in all_features if i != index)))
    try:
        ewma = names.index(FEATURE_REWARD_EWMA)
        age = names.index(FEATURE_TIME_AGE)
    except ValueError as exc:
        raise ValueError("feature list lacks the EWMA or time-age baseline feature") from exc
    specs.extend((("reward_ewma_only", (bias, ewma)), ("time_age_only", (bias, age))))
    return specs


def prequential_predictions(
    features: Sequence[Sequence[float]],
    actions: Sequence[str],
    targets: Sequence[float],
    update_flags: Sequence[bool],
    feature_indices: Sequence[int],
    tail_start: int,
) -> list[dict[str, Any]]:
    """Replay selected-feedback ridge predictions before each update."""

    count = len(features)
    if not (len(actions) == len(targets) == len(update_flags) == count):
        raise ValueError("features, actions, targets and update flags must align")
    if not 0 <= tail_start <= count:
        raise ValueError("tail_start is outside the stream")
    if not feature_indices:
        raise ValueError("at least one feature must be selected")
    model = RetentionValueModel(
        len(feature_indices),
        ridge=EXPECTED_RIDGE,
        discount=EXPECTED_DISCOUNT,
        lambda_cpu=0.0,
    )
    scored: list[dict[str, Any]] = []
    for index, (full_vector, action, target, should_update) in enumerate(
        zip(features, actions, targets, update_flags)
    ):
        vector = np.asarray(full_vector, dtype=np.float64)
        selected_vector = vector[list(feature_indices)]
        prediction = model.predict(selected_vector)[action]
        target_value = _finite(target, f"targets[{index}]")
        if index >= tail_start:
            error = prediction - target_value
            scored.append(
                {
                    "decision_index": index,
                    "selected_action": action,
                    "prediction": float(prediction),
                    "target": target_value,
                    "error": float(error),
                }
            )
        if should_update:
            model.update(action, selected_vector, target_value, 0.0, 1)
    return scored


def _target_values(feedback_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    reward_values: list[float] = []
    cpu_values: list[float] = []
    utility_values: list[float] = []
    for index, row in enumerate(feedback_rows):
        reward_rate = _finite(row.get("reward_rate"), f"feedback[{index}].reward_rate")
        utility_cpu_s = _finite(row.get("utility_cpu_s"), f"feedback[{index}].utility_cpu_s")
        block_steps_value = _finite(row.get("block_steps"), f"feedback[{index}].block_steps")
        if block_steps_value <= 0 or not block_steps_value.is_integer():
            raise ValueError(f"feedback[{index}].block_steps must be a positive integer")
        cpu_rate = utility_cpu_s / int(block_steps_value)
        reward_values.append(reward_rate)
        cpu_values.append(cpu_rate)
        utility_values.append(reward_rate - cpu_rate)
    return {"reward_rate": reward_values, "cpu_rate": cpu_values, "utility": utility_values}


def _discounted_action_mass(
    actions: Sequence[str], update_flags: Sequence[bool]
) -> dict[str, float]:
    if len(actions) != len(update_flags):
        raise ValueError("actions and update flags must align")
    mass = {action: 0.0 for action in ACTIONS}
    for action, should_update in zip(actions, update_flags):
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action}")
        if should_update:
            for key in ACTIONS:
                mass[key] *= EXPECTED_DISCOUNT
            mass[action] += 1.0
    return mass


def _verify_original_replay(
    pre_rows: Sequence[Mapping[str, Any]], feedback_rows: Sequence[Mapping[str, Any]], lambda_cpu: float
) -> dict[str, Any]:
    linear_model = RetentionValueModel(
        6,
        ridge=EXPECTED_RIDGE,
        discount=EXPECTED_DISCOUNT,
        lambda_cpu=lambda_cpu,
    )
    intercept_model = RetentionValueModel(
        1,
        ridge=EXPECTED_RIDGE,
        discount=EXPECTED_DISCOUNT,
        lambda_cpu=lambda_cpu,
    )
    max_linear_abs_error = 0.0
    max_intercept_abs_error = 0.0
    mismatches = 0
    target_max_abs_error = 0.0
    compared_predictions = 0
    for index, (pre, feedback) in enumerate(zip(pre_rows, feedback_rows)):
        features = np.asarray(pre["features"], dtype=np.float64)
        if features.shape != (6,):
            raise ValueError(f"decision {index} feature vector must have six values")
        action = str(pre["selected_action"])
        if action not in ACTIONS or feedback.get("selected_action") != action:
            raise ValueError(f"decision {index} pre-action and feedback actions do not align")
        logged = pre.get("predictions")
        if not isinstance(logged, Mapping):
            raise ValueError(f"decision {index} has no prediction mapping")
        for model_name, model, vector, error_key in (
            ("linear", linear_model, features, "linear"),
            ("intercept", intercept_model, features[[pre["feature_names"].index("bias")]], "intercept"),
        ):
            expected = model.predict(vector)
            logged_values = logged.get(model_name)
            if not isinstance(logged_values, Mapping):
                raise ValueError(f"decision {index} has no {model_name} logged predictions")
            if set(logged_values) != set(ACTIONS):
                raise ValueError(f"decision {index} {model_name} predictions have unexpected actions")
            for action_name in ACTIONS:
                delta = abs(_finite(logged_values[action_name], f"predictions.{model_name}.{action_name}") - expected[action_name])
                if error_key == "linear":
                    max_linear_abs_error = max(max_linear_abs_error, delta)
                else:
                    max_intercept_abs_error = max(max_intercept_abs_error, delta)
                if delta > 1e-9:
                    mismatches += 1
                compared_predictions += 1

        actual_target = _target_values((feedback,))["utility"][0]
        logged_target = _finite(feedback.get("utility_target"), f"feedback[{index}].utility_target")
        target_error = abs(actual_target - logged_target)
        target_max_abs_error = max(target_max_abs_error, target_error)
        if feedback.get("model_updated", True):
            reward_rate = _finite(feedback.get("reward_rate"), f"feedback[{index}].reward_rate")
            utility_cpu_s = _finite(feedback.get("utility_cpu_s"), f"feedback[{index}].utility_cpu_s")
            block_steps = int(_finite(feedback.get("block_steps"), f"feedback[{index}].block_steps"))
            linear_model.update(action, features, reward_rate, utility_cpu_s, block_steps)
            intercept_model.update(
                action,
                features[[pre["feature_names"].index("bias")]],
                reward_rate,
                utility_cpu_s,
                block_steps,
            )
    return {
        "linear_max_abs_prediction_error": max_linear_abs_error,
        "intercept_max_abs_prediction_error": max_intercept_abs_error,
        "compared_action_predictions": compared_predictions,
        "prediction_mismatch_count_at_1e_9": mismatches,
        "max_abs_logged_utility_target_error": target_max_abs_error,
        "replay_matches_logged_predictions": mismatches == 0,
        "logged_utility_targets_match_reward_minus_cpu_rate": target_max_abs_error <= 1e-12,
    }


def _analyse_calibration_stream(
    repo_root: Path,
    calibration_root: Path,
    stream: Mapping[str, Any],
    tail_decisions: int,
    lambda_cpu: float,
) -> dict[str, Any]:
    pre_path = _path_from_summary(calibration_root, stream.get("preactions_path"))
    feedback_path = _path_from_summary(calibration_root, stream.get("feedback_path"))
    if pre_path is None or feedback_path is None:
        raise ValueError("preactions_path and feedback_path are required")
    pre_rows = _read_jsonl(pre_path)
    feedback_rows = _read_jsonl(feedback_path)
    if len(pre_rows) != len(feedback_rows):
        raise ValueError(f"preactions/feedback row count mismatch: {len(pre_rows)} vs {len(feedback_rows)}")
    if int(stream.get("completed_decisions", -1)) != len(pre_rows):
        raise ValueError("completed_decisions does not match preactions row count")
    if len(pre_rows) != int(stream.get("requested_decisions", len(pre_rows))):
        raise ValueError("requested_decisions does not match actual row count")
    if len(pre_rows) < tail_decisions:
        raise ValueError(f"stream has {len(pre_rows)} rows, fewer than tail size {tail_decisions}")
    if not pre_rows:
        raise ValueError("stream contains no decisions")

    feature_names = tuple(pre_rows[0].get("feature_names", ()))
    specs = _feature_specs(feature_names)
    features: list[list[float]] = []
    actions: list[str] = []
    update_flags: list[bool] = []
    propensities: list[float] = []
    for index, (pre, feedback) in enumerate(zip(pre_rows, feedback_rows)):
        if pre.get("decision_index") != index or feedback.get("decision_index") != index:
            raise ValueError(f"decision_index sequence is not 0..N-1 at row {index}")
        if tuple(pre.get("feature_names", ())) != feature_names:
            raise ValueError(f"feature names changed at decision {index}")
        if pre.get("stream_id") != stream.get("stream_id") or feedback.get("stream_id") != stream.get("stream_id"):
            raise ValueError(f"stream_id mismatch at decision {index}")
        vector = np.asarray(pre.get("features"), dtype=np.float64)
        if vector.shape != (len(feature_names),) or not np.isfinite(vector).all():
            raise ValueError(f"invalid feature vector at decision {index}")
        features.append(vector.tolist())
        action = str(pre.get("selected_action"))
        if action not in ACTIONS or feedback.get("selected_action") != action:
            raise ValueError(f"selected action mismatch at decision {index}")
        actions.append(action)
        update_flags.append(bool(feedback.get("model_updated", True)))
        propensities.append(_finite(pre.get("action_propensity"), f"preactions[{index}].action_propensity"))

    targets = _target_values(feedback_rows)
    replay = _verify_original_replay(pre_rows, feedback_rows, lambda_cpu)
    tail_start = len(pre_rows) - tail_decisions
    prediction_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for variant, indices in specs:
        for target_name in TARGETS:
            scored = prequential_predictions(
                features,
                actions,
                targets[target_name],
                update_flags,
                indices,
                tail_start,
            )
            errors = np.asarray([row["error"] for row in scored], dtype=np.float64)
            metrics.append(
                {
                    "stage": "calibration",
                    "pair_kind": stream.get("pair_kind"),
                    "seed": stream.get("seed"),
                    "stream_id": stream.get("stream_id"),
                    "model_variant": variant,
                    "target": target_name,
                    "n_scored": int(errors.size),
                    "tail_start": tail_start,
                    "tail_end_exclusive": len(pre_rows),
                    "mse": float(np.mean(np.square(errors))) if errors.size else None,
                    "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors.size else None,
                    "mean_error_prediction_minus_target": float(np.mean(errors)) if errors.size else None,
                    "mae": float(np.mean(np.abs(errors))) if errors.size else None,
                }
            )
            for row in scored:
                prediction_rows.append(
                    {
                        "pair_kind": stream.get("pair_kind"),
                        "seed": stream.get("seed"),
                        "stream_id": stream.get("stream_id"),
                        "model_variant": variant,
                        "target": target_name,
                        "decision_index": row["decision_index"],
                        "selected_action": row["selected_action"],
                        "prediction": row["prediction"],
                        "observed_target": row["target"],
                        "error": row["error"],
                    }
                )

    action_counts = Counter(actions)
    discounted_mass = _discounted_action_mass(actions, update_flags)
    action_rows: list[dict[str, Any]] = []
    for action in ACTIONS:
        weights = [
            (1.0 / propensity)
            if selected == action and propensity > 0
            else 0.0
            for selected, propensity in zip(actions, propensities)
        ]
        sum_weights = float(sum(weights))
        sum_squared_weights = float(sum(weight * weight for weight in weights))
        action_rows.append(
            {
                "pair_kind": stream.get("pair_kind"),
                "seed": stream.get("seed"),
                "stream_id": stream.get("stream_id"),
                "action": action,
                "selected_count": int(action_counts.get(action, 0)),
                "decisions": len(actions),
                "deterministic_one_step_weight_sum": sum_weights,
                "deterministic_one_step_weight_squared_sum": sum_squared_weights,
                "observed_importance_weight_ess": (
                    float(sum_weights * sum_weights / sum_squared_weights)
                    if sum_squared_weights > 0
                    else 0.0
                ),
                "discounted_observation_mass": discounted_mass[action],
                "discount_factor": EXPECTED_DISCOUNT,
                "discounted_mass_is_ess": False,
                "ess_scope": "observed one-step fixed-action weights only; not whole-policy OPE",
            }
        )
    propensity_values = np.asarray(propensities, dtype=np.float64)
    propensity_rows = [
        {
            "pair_kind": stream.get("pair_kind"),
            "seed": stream.get("seed"),
            "stream_id": stream.get("stream_id"),
            "expected_propensity": 0.25,
            "n_decisions": len(propensities),
            "invalid_count": int(sum(not math.isclose(value, 0.25, rel_tol=0.0, abs_tol=1e-12) for value in propensities)),
            "min_observed_propensity": float(np.min(propensity_values)),
            "max_observed_propensity": float(np.max(propensity_values)),
            "all_uniform_random_propensities_valid": bool(
                all(math.isclose(value, 0.25, rel_tol=0.0, abs_tol=1e-12) for value in propensities)
            ),
        }
    ]
    return {
        "metrics": metrics,
        "prediction_rows": prediction_rows,
        "action_rows": action_rows,
        "propensity_rows": propensity_rows,
        "feature_specs": [
            {"name": name, "feature_indices": list(indices)} for name, indices in specs
        ],
        "replay": {
            "pair_kind": stream.get("pair_kind"),
            "seed": stream.get("seed"),
            "stream_id": stream.get("stream_id"),
            **replay,
        },
        "tail_start": tail_start,
        "row_count": len(pre_rows),
        "source_paths": {
            "preactions": _repo_relative(pre_path, repo_root),
            "feedback": _repo_relative(feedback_path, repo_root),
        },
    }


def _aggregate_metrics(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[(str(row["pair_kind"]), str(row["model_variant"]), str(row["target"]))].append(row)
    result: list[dict[str, Any]] = []
    for (pair_kind, variant, target), rows in sorted(grouped.items()):
        seed_values = np.asarray([float(row["mse"]) for row in rows], dtype=np.float64)
        result.append(
            {
                "pair_kind": pair_kind,
                "model_variant": variant,
                "target": target,
                "seed_count": len(rows),
                "scored_decisions": int(sum(int(row["n_scored"]) for row in rows)),
                "mean_seed_mse": float(np.mean(seed_values)),
                "sd_seed_mse": float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0,
                "pooled_mse": float(
                    sum(float(row["mse"]) * int(row["n_scored"]) for row in rows)
                    / sum(int(row["n_scored"]) for row in rows)
                ),
            }
        )
    for variant in sorted({str(row["model_variant"]) for row in metrics}):
        for target in TARGETS:
            seed_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in metrics:
                if row["model_variant"] == variant and row["target"] == target:
                    seed_rows[str(row["seed"])].append(row)
            pooled_rows = [
                {
                    "mse": float(np.mean([float(item["mse"]) for item in per_seed])),
                    "n_scored": int(sum(int(item["n_scored"]) for item in per_seed)),
                }
                for per_seed in seed_rows.values()
            ]
            if pooled_rows:
                values = np.asarray([float(row["mse"]) for row in pooled_rows], dtype=np.float64)
                result.append(
                    {
                        "pair_kind": "all",
                        "model_variant": variant,
                        "target": target,
                        "seed_count": len(pooled_rows),
                        "scored_decisions": int(sum(row["n_scored"] for row in pooled_rows)),
                        "mean_seed_mse": float(np.mean(values)),
                        "sd_seed_mse": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "pooled_mse": float(
                            sum(float(row["mse"]) * int(row["n_scored"]) for row in pooled_rows)
                            / sum(int(row["n_scored"]) for row in pooled_rows)
                        ),
                    }
                )
    return result


def _bootstrap_difference_rows(
    metrics: Sequence[Mapping[str, Any]], resamples: int, seed: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    available_variants = sorted({str(row["model_variant"]) for row in metrics})
    groups = ("dynamics", "reward_conflict", "all")
    rows: list[dict[str, Any]] = []
    for pair_kind in groups:
        for target in TARGETS:
            for variant in available_variants:
                differences: list[float] = []
                for seed_id in sorted({str(row["seed"]) for row in metrics if row["target"] == target}):
                    applicable = [
                        row
                        for row in metrics
                        if str(row["seed"]) == seed_id
                        and row["target"] == target
                        and (pair_kind == "all" or row["pair_kind"] == pair_kind)
                    ]
                    model_rows = [row for row in applicable if row["model_variant"] == variant]
                    full_rows = [row for row in applicable if row["model_variant"] == "full6"]
                    if model_rows and full_rows:
                        differences.append(
                            float(np.mean([float(row["mse"]) for row in model_rows]))
                            - float(np.mean([float(row["mse"]) for row in full_rows]))
                        )
                values = np.asarray(differences, dtype=np.float64)
                if values.size == 0:
                    continue
                if variant == "full6":
                    mean_difference = 0.0
                    low = high = 0.0
                else:
                    bootstrap_indices = rng.integers(0, values.size, size=(resamples, values.size))
                    bootstrap_means = np.mean(values[bootstrap_indices], axis=1)
                    mean_difference = float(np.mean(values))
                    low, high = np.percentile(bootstrap_means, [2.5, 97.5]).tolist()
                rows.append(
                    {
                        "pair_kind": pair_kind,
                        "target": target,
                        "model_variant": variant,
                        "reference_variant": "full6",
                        "difference_definition": "variant MSE minus full6 MSE",
                        "seed_count": int(values.size),
                        "mean_difference": mean_difference,
                        "bootstrap_95pct_low": float(low),
                        "bootstrap_95pct_high": float(high),
                        "bootstrap_resamples": int(resamples),
                        "bootstrap_seed": int(seed),
                        "interval_scope": "descriptive paired seed bootstrap; not an efficacy test",
                    }
                )
    return rows


def _residual_decomposition(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, str, int, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in predictions:
        key = (str(row["model_variant"]), str(row["pair_kind"]), int(row["seed"]), str(row["decision_index"]))
        keyed[key][str(row["target"])] = row

    variants = sorted({key[0] for key in keyed})
    groups = ("dynamics", "reward_conflict", "all")
    result: list[dict[str, Any]] = []
    for variant in variants:
        for pair_kind in groups:
            points = []
            linearity_errors = []
            residual_linearity_errors = []
            for key, by_target in keyed.items():
                if key[0] != variant or (pair_kind != "all" and key[1] != pair_kind):
                    continue
                if not all(target in by_target for target in TARGETS):
                    continue
                reward = by_target["reward_rate"]
                cpu = by_target["cpu_rate"]
                utility = by_target["utility"]
                er = float(reward["error"])
                ec = float(cpu["error"])
                eu = float(utility["error"])
                points.append((er, ec, eu))
                linearity_errors.append(abs((float(reward["prediction"]) - float(cpu["prediction"])) - float(utility["prediction"])))
                residual_linearity_errors.append(abs(eu - (er - ec)))
            if not points:
                continue
            array = np.asarray(points, dtype=np.float64)
            er, ec, eu = array.T
            cross = float(np.mean(er * ec))
            mse_reward = float(np.mean(er * er))
            mse_cpu = float(np.mean(ec * ec))
            mse_utility = float(np.mean(eu * eu))
            reconstructed = mse_reward + mse_cpu - 2.0 * cross
            covariance = float(np.mean((er - np.mean(er)) * (ec - np.mean(ec))))
            result.append(
                {
                    "pair_kind": pair_kind,
                    "model_variant": variant,
                    "n_scored": int(array.shape[0]),
                    "mean_residual_reward_rate": float(np.mean(er)),
                    "mean_residual_cpu_rate": float(np.mean(ec)),
                    "mean_residual_utility": float(np.mean(eu)),
                    "mse_reward_rate": mse_reward,
                    "mse_cpu_rate": mse_cpu,
                    "mse_utility": mse_utility,
                    "covariance_residual_reward_cpu_rate": covariance,
                    "cross_second_moment_residual_reward_cpu_rate": cross,
                    "utility_mse_reconstructed_from_components": reconstructed,
                    "utility_mse_identity_error": mse_utility - reconstructed,
                    "max_prediction_linearity_abs_error": max(linearity_errors),
                    "max_residual_linearity_abs_error": max(residual_linearity_errors),
                }
            )
    return result


def _utility_mse_component_comparison(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compare full6 and intercept utility MSE by reward, CPU, and cross term."""

    observations: dict[tuple[str, str, int], dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in predictions:
        key = (str(row["pair_kind"]), str(row["seed"]), int(row["decision_index"]))
        observations[key][str(row["model_variant"])][str(row["target"])] = float(row["error"])

    groups: list[tuple[str, str | None]] = []
    seeds = sorted({key[1] for key in observations})
    for pair_kind in ("dynamics", "reward_conflict"):
        groups.extend((pair_kind, seed) for seed in seeds if any(key[0] == pair_kind and key[1] == seed for key in observations))
        groups.append((pair_kind, None))
    groups.extend(("all", seed) for seed in seeds)
    groups.append(("all", None))

    result: list[dict[str, Any]] = []
    for pair_kind, seed in groups:
        selected = [
            (key, model_rows)
            for key, model_rows in observations.items()
            if (pair_kind == "all" or key[0] == pair_kind)
            and (seed is None or key[1] == seed)
            and all(
                variant in model_rows
                and all(target in model_rows[variant] for target in TARGETS)
                for variant in ("full6", "intercept")
            )
        ]
        if not selected:
            continue

        component_metrics: dict[str, dict[str, float]] = {}
        for variant in ("full6", "intercept"):
            reward_errors = np.asarray(
                [model_rows[variant]["reward_rate"] for _, model_rows in selected], dtype=np.float64
            )
            cpu_errors = np.asarray(
                [model_rows[variant]["cpu_rate"] for _, model_rows in selected], dtype=np.float64
            )
            utility_errors = np.asarray(
                [model_rows[variant]["utility"] for _, model_rows in selected], dtype=np.float64
            )
            component_metrics[variant] = {
                "mse_reward_rate": float(np.mean(np.square(reward_errors))),
                "mse_cpu_rate": float(np.mean(np.square(cpu_errors))),
                "residual_cross_term": float(-2.0 * np.mean(reward_errors * cpu_errors)),
                "mse_utility": float(np.mean(np.square(utility_errors))),
            }

        full = component_metrics["full6"]
        intercept = component_metrics["intercept"]
        reward_improvement = intercept["mse_reward_rate"] - full["mse_reward_rate"]
        cpu_improvement = intercept["mse_cpu_rate"] - full["mse_cpu_rate"]
        cross_improvement = intercept["residual_cross_term"] - full["residual_cross_term"]
        utility_improvement = intercept["mse_utility"] - full["mse_utility"]
        shares = (
            [
                100.0 * reward_improvement / utility_improvement,
                100.0 * cpu_improvement / utility_improvement,
                100.0 * cross_improvement / utility_improvement,
            ]
            if abs(utility_improvement) > 1e-18
            else [None, None, None]
        )
        result.append(
            {
                "pair_kind": pair_kind,
                "seed": seed,
                "n_scored": len(selected),
                "full6_mse_reward_rate": full["mse_reward_rate"],
                "intercept_mse_reward_rate": intercept["mse_reward_rate"],
                "full6_mse_cpu_rate": full["mse_cpu_rate"],
                "intercept_mse_cpu_rate": intercept["mse_cpu_rate"],
                "full6_residual_cross_term_minus_2_mean_error_product": full["residual_cross_term"],
                "intercept_residual_cross_term_minus_2_mean_error_product": intercept["residual_cross_term"],
                "full6_mse_utility": full["mse_utility"],
                "intercept_mse_utility": intercept["mse_utility"],
                "utility_mse_improvement_intercept_minus_full6": utility_improvement,
                "reward_mse_improvement_intercept_minus_full6": reward_improvement,
                "cpu_mse_improvement_intercept_minus_full6": cpu_improvement,
                "residual_cross_term_improvement_intercept_minus_full6": cross_improvement,
                "reward_contribution_pct": shares[0],
                "cpu_contribution_pct": shares[1],
                "residual_cross_term_contribution_pct": shares[2],
                "component_sum_identity_error": utility_improvement - reward_improvement - cpu_improvement - cross_improvement,
            }
        )
    return result


def _comparison_coverage(repo_root: Path, comparison_root: Path, summary: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for stream in summary.get("streams", []):
        pre_path = _path_from_summary(comparison_root, stream.get("preactions_path"))
        feedback_path = _path_from_summary(comparison_root, stream.get("feedback_path"))
        pre_count, pre_error = _jsonl_row_count(pre_path) if pre_path else (None, "path missing from summary")
        feedback_count, feedback_error = _jsonl_row_count(feedback_path) if feedback_path else (None, "path missing from summary")
        completed = int(stream.get("completed_decisions", 0))
        rows.append(
            {
                "pair_kind": stream.get("pair_kind"),
                "seed": stream.get("seed"),
                "method": stream.get("method"),
                "stream_id": stream.get("stream_id"),
                "status": stream.get("status"),
                "requested_decisions": stream.get("requested_decisions"),
                "completed_decisions": completed,
                "preactions_rows": pre_count,
                "feedback_rows": feedback_count,
                "preactions_error": pre_error,
                "feedback_error": feedback_error,
                "logs_match_completed_count": (
                    pre_count == completed and feedback_count == completed and pre_error is None and feedback_error is None
                ),
                "use_scope": "descriptive coverage only; no outcome or efficacy analysis",
                "preactions_path": _repo_relative(pre_path, repo_root) if pre_path else None,
                "feedback_path": _repo_relative(feedback_path, repo_root) if feedback_path else None,
            }
        )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("method"))].append(row)
    summaries = [
        {
            "method": method,
            "stream_count": len(method_rows),
            "completed_stream_count": sum(row.get("status") == "completed" for row in method_rows),
            "logs_match_completed_count": sum(bool(row.get("logs_match_completed_count")) for row in method_rows),
            "requested_decisions": sum(int(row.get("requested_decisions") or 0) for row in method_rows),
            "completed_decisions": sum(int(row.get("completed_decisions") or 0) for row in method_rows),
            "preactions_rows": sum(int(row.get("preactions_rows") or 0) for row in method_rows),
            "feedback_rows": sum(int(row.get("feedback_rows") or 0) for row in method_rows),
            "use_scope": "descriptive coverage only; no outcome or efficacy analysis",
        }
        for method, method_rows in sorted(grouped.items())
    ]
    return rows, summaries


def _input_files(root: Path, summary: Mapping[str, Any], *, include_frozen: bool) -> list[Path]:
    paths = [root / "summary.json"]
    if include_frozen:
        frozen = _path_from_summary(root, summary.get("frozen_config_path"))
        if frozen:
            paths.append(frozen)
    for stream in summary.get("streams", []):
        for field in ("preactions_path", "feedback_path"):
            path = _path_from_summary(root, stream.get(field))
            if path:
                paths.append(path)
    return paths


def _make_hash_manifest(paths: Iterable[Path], repo_root: Path) -> dict[str, Any]:
    entries = []
    seen: set[Path] = set()
    for path in paths:
        absolute = path.resolve()
        if absolute in seen:
            continue
        seen.add(absolute)
        exists = absolute.is_file()
        entries.append(
            {
                "path": _repo_relative(absolute, repo_root),
                "exists": exists,
                "bytes": absolute.stat().st_size if exists else None,
                "sha256": _sha256(absolute) if exists else None,
            }
        )
    return {"algorithm": "SHA-256", "file_count": len(entries), "files": entries}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_frozen_config(root: Path, summary: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    frozen_path = _path_from_summary(root, summary.get("frozen_config_path"))
    if frozen_path is None or not frozen_path.is_file():
        raise ValueError("calibration summary must identify an existing frozen config")
    wrapper = _read_json(frozen_path)
    config = wrapper.get("config")
    if not isinstance(config, dict):
        raise ValueError("calibration frozen config has no config object")
    model_config = config.get("value_model")
    if not isinstance(model_config, dict):
        raise ValueError("calibration frozen config has no value_model object")
    for key, expected in (
        ("n_features", 6),
        ("ridge", EXPECTED_RIDGE),
        ("discount", EXPECTED_DISCOUNT),
        ("lambda_cpu", EXPECTED_LAMBDA_CPU),
    ):
        actual = model_config.get(key)
        if float(actual) != float(expected):
            raise ValueError(f"frozen value_model.{key} is {actual!r}; expected {expected!r}")
    return config, frozen_path


def _build_report(analysis: Mapping[str, Any]) -> str:
    input_counts = analysis["input_coverage"]
    replay = analysis["original_replay"]
    prop = analysis["calibration_propensity"]
    comparison = analysis["comparison_coverage"]
    aggregate = analysis["aggregate_metrics"]
    metric_map = {
        (str(row["pair_kind"]), str(row["model_variant"]), str(row["target"])): float(row["mean_seed_mse"])
        for row in aggregate
    }
    difference_map = {
        (str(row["pair_kind"]), str(row["model_variant"]), str(row["target"])): row
        for row in analysis["bootstrap_differences"]
    }
    utility_variants = sorted(
        (
            row
            for row in aggregate
            if row["pair_kind"] == "all" and row["target"] == "utility"
        ),
        key=lambda row: float(row["mean_seed_mse"]),
    )
    action_support = analysis["action_support"]
    component_rows = analysis["utility_mse_component_comparison"]
    pooled_components = next(
        (row for row in component_rows if row["pair_kind"] == "all" and row["seed"] is None),
        None,
    )
    relation_component_rows = [
        row for row in component_rows if row["pair_kind"] in ("dynamics", "reward_conflict") and row["seed"] is None
    ]
    residual_reference = next(
        (
            row
            for row in analysis["residual_decomposition"]
            if row["pair_kind"] == "all" and row["model_variant"] == "full6"
        ),
        None,
    )
    lines = [
        "# Retention prediction diagnostics",
        "",
        "This is an offline, descriptive analysis of existing selected-feedback logs. It ran zero new environment steps.",
        "",
        "## Inputs and scope",
        "",
        f"- Calibration streams: {input_counts['calibration_streams']}; decisions: {input_counts['calibration_decisions']}; tail labels scored per stream: {analysis['tail_decisions']}. ",
        f"- Calibration relations: {', '.join(input_counts['calibration_relations'])}.",
        f"- Original ridge replay mismatches at tolerance 1e-9: {replay['prediction_mismatch_count_at_1e_9']} of {replay['compared_action_predictions']} logged action predictions.",
        f"- Maximum absolute replay error: {replay['max_linear_abs_prediction_error']:.12g}; utility target reconstruction error: {replay['max_abs_logged_utility_target_error']:.12g}.",
        f"- Calibration propensity rows at exactly 0.25 within 1e-12: {prop['valid_rows']} / {prop['total_rows']}.",
        f"- Comparison coverage rows: {comparison['streams']}; complete logs: {comparison['complete_logs']}. Comparison labels were not used for prediction scoring.",
        "",
        "## Prediction scores",
        "",
        "Per-stream MSE for every target and feature set is in `stream_metrics.csv`; relation and pooled seed summaries are in `relation_aggregates.csv`. The scores are prequential and use only the selected action's observed label. At each decision the model predicts first, then the selected feedback updates its state. No unchosen action outcomes are imputed.",
        "",
        "The comparison intervals in `seed_bootstrap_differences.csv` resample the six calibration seed IDs and show descriptive paired MSE differences (model variant minus full6). They are not efficacy tests.",
        "",
        "Mean seed utility MSE by feature set. Every seed has both relations and 32 scored labels per stream, so relation and seed contributions are equally sized (MSE columns are multiplied by 10⁵; differences and intervals by 10⁶):",
        "",
        "| Feature set | Dynamics | Reward conflict | All seeds, both relations | Difference vs full6 [95% seed bootstrap interval] |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in utility_variants:
        variant = str(row["model_variant"])
        difference = difference_map.get(("all", variant, "utility"))
        if difference is not None:
            delta = float(difference["mean_difference"]) * 1e6
            low = float(difference["bootstrap_95pct_low"]) * 1e6
            high = float(difference["bootstrap_95pct_high"]) * 1e6
            delta_cell = f"{delta:.3f} [{low:.3f}, {high:.3f}]"
        else:
            delta_cell = "—"
        lines.append(
            f"| {variant} | {metric_map.get(('dynamics', variant, 'utility'), float('nan')) * 1e5:.3f} "
            f"| {metric_map.get(('reward_conflict', variant, 'utility'), float('nan')) * 1e5:.3f} "
            f"| {float(row['mean_seed_mse']) * 1e5:.3f} | {delta_cell} |"
        )
    if utility_variants:
        best = utility_variants[0]
        best_variant = str(best["model_variant"])
        best_difference = difference_map.get(("all", best_variant, "utility"))
        if best_difference is not None:
            lines.extend(
                [
                    "",
                    f"The lowest pooled utility MSE among these nine feature sets was `{best_variant}` at "
                    f"{float(best['mean_seed_mse']):.8g}; full6 was "
                    f"{metric_map.get(('all', 'full6', 'utility'), float('nan')):.8g}. Its descriptive paired "
                    f"seed-bootstrap difference was {float(best_difference['mean_difference']):.3g} "
                    f"[{float(best_difference['bootstrap_95pct_low']):.3g}, "
                    f"{float(best_difference['bootstrap_95pct_high']):.3g}]. This is a predictive comparison "
                    "within these calibration trajectories, not evidence of policy benefit.",
                ]
            )
    lines.extend([
        "",
        "## Reward and CPU residuals",
        "",
        "`residual_decomposition.csv` reports the reward-rate and CPU-rate residual covariance and verifies the identity for utility residuals. Here utility is reward rate minus `utility_cpu_s / block_steps` because the frozen CPU penalty is 1.",
        "",
    ])
    if residual_reference is not None:
        lines.append(
            "For full6 pooled over all 384 held-out labels, utility MSE was "
            f"{float(residual_reference['mse_utility']):.8g}; the reward/CPU residual covariance was "
            f"{float(residual_reference['covariance_residual_reward_cpu_rate']):.4g}, and the MSE identity "
            f"error was {float(residual_reference['utility_mse_identity_error']):.3g}."
        )
        lines.append("")
    lines.extend(
        [
            "## Full6 versus intercept error components",
            "",
            "The table gives `intercept component MSE − full6 component MSE`, so positive utility improvement means full6 has lower utility MSE. Utility error is reward error minus CPU-rate error; its MSE decomposes exactly into reward MSE, CPU MSE, and the residual cross term `−2 × mean(reward residual × CPU residual)`. Percentages divide each component's improvement by total utility MSE improvement; they may be negative or exceed 100% when components offset.",
            "",
            "Per-seed rows are in units of 10⁻⁶ MSE; the exact component MSEs and deltas are in `utility_mse_component_comparison.csv`.",
            "",
            "| Relation | Seed | Utility improvement | Reward MSE contribution | CPU MSE contribution | Residual cross-term contribution | Reward % | CPU % | Cross-term % |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in component_rows:
        if row["seed"] is None or row["pair_kind"] == "all":
            continue
        values = [
            float(row["utility_mse_improvement_intercept_minus_full6"]),
            float(row["reward_mse_improvement_intercept_minus_full6"]),
            float(row["cpu_mse_improvement_intercept_minus_full6"]),
            float(row["residual_cross_term_improvement_intercept_minus_full6"]),
        ]
        percents = [row["reward_contribution_pct"], row["cpu_contribution_pct"], row["residual_cross_term_contribution_pct"]]
        lines.append(
            f"| {row['pair_kind']} | {row['seed']} | "
            + " | ".join(f"{value * 1e6:.6f}" for value in values)
            + " | "
            + " | ".join("—" if value is None else f"{float(value):.2f}%" for value in percents)
            + " |"
        )
    lines.append("")
    if pooled_components is not None:
        lines.append(
            "Across both relations and all six seeds, full6 reduced utility MSE by "
            f"{float(pooled_components['utility_mse_improvement_intercept_minus_full6']):.6g}; "
            f"{float(pooled_components['reward_contribution_pct']):.2f}% of that reduction came from reward-rate MSE, "
            f"{float(pooled_components['cpu_contribution_pct']):.2f}% from CPU-rate MSE, and "
            f"{float(pooled_components['residual_cross_term_contribution_pct']):.2f}% from the residual cross term."
        )
    for row in relation_component_rows:
        lines.append(
            f"For {row['pair_kind']}, full6 utility MSE improvement was "
            f"{float(row['utility_mse_improvement_intercept_minus_full6']):.6g}; reward, CPU, and residual cross-term "
            f"contributions were {float(row['reward_mse_improvement_intercept_minus_full6']):.6g}, "
            f"{float(row['cpu_mse_improvement_intercept_minus_full6']):.6g}, and "
            f"{float(row['residual_cross_term_improvement_intercept_minus_full6']):.6g}."
        )
    lines.extend(
        [
            "",
            "These component differences describe prediction errors on selected feedback. The replayed action-specific predictors do not separate common state difficulty from action-specific contrast, and this analysis makes no action-ranking claim.",
            "",
        ]
    )
    lines.extend([
        "## Action support",
        "",
        "`action_counts.csv` reports selected action counts, discounted observation mass and arithmetic importance-weight ESS for each deterministic one-step fixed-action weighting. Discounted mass is a weighted count, not ESS. ESS describes observed action support at the logged histories; it does not estimate a multi-step or whole-policy value. No whole-policy off-policy evaluation is reported.",
        "",
        "| Action | Selected rows | Pooled one-step ESS | Mean discounted observation mass per stream |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in action_support:
        lines.append(
            f"| {row['action']} | {row['selected_count']} | {row['pooled_one_step_ess']:.0f} "
            f"| {row['mean_discounted_observation_mass_per_stream']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "These estimates describe prediction of selected feedback along uniform-random calibration trajectories. They do not establish causal benefit, action ranking, or performance under a different policy. The comparison directory is included only for stream and log coverage.",
        "",
        "## Output files",
        "",
        "- `analysis.json`: complete machine-readable summary and audit details.",
        "- `input_hashes.json`: SHA-256 hashes of the calibration and comparison summaries, frozen configs, and selected-feedback logs.",
        "- `prequential_predictions.csv`: every scored observed-action prediction and target.",
        "- `stream_metrics.csv`, `relation_aggregates.csv`, `seed_bootstrap_differences.csv`, `residual_decomposition.csv`.",
        "- `action_counts.csv`, `calibration_propensity.csv`, `comparison_coverage.csv`.",
        "- `utility_mse_component_comparison.csv`: full6 versus intercept MSE attribution by relation and seed.",
        "- `code_snapshot/diagnose_retention_prediction.py`: executed source snapshot.",
        "",
        "## Reproduction",
        "",
        "Run these commands from the repository root in PowerShell:",
        "",
        "```powershell",
        "$env:PYTHONPATH = 'src'",
        ".venv\\Scripts\\python.exe -m pytest tests\\test_diagnose_retention_prediction.py -q",
        ".venv\\Scripts\\python.exe scripts\\diagnose_retention_prediction.py",
        "```",
        "",
    ])
    if analysis["failures"]:
        lines.extend(["## Recorded failures", ""])
        for failure in analysis["failures"]:
            lines.append(f"- {failure['stream_id']}: {failure['error']}")
        lines.append("")
    if analysis["coverage_gaps"]:
        lines.extend(["## Coverage gaps", ""])
        for issue in analysis["coverage_gaps"]:
            lines.append(f"- {issue}")
        lines.append("")
    return "\n".join(lines)


def run_diagnostics(
    repo_root: Path,
    calibration_root: Path,
    comparison_root: Path,
    output_root: Path,
    *,
    tail_decisions: int = DEFAULT_TAIL_DECISIONS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    calibration_root = calibration_root.resolve()
    comparison_root = comparison_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if tail_decisions <= 0:
        raise ValueError("tail_decisions must be positive")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")

    calibration_summary = _read_json(calibration_root / "summary.json")
    comparison_summary = _read_json(comparison_root / "summary.json")
    config, frozen_config_path = _read_frozen_config(calibration_root, calibration_summary)
    lambda_cpu = float(config["value_model"]["lambda_cpu"])
    calibration_streams = calibration_summary.get("streams", [])
    comparison_streams = comparison_summary.get("streams", [])
    if not isinstance(calibration_streams, list) or not isinstance(comparison_streams, list):
        raise ValueError("summary streams fields must be arrays")

    metrics: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    propensity_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    coverage_gaps: list[str] = []
    calibration_decisions = 0
    observed_feature_specs: list[dict[str, Any]] = []
    for stream in calibration_streams:
        stream_id = str(stream.get("stream_id", "unknown"))
        try:
            if stream.get("status") != "completed":
                raise ValueError(f"stream status is {stream.get('status')!r}")
            result = _analyse_calibration_stream(
                repo_root,
                calibration_root,
                stream,
                tail_decisions,
                lambda_cpu,
            )
            metrics.extend(result["metrics"])
            prediction_rows.extend(result["prediction_rows"])
            action_rows.extend(result["action_rows"])
            propensity_rows.extend(result["propensity_rows"])
            replay_rows.append(result["replay"])
            if not observed_feature_specs:
                observed_feature_specs = result["feature_specs"]
            calibration_decisions += int(result["row_count"])
            if not result["replay"]["replay_matches_logged_predictions"]:
                failures.append({"stream_id": stream_id, "error": "original utility replay did not match logged predictions"})
            if not result["replay"]["logged_utility_targets_match_reward_minus_cpu_rate"]:
                failures.append({"stream_id": stream_id, "error": "logged utility targets did not match reward_rate minus CPU_rate"})
            if not result["propensity_rows"][0]["all_uniform_random_propensities_valid"]:
                failures.append({"stream_id": stream_id, "error": "uniform random propensity was not 0.25 for every row"})
        except Exception as exc:  # Persist partial outputs and the exact failure.
            failures.append({"stream_id": stream_id, "error": f"{type(exc).__name__}: {exc}"})

    if len(calibration_streams) != 12:
        coverage_gaps.append(f"calibration summary has {len(calibration_streams)} streams; expected 12")
    if len(comparison_streams) != 80:
        coverage_gaps.append(f"comparison summary has {len(comparison_streams)} streams; observed coverage expected 80")

    comparison_coverage_rows, comparison_coverage_summaries = _comparison_coverage(
        repo_root, comparison_root, comparison_summary
    )
    for row in comparison_coverage_rows:
        if not row["logs_match_completed_count"]:
            coverage_gaps.append(
                f"comparison {row['stream_id']} coverage mismatch: completed={row['completed_decisions']}, "
                f"preactions={row['preactions_rows']}, feedback={row['feedback_rows']}"
            )

    aggregate_rows = _aggregate_metrics(metrics)
    bootstrap_rows = _bootstrap_difference_rows(metrics, bootstrap_resamples, bootstrap_seed)
    residual_rows = _residual_decomposition(prediction_rows)
    component_comparison_rows = _utility_mse_component_comparison(prediction_rows)
    propensity_valid = sum(int(row["n_decisions"]) - int(row["invalid_count"]) for row in propensity_rows)
    propensity_total = sum(int(row["n_decisions"]) for row in propensity_rows)
    replay_summary = {
        "streams_replayed": len(replay_rows),
        "compared_action_predictions": sum(int(row["compared_action_predictions"]) for row in replay_rows),
        "prediction_mismatch_count_at_1e_9": sum(int(row["prediction_mismatch_count_at_1e_9"]) for row in replay_rows),
        "max_linear_abs_prediction_error": max((float(row["linear_max_abs_prediction_error"]) for row in replay_rows), default=None),
        "max_intercept_abs_prediction_error": max((float(row["intercept_max_abs_prediction_error"]) for row in replay_rows), default=None),
        "max_abs_logged_utility_target_error": max((float(row["max_abs_logged_utility_target_error"]) for row in replay_rows), default=None),
        "all_replays_match": bool(replay_rows) and all(bool(row["replay_matches_logged_predictions"]) for row in replay_rows),
    }
    comparison_complete_logs = sum(bool(row["logs_match_completed_count"]) for row in comparison_coverage_rows)
    action_support_rows = []
    for action in ACTIONS:
        matching = [row for row in action_rows if row["action"] == action]
        selected_count = sum(int(row["selected_count"]) for row in matching)
        action_support_rows.append(
            {
                "action": action,
                "selected_count": selected_count,
                "pooled_one_step_ess": float(
                    sum(float(row["observed_importance_weight_ess"]) for row in matching)
                ),
                "mean_discounted_observation_mass_per_stream": float(
                    np.mean([float(row["discounted_observation_mass"]) for row in matching])
                ) if matching else None,
                "discounted_mass_is_ess": False,
            }
        )

    hash_paths = _input_files(calibration_root, calibration_summary, include_frozen=True)
    hash_paths.extend(_input_files(comparison_root, comparison_summary, include_frozen=True))
    input_hashes = _make_hash_manifest(hash_paths, repo_root)
    hashes_path = output_root / "input_hashes.json"
    _write_json(hashes_path, input_hashes)

    snapshot_dir = output_root / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    snapshot_script = snapshot_dir / script_path.name
    shutil.copy2(script_path, snapshot_script)
    code_snapshot = {
        "path": _repo_relative(snapshot_script, repo_root),
        "sha256": _sha256(snapshot_script),
    }
    test_path = repo_root / "tests" / "test_diagnose_retention_prediction.py"
    if test_path.is_file():
        test_snapshot = snapshot_dir / test_path.name
        shutil.copy2(test_path, test_snapshot)
        code_snapshot["test_path"] = _repo_relative(test_snapshot, repo_root)
        code_snapshot["test_sha256"] = _sha256(test_snapshot)

    declared_completed_calibration_decisions = sum(
        int(row.get("completed_decisions", 0)) for row in calibration_streams
    )
    requested_calibration_decisions = sum(
        int(row.get("requested_decisions", 0)) for row in calibration_streams
    )
    analysis_status = "complete" if not failures and not coverage_gaps and len(metrics) == len(calibration_streams) * 9 * 3 else "partial"
    analysis = {
        "format": SCRIPT_VERSION,
        "status": analysis_status,
        "environment_steps_added": 0,
        "analysis_scope": "descriptive prequential selected-feedback prediction on calibration; comparison is coverage only",
        "model_parameters": {
            "n_features_original": 6,
            "ridge": EXPECTED_RIDGE,
            "discount": EXPECTED_DISCOUNT,
            "lambda_cpu": lambda_cpu,
            "initial_model_state": "zero target vectors and ridge identity Gram matrices",
        },
        "tail_decisions": tail_decisions,
        "targets": {
            "reward_rate": "feedback.reward_rate",
            "cpu_rate": "feedback.utility_cpu_s / feedback.block_steps",
            "utility": "reward_rate - cpu_rate (frozen lambda_cpu is 1)",
        },
        "scoring": {
            "sequence": "predict selected action before reading that decision's selected feedback update; then update only selected action if model_updated",
            "outcome_scope": "selected-action labels only; no unchosen-action outcomes imputed",
            "normalization": "none; raw logged features are used without full-stream normalization",
            "feature_specs": observed_feature_specs,
        },
        "input_coverage": {
            "calibration_streams": len(calibration_streams),
            "calibration_decisions": calibration_decisions,
            "calibration_requested_decisions": requested_calibration_decisions,
            "calibration_summary_completed_decisions": declared_completed_calibration_decisions,
            "calibration_relations": sorted({str(row.get("pair_kind")) for row in calibration_streams}),
            "comparison_streams": len(comparison_streams),
            "comparison_relations": sorted({str(row.get("pair_kind")) for row in comparison_streams}),
            "comparison_methods": sorted({str(row.get("method")) for row in comparison_streams}),
        },
        "original_replay": replay_summary,
        "original_replay_per_stream": replay_rows,
        "calibration_propensity": {
            "expected": 0.25,
            "valid_rows": propensity_valid,
            "total_rows": propensity_total,
            "all_valid": propensity_valid == propensity_total,
        },
        "action_support": action_support_rows,
        "comparison_coverage": {
            "streams": len(comparison_coverage_rows),
            "complete_logs": comparison_complete_logs,
            "by_method": comparison_coverage_summaries,
            "scope": "descriptive coverage only; no comparison outcome analysis",
        },
        "aggregate_metrics": aggregate_rows,
        "bootstrap_differences": bootstrap_rows,
        "residual_decomposition": residual_rows,
        "utility_mse_component_comparison": component_comparison_rows,
        "input_hash_manifest": _repo_relative(hashes_path, repo_root),
        "input_hash_file_count": input_hashes["file_count"],
        "code_snapshot": code_snapshot,
        "bootstrap": {"resamples": bootstrap_resamples, "seed": bootstrap_seed, "interval": "percentile 95 percent"},
        "failures": failures,
        "coverage_gaps": coverage_gaps,
        "interpretation_limits": [
            "Prediction errors are descriptive for selected feedback on uniform-random calibration trajectories.",
            "They do not establish causal benefit or action ranking.",
            "No per-decision unchosen outcomes were created.",
            "No multi-step or whole-policy off-policy evaluation was performed.",
            "Comparison outputs are used only to report stream and log coverage.",
            "Discounted evidence mass is not an effective sample size and is not reported as ESS.",
            "Action-specific selected-feedback prediction errors do not distinguish common state difficulty from action-specific contrast, and do not support action ranking.",
        ],
    }

    _write_csv(output_root / "stream_metrics.csv", metrics)
    _write_csv(output_root / "prequential_predictions.csv", prediction_rows)
    _write_csv(output_root / "relation_aggregates.csv", aggregate_rows)
    _write_csv(output_root / "seed_bootstrap_differences.csv", bootstrap_rows)
    _write_csv(output_root / "residual_decomposition.csv", residual_rows)
    _write_csv(output_root / "utility_mse_component_comparison.csv", component_comparison_rows)
    _write_csv(output_root / "action_counts.csv", action_rows)
    _write_csv(output_root / "calibration_propensity.csv", propensity_rows)
    _write_csv(output_root / "comparison_coverage.csv", comparison_coverage_rows)
    _write_csv(output_root / "comparison_coverage_by_method.csv", comparison_coverage_summaries)
    _write_json(output_root / "analysis.json", analysis)
    (output_root / "report.md").write_text(_build_report(analysis), encoding="utf-8")
    return analysis


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument("--calibration-root", type=Path, default=Path("outputs/retention_value_calibration_main"))
    parser.add_argument("--comparison-root", type=Path, default=Path("outputs/retention_value_comparison_main"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/retention_prediction_diagnostics"))
    parser.add_argument("--tail-decisions", type=int, default=DEFAULT_TAIL_DECISIONS)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    calibration_root = args.calibration_root if args.calibration_root.is_absolute() else repo_root / args.calibration_root
    comparison_root = args.comparison_root if args.comparison_root.is_absolute() else repo_root / args.comparison_root
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    result = run_diagnostics(
        repo_root,
        calibration_root,
        comparison_root,
        output_root,
        tail_decisions=args.tail_decisions,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"status": result["status"], "output_root": str(output_root.resolve()), "failures": len(result["failures"]), "coverage_gaps": len(result["coverage_gaps"])}))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
