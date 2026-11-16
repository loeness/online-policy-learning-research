#!/usr/bin/env python3
"""Independent, read-only E 162 statistics and crossfit audit.

The regular CLI reads only the three supplied JSON packets.  ``--self-check``
uses generated synthetic packets and never opens an E 161 feature or crossfit
artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("crossfit", "fixed_keep", "fixed_replace", "intercept_only", "frequency_rule")
CONTROLS = ("fixed_keep", "fixed_replace", "intercept_only", "frequency_rule")
FEATURE_NAMES = ("recent_frequency_contrast", "new_fit_gain", "old_prior_loss")
STRATA = (("A", "old_major"), ("A", "new_major"), ("B", "old_major"), ("B", "new_major"))
FUTURES = ("old_major", "new_major", "balanced")
ARMS = ("keep", "replace_behavioral")
STAT_TOL = 1e-10
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 162033


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON in {path}: {exc}") from exc


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_object(value: Any, label: str, keys: set[str] | None = None) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    if keys is not None:
        missing, extra = keys - set(value), set(value) - keys
        require(not missing and not extra, f"{label} keys mismatch (missing={sorted(missing)}, extra={sorted(extra)})")
    return value


def finite_number(value: Any, label: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{label} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def stable_mean(values: list[float], label: str) -> float:
    require(bool(values), f"cannot average empty values for {label}")
    magnitude = max(abs(value) for value in values)
    if magnitude == 0.0:
        return 0.0
    result = magnitude * math.fsum(value / magnitude for value in values) / len(values)
    require(math.isfinite(result), f"non-finite mean for {label}")
    return result


def load_context() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    preparation = root / "experiments" / "exp162" / "preparation"
    delivery = root / "research" / "appendix" / "H033" / "delivery_v1"
    snapshot = delivery / "source_snapshot"

    freeze_path = preparation / "freeze.json"
    freeze = require_object(read_json(freeze_path), "E 162 freeze")
    declared = require_object(freeze.get("files_sha256"), "freeze.files_sha256")

    fixed_paths = {
        "protocol": preparation / "protocol.md",
        "registration_budget": preparation / "budget_at_registration.json",
        "delivery_manifest": delivery / "manifest.json",
        "method_v1": root / "research" / "appendix" / "H033" / "METHOD_V1.md",
    }
    file_hashes: dict[str, dict[str, str]] = {}
    for name, path in fixed_paths.items():
        require(path.is_file(), f"missing provenance file: {path}")
        digest = sha256_file(path)
        relative = path.relative_to(root).as_posix()
        file_hashes[name] = {"path": relative, "sha256": digest}
        frozen_digest = declared.get(relative)
        if frozen_digest is not None:
            require(digest == frozen_digest, f"frozen hash mismatch for {relative}")

    handoff_path = root / "research" / "appendix" / "H033" / "HANDOFF_V1.md"
    require(handoff_path.is_file(), f"missing method handoff: {handoff_path}")
    file_hashes["handoff_v1"] = {
        "path": handoff_path.relative_to(root).as_posix(),
        "sha256": sha256_file(handoff_path),
    }
    file_hashes["freeze"] = {"path": freeze_path.relative_to(root).as_posix(), "sha256": sha256_file(freeze_path)}
    current_budget = root / "experiments" / "exp162" / "budget.json"
    require(current_budget.is_file(), f"missing E 162 budget: {current_budget}")
    file_hashes["current_budget"] = {
        "path": current_budget.relative_to(root).as_posix(),
        "sha256": sha256_file(current_budget),
    }

    manifest = read_json(delivery / "manifest.json")
    require_object(manifest, "delivery manifest")
    require(manifest.get("source_identity_sha256") == freeze.get("source_identity_sha256"), "delivery identity differs from E 162 freeze")
    require(freeze.get("experiment_id") == "E162", "freeze is not for E 162")
    require(freeze.get("hypothesis_id") == "H033", "freeze is not for H 033")

    snapshot_entries = manifest.get("snapshot_files")
    require(isinstance(snapshot_entries, list) and bool(snapshot_entries), "delivery manifest has no snapshot file list")
    config_relative = "configs/h033_diagnostic_v1.json"
    snapshot_hashes: list[dict[str, str]] = []
    config_digest: str | None = None
    seen_paths: set[str] = set()
    for index, raw_entry in enumerate(snapshot_entries):
        entry = require_object(raw_entry, f"manifest.snapshot_files[{index}]")
        relative = entry.get("path")
        expected_digest = entry.get("sha256")
        require(isinstance(relative, str) and relative not in seen_paths, "invalid/duplicate delivery snapshot path")
        require(isinstance(expected_digest, str) and len(expected_digest) == 64, f"invalid snapshot hash for {relative}")
        seen_paths.add(relative)
        source_file = snapshot / Path(relative)
        require(source_file.is_file(), f"missing frozen snapshot file: {relative}")
        actual_digest = sha256_file(source_file)
        require(actual_digest == expected_digest, f"frozen snapshot hash mismatch: {relative}")
        snapshot_hashes.append({"path": (Path("research/appendix/H033/delivery_v1/source_snapshot") / relative).as_posix(), "sha256": actual_digest})
        if relative == config_relative:
            config_digest = actual_digest

    require(config_digest is not None, "frozen snapshot omits diagnostic config")
    config_path = snapshot / config_relative
    config = require_object(read_json(config_path), "frozen config")
    require(config_digest == sha256_file(config_path), "diagnostic config hash mismatch")
    require(config.get("format") == "h033_diagnostic_config_v1", "unexpected frozen config format")
    require(config.get("feature_names") == list(FEATURE_NAMES), "frozen feature names differ from H033 v1")
    require(config.get("primary_future_regimes") == ["old_major", "new_major"], "frozen primary regimes differ from H033 v1")
    require(config.get("secondary_future_regime") == "balanced", "frozen secondary regime differs from H033 v1")
    require(config.get("replicas") == [0, 1] and config.get("horizon") == 6144, "frozen outcome protocol differs from H033 v1")
    require(config.get("folds") == 4 and config.get("ridge_alpha") == 1.0, "frozen ridge protocol differs from H033 v1")
    require(config.get("scale_floor") == 1e-12 and config.get("replace_threshold") == 0.0, "frozen scale/decision rule differs from H033 v1")
    require(isinstance(config.get("seeds"), list) and len(config["seeds"]) == 16, "frozen config must contain 16 seeds")
    require(all(isinstance(seed, int) and not isinstance(seed, bool) for seed in config["seeds"]), "frozen seeds must be integers")
    require(len(set(config["seeds"])) == 16, "frozen seeds must be unique")
    method_sha = file_hashes["method_v1"]["sha256"]
    require(method_sha == declared.get("research/appendix/H033/METHOD_V1.md"), "METHOD_V1 hash is not frozen by E 162")
    manifest_sha = file_hashes["delivery_manifest"]["sha256"]
    require(manifest_sha == declared.get("research/appendix/H033/delivery_v1/manifest.json"), "delivery manifest hash is not frozen by E 162")
    protocol_sha = file_hashes["protocol"]["sha256"]
    require(protocol_sha == declared.get("experiments/exp162/preparation/protocol.md"), "protocol hash is not frozen by E 162")
    registration_budget_rel = "experiments/exp162/preparation/budget_at_registration.json"
    require(file_hashes["registration_budget"]["sha256"] == declared.get(registration_budget_rel), "registration budget hash is not frozen by E 162")

    stats_path = Path(__file__).resolve()
    file_hashes["statistics_source"] = {
        "path": stats_path.relative_to(root).as_posix(),
        "sha256": sha256_file(stats_path),
    }
    return {
        "root": root,
        "config": config,
        "config_sha256": config_digest,
        "provenance": {
            "git_commit": manifest.get("git_commit"),
            "source_identity_sha256": manifest.get("source_identity_sha256"),
            "file_hashes": file_hashes,
            "frozen_delivery_snapshot_files": snapshot_hashes,
        },
    }


def validate_features(packet: Any, config: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[tuple[str, str], list[int]]]:
    packet = require_object(packet, "feature packet", {"format", "rows", "metadata"})
    require(packet["format"] == "h033_features_v1", "feature packet format must be h033_features_v1")
    require(isinstance(packet["metadata"], dict), "feature packet metadata must be an object")
    rows = packet["rows"]
    require(isinstance(rows, list) and len(rows) == 16, "feature packet must have exactly 16 rows")
    seed_order = config.get("seeds")
    require(isinstance(seed_order, list) and len(seed_order) == 16, "frozen config must contain 16 seeds")
    seeds = [int(seed) for seed in seed_order]
    require(len(set(seeds)) == 16, "frozen config seeds are not unique")
    by_seed: dict[int, dict[str, Any]] = {}
    strata: dict[tuple[str, str], list[int]] = {stratum: [] for stratum in STRATA}

    for index, raw in enumerate(rows):
        row = require_object(raw, f"feature rows[{index}]", {"seed", "old_family", "history_regime", "features", "provenance"})
        seed = row["seed"]
        require(isinstance(seed, int) and not isinstance(seed, bool), f"feature rows[{index}].seed must be integer")
        require(seed in seeds and seed not in by_seed, f"unexpected/duplicate feature seed {seed}")
        family, history = row["old_family"], row["history_regime"]
        stratum = (family, history)
        require(stratum in strata, f"invalid feature stratum {stratum}")
        require(isinstance(row["provenance"], dict), f"feature rows[{index}].provenance must be an object")
        features = require_object(row["features"], f"feature rows[{index}].features", set(FEATURE_NAMES))
        clean_features = {name: finite_number(features[name], f"seed {seed} feature {name}") for name in FEATURE_NAMES}
        by_seed[seed] = {"old_family": family, "history_regime": history, "features": clean_features}
        strata[stratum].append(seed)

    require(set(by_seed) == set(seeds), "feature seeds differ from frozen config")
    for stratum, group_seeds in strata.items():
        require(len(group_seeds) == 4, f"feature stratum {stratum} must contain exactly 4 seeds")
        group_seeds.sort()
    return by_seed, strata


def validate_outcomes(packet: Any, config: dict[str, Any]) -> dict[tuple[int, str, int, str], float]:
    packet = require_object(packet, "outcome packet", {"format", "rows", "metadata"})
    require(packet["format"] == "h033_outcomes_v1", "outcome packet format must be h033_outcomes_v1")
    require(isinstance(packet["metadata"], dict), "outcome packet metadata must be an object")
    rows = packet["rows"]
    require(isinstance(rows, list) and len(rows) == 192, "outcome packet must have exactly 192 rows")
    seeds = [int(seed) for seed in config["seeds"]]
    cells: dict[tuple[int, str, int, str], float] = {}
    primary_rows = 0
    for index, raw in enumerate(rows):
        row = require_object(
            raw,
            f"outcome rows[{index}]",
            {"seed", "future_regime", "replica", "arm", "horizon", "reward_per_step"},
        )
        seed = row["seed"]
        replica, horizon = row["replica"], row["horizon"]
        require(isinstance(seed, int) and not isinstance(seed, bool) and seed in seeds, f"invalid outcome seed at row {index}")
        require(isinstance(replica, int) and not isinstance(replica, bool) and replica in (0, 1), f"invalid replica at row {index}")
        require(isinstance(horizon, int) and not isinstance(horizon, bool) and horizon == 6144, f"invalid horizon at row {index}")
        future, arm = row["future_regime"], row["arm"]
        require(future in FUTURES and arm in ARMS, f"invalid future regime or arm at row {index}")
        key = (seed, future, replica, arm)
        require(key not in cells, f"duplicate outcome cell {key}")
        cells[key] = finite_number(row["reward_per_step"], f"outcome reward {key}")
        primary_rows += int(future in ("old_major", "new_major"))
    expected = {
        (seed, future, replica, arm)
        for seed in seeds
        for future in FUTURES
        for replica in (0, 1)
        for arm in ARMS
    }
    require(set(cells) == expected, f"outcome cells incomplete: missing={len(expected - set(cells))}, extra={len(set(cells) - expected)}")
    require(primary_rows == 128, f"expected 128 primary rows, got {primary_rows}")
    return cells


def make_targets(cells: dict[tuple[int, str, int, str], float], config: dict[str, Any]) -> dict[int, dict[str, float]]:
    targets: dict[int, dict[str, float]] = {}
    for raw_seed in config["seeds"]:
        seed = int(raw_seed)
        means: dict[str, float] = {}
        for arm in ARMS:
            means[arm] = stable_mean(
                [cells[(seed, future, replica, arm)] for future in ("old_major", "new_major") for replica in (0, 1)],
                f"seed {seed} primary {arm}",
            )
        balanced_means = {
            arm: stable_mean([cells[(seed, "balanced", replica, arm)] for replica in (0, 1)], f"seed {seed} balanced {arm}")
            for arm in ARMS
        }
        targets[seed] = {
            "primary_keep": means["keep"],
            "primary_replace": means["replace_behavioral"],
            "target_delta": means["replace_behavioral"] - means["keep"],
            "balanced_keep": balanced_means["keep"],
            "balanced_replace": balanced_means["replace_behavioral"],
            "balanced_delta": balanced_means["replace_behavioral"] - balanced_means["keep"],
        }
    return targets


def make_folds(strata: dict[tuple[str, str], list[int]]) -> list[list[int]]:
    folds: list[list[int]] = [[] for _ in range(4)]
    for stratum in STRATA:
        seeds = sorted(strata[stratum])
        require(len(seeds) == 4, f"stratum {stratum} must contain exactly 4 seeds")
        for fold, seed in enumerate(seeds):
            folds[fold].append(seed)
    return [sorted(test_seeds) for test_seeds in folds]


def fit_fold(train_seeds: list[int], features: dict[int, dict[str, Any]], targets: dict[int, dict[str, float]]) -> dict[str, Any]:
    x = np.asarray([[features[seed]["features"][name] for name in FEATURE_NAMES] for seed in train_seeds], dtype=np.float64)
    y = np.asarray([targets[seed]["target_delta"] for seed in train_seeds], dtype=np.float64)
    means = x.mean(axis=0)
    raw_scales = x.std(axis=0, ddof=0)
    scales = np.where(raw_scales < 1e-12, 1.0, raw_scales)
    z = (x - means) / scales
    intercept = float(y.mean())
    beta = np.linalg.solve(z.T @ z + np.eye(3, dtype=np.float64), z.T @ (y - intercept))
    values = [*means.tolist(), *scales.tolist(), *beta.tolist(), intercept]
    require(all(math.isfinite(float(value)) for value in values), "non-finite fold parameters")
    return {
        "mean": means.tolist(),
        "scale": scales.tolist(),
        "beta": beta.tolist(),
        "intercept": intercept,
    }


def selected_values(target: dict[str, float], decision: str) -> tuple[float, float]:
    if decision == "keep":
        return target["primary_keep"], target["balanced_keep"]
    if decision == "replace":
        return target["primary_replace"], target["balanced_replace"]
    raise ValueError(f"unknown decision {decision}")


def recompute(feature_packet: Any, outcome_packet: Any, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    features, strata = validate_features(feature_packet, config)
    cells = validate_outcomes(outcome_packet, config)
    targets = make_targets(cells, config)
    test_folds = make_folds(strata)
    fold_models: list[dict[str, Any]] = []
    fold_for_seed: dict[int, int] = {}
    seeds = sorted(int(seed) for seed in config["seeds"])

    for fold, test_seeds in enumerate(test_folds):
        train_seeds = sorted(set(seeds) - set(test_seeds))
        model = fit_fold(train_seeds, features, targets)
        model_packet = {
            "fold": fold,
            "train_seeds": train_seeds,
            "test_seeds": test_seeds,
            **model,
        }
        fold_models.append(model_packet)
        for seed in test_seeds:
            fold_for_seed[seed] = fold

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        fold = fold_for_seed[seed]
        model = fold_models[fold]
        x = np.asarray([features[seed]["features"][name] for name in FEATURE_NAMES], dtype=np.float64)
        mean = np.asarray(model["mean"], dtype=np.float64)
        scale = np.asarray(model["scale"], dtype=np.float64)
        beta = np.asarray(model["beta"], dtype=np.float64)
        predicted = float(model["intercept"] + np.dot((x - mean) / scale, beta))
        require(math.isfinite(predicted), f"non-finite ridge prediction for seed {seed}")
        freq = features[seed]["features"]["recent_frequency_contrast"]
        decisions = {
            "crossfit": "replace" if predicted > 0.0 else "keep",
            "fixed_keep": "keep",
            "fixed_replace": "replace",
            "intercept_only": "replace" if model["intercept"] > 0.0 else "keep",
            "frequency_rule": "replace" if freq > 0.0 else "keep",
        }
        target = targets[seed]
        primary: dict[str, float] = {}
        balanced: dict[str, float] = {}
        primary_gain: dict[str, float] = {}
        balanced_gain: dict[str, float] = {}
        for method, decision in decisions.items():
            primary[method], balanced[method] = selected_values(target, decision)
            primary_gain[method] = primary[method] - target["primary_keep"]
            balanced_gain[method] = balanced[method] - target["balanced_keep"]
        rows.append(
            {
                "seed": seed,
                "fold": fold,
                "features": {name: features[seed]["features"][name] for name in FEATURE_NAMES},
                "predicted_delta": predicted,
                "target_delta": target["target_delta"],
                "decisions": decisions,
                "values_primary": primary,
                "values_balanced": balanced,
                "primary_gain_vs_keep": primary_gain,
            }
        )

    aggregates: dict[str, dict[str, float]] = {}
    for method in METHODS:
        aggregates[method] = {
            "mean_selected_primary": stable_mean([row["values_primary"][method] for row in rows], f"{method} selected primary"),
            "mean_primary_gain_vs_keep": stable_mean([row["primary_gain_vs_keep"][method] for row in rows], f"{method} primary gain"),
            "mean_selected_balanced": stable_mean([row["values_balanced"][method] for row in rows], f"{method} selected balanced"),
            "mean_balanced_gain_vs_keep": stable_mean([balanced_gain[method] for balanced_gain in [
                {m: row["values_balanced"][m] - targets[row["seed"]]["balanced_keep"] for m in METHODS}
                for row in rows
            ]], f"{method} balanced gain"),
        }
    expected = {
        "format": "h033_crossfit_v1",
        "scope": "exploratory_reuse_not_confirmation",
        "feature_names": list(FEATURE_NAMES),
        "folds": fold_models,
        "rows": rows,
        "aggregates": aggregates,
    }
    return expected, {"features": features, "strata": strata, "targets": targets}


def compare_tree(expected: Any, actual: Any, path: str = "crossfit") -> tuple[int, float]:
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{path} must be an object")
        require(set(expected) == set(actual), f"{path} keys differ (expected={sorted(expected)}, actual={sorted(actual)})")
        checks, maximum = 0, 0.0
        for key in expected:
            child_checks, child_max = compare_tree(expected[key], actual[key], f"{path}.{key}")
            checks += child_checks
            maximum = max(maximum, child_max)
        return checks, maximum
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(expected) == len(actual), f"{path} length differs")
        checks, maximum = 0, 0.0
        for index, (left, right) in enumerate(zip(expected, actual)):
            child_checks, child_max = compare_tree(left, right, f"{path}[{index}]")
            checks += child_checks
            maximum = max(maximum, child_max)
        return checks, maximum
    if isinstance(expected, bool) or isinstance(expected, str) or expected is None:
        require(type(expected) is type(actual) and expected == actual, f"{path} differs: expected {expected!r}, got {actual!r}")
        return 1, 0.0
    if isinstance(expected, int):
        require(isinstance(actual, int) and not isinstance(actual, bool) and expected == actual, f"{path} differs: expected {expected!r}, got {actual!r}")
        return 1, 0.0
    left, right = finite_number(expected, path), finite_number(actual, path)
    difference = abs(left - right)
    require(math.isclose(left, right, rel_tol=STAT_TOL, abs_tol=STAT_TOL), f"{path} differs: expected {left:.17g}, got {right:.17g}")
    return 1, difference


def validate_crossfit_envelope(packet: Any, feature_hash: str, outcome_hash: str, config_hash: str) -> dict[str, Any]:
    packet = require_object(packet, "crossfit packet")
    required = {"format", "scope", "feature_names", "folds", "rows", "aggregates", "input_packets", "execution"}
    require(set(packet) == required, f"crossfit packet keys mismatch (missing={sorted(required - set(packet))}, extra={sorted(set(packet) - required)})")
    require(packet.get("format") == "h033_crossfit_v1", "crossfit format must be h033_crossfit_v1")
    require(packet.get("scope") == "exploratory_reuse_not_confirmation", "crossfit scope changed")
    inputs = require_object(packet.get("input_packets"), "crossfit.input_packets", {"features", "outcomes"})
    for label, expected_hash in (("features", feature_hash), ("outcomes", outcome_hash)):
        ref = require_object(inputs[label], f"crossfit input {label}", {"path", "sha256"})
        require(isinstance(ref["path"], str) and bool(ref["path"]), f"crossfit input {label} path missing")
        require(ref["sha256"] == expected_hash, f"crossfit {label} hash does not match supplied packet")
    execution = require_object(packet.get("execution"), "crossfit.execution")
    require(execution.get("stage") == "crossfit", "crossfit execution stage is invalid")
    require(execution.get("environment_interactions") == 0, "crossfit reports nonzero environment interactions")
    config_ref = require_object(execution.get("config"), "crossfit.execution.config")
    require(config_ref.get("sha256") == config_hash, "crossfit config hash does not match frozen H033 config")
    return packet


def feature_ranges(features: dict[int, dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in FEATURE_NAMES:
        values = [features[seed]["features"][name] for seed in sorted(features)]
        result[name] = {"min": min(values), "max": max(values)}
    return result


def policy_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        primary = [row["values_primary"][method] for row in rows]
        primary_gain = [row["primary_gain_vs_keep"][method] for row in rows]
        balanced = [row["values_balanced"][method] for row in rows]
        balanced_gain = [row["values_balanced"][method] - row["target_balanced_keep"] for row in rows]
        result[method] = {
            "selection_counts": {
                "keep": sum(row["decisions"][method] == "keep" for row in rows),
                "replace": sum(row["decisions"][method] == "replace" for row in rows),
            },
            "mean_selected_primary": stable_mean(primary, f"{method} primary values"),
            "mean_primary_gain_vs_keep": stable_mean(primary_gain, f"{method} primary gains"),
            "mean_selected_balanced": stable_mean(balanced, f"{method} balanced values"),
            "mean_balanced_gain_vs_keep": stable_mean(balanced_gain, f"{method} balanced gains"),
        }
    return result


def conditional_bootstrap(rows: list[dict[str, Any]], strata: dict[tuple[str, str], list[int]]) -> dict[str, Any]:
    seed_to_position = {row["seed"]: index for index, row in enumerate(rows)}
    matrix = np.asarray(
        [[row["paired_primary_differences"][control] for control in CONTROLS] for row in rows],
        dtype=np.float64,
    )
    grouped_positions = [np.asarray([seed_to_position[seed] for seed in sorted(strata[stratum])], dtype=np.int64) for stratum in STRATA]
    require(all(len(group) == 4 for group in grouped_positions), "bootstrap requires 4 seeds per frozen stratum")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws_by_stratum = rng.integers(0, 4, size=(BOOTSTRAP_DRAWS, len(STRATA), 4), endpoint=False)
    joint_indices = np.concatenate(
        [group[draws_by_stratum[:, stratum_index, :]] for stratum_index, group in enumerate(grouped_positions)],
        axis=1,
    )
    draw_g = matrix[joint_indices].mean(axis=1)
    draw_g_min = draw_g.min(axis=1)
    q = np.quantile(draw_g, [0.025, 0.975], axis=0)
    q_min = np.quantile(draw_g_min, [0.025, 0.975])
    return {
        "method": "conditional stratified percentile bootstrap of fixed OOF values",
        "resampling_unit": "seed within the four family-by-history strata; n=4 sampled with replacement per stratum",
        "draws": BOOTSTRAP_DRAWS,
        "bit_generator": "PCG64",
        "seed": BOOTSTRAP_SEED,
        "quantile_method": "NumPy default linear",
        "G_b_interval_2_5_97_5": {
            control: [float(q[0, index]), float(q[1, index])] for index, control in enumerate(CONTROLS)
        },
        "G_min_interval_2_5_97_5": [float(q_min[0]), float(q_min[1])],
        "interpretation": "Conditional resampling range for fixed held-out values only; no refitting, training uncertainty, formal 95% coverage, or p-value claim.",
    }


def build_analysis(
    feature_packet: Any,
    outcome_packet: Any,
    crossfit_packet: Any,
    hashes: dict[str, str],
    paths: dict[str, str],
    context: dict[str, Any],
) -> dict[str, Any]:
    config = context["config"]
    expected, validated = recompute(feature_packet, outcome_packet, config)
    crossfit_packet = validate_crossfit_envelope(
        crossfit_packet, hashes["features"], hashes["outcomes"], context["config_sha256"]
    )
    comparisons, maximum_difference = compare_tree(expected, {key: crossfit_packet[key] for key in expected})
    features, strata, targets = validated["features"], validated["strata"], validated["targets"]

    rows: list[dict[str, Any]] = []
    for computed_row in expected["rows"]:
        seed = computed_row["seed"]
        target = targets[seed]
        values_primary = computed_row["values_primary"]
        paired = {control: values_primary["crossfit"] - values_primary[control] for control in CONTROLS}
        rows.append(
            {
                "seed": seed,
                "old_family": features[seed]["old_family"],
                "history_regime": features[seed]["history_regime"],
                "fold": computed_row["fold"],
                "features": computed_row["features"],
                "target_primary_keep": target["primary_keep"],
                "target_primary_replace": target["primary_replace"],
                "target_delta": target["target_delta"],
                "target_balanced_keep": target["balanced_keep"],
                "target_balanced_replace": target["balanced_replace"],
                "target_balanced_delta": target["balanced_delta"],
                "predicted_delta": computed_row["predicted_delta"],
                "decisions": computed_row["decisions"],
                "values_primary": values_primary,
                "primary_gain_vs_keep": computed_row["primary_gain_vs_keep"],
                "values_balanced": computed_row["values_balanced"],
                "balanced_gain_vs_keep": {
                    method: computed_row["values_balanced"][method] - target["balanced_keep"] for method in METHODS
                },
                "paired_primary_differences": paired,
                "squared_prediction_error": (computed_row["predicted_delta"] - target["target_delta"]) ** 2,
            }
        )

    policies = policy_summary(rows)
    fold_results: list[dict[str, Any]] = []
    for fold in range(4):
        fold_rows = [row for row in rows if row["fold"] == fold]
        means = {control: stable_mean([row["paired_primary_differences"][control] for row in fold_rows], f"fold {fold} {control}") for control in CONTROLS}
        worst = min(means.values())
        fold_results.append({"fold": fold, "test_seeds": [row["seed"] for row in fold_rows], "G_b_fold_means": means, "fold_worst_case": worst, "positive": worst > 0.0})

    g_b = {control: policies["crossfit"]["mean_selected_primary"] - policies[control]["mean_selected_primary"] for control in CONTROLS}
    g_min = min(g_b.values())
    positive_folds = sum(row["positive"] for row in fold_results)
    status = "PROMISING_EXPLORATORY" if g_min >= 0.002 and positive_folds >= 3 else "NO_ADVANCEMENT"
    bootstrap = conditional_bootstrap(rows, strata)
    mse = stable_mean([row["squared_prediction_error"] for row in rows], "crossfit OOF MSE")

    provenance = json.loads(json.dumps(context["provenance"]))
    for label in ("features", "outcomes", "crossfit"):
        provenance["file_hashes"][f"input_{label}"] = {"path": paths[label], "sha256": hashes[label]}
    per_seed = []
    for row in rows:
        clean = {key: value for key, value in row.items() if key not in ("target_balanced_keep", "target_balanced_replace", "target_balanced_delta", "paired_primary_differences", "squared_prediction_error")}
        clean["balanced_targets"] = {
            "keep": row["target_balanced_keep"],
            "replace": row["target_balanced_replace"],
            "delta": row["target_balanced_delta"],
        }
        clean["paired_primary_differences_ridge_minus_control"] = row["paired_primary_differences"]
        clean["squared_prediction_error_diagnostic_only"] = row["squared_prediction_error"]
        per_seed.append(clean)

    fold_models = expected["folds"]
    return {
        "format": "e162_h033_independent_statistics_v1",
        "scope": "read_only_independent_recomputation_of_frozen_oof_statistics",
        "experiment": "E162",
        "hypothesis": "H033",
        "input_data": {
            label: {"path": paths[label], "sha256": hashes[label], "format": packet.get("format"), "row_count": len(packet.get("rows", []))}
            for label, packet in (("features", feature_packet), ("outcomes", outcome_packet))
        } | {"crossfit": {"path": paths["crossfit"], "sha256": hashes["crossfit"], "format": crossfit_packet.get("format"), "seed_rows": len(crossfit_packet.get("rows", []))}},
        "provenance": provenance,
        "validation": {
            "configured_seeds": sorted(int(seed) for seed in config["seeds"]),
            "seed_count": len(rows),
            "outcome_cell_count": 192,
            "primary_outcome_cell_count": 128,
            "balanced_outcome_cell_count": 64,
            "folds": 4,
            "train_seeds_per_fold": 12,
            "test_seeds_per_fold": 4,
            "strata": {f"{family}×{history}": sorted(seeds) for (family, history), seeds in strata.items()},
            "crossfit_input_hashes_match_supplied_packets": True,
            "crossfit_recomputation": {"matched_all_reported_folds_rows_and_aggregates": True, "numeric_fields_checked": comparisons, "maximum_absolute_numeric_difference": maximum_difference, "tolerance": {"relative": STAT_TOL, "absolute": STAT_TOL}},
        },
        "feature_ranges": feature_ranges(features),
        "fold_models": fold_models,
        "policy_summaries": policies,
        "paired_primary_gains_vs_controls": {
            "definition": "G_b = mean over the same 16 seeds of V_ridge_crossfit - V_control",
            "G_b": g_b,
            "G_min": g_min,
            "G_min_control": min(g_b, key=g_b.get),
        },
        "fold_stability": {
            "folds": fold_results,
            "positive_fold_count": positive_folds,
            "required_positive_folds": 3,
        },
        "screening_decision": {
            "status": status,
            "G_min_threshold": 0.002,
            "G_min_pass": g_min >= 0.002,
            "positive_fold_requirement": "at least 3 of 4 fold_worst_case values > 0",
            "fold_stability_pass": positive_folds >= 3,
            "gate_pass": status == "PROMISING_EXPLORATORY",
            "meaning": "Operational Stage A screen only; not a significance test, independent confirmation, or authorization for another phase.",
        },
        "conditional_bootstrap": bootstrap,
        "mse_diagnostic_only": {"primary_oof_mean_squared_error": mse, "use": "diagnostic only; excluded from advancement gate"},
        "per_seed": per_seed,
        "limitations": [
            "Sixteen complete seeds are the independent evaluation units; the 192 outcome cells and four overlapping training folds are not independent replications.",
            "Balanced outcomes score the fixed held-out decisions only and do not enter targets, fitting, scaling, thresholds, or advancement criteria.",
            "The conditional bootstrap resamples fixed OOF values within the four strata; it does not refit models or capture training uncertainty and makes no formal confidence-coverage or p-value claim.",
            "A positive result is exploratory reuse of previously observed E161 labels and does not confirm deployment performance or establish the broader H033 claim.",
        ],
    }


def render_report(metrics: dict[str, Any]) -> str:
    policies = metrics["policy_summaries"]
    lines = [
        "# E162 independent statistics report",
        "",
        f"Status: **{metrics['screening_decision']['status']}**.",
        "",
        "This is a read-only audit of the frozen 16-seed, four-fold H033 v1 output. The script independently recomputed primary arm means, the fixed standardized ridge OOF predictions, actions, selected values, and aggregates from the supplied packets, then compared them with the crossfit packet.",
        "",
        "## Provenance",
        "",
        "Input packet hashes and all frozen H033 delivery snapshot hashes, including this statistics source hash, are recorded in `metrics.json` under `provenance.file_hashes` and `provenance.frozen_delivery_snapshot_files`.",
        "",
        "| Input | SHA256 |",
        "| --- | --- |",
    ]
    for label in ("features", "outcomes", "crossfit"):
        item = metrics["input_data"][label]
        lines.append(f"| {label} (`{item['path']}`) | `{item['sha256']}` |")
    lines += [
        "",
        f"The independent recomputation matched {metrics['validation']['crossfit_recomputation']['numeric_fields_checked']} reported fields; maximum absolute numeric difference was {metrics['validation']['crossfit_recomputation']['maximum_absolute_numeric_difference']:.3g} (tolerance: relative and absolute {STAT_TOL:g}).",
        "",
        "## Policy results",
        "",
        "Values are reward per step averaged over the 16 seed-level selections. Balanced is secondary and scores the same decisions.",
        "",
        "| Policy | Keep | Replace | Primary value | Primary gain vs keep | Balanced value | Balanced gain vs keep |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        summary = policies[method]
        counts = summary["selection_counts"]
        lines.append(
            f"| {method} | {counts['keep']} | {counts['replace']} | {summary['mean_selected_primary']:.8f} | {summary['mean_primary_gain_vs_keep']:.8f} | {summary['mean_selected_balanced']:.8f} | {summary['mean_balanced_gain_vs_keep']:.8f} |"
        )
    lines += [
        "",
        "### Paired primary value differences",
        "",
        "| Control | G_b = mean(ridge − control) |",
        "| --- | ---: |",
    ]
    for control in CONTROLS:
        lines.append(f"| {control} | {metrics['paired_primary_gains_vs_controls']['G_b'][control]:.8f} |")
    lines += [
        "",
        f"G_min = **{metrics['paired_primary_gains_vs_controls']['G_min']:.8f}** against `{metrics['paired_primary_gains_vs_controls']['G_min_control']}`. The fixed screen requires G_min ≥ 0.002 and at least 3 of 4 fold worst-case means > 0.",
        "",
        "| Fold | Test seeds | Fold worst-case min_b mean(d) | Positive |",
        "| ---: | --- | ---: | --- |",
    ]
    for fold in metrics["fold_stability"]["folds"]:
        lines.append(f"| {fold['fold']} | {', '.join(str(seed) for seed in fold['test_seeds'])} | {fold['fold_worst_case']:.8f} | {fold['positive']} |")
    lines += [
        "",
        "## Frozen ridge folds and feature ranges",
        "",
        "Feature ranges cover all 16 input rows. Each fold's training-only means, scales, coefficients, intercept, and seed membership are fully listed in `metrics.json` under `fold_models`.",
        "",
        "| Feature | Minimum | Maximum |",
        "| --- | ---: | ---: |",
    ]
    for name, values in metrics["feature_ranges"].items():
        lines.append(f"| {name} | {values['min']:.10g} | {values['max']:.10g} |")
    lines += [
        "",
        f"Primary OOF MSE is {metrics['mse_diagnostic_only']['primary_oof_mean_squared_error']:.8g}; it is diagnostic only and did not enter the advancement gate.",
        "",
        "## Conditional bootstrap",
        "",
        f"The fixed OOF paired values were resampled {BOOTSTRAP_DRAWS:,} times, drawing 4 seeds with replacement within each family-by-history stratum and using the same sampled seed indices for every contrast and G_min. NumPy PCG64 seed: {BOOTSTRAP_SEED}; quantiles use NumPy's default linear method.",
        "",
        "| Quantity | 2.5% | 97.5% |",
        "| --- | ---: | ---: |",
    ]
    bootstrap = metrics["conditional_bootstrap"]
    for control in CONTROLS:
        interval = bootstrap["G_b_interval_2_5_97_5"][control]
        lines.append(f"| G_b: {control} | {interval[0]:.8f} | {interval[1]:.8f} |")
    interval = bootstrap["G_min_interval_2_5_97_5"]
    lines.append(f"| G_min | {interval[0]:.8f} | {interval[1]:.8f} |")
    lines += [
        "",
        "These are conditional resampling ranges for already fixed held-out values. They do not refit the predictor, include training uncertainty, claim formal 95% coverage, or provide p-values.",
        "",
        "All per-seed predictions, features, actions, primary and balanced selected values, gains, and paired differences are in `metrics.json` and `per_seed.csv`.",
        "",
        "The result is exploratory reuse of E161 outcomes. A PROMISING_EXPLORATORY status only meets the registered Stage A screen; it does not establish confirmation or authorize another phase.",
        "",
    ]
    return "\n".join(lines)


def csv_bytes(metrics: dict[str, Any]) -> bytes:
    methods = METHODS
    labels = [
        "seed", "old_family", "history_regime", "fold",
        *[f"feature_{name}" for name in FEATURE_NAMES],
        "target_primary_keep", "target_primary_replace", "target_delta",
        "target_balanced_keep", "target_balanced_replace", "target_balanced_delta",
        "predicted_delta", "squared_prediction_error_diagnostic_only",
    ]
    for method in methods:
        labels += [
            f"decision_{method}", f"value_primary_{method}", f"gain_primary_vs_keep_{method}",
            f"value_balanced_{method}", f"gain_balanced_vs_keep_{method}",
        ]
    for control in CONTROLS:
        labels.append(f"paired_difference_ridge_minus_{control}_primary")
    labels += [f"provenance_sha256_{label}" for label in sorted(metrics["provenance"]["file_hashes"])]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=labels, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    hashes = metrics["provenance"]["file_hashes"]
    for row in metrics["per_seed"]:
        record: dict[str, Any] = {
            "seed": row["seed"], "old_family": row["old_family"], "history_regime": row["history_regime"], "fold": row["fold"],
            "target_primary_keep": row["target_primary_keep"], "target_primary_replace": row["target_primary_replace"], "target_delta": row["target_delta"],
            "target_balanced_keep": row["balanced_targets"]["keep"], "target_balanced_replace": row["balanced_targets"]["replace"], "target_balanced_delta": row["balanced_targets"]["delta"],
            "predicted_delta": row["predicted_delta"], "squared_prediction_error_diagnostic_only": row["squared_prediction_error_diagnostic_only"],
        }
        for name in FEATURE_NAMES:
            record[f"feature_{name}"] = row["features"][name]
        for method in methods:
            record[f"decision_{method}"] = row["decisions"][method]
            record[f"value_primary_{method}"] = row["values_primary"][method]
            record[f"gain_primary_vs_keep_{method}"] = row["primary_gain_vs_keep"][method]
            record[f"value_balanced_{method}"] = row["values_balanced"][method]
            record[f"gain_balanced_vs_keep_{method}"] = row["balanced_gain_vs_keep"][method]
        for control in CONTROLS:
            record[f"paired_difference_ridge_minus_{control}_primary"] = row["paired_primary_differences_ridge_minus_control"][control]
        for label, info in hashes.items():
            record[f"provenance_sha256_{label}"] = info["sha256"]
        writer.writerow(record)
    return output.getvalue().encode("utf-8")


def analyze_paths(features_path: Path, outcomes_path: Path, crossfit_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    for label, path in (("features", features_path), ("outcomes", outcomes_path), ("crossfit", crossfit_path)):
        require(path.is_file(), f"{label} input is not a file: {path}")
    paths = {label: str(path.resolve(strict=True)) for label, path in (("features", features_path), ("outcomes", outcomes_path), ("crossfit", crossfit_path))}
    hashes = {label: sha256_file(path) for label, path in (("features", features_path), ("outcomes", outcomes_path), ("crossfit", crossfit_path))}
    packets = {label: read_json(path) for label, path in (("features", features_path), ("outcomes", outcomes_path), ("crossfit", crossfit_path))}
    context = load_context()
    metrics = build_analysis(packets["features"], packets["outcomes"], packets["crossfit"], hashes, paths, context)
    return metrics, context


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently audit frozen E162 H033 crossfit output and report statistics.")
    parser.add_argument("--features")
    parser.add_argument("--outcomes")
    parser.add_argument("--crossfit")
    parser.add_argument("--output", help="new output directory for metrics.json, report.md, and per_seed.csv")
    parser.add_argument("--self-check", action="store_true", help="run synthetic-only hand calculations and reproducibility checks")
    args = parser.parse_args(argv)
    if args.self_check:
        return run_synthetic_check()
    if not all((args.features, args.outcomes, args.crossfit, args.output)):
        parser.error("--features, --outcomes, --crossfit, and --output are all required")

    output_dir = Path(args.output).expanduser().absolute()
    try:
        require(not output_dir.exists(), f"refusing to overwrite existing output directory: {output_dir}")
        metrics, _ = analyze_paths(Path(args.features).expanduser(), Path(args.outcomes).expanduser(), Path(args.crossfit).expanduser())
        metrics_payload = json_bytes(metrics)
        report_payload = render_report(metrics).encode("utf-8")
        csv_payload = csv_bytes(metrics)
        output_dir.mkdir(parents=True, exist_ok=False)
        for name, payload in (("metrics.json", metrics_payload), ("report.md", report_payload), ("per_seed.csv", csv_payload)):
            with (output_dir / name).open("xb") as stream:
                stream.write(payload)
        print(f"E162 statistics written to {output_dir}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        print(f"statistics error: {exc}", file=sys.stderr)
        return 2


def synthetic_packets(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    seeds = sorted(int(seed) for seed in config["seeds"])
    features_rows: list[dict[str, Any]] = []
    index_by_seed = {seed: index for index, seed in enumerate(seeds)}
    for stratum_index, (family, history) in enumerate(STRATA):
        group = seeds[stratum_index * 4 : (stratum_index + 1) * 4]
        for seed in group:
            position = index_by_seed[seed]
            features_rows.append(
                {
                    "seed": seed,
                    "old_family": family,
                    "history_regime": history,
                    "features": {
                        "recent_frequency_contrast": (position - 7.5) / 8.0 + stratum_index * 0.04,
                        "new_fit_gain": ((position * 3) % 11 - 5) / 5.0,
                        "old_prior_loss": ((position * 7) % 13) / 4.0,
                    },
                    "provenance": {"source": "synthetic_hand_check", "seed_index": position},
                }
            )
    features_rows.sort(key=lambda row: row["seed"])
    feature_packet = {"format": "h033_features_v1", "rows": features_rows, "metadata": {"source": "synthetic_hand_check"}}

    outcome_rows: list[dict[str, Any]] = []
    for seed in seeds:
        position = index_by_seed[seed]
        for future in FUTURES:
            future_sign = -1 if future == "old_major" else 1 if future == "new_major" else 0
            for replica in (0, 1):
                replica_sign = -1 if replica == 0 else 1
                if future == "balanced":
                    keep = 0.7 + 0.03 * position + 0.004 * replica_sign
                    delta = -0.01 + 0.001 * position
                else:
                    keep = 1.0 + 0.1 * position + 0.01 * future_sign + 0.002 * replica_sign
                    delta = 0.004 + 0.01 * (position - 7.5) + 0.002 * future_sign + 0.001 * replica_sign
                for arm in ARMS:
                    reward = keep if arm == "keep" else keep + delta
                    outcome_rows.append(
                        {"seed": seed, "future_regime": future, "replica": replica, "arm": arm, "horizon": 6144, "reward_per_step": reward}
                    )
    outcome_packet = {"format": "h033_outcomes_v1", "rows": outcome_rows, "metadata": {"source": "synthetic_hand_check"}}
    return feature_packet, outcome_packet


def run_synthetic_check() -> int:
    """Exercise the CLI with fake packets and persist compact verification evidence."""
    try:
        context = load_context()
        root: Path = context["root"]
        preparation = root / "experiments" / "exp162" / "preparation"
        config = context["config"]
        feature_packet, outcome_packet = synthetic_packets(config)
        frozen_src = root / "research" / "appendix" / "H033" / "delivery_v1" / "source_snapshot" / "src"
        # Keep the frozen delivery snapshot byte-for-byte unchanged during the
        # synthetic expected-output generation.
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(frozen_src))
        from online_rl.h033_diagnostic import crossfit_diagnostic  # synthetic packets only

        with tempfile.TemporaryDirectory(prefix=".statistics-check-", dir=preparation) as temporary:
            temp = Path(temporary)
            feature_path, outcome_path, crossfit_path = temp / "features.json", temp / "outcomes.json", temp / "crossfit.json"
            feature_path.write_bytes(json_bytes(feature_packet))
            outcome_path.write_bytes(json_bytes(outcome_packet))
            crossfit_packet = crossfit_diagnostic(feature_packet, outcome_packet, config)
            crossfit_packet["input_packets"] = {
                "features": {"path": str(feature_path.resolve()), "sha256": sha256_file(feature_path)},
                "outcomes": {"path": str(outcome_path.resolve()), "sha256": sha256_file(outcome_path)},
            }
            crossfit_packet["execution"] = {"stage": "crossfit", "config": {"path": "synthetic", "sha256": context["config_sha256"]}, "cpu_s": 0.0, "wall_s": 0.0, "environment_interactions": 0, "formal_statistics": "owned_by_experiment_chat"}
            crossfit_path.write_bytes(json_bytes(crossfit_packet))
            synthetic_hashes = {"features": sha256_file(feature_path), "outcomes": sha256_file(outcome_path), "crossfit": sha256_file(crossfit_path)}
            output_dir = temp / "synthetic-report"
            args = ["--features", str(feature_path), "--outcomes", str(outcome_path), "--crossfit", str(crossfit_path), "--output", str(output_dir)]
            require(main(args) == 0, "synthetic CLI run failed")
            metrics = read_json(output_dir / "metrics.json")
            require((output_dir / "report.md").is_file() and (output_dir / "per_seed.csv").is_file(), "synthetic CLI outputs are incomplete")

            seeds = sorted(int(seed) for seed in config["seeds"])
            for position, seed in enumerate(seeds):
                row = metrics["per_seed"][position]
                expected_keep = 1.0 + 0.1 * position
                expected_delta = 0.004 + 0.01 * (position - 7.5)
                expected_balanced_keep = 0.7 + 0.03 * position
                expected_balanced_delta = -0.01 + 0.001 * position
                require(math.isclose(row["target_primary_keep"], expected_keep, abs_tol=1e-12), f"hand primary keep mismatch at seed {seed}")
                require(math.isclose(row["target_delta"], expected_delta, abs_tol=1e-12), f"hand primary delta mismatch at seed {seed}")
                require(math.isclose(row["balanced_targets"]["keep"], expected_balanced_keep, abs_tol=1e-12), f"hand balanced keep mismatch at seed {seed}")
                require(math.isclose(row["balanced_targets"]["delta"], expected_balanced_delta, abs_tol=1e-12), f"hand balanced delta mismatch at seed {seed}")
            require(math.isclose(metrics["policy_summaries"]["fixed_keep"]["mean_selected_primary"], 1.75, abs_tol=1e-12), "hand mean fixed_keep value mismatch")
            require(math.isclose(metrics["policy_summaries"]["fixed_replace"]["mean_selected_primary"], 1.754, abs_tol=1e-12), "hand mean fixed_replace value mismatch")
            require(math.isclose(metrics["policy_summaries"]["fixed_replace"]["mean_primary_gain_vs_keep"], 0.004, abs_tol=1e-12), "hand fixed_replace gain mismatch")
            require(all(sum(summary["selection_counts"].values()) == 16 for summary in metrics["policy_summaries"].values()), "selection counts do not sum to 16")

            manual_g = {
                control: stable_mean([row["values_primary"]["crossfit"] - row["values_primary"][control] for row in metrics["per_seed"]], control)
                for control in CONTROLS
            }
            for control in CONTROLS:
                require(math.isclose(manual_g[control], metrics["paired_primary_gains_vs_controls"]["G_b"][control], abs_tol=1e-12), f"hand G_b mismatch for {control}")
            manual_fold_positive = 0
            for fold in range(4):
                fold_rows = [row for row in metrics["per_seed"] if row["fold"] == fold]
                worst = min(stable_mean([row["values_primary"]["crossfit"] - row["values_primary"][control] for row in fold_rows], control) for control in CONTROLS)
                actual_fold = metrics["fold_stability"]["folds"][fold]
                require(math.isclose(worst, actual_fold["fold_worst_case"], abs_tol=1e-12), f"hand fold worst-case mismatch for fold {fold}")
                manual_fold_positive += int(worst > 0.0)
            manual_status = "PROMISING_EXPLORATORY" if min(manual_g.values()) >= 0.002 and manual_fold_positive >= 3 else "NO_ADVANCEMENT"
            require(metrics["screening_decision"]["status"] == manual_status, "screening gate hand-check mismatch")

            synthetic_strata = {stratum: [] for stratum in STRATA}
            for row in feature_packet["rows"]:
                synthetic_strata[(row["old_family"], row["history_regime"])].append(row["seed"])
            bootstrap_rows = [
                {"seed": row["seed"], "paired_primary_differences": row["paired_primary_differences_ridge_minus_control"]}
                for row in metrics["per_seed"]
            ]
            boot_a = conditional_bootstrap(bootstrap_rows, synthetic_strata)
            boot_b = conditional_bootstrap(bootstrap_rows, synthetic_strata)
            require(boot_a == boot_b == metrics["conditional_bootstrap"], "bootstrap intervals are not reproducible")
            boot_hash = sha256_bytes(json_bytes(boot_a))

            previous = (output_dir / "metrics.json").read_bytes()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                overwrite_code = main(args)
            require(overwrite_code == 2 and (output_dir / "metrics.json").read_bytes() == previous, "output directory overwrite guard failed")

            check = {
                "format": "e162_h033_statistics_synthetic_check_v1",
                "scope": "synthetic_only_no_real_feature_or_crossfit_packets_read",
                "experiment": "E162",
                "hypothesis": "H033",
                "statistics_source_sha256": context["provenance"]["file_hashes"]["statistics_source"]["sha256"],
                "synthetic_packet_hashes": synthetic_hashes,
                "checks": [
                    {"name": "192_outcomes_128_primary_and_complete_16_seed_schema", "status": "PASS", "expected": {"outcomes": 192, "primary": 128, "balanced": 64, "seeds": 16}, "observed": {"outcomes": metrics["validation"]["outcome_cell_count"], "primary": metrics["validation"]["primary_outcome_cell_count"], "balanced": metrics["validation"]["balanced_outcome_cell_count"], "seeds": metrics["validation"]["seed_count"]}},
                    {"name": "hand_calculated_primary_and_balanced_arm_means", "status": "PASS", "fixed_keep_mean_primary": 1.75, "fixed_replace_mean_primary": 1.754, "fixed_replace_gain_vs_keep": 0.004, "all_16_seed_arm_means_checked": True},
                    {"name": "independent_crossfit_models_predictions_actions_values_and_aggregates", "status": "PASS", "numeric_fields_checked": metrics["validation"]["crossfit_recomputation"]["numeric_fields_checked"], "max_abs_difference": metrics["validation"]["crossfit_recomputation"]["maximum_absolute_numeric_difference"]},
                    {"name": "hand_calculated_paired_gains_fold_gate_and_selection_counts", "status": "PASS", "G_b": manual_g, "G_min": min(manual_g.values()), "positive_fold_count": manual_fold_positive, "synthetic_status": manual_status},
                    {"name": "conditional_bootstrap_reproducibility", "status": "PASS", "draws": BOOTSTRAP_DRAWS, "generator": "PCG64", "seed": BOOTSTRAP_SEED, "intervals_sha256": boot_hash, "intervals": boot_a},
                    {"name": "new_output_directory_and_refuse_overwrite", "status": "PASS"},
                ],
                "synthetic_only_note": "The frozen crossfit implementation was used only to create an expected output from generated synthetic packets. The independent statistics implementation recomputed it separately. No E161 feature packets, E161 crossfit output, extraction, fitting on real inputs, simulations, or remote commands were used.",
            }
            check_path = preparation / "statistics_check.json"
            with check_path.open("wb") as stream:
                stream.write(json_bytes(check))
        print(f"Synthetic statistics checks passed: {preparation / 'statistics_check.json'}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError, np.linalg.LinAlgError) as exc:
        print(f"synthetic statistics check failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
