"""Pure, exploratory E165 report construction from normalized score values.

The input is an experiment-owned logical-score payload.  It is not a
scientific-helper API response and this module does not load artifacts, score
candidates, choose arms, or make an advancement decision.
"""

import math


INPUT_FORMAT = "e165_logical_scores_v1"
OUTPUT_FORMAT = "e165_exploratory_report_v1"
SCORE_FORMAT = "h036_score_metrics_v1"
EXPECTED_SEEDS = (
    26201, 26202, 26203, 26204, 26205, 26206, 26207, 26208,
    26209, 26210, 26211, 26212, 26213, 26215, 26216, 26218,
)
ARMS = ("C", "D", "S0", "S1")
TRAINING_ARMS = ("D", "S0", "S1")
ORIGINS = {"NEW_SCORE", "REUSED_E163_D", "ALIAS"}
SCORE_VIEWS = ("pre_codec_fresh", "encoded_fresh", "W0_fresh")
OLD_KL_VIEWS = ("pre_codec_fresh", "encoded_fresh")
G_NEW_VIEWS = ("pre_codec_fresh", "encoded_fresh")
SPLITS = ("train", "holdout")
METRIC_KINDS = ("old", "new")
TRAINING_FIELDS = ("fullL", "new_train_kl", "old_train_kl", "ridge")
COMPARISONS = (("S1-S0", "S0", "S1"), ("S1-D", "D", "S1"))
LAMBDA_VALUES = {"C": 0, "D": 1, "S0": 1, "S1": 1}
FLOAT32_BOUND_TOLERANCE = 1e-12
WITNESS_TOLERANCE = 1e-8
RATIO_DENOMINATOR_FLOOR = 1e-10


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(label + " must be an object")
    return value


def _non_empty_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " must be a non-empty string")
    return value


def _sha256(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)):
        raise ValueError(label + " must be a 64-character SHA256 hex digest")
    return value


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(label + " must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ValueError(label + " must be a finite number") from exc
    if not finite:
        raise ValueError(label + " must be a finite number")
    return value


def _score_values(metrics, seed, expected_lambda, expected_prepared_id, label):
    metrics = _mapping(metrics, label)
    if metrics.get("format") != SCORE_FORMAT:
        raise ValueError(label + ".format must be " + SCORE_FORMAT)
    metric_seed = metrics.get("seed")
    if isinstance(metric_seed, bool) or not isinstance(metric_seed, int) or metric_seed != seed:
        raise ValueError(label + ".seed does not match its logical prefix")
    lambda_value = metrics.get("lambda_value")
    if isinstance(lambda_value, bool) or not isinstance(lambda_value, (int, float)) or lambda_value != expected_lambda:
        raise ValueError(label + ".lambda_value does not match the arm contract")
    if metrics.get("binding") != "wildcard":
        raise ValueError(label + ".binding must be wildcard")
    prepared_id = _non_empty_string(metrics.get("prepared_id"), label + ".prepared_id")
    if prepared_id != expected_prepared_id:
        raise ValueError(label + ".prepared_id does not match its logical prefix")

    views = {
        view: {
            key: _finite_number(source.get(key), label + "." + view + "." + key)
            for split in SPLITS for kind in METRIC_KINDS for key in (kind + "_" + split)
        }
        for view in SCORE_VIEWS
        for source in (_mapping(metrics.get(view), label + "." + view),)
    }
    aliases = {
        "K_old": views["encoded_fresh"]["old_holdout"],
        "K_new": views["encoded_fresh"]["new_holdout"],
        "K_new_W0": views["W0_fresh"]["new_holdout"],
    }
    legacy = {}
    for key, expected in aliases.items():
        actual = _finite_number(metrics.get(key), label + "." + key)
        if actual != expected:
            raise ValueError(label + "." + key + " disagrees with its nested metric")
        legacy[key] = actual

    derived_holdout_gain = _finite_number(aliases["K_new_W0"] - aliases["K_new"], label + ".derived_G_new")
    legacy_gain = _finite_number(metrics.get("G_new"), label + ".G_new")
    if legacy_gain != derived_holdout_gain:
        raise ValueError(label + ".G_new disagrees with K_new_W0 - K_new")
    legacy["G_new"] = legacy_gain

    gains = {
        view: {split: _finite_number(views["W0_fresh"]["new_" + split] - views[view]["new_" + split],
                                     label + ".G_new." + view + "." + split) for split in SPLITS}
        for view in G_NEW_VIEWS
    }
    if gains["encoded_fresh"]["holdout"] != legacy_gain:
        raise ValueError(label + ".G_new disagrees with encoded holdout gain")

    # Retain every canonical h036_score_metrics_v1 metric in normalized C rows.
    canonical_metrics = {"format": SCORE_FORMAT, "seed": seed, "lambda_value": expected_lambda,
                         "binding": "wildcard", "prepared_id": prepared_id}
    canonical_metrics.update(views)
    canonical_metrics.update(legacy)
    return {
        "prepared_id": prepared_id,
        "views": views,
        "old_kl": {view: {split: views[view]["old_" + split] for split in SPLITS} for view in OLD_KL_VIEWS},
        "G_new": gains,
        "legacy": legacy,
        "canonical_metrics": canonical_metrics,
    }


def _validate_payload(payload):
    payload = _mapping(payload, "payload")
    if payload.get("format") != INPUT_FORMAT:
        raise ValueError("payload.format must be " + INPUT_FORMAT)
    source_identity = _non_empty_string(
        payload.get("source_identity_sha256"), "source_identity_sha256"
    )
    cohort_lock = _sha256(payload.get("cohort_lock_sha256"), "cohort_lock_sha256")
    prefixes = payload.get("prefixes")
    if not isinstance(prefixes, list) or len(prefixes) != len(EXPECTED_SEEDS):
        raise ValueError("payload must contain exactly 16 logical prefixes")

    parsed = []
    for index, raw_prefix in enumerate(prefixes):
        label = "prefixes[" + str(index) + "]"
        raw_prefix = _mapping(raw_prefix, label)
        seed = raw_prefix.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(label + ".seed must be an integer")
        if seed != EXPECTED_SEEDS[index]:
            raise ValueError("logical prefix order/seed set differs from the fixed 16-prefix set")
        prepared_id = _non_empty_string(raw_prefix.get("prepared_id"), label + ".prepared_id")
        budget = _finite_number(raw_prefix.get("b"), label + ".b")

        training_source = _mapping(raw_prefix.get("training"), label + ".training")
        training = {
            arm: {
                field: _finite_number(arm_values.get(field), label + ".training." + arm + "." + field)
                for field in TRAINING_FIELDS
            }
            for arm in TRAINING_ARMS
            for arm_values in (_mapping(training_source.get(arm), label + ".training." + arm),)
        }

        hashes_source = _mapping(raw_prefix.get("candidate_sha256"), label + ".candidate_sha256")
        candidate_hashes = {
            arm: _sha256(hashes_source.get(arm), label + ".candidate_sha256." + arm)
            for arm in TRAINING_ARMS
        }

        metrics_source = _mapping(raw_prefix.get("metrics"), label + ".metrics")
        metrics = {arm: _score_values(metrics_source.get(arm), seed, LAMBDA_VALUES[arm], prepared_id,
                                      label + ".metrics." + arm) for arm in ARMS}
        reference_view = metrics["C"]["views"]["W0_fresh"]
        for arm in ARMS[1:]:
            arm_view = metrics[arm]["views"]["W0_fresh"]
            for key in ("old_train", "new_train", "old_holdout", "new_holdout"):
                if arm_view[key] != reference_view[key]:
                    raise ValueError(label + ".metrics." + arm + " W0_fresh differs from C")

        origins_source = _mapping(raw_prefix.get("origins"), label + ".origins")
        origins = {arm: origins_source.get(arm) for arm in ("S0", "S1")}
        for arm, origin in origins.items():
            if not isinstance(origin, str) or origin not in ORIGINS:
                raise ValueError(label + ".origins." + arm + " is not a permitted origin")

        references_source = _mapping(raw_prefix.get("references"), label + ".references")
        references = {arm: _mapping(references_source.get(arm), label + ".references." + arm)
                      for arm in ARMS}
        for arm, reference in references.items():
            if not reference:
                raise ValueError(label + ".references." + arm + " must be non-empty")

        parsed.append({"seed": seed, "prepared_id": prepared_id, "b": budget, "training": training,
                       "candidate_sha256": candidate_hashes, "metrics": metrics,
                       "origins": origins, "references": references})
    return source_identity, cohort_lock, parsed


def _direction(delta, metric):
    if delta == 0:
        return "unchanged"
    if metric == "old_kl":
        return "improved" if delta < 0 else "degraded"
    return "improved" if delta > 0 else "degraded"


def _paired_delta(left, right, metric, label):
    delta = _finite_number(right - left, label + ".delta")
    return {
        "from_value": left,
        "to_value": right,
        "signed_delta_to_minus_from": delta,
        "absolute_delta": abs(delta),
        "classification": _direction(delta, metric),
    }


def _training_delta(left, right, label):
    delta = _finite_number(right - left, label + ".delta")
    return {
        "from_value": left,
        "to_value": right,
        "signed_delta_to_minus_from": delta,
        "absolute_delta": abs(delta),
    }


def _mean(values, label):
    try:
        result = math.fsum(values) / len(values)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(label + " cannot be represented as a finite mean") from exc
    return _finite_number(result, label)


def _summary_bucket():
    return {"improved": 0, "degraded": 0, "unchanged": 0}


def _build_comparisons(prefixes):
    comparisons = []
    for comparison, from_arm, to_arm in COMPARISONS:
        rows = []
        for prefix in prefixes:
            training_changes = {
                field: _training_delta(
                    prefix["training"][from_arm][field],
                    prefix["training"][to_arm][field],
                    comparison + "." + field,
                )
                for field in TRAINING_FIELDS
            }
            old_changes = {
                view: {
                    split: _paired_delta(
                        prefix["metrics"][from_arm]["old_kl"][view][split],
                        prefix["metrics"][to_arm]["old_kl"][view][split],
                        "old_kl",
                        comparison + ".old_kl." + view + "." + split,
                    )
                    for split in SPLITS
                }
                for view in OLD_KL_VIEWS
            }
            gain_changes = {
                view: {
                    split: _paired_delta(
                        prefix["metrics"][from_arm]["G_new"][view][split],
                        prefix["metrics"][to_arm]["G_new"][view][split],
                        "G_new",
                        comparison + ".G_new." + view + "." + split,
                    )
                    for split in SPLITS
                }
                for view in G_NEW_VIEWS
            }
            sources = {
                arm: {
                    "reference": prefix["references"][arm],
                    "candidate_sha256": prefix["candidate_sha256"][arm]
                    if arm in TRAINING_ARMS else None,
                    "origin": prefix["origins"].get(arm),
                }
                for arm in (from_arm, to_arm)
            }
            old_delta = old_changes["encoded_fresh"]["holdout"]["signed_delta_to_minus_from"]
            gain_delta = gain_changes["encoded_fresh"]["holdout"]["signed_delta_to_minus_from"]
            rows.append({
                "seed": prefix["seed"],
                "prepared_id": prefix["prepared_id"],
                "sources": sources,
                "training_changes": training_changes,
                "old_kl_changes": old_changes,
                "G_new_changes": gain_changes,
                "encoded_holdout_joint_direction": old_delta < 0 and gain_delta > 0,
            })

        training_summary = {}
        for field in TRAINING_FIELDS:
            deltas = [row["training_changes"][field]["signed_delta_to_minus_from"] for row in rows]
            absolute_deltas = [row["training_changes"][field]["absolute_delta"] for row in rows]
            training_summary[field] = {
                "logical_prefix_count": len(rows),
                "mean_signed_delta": _mean(deltas, comparison + "." + field + ".mean_delta"),
                "mean_absolute_delta": _mean(absolute_deltas, comparison + "." + field + ".mean_absolute_delta"),
            }

        def score_summary(field, views, direction_metric):
            result = {}
            for view in views:
                result[view] = {}
                for split in SPLITS:
                    records = [row[field][view][split] for row in rows]
                    counts = _summary_bucket()
                    for record in records:
                        counts[record["classification"]] += 1
                    deltas = [record["signed_delta_to_minus_from"] for record in records]
                    absolute_deltas = [record["absolute_delta"] for record in records]
                    result[view][split] = {
                        "direction": "lower_is_better" if direction_metric == "old_kl" else "higher_is_improvement",
                        "logical_prefix_count": len(records),
                        "mean_signed_delta": _mean(deltas, comparison + "." + field + "." + view + "." + split),
                        "mean_absolute_delta": _mean(absolute_deltas, comparison + "." + field + "." + view + "." + split + ".abs"),
                        "counts": counts,
                    }
            return result

        comparisons.append({
            "comparison": comparison,
            "from_arm": from_arm,
            "to_arm": to_arm,
            "pairing_unit": "logical_prefix",
            "logical_prefix_count": len(rows),
            "per_prefix": rows,
            "summary": {
                "training": training_summary,
                "old_kl": score_summary("old_kl_changes", OLD_KL_VIEWS, "old_kl"),
                "G_new": score_summary("G_new_changes", G_NEW_VIEWS, "G_new"),
            },
            "encoded_holdout_joint_direction_diagnostic": {
                "definition": "Descriptive only: old-KL delta < 0 and G_new delta > 0; no threshold.",
                "descriptive_count": sum(1 for row in rows if row["encoded_holdout_joint_direction"]),
                "logical_prefix_count": len(rows),
            },
        })
    return comparisons


def _build_witness_report(prefixes):
    rows = []
    for prefix in prefixes:
        d = prefix["training"]["D"]
        s1 = prefix["training"]["S1"]
        full_l_delta = _finite_number(s1["fullL"] - d["fullL"], "witness.fullL_delta")
        old_kl_delta = _finite_number(s1["old_train_kl"] - d["old_train_kl"], "witness.old_train_kl_delta")
        parameters_differ = prefix["candidate_sha256"]["S1"] != prefix["candidate_sha256"]["D"]
        bound_satisfied = s1["new_train_kl"] <= prefix["b"] + FLOAT32_BOUND_TOLERANCE
        full_l_condition = full_l_delta < -WITNESS_TOLERANCE
        old_kl_condition = old_kl_delta < -WITNESS_TOLERANCE
        rows.append({
            "seed": prefix["seed"], "prepared_id": prefix["prepared_id"], "b": prefix["b"],
            "S1_new_train_kl": s1["new_train_kl"], "S1_minus_D_fullL": full_l_delta,
            "S1_minus_D_old_train_kl": old_kl_delta,
            "S1_candidate_sha256": prefix["candidate_sha256"]["S1"],
            "D_candidate_sha256": prefix["candidate_sha256"]["D"],
            "parameters_differ_from_D": parameters_differ,
            "float32_new_train_bound_satisfied": bound_satisfied,
            "fullL_improved_vs_D_by_more_than_1e_8": full_l_condition,
            "old_train_kl_improved_vs_D_by_more_than_1e_8": old_kl_condition,
            "witnessed": parameters_differ and bound_satisfied and full_l_condition and old_kl_condition,
        })
    return {
        "definition": (
            "Final-float32 S1 relative to D: candidate SHA differs, S1.new_train_kl <= b + 1e-12, "
            "S1.fullL < D.fullL - 1e-8, and S1.old_train_kl < D.old_train_kl - 1e-8."
        ),
        "logical_prefix_count": len(rows),
        "witness_count": sum(1 for row in rows if row["witnessed"]),
        "per_prefix": rows,
    }


def _build_deployment_report(prefixes):
    rows = []
    arm_summaries = {arm: {"signed_excesses": [], "violations": [], "feasible_count": 0}
                     for arm in ("S0", "S1")}
    for prefix in prefixes:
        row = {"seed": prefix["seed"], "prepared_id": prefix["prepared_id"], "arms": {}}
        for arm in ("S0", "S1"):
            metric_value = prefix["metrics"][arm]["views"]["encoded_fresh"]["new_train"]
            signed_excess = _finite_number(metric_value - prefix["b"], "deployment.signed_excess")
            violation = _finite_number(max(0.0, signed_excess), "deployment.violation")
            feasible = metric_value <= prefix["b"] + FLOAT32_BOUND_TOLERANCE
            arm_summary = arm_summaries[arm]
            arm_summary["signed_excesses"].append(signed_excess)
            arm_summary["violations"].append(violation)
            arm_summary["feasible_count"] += int(feasible)
            row["arms"][arm] = {"encoded_fresh_new_train_kl": metric_value, "b": prefix["b"],
                                "signed_excess_K_minus_b": signed_excess, "violation": violation,
                                "feasible_with_tolerance": feasible}
        rows.append(row)
    summary_by_arm = {}
    for arm, values in arm_summaries.items():
        summary_by_arm[arm] = {
            "logical_prefix_count": len(prefixes),
            "feasible_count": values["feasible_count"],
            "mean_signed_excess": _mean(values["signed_excesses"], arm + ".mean_signed_excess"),
            "mean_violation": _mean(values["violations"], arm + ".mean_violation"),
        }
    return {
        "constraint": "encoded_fresh.new_train_kl <= b + 1e-12",
        "per_prefix": rows,
        "summary_by_arm": summary_by_arm,
    }


def _ratio(numerator, denominator, label):
    if denominator <= RATIO_DENOMINATOR_FLOOR:
        return None
    return _finite_number(numerator / denominator, label)


def _build_c_reference_report(prefixes):
    rows = []
    old_ratios = []
    gain_ratios = []
    for prefix in prefixes:
        c_values = prefix["metrics"]["C"]
        s1_values = prefix["metrics"]["S1"]
        c_old = c_values["legacy"]["K_old"]
        c_gain = c_values["legacy"]["G_new"]
        old_ratio = _ratio(
            s1_values["legacy"]["K_old"], c_old,
            "seed " + str(prefix["seed"]) + ".S1_over_C.K_old",
        )
        gain_ratio = _ratio(
            s1_values["legacy"]["G_new"], c_gain,
            "seed " + str(prefix["seed"]) + ".S1_over_C.G_new",
        )
        if old_ratio is not None:
            old_ratios.append(old_ratio)
        if gain_ratio is not None:
            gain_ratios.append(gain_ratio)
        rows.append({
            "seed": prefix["seed"],
            "prepared_id": prefix["prepared_id"],
            "reference": prefix["references"]["C"],
            "canonical_score_metrics": c_values["canonical_metrics"],
            "original_legacy_values": dict(c_values["legacy"]),
            "reported_values": {
                "old_kl": c_values["old_kl"],
                "G_new": c_values["G_new"],
                "W0_fresh": c_values["views"]["W0_fresh"],
            },
            "S1_over_C_descriptive_ratios": {
                "K_old": {
                    "defined": old_ratio is not None,
                    "value": old_ratio,
                    "definition": "S1.K_old / C.K_old when C.K_old > 1e-10",
                },
                "G_new": {
                    "defined": gain_ratio is not None,
                    "value": gain_ratio,
                    "definition": "S1.G_new / C.G_new when C.G_new > 1e-10",
                },
            },
        })
    return {
        "interpretation": (
            "C is the original lambda=0 performance reference. Retain every prefix and all "
            "canonical C score metrics regardless of whether either ratio is defined. Ratios "
            "are descriptive KL diagnostics; G_new is not reward."
        ),
        "logical_prefix_count": len(rows),
        "per_prefix": rows,
        "ratio_summary": {
            "S1_K_old_over_C_K_old": {
                "defined_count": len(old_ratios),
                "logical_prefix_count": len(rows),
                "mean_of_defined_ratios": _mean(old_ratios, "C ratio mean K_old")
                if old_ratios else None,
                "denominator_condition": "C.K_old > 1e-10",
            },
            "S1_G_new_over_C_G_new": {
                "defined_count": len(gain_ratios),
                "logical_prefix_count": len(rows),
                "mean_of_defined_ratios": _mean(gain_ratios, "C ratio mean G_new")
                if gain_ratios else None,
                "denominator_condition": "C.G_new > 1e-10",
            },
        },
    }


def build_report(payload):
    """Build the frozen E165 exploratory report without selecting or promoting.

    Comparison deltas are always to-arm minus from-arm. Old KL is improved by a
    negative delta; G_new is improved by a positive delta. Direction counts use
    exact zero only. The report performs consistency checks on normalized score
    fields but does not call a scientific helper, scorer, or audit routine.
    """
    source_identity, cohort_lock, prefixes = _validate_payload(payload)
    comparisons = _build_comparisons(prefixes)
    witness = _build_witness_report(prefixes)
    deployment = _build_deployment_report(prefixes)
    c_reference = _build_c_reference_report(prefixes)
    return {
        "format": OUTPUT_FORMAT,
        "status": "EXPLORATORY_ONLY",
        "exploratory_only": True,
        "no_advancement_decision": True,
        "source_identity_sha256": source_identity,
        "cohort_lock_sha256": cohort_lock,
        "logical_prefix_count": len(prefixes),
        "pairing_unit": "logical_prefix",
        "comparison_orientation": "to_arm minus from_arm",
        "classification_rule": "exact signed delta zero only; no tolerance",
        "score_interpretation": "G_new is a KL-derived diagnostic, not reward",
        "comparisons": comparisons,
        "final32_witness": witness,
        "encoded_train_bound_diagnostic": deployment,
        "C_reference_summary": c_reference,
    }
