"""Pure, exploratory E164 paired report construction.

This module consumes only the frozen logical-score proposal described by
report_contract.json. It does not load scientific artifacts, call a scorer,
select an arm, or make an advancement decision.
"""

import math


INPUT_FORMAT = "e164_logical_scores_v1"
OUTPUT_FORMAT = "e164_exploratory_report_v1"
SCORE_FORMAT = "h036_score_metrics_v1"
EXPECTED_SEEDS = (
    26201, 26202, 26203, 26204, 26205, 26206, 26207, 26208,
    26209, 26210, 26211, 26212, 26213, 26215, 26216, 26218,
)
ARMS = ("V", "UG", "UR")
ORIGINS = {"REUSED_E163", "NEW_SCORE", "ALIAS"}
SPLITS = ("train", "holdout")
OLD_KL_VIEWS = ("pre_codec_fresh", "encoded_fresh")
G_NEW_VIEWS = ("pre_codec_fresh", "encoded_fresh")
ALL_FRESH_VIEWS = OLD_KL_VIEWS + ("W0_fresh",)
COMPARISONS = (("UG-V", "V", "UG"), ("UR-UG", "UG", "UR"))


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(label + " must be an object")
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


def _score_values(metrics, seed, expected_lambda, label):
    metrics = _mapping(metrics, label + ".metrics")
    if metrics.get("format") != SCORE_FORMAT:
        raise ValueError(label + " has an unsupported score metrics format")
    if metrics.get("seed") != seed:
        raise ValueError(label + " seed does not match its logical prefix")
    lambda_value = metrics.get("lambda_value")
    if (isinstance(lambda_value, bool) or not isinstance(lambda_value, (int, float))
            or lambda_value != expected_lambda):
        raise ValueError(label + " lambda does not match the E164 arm contract")
    if metrics.get("binding") != "wildcard":
        raise ValueError(label + " binding must be wildcard")
    prepared_id = metrics.get("prepared_id")
    if not isinstance(prepared_id, str) or not prepared_id.strip():
        raise ValueError(label + " prepared_id must be a non-empty string")

    views = {}
    for view in ALL_FRESH_VIEWS:
        source = _mapping(metrics.get(view), label + "." + view)
        view_values = {}
        for split in SPLITS:
            for kind in ("old", "new"):
                key = kind + "_" + split
                view_values[key] = _finite_number(
                    source.get(key), label + "." + view + "." + key
                )
        views[view] = view_values

    aliases = {
        "K_old": views["encoded_fresh"]["old_holdout"],
        "K_new": views["encoded_fresh"]["new_holdout"],
        "K_new_W0": views["W0_fresh"]["new_holdout"],
    }
    for key, expected in aliases.items():
        actual = _finite_number(metrics.get(key), label + "." + key)
        if actual != expected:
            raise ValueError(label + "." + key + " disagrees with its legacy nested metric")

    derived_holdout_gain = aliases["K_new_W0"] - aliases["K_new"]
    legacy_gain = _finite_number(metrics.get("G_new"), label + ".G_new")
    if legacy_gain != derived_holdout_gain:
        raise ValueError(label + ".G_new disagrees with K_new_W0 - K_new")

    gains = {
        view: {
            split: views["W0_fresh"]["new_" + split] - views[view]["new_" + split]
            for split in SPLITS
        }
        for view in G_NEW_VIEWS
    }
    for view, split_values in gains.items():
        for split, value in split_values.items():
            _finite_number(value, label + ".G_new." + view + "." + split)
    if gains["encoded_fresh"]["holdout"] != legacy_gain:
        raise ValueError(label + ".G_new disagrees with encoded holdout gain")
    return {
        "prepared_id": prepared_id,
        "old_kl": {
            view: {split: views[view]["old_" + split] for split in SPLITS}
            for view in OLD_KL_VIEWS
        },
        "G_new": gains,
        "W0_fresh": views["W0_fresh"],
    }


def _direction(delta, metric):
    if delta == 0:
        return "unchanged"
    if metric == "old_kl":
        return "improved" if delta < 0 else "degraded"
    return "improved" if delta > 0 else "degraded"


def _pair(left, right, metric):
    delta = right - left
    _finite_number(delta, metric + " delta")
    return {
        "from_value": left,
        "to_value": right,
        "delta_to_minus_from": delta,
        "absolute_delta": abs(delta),
        "classification": _direction(delta, metric),
    }


def _summary_bucket():
    return {"improved": 0, "degraded": 0, "unchanged": 0}


def _validate_payload(payload):
    payload = _mapping(payload, "payload")
    if payload.get("format") != INPUT_FORMAT:
        raise ValueError("payload format must be " + INPUT_FORMAT)
    if payload.get("source_freeze_state") != "FROZEN":
        raise ValueError("source normalization must be frozen before report construction")
    freeze_id = payload.get("source_freeze_id")
    if not isinstance(freeze_id, str) or not freeze_id.strip():
        raise ValueError("source_freeze_id must be a non-empty frozen identity")
    prefixes = payload.get("prefixes")
    if not isinstance(prefixes, list) or len(prefixes) != len(EXPECTED_SEEDS):
        raise ValueError("payload must contain exactly 16 logical prefixes")

    by_seed = {}
    for index, prefix in enumerate(prefixes):
        prefix = _mapping(prefix, "prefixes[" + str(index) + "]")
        seed = prefix.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("logical prefix seed must be an integer")
        if seed in by_seed:
            raise ValueError("duplicate logical prefix seed " + str(seed))
        by_seed[seed] = prefix
    actual_seeds = tuple(prefix.get("seed") for prefix in prefixes)
    if actual_seeds != EXPECTED_SEEDS:
        raise ValueError("logical prefix order/seed set differs from the frozen 16-prefix set")

    parsed = []
    for prefix in prefixes:
        seed = prefix["seed"]
        c_metrics = _mapping(prefix.get("C"), "seed " + str(seed) + ".C")
        c_values = _score_values(c_metrics, seed, 0, "seed " + str(seed) + ".C")
        c_reference = _mapping(prefix.get("C_reference"), "seed " + str(seed) + ".C_reference")
        if not c_reference:
            raise ValueError("C_reference must preserve the original source identity")
        arms = {}
        for arm in ARMS:
            wrapper = _mapping(prefix.get(arm), "seed " + str(seed) + "." + arm)
            origin = wrapper.get("origin")
            if not isinstance(origin, str) or origin not in ORIGINS:
                raise ValueError("seed " + str(seed) + "." + arm + " has an unknown origin")
            reference = _mapping(wrapper.get("reference"), "seed " + str(seed) + "." + arm + ".reference")
            if not reference:
                raise ValueError("arm reference must preserve the original source identity")
            arms[arm] = {
                "values": _score_values(
                    wrapper.get("metrics"), seed, 1, "seed " + str(seed) + "." + arm
                ),
                "origin": origin,
                "reference": reference,
            }
        all_values = {"C": c_values}
        all_values.update({arm: item["values"] for arm, item in arms.items()})
        prepared_ids = {item["prepared_id"] for item in all_values.values()}
        if len(prepared_ids) != 1:
            raise ValueError("seed " + str(seed) + " C/V/UG/UR prepared_id values differ")
        baseline_keys = ("old_train", "new_train", "old_holdout", "new_holdout")
        reference_w0 = c_values["W0_fresh"]
        for arm, item in all_values.items():
            for key in baseline_keys:
                if item["W0_fresh"][key] != reference_w0[key]:
                    raise ValueError("seed " + str(seed) + " W0_fresh baseline differs for " + arm + "." + key)
        parsed.append({
            "seed": seed,
            "prepared_id": next(iter(prepared_ids)),
            "C_reference": c_reference,
            "arms": arms,
        })
    return freeze_id, parsed


def build_report(payload):
    """Build the fixed two-comparison report from frozen logical score metrics.

    Pairwise deltas are always right arm minus left arm. Lower old KL is an
    improvement; higher G_new is an improvement. Equality is exact zero with
    no tolerance. The function has no advancement, arm-selection, or reward
    logic.
    """
    freeze_id, prefixes = _validate_payload(payload)
    comparisons = []
    old_keys = tuple((view, split) for view in OLD_KL_VIEWS for split in SPLITS)

    for comparison, left_arm, right_arm in COMPARISONS:
        counts_old = {view: {split: _summary_bucket() for split in SPLITS} for view in OLD_KL_VIEWS}
        counts_gain = {view: {split: _summary_bucket() for split in SPLITS} for view in G_NEW_VIEWS}
        rows = []
        for prefix in prefixes:
            left = prefix["arms"][left_arm]
            right = prefix["arms"][right_arm]
            old_changes = {view: {} for view in OLD_KL_VIEWS}
            for view, split in old_keys:
                record = _pair(
                    left["values"]["old_kl"][view][split],
                    right["values"]["old_kl"][view][split],
                    "old_kl",
                )
                old_changes[view][split] = record
                counts_old[view][split][record["classification"]] += 1

            gain_changes = {view: {} for view in G_NEW_VIEWS}
            for view in G_NEW_VIEWS:
                for split in SPLITS:
                    record = _pair(
                        left["values"]["G_new"][view][split],
                        right["values"]["G_new"][view][split],
                        "G_new",
                    )
                    gain_changes[view][split] = record
                    counts_gain[view][split][record["classification"]] += 1

            rows.append({
                "seed": prefix["seed"],
                "prepared_id": prefix["prepared_id"],
                "sources": {
                    left_arm: {"origin": left["origin"], "reference": left["reference"]},
                    right_arm: {"origin": right["origin"], "reference": right["reference"]},
                    "C_reference": prefix["C_reference"],
                },
                "old_kl_changes": old_changes,
                "G_new_changes": gain_changes,
            })

        comparisons.append({
            "comparison": comparison,
            "from_arm": left_arm,
            "to_arm": right_arm,
            "logical_prefix_count": len(rows),
            "pairing_unit": "logical_prefix",
            "paired_rows": rows,
            "summary": {
                "old_kl": {
                    view: {
                        split: {
                            "direction": "lower_is_better",
                            "logical_prefix_count": len(rows),
                            "counts": counts_old[view][split],
                        }
                        for split in SPLITS
                    }
                    for view in OLD_KL_VIEWS
                },
                "G_new": {
                    view: {
                        split: {
                            "direction": "higher_is_better",
                            "logical_prefix_count": len(rows),
                            "counts": counts_gain[view][split],
                        }
                        for split in SPLITS
                    }
                    for view in G_NEW_VIEWS
                },
            },
        })

    return {
        "format": OUTPUT_FORMAT,
        "status": "EXPLORATORY_ONLY",
        "exploratory_only": True,
        "no_advancement_decision": True,
        "source_contract": INPUT_FORMAT,
        "source_freeze_id": freeze_id,
        "logical_prefix_count": len(prefixes),
        "pairing_unit": "logical_prefix",
        "classification_rule": "exact delta zero only; no tolerance",
        "comparisons": comparisons,
    }
