from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import audit_retention_value_stream as audit_stream


def _blank_config() -> dict[str, object]:
    return {
        "value_model": {"n_features": 6, "ridge": 1.0, "discount": 0.97, "lambda_cpu": 1.0},
    }


def test_random_selector_replay_uses_logged_preaction_rng_and_predictions() -> None:
    spec = {"stage": "calibration", "pilot": True, "pair_kind": "dynamics", "seed": 23089,
            "method": "random", "decisions": 4}
    rngs = audit_stream._expected_rngs(spec)
    preview = audit_stream._expected_rngs(spec)
    stats = audit_stream.BoundedRetentionStats(("A0+", "A1+", "A2+", "B0+", "B1+", "B2+"))
    models = audit_stream._models(_blank_config())
    features = np.asarray([1.0, 0.0, 0.25, 0.0, 0.0, 0.0])
    row = {
        "rng_state_before": audit_stream._rng_states(preview),
        "predictions": {
            "linear": models["linear"].predict(features),
            "intercept": models["intercept"].predict(np.asarray([1.0])),
        },
        "selected_action": audit_stream.ACTIONS[int(preview["uniform"].integers(4))],
        "action_propensity": 0.25,
        "selection_rule": "uniform_random",
        "explored": True,
    }
    audit = audit_stream.Audit(Path("unused"))

    action = audit_stream._check_selection(audit, row, "random", 0, features, stats, models, rngs, "decision")

    assert audit.errors == []
    assert action == row["selected_action"]

    row["selected_action"] = "none" if action != "none" else "mean"
    bad_audit = audit_stream.Audit(Path("unused"))
    audit_stream._check_selection(
        bad_audit, row, "random", 0, features, stats,
        audit_stream._models(_blank_config()), audit_stream._expected_rngs(spec), "decision",
    )
    assert "selected_action_replay" in {error["code"] for error in bad_audit.errors}


def test_candidate_sources_must_be_two_current_distinct_generations_with_fresh_fit() -> None:
    prior = [
        {"bout_index": 2, "task": "B0+", "group": 1, "variant": 0, "slot": 0,
         "generation": 3, "fitted_steps_after": 512},
        {"bout_index": 3, "task": "B1+", "group": 1, "variant": 1, "slot": 1,
         "generation": 4, "fitted_steps_after": 512},
    ]
    sources = [
        {"bout": 2, "task": "B0+", "group": 1, "variant": 0, "slot": 0, "generation": 3,
         "fitted_steps": 512, "fitted_steps_before_source_pair": 0,
         "fitted_steps_added_in_source_pair": 512},
        {"bout": 3, "task": "B1+", "group": 1, "variant": 1, "slot": 1, "generation": 4,
         "fitted_steps": 512, "fitted_steps_before_source_pair": 0,
         "fitted_steps_added_in_source_pair": 512},
    ]
    row = {
        "candidate_provenance": {
            "sources": sources, "candidate_group": 1, "source_fit_steps_mean": 512.0,
            "candidate_sha256": "a" * 64, "candidate_matrix_persisted": False,
        },
        "candidate_group": 1,
        "source_fit_steps": 512.0,
    }
    audit = audit_stream.Audit(Path("unused"))

    assert audit_stream._check_candidate(audit, row, prior, minimum_fit=512, where="decision") == (1, 512.0)
    assert audit.errors == []

    sources[0]["fitted_steps_added_in_source_pair"] = 256
    bad_audit = audit_stream.Audit(Path("unused"))
    audit_stream._check_candidate(bad_audit, row, prior, minimum_fit=512, where="decision")
    assert "candidate_source_fitted_budget" in {error["code"] for error in bad_audit.errors}


def test_protocol_provenance_accepts_only_a_clarification_append() -> None:
    baseline = b"Frozen protocol\n"
    heading = b"\n## Pre-main fidelity clarification (2026-10-19)\n\n"

    assert audit_stream._is_protocol_clarification_append(baseline, baseline + heading + b"Clarification.\n")
    assert not audit_stream._is_protocol_clarification_append(baseline, b"Changed protocol\n" + heading)
    assert not audit_stream._is_protocol_clarification_append(baseline, baseline + b"\nChanged criteria.\n")


def test_matched_learners_must_share_the_behavior_prefix() -> None:
    base = {
        "pair_kind": "dynamics", "seed": 23089, "completed_decisions": 4,
        "prefix_behavior_rows": 9216, "prefix_behavior_sha256": "same",
        "preactions": [
            {"selected_action": "keep", "rng_state_before": {"x": index}}
            for index in range(4)
        ],
    }
    streams = [
        {**base, "method": "learned_linear"},
        {**base, "method": "learned_intercept"},
    ]
    audit = audit_stream.Audit(Path("unused"))
    audit_stream._audit_matched_learners(audit, streams, {})
    assert audit.errors == []

    streams[1]["prefix_behavior_sha256"] = "different"
    bad_audit = audit_stream.Audit(Path("unused"))
    audit_stream._audit_matched_learners(bad_audit, streams, {})
    assert "matched_environment_prefix" in {error["code"] for error in bad_audit.errors}
