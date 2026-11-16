"""Synthetic complete-codec and persistent-package checks for H036."""
from __future__ import annotations

import copy
import io
import json
import random

import numpy as np
import pytest
import torch

from online_rl import retention_behavioral_candidate as behavioral
from online_rl import retention_protected_candidate as protected
from online_rl import retention_protected_inputs as inputs
from online_rl.write_decision_policy import decode_actor
from test_retention_protected_inputs import CONFIG, synthetic_payload


def prepared_payload(*, oversize=False):
    checkpoint, actor, sidecar = synthetic_payload()
    rng = np.random.default_rng(312)
    core = checkpoint["core_state"]
    for name, shape in (
        ("mlp_extractor.value_net.0.weight", (64, 12)), ("mlp_extractor.value_net.0.bias", (64,)),
        ("mlp_extractor.value_net.2.weight", (64, 64)), ("mlp_extractor.value_net.2.bias", (64,)),
        ("value_net.weight", (1, 64)), ("value_net.bias", (1,)),
    ):
        core[name] = torch.from_numpy(rng.normal(0, 0.01, shape).astype(np.float32))
    core["seen_mask"] = torch.zeros(9, dtype=torch.bool)
    checkpoint["optimizer_state"] = {
        "state": {index: {"step": torch.tensor(3.0), "exp_avg": torch.tensor([index + .5])} for index in range(9)},
        "param_groups": [
            {"name": "local", "params": [0, 1], "lr": .003},
            {"name": "shared", "params": [2], "lr": 0.0},
            {"name": "critic", "params": list(range(3, 9)), "lr": .003},
        ],
    }
    if oversize:
        checkpoint["extra_persistent_blob"] = bytes(262144)
    return inputs.prepare_from_payload(
        checkpoint, actor, sidecar, foundation_bytes=CONFIG["foundation"]["bytes"],
        foundation_sha256=CONFIG["foundation"]["sha256"],
    )


def fixed_fit(prepared, *, lam=0):
    candidate = np.asarray([[.39117, -.12817, .05793], [.10273, .29161, -.04913],
                            [-.07317, .04239, .21871]], dtype=np.float32)
    return {"format": "h036_fit_v1", "prepared_id": prepared["prepared_id"], "lambda_value": lam,
            "candidate": candidate, "diagnostics": {
                "candidate_sha256": protected._candidate_hash(candidate, np.dtype(np.float32))}}


def independently_mean_kl(teacher, features, coefficient, indices, owners=None):
    probability = behavioral.candidate_probabilities(features, coefficient)
    rows = np.maximum(np.sum(teacher * np.log(teacher / probability), axis=1), 0)
    if owners is None:
        return float(rows[indices].mean())
    return float(sum(rows[indices][owners[indices] == owner].mean() for owner in (0, 1)) / 2)


def test_full_codec_fresh_slot_metrics_match_independent_decoded_parent_calculation():
    prepared = prepared_payload()
    fit = fixed_fit(prepared)
    before = inputs._value_digest(prepared)
    fit_before = inputs._value_digest(fit)
    score = inputs.encode_and_score_stage0(prepared, fit, "wildcard")
    metric = score["metrics"]
    local, shared, groups = decode_actor(score["deployment"]["actor_bytes"])
    coefficient = shared.reshape(3, 3)
    expected_old = independently_mean_kl(prepared["old_teacher"], prepared["old_features"], coefficient,
                                       prepared["old_holdout_indices"])
    expected_new = independently_mean_kl(prepared["new_teacher"], prepared["new_features"], coefficient,
                                       prepared["new_holdout_indices"], prepared["new_owners"])
    assert metric["K_old"] == pytest.approx(expected_old, abs=1e-15)
    assert metric["K_new"] == pytest.approx(expected_new, abs=1e-15)
    assert metric["G_new"] == metric["K_new_W0"] - metric["K_new"]
    assert metric["fresh_physical_groups"] == [0, 0]
    assert metric["fresh_local_is_zero"] is True
    assert not np.array_equal(coefficient, fit["candidate"])
    assert abs(metric["K_old"] - metric["pre_codec_fresh"]["old_holdout"]) > 1e-12
    assert np.max(np.abs(coefficient - fit["candidate"])) < .003
    assert np.max(np.abs(coefficient - (fit["candidate"] + prepared["old_parent"]) / 2)) > .01
    assert np.any(local != 0)  # only the evaluation copy, not deployed residents, was cleared
    assert groups.tolist() == [0, 0]
    assert inputs._value_digest(prepared) == before
    assert inputs._value_digest(fit) == fit_before
    json.dumps(metric, allow_nan=False)


def test_b_c_share_one_fit_and_actor_bytes_but_old_inheritance_follows_binding():
    prepared = prepared_payload()
    fit = fixed_fit(prepared)
    new = inputs.encode_and_score_stage0(prepared, fit, "new")
    wildcard = inputs.encode_and_score_stage0(prepared, fit, "wildcard")
    assert new["deployment"]["actor_bytes"] == wildcard["deployment"]["actor_bytes"]
    assert new["metrics"]["K_new"] == wildcard["metrics"]["K_new"]
    assert new["metrics"]["fresh_physical_groups"] == [-1, 0]
    assert new["metrics"]["K_old"] == new["metrics"]["W0_fresh"]["old_holdout"]
    assert new["metrics"]["K_old"] != wildcard["metrics"]["K_old"]


def test_package_counts_actual_bytes_clears_only_actor_and_preserves_all_other_learner_state():
    prepared = prepared_payload()
    score = inputs.encode_and_score_stage0(prepared, fixed_fit(prepared), "wildcard")
    deployment, metrics = score["deployment"], score["metrics"]
    checkpoint = torch.load(io.BytesIO(deployment["checkpoint_bytes"]), weights_only=False, map_location="cpu")
    original = prepared["prefix_checkpoint"]
    assert set(checkpoint) == set(original)
    assert checkpoint["parent_group"] == -2
    assert set(checkpoint["optimizer_state"]["state"]) == set(range(3, 9))
    for index in range(3, 9):
        assert inputs._value_digest(checkpoint["optimizer_state"]["state"][index]) == inputs._value_digest(original["optimizer_state"]["state"][index])
    for key in set(original) - {"core_state", "optimizer_state", "parent_group"}:
        assert inputs._value_digest(checkpoint[key]) == inputs._value_digest(original[key])
    for key in set(original["core_state"]) - {"_nodes.1.A", "_nodes.2.A", "_shared_coefficients.0.C", "slot_groups"}:
        torch.testing.assert_close(checkpoint["core_state"][key], original["core_state"][key], rtol=0, atol=0)
    capacity = metrics["capacity"]
    assert capacity["full_persistent_bytes"] == (
        55477 + len(deployment["checkpoint_bytes"]) + 89 + len(prepared["prefix_evidence_bytes"]))
    assert capacity["checkpoint_bytes"] == len(deployment["checkpoint_bytes"])
    assert capacity["full_persistent_bytes"] <= 262144
    assert deployment["evidence_bytes"] == prepared["prefix_evidence_bytes"]
    assert metrics["optimizer"]["critic_state_preserved"] is True


def test_resident_drift_is_computed_separately_from_fresh_prior_loss():
    prepared = prepared_payload()
    score = inputs.encode_and_score_stage0(prepared, fixed_fit(prepared), "wildcard")
    local, shared, groups = decode_actor(score["deployment"]["actor_bytes"])
    from online_rl.storage_decision import effective_matrices
    effective = effective_matrices(local, shared, groups)[prepared["owner_slots"]]
    probabilities = behavioral._teacher_probabilities(
        prepared["new_features"], effective, prepared["new_owners"],
        behavioral._bands_from_observations(prepared["new_observations"]),
    )
    expected_max = np.max(np.abs(probabilities - prepared["new_teacher"]))
    assert score["metrics"]["resident_drift"]["probability_max_abs"] == expected_max
    assert score["metrics"]["resident_drift"]["KL_by_owner_all128"][0] < score["metrics"]["K_new"]


def test_invalid_fit_binding_and_full_package_overflow_are_errors_not_fallbacks():
    prepared = prepared_payload()
    bad_fit = fixed_fit(prepared)
    bad_fit["prepared_id"] = "other-input"
    with pytest.raises(ValueError, match="does not belong"):
        inputs.encode_and_score_stage0(prepared, bad_fit, "wildcard")
    bad_hash = fixed_fit(prepared)
    bad_hash["candidate"][0, 0] += .01
    with pytest.raises(ValueError, match="hash"):
        inputs.encode_and_score_stage0(prepared, bad_hash, "wildcard")
    with pytest.raises(ValueError, match="binding"):
        inputs.encode_and_score_stage0(prepared, fixed_fit(prepared), "mean")
    oversized = prepared_payload(oversize=True)
    with pytest.raises(ValueError, match="exceeds 262144"):
        inputs.encode_and_score_stage0(oversized, fixed_fit(oversized), "wildcard")


def test_end_to_end_synthetic_fits_and_scores_preserve_rng_and_do_not_train_on_holdout():
    prepared = prepared_payload()
    rng_before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    for lam in (0, 1):
        result = protected.fit_protected_candidate(prepared, lam)
        changed = copy.deepcopy(prepared)
        for key, indices in (("new_teacher", changed["new_holdout_indices"]),
                             ("old_teacher", changed["old_holdout_indices"])):
            replacement = changed[key].copy()
            replacement[indices] = np.roll(replacement[indices], 1, axis=1)
            changed[key] = replacement
        second = protected.fit_protected_candidate(changed, lam)
        np.testing.assert_array_equal(result["candidate"], second["candidate"])
        score = inputs.encode_and_score_stage0(prepared, result, "wildcard")
        assert score["metrics"]["lambda_value"] == lam
        assert score["metrics"]["capacity"]["actor_bytes"] == 89
        assert result["diagnostics"]["closure_count"] <= 256
    assert random.getstate() == rng_before[0]
    now_numpy = np.random.get_state()
    assert now_numpy[0] == rng_before[1][0] and np.array_equal(now_numpy[1], rng_before[1][1])
    assert now_numpy[2:] == rng_before[1][2:]
    assert torch.equal(torch.get_rng_state(), rng_before[2])
