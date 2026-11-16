"""Necessary synthetic-only checks for the H036 acceptance diagnostic."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl import retention_acceptance_diagnostic as diagnostic  # noqa: E402
from online_rl import retention_acceptance_replay as replay  # noqa: E402
from online_rl import retention_protected_candidate as protected  # noqa: E402
from online_rl import retention_protected_inputs as inputs  # noqa: E402
from test_retention_protected_score import prepared_payload  # noqa: E402
from test_retention_protected_inputs import (  # noqa: E402
    CONFIG, FOUNDATION_SHA256, synthetic_payload,
)


def _hash(array: np.ndarray, dtype: np.dtype) -> str:
    return protected._candidate_hash(array, dtype)


def _canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _prepared_with_identity(seed: int) -> dict:
    checkpoint, actor, sidecar = synthetic_payload(seed)
    rng = np.random.default_rng(312)
    core = checkpoint["core_state"]
    for name, shape in (
        ("mlp_extractor.value_net.0.weight", (64, 12)),
        ("mlp_extractor.value_net.0.bias", (64,)),
        ("mlp_extractor.value_net.2.weight", (64, 64)),
        ("mlp_extractor.value_net.2.bias", (64,)),
        ("value_net.weight", (1, 64)), ("value_net.bias", (1,)),
    ):
        core[name] = torch.from_numpy(rng.normal(0, 0.01, shape).astype(np.float32))
    core["seen_mask"] = torch.zeros(9, dtype=torch.bool)
    checkpoint["optimizer_state"] = {
        "state": {index: {"step": torch.tensor(3.0), "exp_avg": torch.tensor([index + .5])}
                  for index in range(9)},
        "param_groups": [
            {"name": "local", "params": [0, 1], "lr": .003},
            {"name": "shared", "params": [2], "lr": 0.0},
            {"name": "critic", "params": list(range(3, 9)), "lr": .003},
        ],
    }
    checkpoint_stream = io.BytesIO()
    torch.save(checkpoint, checkpoint_stream)
    evidence_bytes = inputs._serialize_sidecar(sidecar)
    source_files = {
        "checkpoint.pt": checkpoint_stream.getvalue(),
        "actor.bin": actor,
        "evidence.npz": evidence_bytes,
    }
    files = {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for name, data in source_files.items()
    }
    return inputs.prepare_from_payload(
        checkpoint, actor, sidecar,
        foundation_bytes=CONFIG["foundation"]["bytes"],
        foundation_sha256=FOUNDATION_SHA256,
        input_identity={
            "logical_case": f"s{seed}_dynamics_late", "kind": "synthetic",
            "files": files,
        },
    )


def _synthetic_trace(
    view: diagnostic.TrainingView,
    *,
    f64: np.ndarray | None = None,
    u64: np.ndarray | None = None,
    closure_new_kl: float | None = None,
    tie_count: int = 0,
    baseline_new_kl: float = 0.5,
) -> dict:
    """Build a tiny, internally consistent trace for gate and tie cases."""
    c0 = np.asarray(view.c0, dtype=np.float64)
    baseline = float(baseline_new_kl)
    if tie_count:
        f64 = u64 = c0.copy()
        records = [
            {
                "index": index, "fullL": 1.0, "newKL": baseline, "oldKL": 0.7,
                "ridge": 0.0, "accepted": True, "reject_reason": None,
                "candidate_hash": _hash(c0, np.dtype(np.float64)), "u_updated": True,
            }
            for index in range(1, tie_count + 1)
        ]
        f_origin = u_origin = tie_count
        f_l = u_l = 1.0
        initial_l = 1.0
    elif f64 is None and u64 is None:
        f64 = u64 = c0.copy()
        records = [{
            "index": 1, "fullL": 1.0, "newKL": baseline, "oldKL": 0.7,
            "ridge": 0.0, "accepted": True, "reject_reason": None,
            "candidate_hash": _hash(c0, np.dtype(np.float64)), "u_updated": True,
        }]
        f_origin = u_origin = 1
        f_l = u_l = initial_l = 1.0
    else:
        assert f64 is not None and u64 is not None
        chosen_kl = baseline + 2.0e-12 if closure_new_kl is None else closure_new_kl
        records = [
            {
                "index": 1, "fullL": 1.0, "newKL": baseline, "oldKL": 0.7,
                "ridge": 0.0, "accepted": True, "reject_reason": None,
                "candidate_hash": _hash(c0, np.dtype(np.float64)), "u_updated": True,
            },
            {
                "index": 2, "fullL": 0.9, "newKL": float(chosen_kl), "oldKL": 0.6,
                "ridge": 0.0, "accepted": False, "reject_reason": "new_train_kl_gate",
                "candidate_hash": _hash(u64, np.dtype(np.float64)), "u_updated": True,
            },
        ]
        f_origin, u_origin = 1, 2
        f_l, u_l, initial_l = 1.0, 0.9, 1.0

    candidates = {
        "C0": np.array(c0, dtype="<f8", copy=True),
        "F64": np.array(f64, dtype="<f8", copy=True),
        "U64": np.array(u64, dtype="<f8", copy=True),
    }
    for array in candidates.values():
        array.setflags(write=False)
    origins = {"C0": 0, "F64": f_origin, "U64": u_origin}
    f_hash = _hash(candidates["F64"], np.dtype(np.float64))
    u_hash = _hash(candidates["U64"], np.dtype(np.float64))
    diagnostics_data = {
        "closure_count": len(records), "iterations": 0,
        "termination_reason": "optimizer_completed", "closure_records": records,
        "optimizer_max_iterations": 64, "optimizer_max_evaluations": 256,
        "optimizer_history_size": 10, "optimizer_learning_rate": 1.0,
        "optimizer_tolerance_grad": 1.0e-9, "optimizer_tolerance_change": 1.0e-12,
        "prepared_id": view.prepared_id, "seed": view.seed,
        "initial_fullL": initial_l,
        "initial_candidate_sha256": _hash(c0, np.dtype(np.float64)),
        "initial_metrics": {"new_train_kl": baseline, "old_train_kl": 0.7, "ridge": 0.0},
        "best_float64_candidate_sha256": f_hash, "best_float64_fullL": f_l,
        "best_float64_origin_index": f_origin,
        "u_best_float64_candidate_sha256": u_hash, "u_best_float64_fullL": u_l,
        "u_best_float64_origin_index": u_origin,
        "selected_float32_candidate_sha256": _hash(candidates["F64"].astype(np.float32), np.dtype(np.float32)),
        "candidate_sha256": _hash(candidates["F64"].astype(np.float32), np.dtype(np.float32)),
        "fallback_applied": False, "fallback_reason": None,
    }
    path_sha = diagnostic._trace_path_sha256(diagnostics_data)
    trace = {
        "format": "h036_selection_trace_v1", "seed": view.seed,
        "prepared_id": view.prepared_id,
        "input_identity": {"synthetic": True}, "lambda_value": 1,
        "training_hash": view.training_hash, "origin_indices": origins,
        "candidates": candidates, "path_sha256": path_sha,
        "diagnostics": diagnostics_data,
    }
    trace["trace_sha256"] = diagnostic._trace_digest(trace)
    return trace


def _lock_with_gate_values(monkeypatch, view, trace, values):
    def fake_owner_kl(_features, _teacher, _owners, coefficient):
        coefficient = np.asarray(coefficient, dtype=np.float64)
        if np.array_equal(coefficient, view.c0):
            value = values.get("C0", 0.5)
        elif np.array_equal(coefficient, trace["candidates"]["F64"].astype(np.float32).astype(np.float64)):
            value = values.get("F64", 0.5)
        elif np.array_equal(coefficient, trace["candidates"]["U64"].astype(np.float32).astype(np.float64)):
            value = values.get("U64", 0.5)
        else:
            raise AssertionError("unexpected candidate reached the synthetic KL gate")
        return [float(value), float(value)], float(value)

    monkeypatch.setattr(diagnostic.protected, "_owner_kl_diagnostics", fake_owner_kl)
    return diagnostic.lock_acceptance_probes(view, trace)


def _reference(key: str, data: bytes, path: str | None = None) -> dict:
    ref = {"key": key, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if path is not None:
        ref["path"] = path
    return ref


def _synthetic_bindings(prepared_by_seed, trace_by_seed, locked_by_seed, fixture_score):
    config = json.loads((ROOT / "configs" / "h036_acceptance_diagnostic_v1.json").read_text())
    source_identity = config["v1_source_identity_sha256"]
    foundation = {
        "key": "synthetic-foundation", "bytes": CONFIG["foundation"]["bytes"],
        "sha256": FOUNDATION_SHA256,
    }
    actor_bytes = fixture_score["deployment"]["actor_bytes"]
    actor_ref = _reference("actor.bin", actor_bytes)
    base_metrics = fixture_score["metrics"]
    expected_metrics_bytes = {}
    records = {}

    def make_arm(seed: int, arm_name: str, candidate_bytes: bytes, helper_hash: str,
                 best64_hash: str, metrics_ref: dict, diagnostics_bytes: bytes):
        raw_ref = _reference("candidate_f32", candidate_bytes)
        artifacts = {
            "candidate_f32": raw_ref,
            "candidate_json": _reference("candidate_json", _canonical_bytes({"candidate": list(candidate_bytes)})),
            "metrics": copy.deepcopy(metrics_ref),
            "diagnostics": _reference("diagnostics", diagnostics_bytes),
            "actor": copy.deepcopy(actor_ref),
            "checkpoint": _reference("checkpoint", b"synthetic checkpoint artifact"),
            "persistent_references": _reference("persistent_references", b"synthetic persistent reference"),
        }
        return {
            "lambda_value": 0 if arm_name == "C" else 1,
            "prepared_id": prepared_by_seed[seed]["prepared_id"], "binding": "wildcard",
            "helper_candidate_sha256": helper_hash,
            "raw_candidate_sha256": raw_ref["sha256"],
            "selected32_helper_sha256": helper_hash,
            "best64_helper_sha256": best64_hash,
            "fallback_applied": False, "artifacts": artifacts,
        }

    for seed in replay.COHORT_SEEDS:
        prepared = prepared_by_seed[seed]
        locked = locked_by_seed[seed]
        candidate = locked["probes"]["V"]["candidate"]
        candidate_bytes = np.ascontiguousarray(candidate, dtype="<f4").tobytes()
        helper_hash = locked["probes"]["V"]["candidate_sha256"]
        metrics = copy.deepcopy(base_metrics)
        metrics["seed"] = seed
        metrics["prepared_id"] = prepared["prepared_id"]
        metrics["candidate_sha256"] = helper_hash
        metrics_bytes = _canonical_bytes(metrics)
        if seed in replay.RECOVERY_SEEDS:
            expected_metrics_bytes[str(seed)] = metrics_bytes
        metrics_ref = _reference("metrics", metrics_bytes)
        diagnostics_bytes = _canonical_bytes(trace_by_seed[seed]["diagnostics"])
        d_arm = make_arm(
            seed, "D", candidate_bytes, helper_hash,
            trace_by_seed[seed]["diagnostics"]["best_float64_candidate_sha256"],
            metrics_ref, diagnostics_bytes,
        )
        c_arm = make_arm(seed, "C", candidate_bytes, helper_hash, d_arm["best64_helper_sha256"],
                         metrics_ref, diagnostics_bytes)
        identity_files = prepared["input_identity"]["files"]
        inputs_ref = {
            name: {**details, "key": name, "path": f"synthetic/{seed}/{name}"}
            for name, details in identity_files.items()
        }
        prep_ref = _reference(
            f"prepare-{seed}", f"synthetic preparation {seed}".encode(),
            f"synthetic/{seed}/prepare.json",
        )
        records[str(seed)] = {
            "seed": seed, "logical_case": f"s{seed}_dynamics_late",
            "prepared_id": prepared["prepared_id"], "binding": "wildcard",
            "foundation": copy.deepcopy(foundation),
            "preparation_reference": prep_ref, "inputs": inputs_ref,
            "arms": {"C": c_arm, "D": d_arm},
        }
    bindings = {
        "format": "h036_acceptance_bindings_v1",
        "v1_source_identity_sha256": source_identity,
        "run002_provenance": _reference(
            "run002 provenance", b"synthetic run002 provenance", "synthetic/run002/provenance.json"),
        "run002_mirror_manifest": _reference(
            "run002 mirror", b"synthetic run002 mirror", "synthetic/run002/mirror.json"),
        "v1_manifest": _reference("v1 manifest", b"synthetic v1 manifest", "synthetic/v1/manifest.json"),
        "foundation": copy.deepcopy(foundation),
        "original_scientific_files": [
            _reference("source manifest", b"synthetic source manifest", "synthetic/v1/source.json"),
            _reference("source config", b"synthetic source config", "synthetic/v1/config.json"),
        ],
        "records": records,
    }
    return bindings, expected_metrics_bytes


def test_training_only_fit_matches_original_lambda_one_path_and_preserves_rng():
    prepared = prepared_payload()
    view = diagnostic.make_training_view(prepared)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    original = protected.fit_protected_candidate(prepared, 1)
    trace = diagnostic.fit_selection_trace(view)

    old_diag = original["diagnostics"]
    new_diag = trace["diagnostics"]
    assert new_diag["closure_count"] == old_diag["closure_count"]
    assert new_diag["iterations"] == old_diag["iterations"]
    assert new_diag["termination_reason"] == old_diag["termination_reason"]
    assert np.array_equal(trace["candidates"]["F64"], np.asarray(old_diag["best_float64_candidate"]))
    assert _hash(trace["candidates"]["F64"], np.dtype(np.float64)) == old_diag["best_float64_candidate_sha256"]
    assert new_diag["candidate_sha256"] == _hash(original["candidate"], np.dtype(np.float32))
    assert new_diag["candidate_sha256"] == old_diag["candidate_sha256"]
    assert new_diag["selected_float32_candidate_sha256"] == old_diag["selected_float32_candidate_sha256"]
    assert new_diag["fallback_applied"] == old_diag["fallback_applied"]
    assert new_diag["fallback_reason"] == old_diag["fallback_reason"]
    assert new_diag["returned_is_C0"] == old_diag["returned_is_C0"]
    assert new_diag["initial_fullL"] == old_diag["initial_fullL"]
    assert trace["path_sha256"] == replay.replay_diagnostics(old_diag)["path_sha256"]
    assert new_diag["initial_candidate_sha256"] == _hash(view.c0, np.dtype(np.float64))
    assert new_diag["closure_records"][0]["candidate_hash"] == new_diag["initial_candidate_sha256"]
    assert new_diag["training_view_array_bytes"] > new_diag["teacher_array_bytes"]
    assert new_diag["candidate_array_bytes"] == 3 * 9 * np.dtype(np.float64).itemsize
    assert new_diag["initial_metrics"]["new_train_kl"] == old_diag["initial_metrics"]["new_train_kl"]
    assert new_diag["initial_metrics"]["old_train_kl"] == old_diag["initial_metrics"]["old_train_kl"]
    for old_row, new_row in zip(old_diag["closure_records"], new_diag["closure_records"]):
        for key in ("index", "fullL", "newKL", "oldKL", "ridge", "accepted", "reject_reason", "candidate_hash"):
            assert new_row[key] == old_row[key]
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert "new_holdout_kl" not in new_diag["initial_metrics"]
    assert "new_holdout_kl" not in new_diag["final_metrics"]


def test_training_view_ignores_holdout_and_future_fields_but_retains_provenance(monkeypatch):
    prepared_a = prepared_payload()
    prepared_b = copy.deepcopy(prepared_a)
    new_teacher = prepared_b["new_teacher"].copy()
    new_teacher[96:128] = np.roll(new_teacher[96:128], 1, axis=1)
    prepared_b["new_teacher"] = new_teacher
    old_teacher = prepared_b["old_teacher"].copy()
    old_teacher[96:128] = np.roll(old_teacher[96:128], 2, axis=1)
    prepared_b["old_teacher"] = old_teacher
    prepared_b["future_result"] = {"score": -999}
    prepared_b["input_identity"]["synthetic_note"] = "changed provenance"

    view_a = diagnostic.make_training_view(prepared_a)
    view_b = diagnostic.make_training_view(prepared_b)
    assert view_a.training_hash == view_b.training_hash
    assert view_a.input_identity_json != view_b.input_identity_json
    assert not hasattr(view_a, "new_holdout_teacher")
    assert not hasattr(view_a, "future_result")
    with pytest.raises(TypeError):
        diagnostic.fit_selection_trace({"training_view": view_a, "holdout": prepared_a["new_teacher"]})

    monkeypatch.setattr(torch.optim.LBFGS, "step", lambda _self, closure: closure())
    trace_a = diagnostic.fit_selection_trace(view_a)
    trace_b = diagnostic.fit_selection_trace(view_b)
    assert trace_a["path_sha256"] == trace_b["path_sha256"]
    assert trace_a["trace_sha256"] == trace_b["trace_sha256"]
    for label in ("C0", "F64", "U64"):
        assert np.array_equal(trace_a["candidates"][label], trace_b["candidates"][label])

    invalid_indices = view_a.new_source_indices.copy()
    invalid_indices[0] = 96
    invalid_hash = diagnostic._training_hash(
        view_a.seed, view_a.new_features, view_a.new_teacher, view_a.new_owners,
        view_a.new_weights, view_a.old_features, view_a.old_teacher, view_a.old_weights,
        invalid_indices, view_a.old_source_indices, view_a.mean64, view_a.c0,
    )
    invalid_view = dataclasses.replace(
        view_a, new_source_indices=invalid_indices, training_hash=invalid_hash,
    )
    with pytest.raises(ValueError, match="frozen train split"):
        diagnostic.fit_selection_trace(invalid_view)


@pytest.mark.parametrize("case", ["identical", "closure_only", "post_cast_gate"])
def test_gate_probes_keep_selection_and_post_cast_gate_separate(monkeypatch, case):
    view = diagnostic.make_training_view(prepared_payload())
    c0 = view.c0.copy()
    if case == "identical":
        trace = _synthetic_trace(view, tie_count=1)
        locked = _lock_with_gate_values(monkeypatch, view, trace, {"C0": 0.5})
        assert locked["probes"]["V"]["raw_sha256"] == locked["probes"]["UG"]["raw_sha256"]
        assert locked["probes"]["UG"]["raw_sha256"] == locked["probes"]["UR"]["raw_sha256"]
    elif case == "closure_only":
        u64 = c0 + np.full((3, 3), 0.02, dtype=np.float64)
        trace = _synthetic_trace(view, f64=c0.copy(), u64=u64)
        locked = _lock_with_gate_values(monkeypatch, view, trace, {"C0": 0.5, "U64": 0.5})
        probes = locked["probes"]
        assert probes["V"]["candidate_origin_index"] == 1
        assert probes["UG"]["candidate_origin_index"] == 2
        assert probes["UR"]["candidate_origin_index"] == 2
        assert probes["V"]["raw_sha256"] != probes["UG"]["raw_sha256"]
        assert probes["UG"]["raw_sha256"] == probes["UR"]["raw_sha256"]
        assert probes["UG"]["post_cast_gate_pass"] is True
    else:
        u64 = c0 + np.full((3, 3), 0.03, dtype=np.float64)
        trace = _synthetic_trace(view, f64=c0.copy(), u64=u64)
        locked = _lock_with_gate_values(monkeypatch, view, trace, {"C0": 0.5, "U64": 0.5 + 2e-12})
        probes = locked["probes"]
        assert probes["UG"]["post_cast_gate_enabled"] is True
        assert probes["UG"]["post_cast_gate_pass"] is False
        assert probes["UG"]["fallback_applied"] is True
        assert np.array_equal(probes["UG"]["candidate"], c0.astype(np.float32))
        assert probes["UR"]["post_cast_gate_enabled"] is False
        assert probes["UR"]["post_cast_gate_pass"] is False
        assert probes["UR"]["fallback_applied"] is False
        assert np.array_equal(probes["UR"]["candidate"], u64.astype(np.float32))
        assert probes["UG"]["raw_sha256"] != probes["UR"]["raw_sha256"]


def test_exact_ties_choose_last_closure_and_lock_hash_detects_byte_edits():
    view = diagnostic.make_training_view(prepared_payload())
    trace = _synthetic_trace(view, tie_count=2)
    diagnostic._validate_trace(view, trace)
    assert trace["origin_indices"]["F64"] == 2
    assert trace["origin_indices"]["U64"] == 2
    assert np.array_equal(trace["candidates"]["F64"], view.c0)
    locked = diagnostic.lock_acceptance_probes(view, trace)
    assert locked["probes"]["V"]["aliases"] == ["UG", "UR", "V"]
    altered_metadata = copy.deepcopy(locked)
    altered_metadata["probes"]["UG"]["source"]["origin_index"] += 1
    with pytest.raises(ValueError, match="SHA-256"):
        diagnostic.locked_probe_manifest(altered_metadata)
    candidate = locked["probes"]["UG"]["candidate"]
    candidate.setflags(write=True)
    candidate[0, 0] += np.float32(1.0)
    with pytest.raises(ValueError, match="SHA-256"):
        diagnostic._validate_locked(locked)


def test_nonfinite_training_input_fails_closed():
    prepared = prepared_payload()
    changed = prepared["new_teacher"].copy()
    changed[0, 0] = np.nan
    prepared["new_teacher"] = changed
    with pytest.raises((FloatingPointError, ValueError)):
        diagnostic.make_training_view(prepared)


def test_budget_exhaustion_is_a_legal_incumbent_return(monkeypatch):
    prepared = prepared_payload()
    view = diagnostic.make_training_view(prepared)
    monkeypatch.setattr(
        protected, "protected_objective_and_gradient",
        lambda candidate, *args: (1.0, np.zeros((3, 3), dtype=np.float64)),
    )
    monkeypatch.setattr(
        protected, "_owner_kl_diagnostics",
        lambda *args: ([0.25, 0.25], 0.25),
    )
    monkeypatch.setattr(protected, "_weighted_forward_kl", lambda *args: 0.5)
    def exhaust(self, closure):
        for _ in range(257):
            closure()

    monkeypatch.setattr(torch.optim.LBFGS, "step", exhaust)
    trace = diagnostic.fit_selection_trace(view)
    assert trace["diagnostics"]["closure_count"] == 256
    assert trace["diagnostics"]["termination_reason"] == "evaluation_budget_exhausted"
    assert trace["origin_indices"]["F64"] == 256
    assert trace["origin_indices"]["U64"] == 256
    assert np.array_equal(trace["candidates"]["F64"], view.c0)


def test_full_synthetic_cohort_locks_before_anchor_and_counterfactual_permit():
    prepared_by_seed = {seed: _prepared_with_identity(seed) for seed in replay.COHORT_SEEDS}
    trace_by_seed = {}
    locked_by_seed = {}
    for seed, prepared in prepared_by_seed.items():
        view = diagnostic.make_training_view(prepared)
        baseline = protected._owner_kl_diagnostics(
            view.new_features, view.new_teacher, view.new_owners, view.c0,
        )[1]
        if seed in replay.RECOVERY_SEEDS:
            u64 = view.c0 + np.full((3, 3), 0.02, dtype=np.float64)
            trace = _synthetic_trace(view, f64=view.c0.copy(), u64=u64,
                                     baseline_new_kl=baseline)
        else:
            trace = _synthetic_trace(view, tie_count=1, baseline_new_kl=baseline)
        locked = diagnostic.lock_acceptance_probes(view, trace)
        exported = diagnostic.locked_probe_manifest(locked)
        assert json.loads(json.dumps(exported, sort_keys=True)) == exported
        assert "candidate" not in exported["probes"]["V"]
        trace_by_seed[seed] = trace
        locked_by_seed[seed] = locked

    records = {seed: trace["diagnostics"] for seed, trace in trace_by_seed.items()}
    manifest = replay.build_recovery_manifest(records)

    # Materialize one prior synthetic D score artifact. The new V anchor below
    # goes through the locked adapter only after the full cohort is sealed.
    recovered_seed = replay.RECOVERY_SEEDS[0]
    prepared = prepared_by_seed[recovered_seed]
    v_candidate = np.array(locked_by_seed[recovered_seed]["probes"]["V"]["candidate"], copy=True)
    historical_fit = {
        "format": "h036_fit_v1", "prepared_id": prepared["prepared_id"],
        "lambda_value": 1, "candidate": v_candidate,
        "diagnostics": {"candidate_sha256": _hash(v_candidate, np.dtype(np.float32))},
    }
    fixture_score = inputs.encode_and_score_stage0(prepared, historical_fit, "wildcard")
    bindings, expected_metrics_bytes = _synthetic_bindings(
        prepared_by_seed, trace_by_seed, locked_by_seed, fixture_score,
    )
    recovered_locks = {
        seed: diagnostic.locked_probe_manifest(locked_by_seed[seed])
        for seed in replay.RECOVERY_SEEDS
    }
    cohort = replay.lock_cohort(manifest, recovered_locks, bindings)
    anchor_permit = replay.make_anchor_permit(cohort)

    with pytest.raises(PermissionError):
        diagnostic.encode_and_score_locked_probe(prepared, locked_by_seed[recovered_seed], "V")
    anchor = diagnostic.encode_and_score_locked_probe(
        prepared, locked_by_seed[recovered_seed], "V", stage_permit=anchor_permit,
    )
    assert anchor["format"] == "h036_stage0_score_v1"
    assert anchor["diagnostic_only"] is True
    assert anchor["selection_source"]["stage"] == "anchor"
    assert anchor["selection_source"]["label"] == "V"
    assert len(anchor["deployment"]["actor_bytes"]) == 89
    assert anchor["metrics"]["candidate_sha256"] == locked_by_seed[recovered_seed]["probes"]["V"]["candidate_sha256"]
    with pytest.raises(ValueError, match="only V"):
        diagnostic.encode_and_score_locked_probe(
            prepared, locked_by_seed[recovered_seed], "UG", stage_permit=anchor_permit,
        )

    anchor_scores = {}
    for seed in replay.RECOVERY_SEEDS:
        if seed == recovered_seed:
            score_metrics = anchor["metrics"]
            actor_bytes = anchor["deployment"]["actor_bytes"]
        else:
            score_metrics = json.loads(expected_metrics_bytes[str(seed)].decode("utf-8"))
            actor_bytes = fixture_score["deployment"]["actor_bytes"]
        anchor_scores[str(seed)] = {"metrics": score_metrics, "deployment": actor_bytes}
    counterfactual_permit = replay.confirm_anchors(
        cohort, anchor_scores, expected_metrics_bytes, bindings,
    )
    assert counterfactual_permit["stage"] == "counterfactual"
    assert set(counterfactual_permit["anchors_pass_token"]["records"]) == {
        str(seed) for seed in replay.RECOVERY_SEEDS
    }
    replay.validate_stage_permit(
        counterfactual_permit,
        diagnostic.locked_probe_manifest(locked_by_seed[recovered_seed]),
        "UG",
    )
