"""Remote-only E165 checks; do not execute research checks on this workstation.

The CHECKS owner runs this file once in the admitted remote environment. The
tests below use synthetic coefficients and mocked oracles except for the one
explicitly small frozen training-objective consistency check.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from online_rl import retention_acceptance_diagnostic as acceptance
from online_rl import retention_behavioral_candidate as behavioral
from online_rl import retention_budgeted_search as budgeted
from online_rl import retention_protected_candidate as protected
from online_rl import retention_protected_inputs as inputs


SEEDS = tuple(range(26201, 26214)) + (26215, 26216, 26218)
RECOVERED_SEEDS = {26204, 26206, 26209, 26212, 26213}
HELPER_KEYS = (
    "protected_objective_and_gradient",
    "behavioral_objective_and_gradient",
    "gradient_evaluations",
    "new_kl",
    "old_kl",
    "ridge",
)
_CURRENT_CALLS: dict[str, int] = {}


@pytest.fixture(autouse=True)
def _record_frozen_call_counts(request, monkeypatch):
    """Emit per-node real-helper and mock-oracle counts into remote JUnit XML."""
    counts = {
        "real_budgeted_training_oracle_calls": 0,
        "real_protected_objective_and_gradient_calls": 0,
        "real_owner_kl_diagnostics_calls": 0,
        "real_stage0_scorer_calls": 0,
        "real_protected_fit_calls": 0,
        "real_behavioral_fit_calls": 0,
        "real_fit_calls": 0,
        "mock_training_oracle_calls": 0,
        "mock_stage0_scorer_calls": 0,
    }
    _CURRENT_CALLS.clear()
    _CURRENT_CALLS.update(counts)

    original_budgeted = budgeted._compute_training_metrics
    original_objective = protected.protected_objective_and_gradient
    original_owner_kl = protected._owner_kl_diagnostics

    def counted_budgeted(*args, **kwargs):
        _CURRENT_CALLS["real_budgeted_training_oracle_calls"] += 1
        return original_budgeted(*args, **kwargs)

    def counted_objective(*args, **kwargs):
        _CURRENT_CALLS["real_protected_objective_and_gradient_calls"] += 1
        return original_objective(*args, **kwargs)

    def counted_owner_kl(*args, **kwargs):
        _CURRENT_CALLS["real_owner_kl_diagnostics_calls"] += 1
        return original_owner_kl(*args, **kwargs)

    def counted_scorer(*args, **kwargs):
        _CURRENT_CALLS["real_stage0_scorer_calls"] += 1
        raise AssertionError("E165 CHECKS forbids calling the real stage0 scorer")

    def counted_protected_fit(*args, **kwargs):
        _CURRENT_CALLS["real_protected_fit_calls"] += 1
        _CURRENT_CALLS["real_fit_calls"] += 1
        raise AssertionError("E165 CHECKS forbids protected fitting")

    def counted_behavioral_fit(*args, **kwargs):
        _CURRENT_CALLS["real_behavioral_fit_calls"] += 1
        _CURRENT_CALLS["real_fit_calls"] += 1
        raise AssertionError("E165 CHECKS forbids behavioral fitting")

    monkeypatch.setattr(budgeted, "_compute_training_metrics", counted_budgeted)
    monkeypatch.setattr(protected, "protected_objective_and_gradient", counted_objective)
    monkeypatch.setattr(protected, "_owner_kl_diagnostics", counted_owner_kl)
    monkeypatch.setattr(inputs, "encode_and_score_stage0", counted_scorer)
    monkeypatch.setattr(protected, "fit_protected_candidate", counted_protected_fit)
    monkeypatch.setattr(behavioral, "fit_behavioral_candidate", counted_behavioral_fit)
    try:
        yield
    finally:
        properties = {
            **_CURRENT_CALLS,
            "synthetic_full_training_oracle_evaluations": _CURRENT_CALLS[
                "real_budgeted_training_oracle_calls"
            ],
            "synthetic_complete_score_calls": _CURRENT_CALLS["real_stage0_scorer_calls"],
        }
        for name, value in properties.items():
            property_name = name if name.startswith("synthetic_") else f"e165.{name}"
            request.node.user_properties.append((property_name, int(value)))


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def _features(rows: int) -> behavioral.FrozenPolicyFeatures:
    return behavioral.FrozenPolicyFeatures(
        base_logits=np.zeros((rows, 4), dtype=np.float64),
        source_psi=np.zeros((rows, 3), dtype=np.float64),
        contrast_basis=np.zeros((4, 3), dtype=np.float64),
        temperature=2.0,
        epsilon=0.05,
    )


def _identity(seed: int) -> dict:
    return {
        "logical_case": f"s{seed}_dynamics_late",
        "files": {
            name: {"bytes": 1, "sha256": hashlib.sha256(name.encode()).hexdigest()}
            for name in ("checkpoint.pt", "actor.bin", "evidence.npz")
        },
    }


def _view(seed: int = 26201) -> acceptance.TrainingView:
    new_features = _features(192)
    old_features = _features(96)
    new_teacher = _readonly(np.full((192, 4), 0.25, dtype=np.float64))
    old_teacher = _readonly(np.full((96, 4), 0.25, dtype=np.float64))
    owners = _readonly(np.repeat(np.arange(2, dtype=np.int64), 96))
    new_weights = _readonly(np.full(192, 0.5 / 96, dtype=np.float64))
    old_weights = _readonly(np.full(96, 1.0 / 96, dtype=np.float64))
    new_source_indices = _readonly(np.concatenate((
        np.arange(96, dtype=np.int64), np.arange(128, 224, dtype=np.int64),
    )))
    old_source_indices = _readonly(np.arange(96, dtype=np.int64))
    mean64 = _readonly(np.zeros((3, 3), dtype=np.float64))
    c0 = _readonly(mean64.astype(np.float32).astype(np.float64))
    training_hash = acceptance._training_hash(
        seed, new_features, new_teacher, owners, new_weights, old_features,
        old_teacher, old_weights, new_source_indices, old_source_indices,
        mean64, c0,
    )
    return acceptance.TrainingView(
        seed=seed,
        prepared_id=f"prepared-{seed}",
        input_identity_json=json.dumps(_identity(seed), sort_keys=True),
        training_hash=training_hash,
        new_features=new_features,
        new_teacher=new_teacher,
        new_owners=owners,
        new_weights=new_weights,
        old_features=old_features,
        old_teacher=old_teacher,
        old_weights=old_weights,
        new_source_indices=new_source_indices,
        old_source_indices=old_source_indices,
        mean64=mean64,
        c0=c0,
    )


def _prepared(seed: int = 26201) -> dict:
    """Build a minimal v1 training input with deliberately hostile holdout data."""
    class Unreadable:
        def __array__(self, *_args, **_kwargs):
            raise AssertionError("holdout sentinel was converted to an array")

        def __getitem__(self, *_args, **_kwargs):
            raise AssertionError("holdout sentinel was indexed")

        def __iter__(self):
            raise AssertionError("holdout sentinel was iterated")

        def __deepcopy__(self, _memo):
            return self

    return {
        "format": "h036_prepared_v1",
        "seed": seed,
        "prepared_id": f"prepared-{seed}",
        "input_identity": _identity(seed),
        "new_train_indices": np.concatenate((
            np.arange(96, dtype=np.int64), np.arange(128, 224, dtype=np.int64),
        )),
        "old_train_indices": np.arange(96, dtype=np.int64),
        "new_features": _features(256),
        "old_features": _features(128),
        "new_teacher": np.full((256, 4), 0.25, dtype=np.float64),
        "old_teacher": np.full((128, 4), 0.25, dtype=np.float64),
        "new_owners": np.repeat(np.arange(2, dtype=np.int64), 128),
        "mean64": np.zeros((3, 3), dtype=np.float64),
        "new_holdout_observations": Unreadable(),
        "future_result": Unreadable(),
    }


def _coeff(value: float | np.ndarray) -> np.ndarray:
    if np.isscalar(value):
        result = np.full((3, 3), value, dtype=np.float32)
    else:
        result = np.asarray(value, dtype=np.float32).reshape(3, 3)
    result.setflags(write=False)
    return result


def _expected_training(
    c0=(20.0, 3.0, 17.0, 0.0),
    c=(12.0, 2.0, 10.0, 0.0),
    d=(10.0, 3.0, 7.0, 0.0),
) -> dict:
    def full(value):
        full_l, new_kl, old_kl, ridge = value
        return {"fullL": full_l, "new_train_kl": new_kl,
                "old_train_kl": old_kl, "ridge": ridge}

    return {
        "C0": full(c0),
        "C": {"new_train_kl": c[1], "old_train_kl": c[2], "ridge": c[3]},
        "D": full(d),
    }


def _anchor(
    view: acceptance.TrainingView,
    *,
    c: float = 1.0,
    d: float = 4.0,
    u: float = 3.0,
    expected_training: dict | None = None,
) -> budgeted.AnchorSet:
    refs = ({
        "path": f"synthetic/{view.seed}/anchor.bin",
        "bytes": 36,
        "sha256": "a" * 64,
        "role": "synthetic_anchor",
    },)
    return budgeted.AnchorSet(
        seed=view.seed,
        prepared_id=view.prepared_id,
        input_identity=json.loads(view.input_identity_json),
        C=_coeff(c),
        D=_coeff(d),
        U32=_coeff(u),
        expected_training=expected_training or _expected_training(),
        source_references=refs,
        v1_source_identity_sha256=budgeted._V1_SOURCE_IDENTITY_SHA256,
        e164_source_identity_sha256=budgeted._E164_SOURCE_IDENTITY_SHA256,
        e164_cohort_manifest_sha256=budgeted._E164_COHORT_MANIFEST_SHA256,
        manifest_sha256=budgeted._ANCHOR_MANIFEST_SHA256,
    )


def _metrics(full_l: float, new_kl: float, old_kl: float, ridge: float = 0.0) -> dict:
    return {
        "fullL": full_l,
        "new_train_kl": new_kl,
        "old_train_kl": old_kl,
        "ridge": ridge,
        "helper_counts": {key: 0 for key in HELPER_KEYS},
    }


def _candidate_key(candidate: np.ndarray) -> bytes:
    return np.ascontiguousarray(candidate, dtype="<f8").tobytes()


def _patch_oracle(monkeypatch, table: dict[bytes, dict], *, default=None):
    calls: list[bytes] = []

    def fake(_view, candidate64):
        key = _candidate_key(candidate64)
        calls.append(key)
        _CURRENT_CALLS["mock_training_oracle_calls"] += 1
        if key in table:
            return copy.deepcopy(table[key])
        if default is not None:
            return copy.deepcopy(default)
        raise AssertionError(f"unexpected training oracle candidate {np.asarray(candidate64)[0, 0]}")

    monkeypatch.setattr(budgeted, "_compute_training_metrics", fake)
    return calls


def _table_for(values: dict[float, dict]) -> dict[bytes, dict]:
    return {
        _candidate_key(np.full((3, 3), value, dtype=np.float64)): result
        for value, result in values.items()
    }


def _cohort_fixture(monkeypatch):
    """Create all 16 synthetic locks using one cheap, cached oracle case each."""
    locked_pairs = []
    by_seed = {}
    for seed in SEEDS:
        view = _view(seed)
        if seed == 26201:
            next_f32 = np.nextafter(np.float32(1.0), np.float32(2.0))
            anchor = _anchor(
                view, c=1.0, d=2.0, u=next_f32,
                expected_training=_expected_training(
                    c0=(20.0, 3.0, 17.0, 0.0),
                    c=(12.0, 4.0, 8.0, 0.0),
                    d=(10.0, 3.0, 7.0, 0.0),
                ),
            )
            midpoint = 1.0 + (float(next_f32) - 1.0) / 2.0
            table = _table_for({
                0.0: _metrics(20.0, 3.0, 17.0),
                1.0: _metrics(12.0, 4.0, 8.0),
                2.0: _metrics(10.0, 3.0, 7.0),
                float(next_f32): _metrics(8.0, 4.0, 4.0),
                midpoint: _metrics(9.0, 2.0, 7.0),
            })
        else:
            anchor = _anchor(view)
            table = _table_for({
                0.0: _metrics(20.0, 3.0, 17.0),
                1.0: _metrics(12.0, 2.0, 10.0),
                4.0: _metrics(10.0, 3.0, 7.0),
                3.0: _metrics(9.0, 3.0, 6.0),
            })
        calls = _patch_oracle(monkeypatch, table)
        trace = budgeted.search_budgeted_pair(view, anchor)
        locked = budgeted.lock_budgeted_pair(view, trace)
        assert len(calls) <= 5
        locked_pairs.append(locked)
        by_seed[seed] = (view, anchor, trace, locked)
    return locked_pairs, by_seed


def _canonical_json(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _sci_sha(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype=np.dtype(np.float32))
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode("ascii") + b"\0")
    digest.update(repr(canonical.shape).encode("ascii") + b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _synthetic_anchor_manifest(root: Path) -> dict:
    """Use only the checked-in manifest schema; replace every payload with toy bytes."""
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "research" / "appendix" / "H036" / "delivery_budgeted_v1"
        / "anchor_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payloads: dict[str, bytes] = {}

    def put(path: str, data: bytes) -> None:
        payloads[path] = data

    for case in manifest["cases"]:
        seed = case["seed"]
        identity_files = case["input_identity"]["files"]
        for basename in ("checkpoint.pt", "actor.bin", "evidence.npz"):
            path = case["prefix_files"][basename]
            data = f"synthetic prefix {seed} {basename}".encode("ascii")
            put(path, data)
            identity_files[basename] = {
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            }

        for label, value in (("C", 1.0 + seed / 10000.0), ("D", 4.0 + seed / 10000.0)):
            reference = case[label]
            candidate = np.full((3, 3), value, dtype="<f4")
            raw = candidate.tobytes(order="C")
            scientific_sha = _sci_sha(candidate)
            raw_sha = hashlib.sha256(raw).hexdigest()
            reference["candidate_sha256"] = scientific_sha
            reference["raw_sha256"] = raw_sha
            put(reference["candidate_path"], raw)
            put(reference["metadata_path"], _canonical_json({
                "format": "synthetic_candidate_metadata_v1",
                "prepared_id": case["prepared_id"],
                "lambda_value": 0 if label == "C" else 1,
                "dtype_str": "<f4",
                "dtype": "float32",
                "shape": [3, 3],
                "order": "C",
                "byteorder": "little",
                "scientific_candidate_sha256": scientific_sha,
                "raw_file_sha256": raw_sha,
            }))
            put(reference["metrics_path"], _canonical_json({
                "format": "h036_score_metrics_v1",
                "seed": seed,
                "prepared_id": case["prepared_id"],
                "candidate_sha256": scientific_sha,
            }))
            put(reference["persistent_references_path"], _canonical_json({"synthetic": True}))

        expected = case["expected_training"]
        put(case["C"]["diagnostics_path"], _canonical_json({
            "final_metrics": expected["C"],
            "final_float32_ridge": expected["C"]["ridge"],
        }))
        put(case["D"]["diagnostics_path"], _canonical_json({
            "initial_metrics": expected["C0"],
            "initial_fullL": expected["C0"]["fullL"],
            "final_metrics": expected["D"],
            "final_float32_ridge": expected["D"]["ridge"],
            "final_float32_fullL": expected["D"]["fullL"],
        }))

        if seed in RECOVERED_SEEDS:
            reference = case["U32"]["references"]
            candidate = np.full((3, 3), 7.0 + seed / 10000.0, dtype="<f4")
            raw = candidate.tobytes(order="C")
            scientific_sha = _sci_sha(candidate)
            raw_sha = hashlib.sha256(raw).hexdigest()
            reference["candidate_sha256"] = scientific_sha
            reference["raw_sha256"] = raw_sha
            put(case["U32"]["candidate_path"], raw)
            put(reference["lock_path"], _canonical_json({
                "seed": seed,
                "prepared_id": case["prepared_id"],
                "probes": {"UR": {
                    "candidate_sha256": scientific_sha,
                    "raw_sha256": raw_sha,
                }},
            }))
            put(reference["trace_artifacts_path"], _canonical_json({"synthetic": True}))
            put(reference["trace_path"], _canonical_json({"synthetic": True}))
        else:
            d_ref = case["D"]
            reference = case["U32"]["references"]
            reference["candidate_sha256"] = d_ref["candidate_sha256"]
            reference["raw_sha256"] = d_ref["raw_sha256"]
            reference["D_candidate_sha256"] = d_ref["candidate_sha256"]
            reference["D_raw_sha256"] = d_ref["raw_sha256"]
            # Filled after the synthetic D score payload and manifest entry are known.

    cohort_cases: dict[str, dict] = {}
    for case in manifest["cases"]:
        seed = case["seed"]
        references = case["U32"]["references"]
        probe = {
            "candidate_sha256": references["candidate_sha256"],
            "raw_sha256": references["raw_sha256"],
        }
        if seed not in RECOVERED_SEEDS:
            d_metrics_path = case["D"]["metrics_path"]
            d_metrics_bytes = payloads[d_metrics_path]
            score_ref = {
                "path": d_metrics_path,
                "bytes": len(d_metrics_bytes),
                "sha256": hashlib.sha256(d_metrics_bytes).hexdigest(),
            }
            references["score_ref"] = score_ref
            probe["score_ref"] = score_ref
        cohort_cases[str(seed)] = {"probes": {"UR": probe}}

    cohort_entry = next(entry for entry in manifest["entries"] if entry["role"] == "e164_cohort")
    cohort_data = _canonical_json({
        "manifest_sha256": manifest["e164_cohort_manifest_sha256"],
        "cases": cohort_cases,
    })
    put(cohort_entry["path"], cohort_data)

    for entry in manifest["entries"]:
        path = entry["path"]
        role = entry["role"]
        data = payloads.get(path)
        if data is None:
            if role == "e164_environment":
                data = _canonical_json({
                    "source_identity_sha256": manifest["e164_source_identity_sha256"],
                })
            else:
                data = _canonical_json({"synthetic_role": role, "path": path})
            put(path, data)
        entry["bytes"] = len(data)
        entry["sha256"] = hashlib.sha256(data).hexdigest()
        target = root.joinpath(*path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return manifest


def test_frozen_training_objective_matches_hand_computed_ridge():
    view = _view()
    candidate = np.ones((3, 3), dtype=np.float64)

    # Zero source features make both uniform teachers exactly match the policy;
    # only the frozen ridge remains: 1e-4 * mean((1 - 0)^2) = 1e-4.
    observed = budgeted._compute_training_metrics(view, candidate)
    assert observed["new_train_kl"] == pytest.approx(0.0, abs=1e-14)
    assert observed["old_train_kl"] == pytest.approx(0.0, abs=1e-14)
    assert observed["ridge"] == pytest.approx(1.0e-4, abs=1e-14)
    assert observed["fullL"] == pytest.approx(
        observed["new_train_kl"] + observed["old_train_kl"] + 1.0e-4,
        abs=1e-14,
    )
    assert set(observed["helper_counts"]) == set(HELPER_KEYS)
    assert observed["helper_counts"] == {
        "protected_objective_and_gradient": 1,
        "behavioral_objective_and_gradient": 2,
        "gradient_evaluations": 2,
        "new_kl": 1,
        "old_kl": 1,
        "ridge": 1,
    }
    assert _CURRENT_CALLS["real_budgeted_training_oracle_calls"] == 1
    assert _CURRENT_CALLS["real_protected_objective_and_gradient_calls"] == 1
    assert _CURRENT_CALLS["real_owner_kl_diagnostics_calls"] == 1
    assert _CURRENT_CALLS["real_stage0_scorer_calls"] == 0


def test_training_view_projection_ignores_holdout_and_search_rejects_prepared():
    prepared_a = _prepared()
    prepared_b = copy.deepcopy(prepared_a)
    prepared_b["new_teacher"][96:128] = np.array([0.1, 0.2, 0.3, 0.4])
    prepared_b["new_teacher"][224:256] = np.array([0.4, 0.3, 0.2, 0.1])
    prepared_b["old_teacher"][96:128] = np.array([0.3, 0.1, 0.4, 0.2])

    for teacher_name, holdout_slice in (
        ("new_teacher", slice(96, 128)),
        ("new_teacher", slice(224, 256)),
        ("old_teacher", slice(96, 128)),
    ):
        assert not np.array_equal(
            prepared_a[teacher_name][holdout_slice],
            prepared_b[teacher_name][holdout_slice],
        )
        assert np.allclose(
            prepared_b[teacher_name][holdout_slice].sum(axis=1), 1.0,
        )

    view_a = budgeted.make_training_view(prepared_a)
    view_b = budgeted.make_training_view(prepared_b)
    assert view_a.training_hash == view_b.training_hash
    assert np.array_equal(
        view_a.new_source_indices,
        np.concatenate((np.arange(96), np.arange(128, 224))),
    )
    assert np.array_equal(view_a.old_source_indices, np.arange(96))
    assert not hasattr(view_a, "new_holdout_teacher")
    assert not hasattr(view_a, "future_result")

    # The selection entry point accepts only the projected training type.
    with pytest.raises(TypeError):
        budgeted.search_budgeted_pair(prepared_a, _anchor(view_a))


def test_anchor_loader_checks_raw_bytes_three_input_identity_and_path_boundary(tmp_path, monkeypatch):
    manifest = _synthetic_anchor_manifest(tmp_path)
    def pin_synthetic_manifest(value):
        monkeypatch.setattr(
            budgeted, "_ANCHOR_MANIFEST_SHA256",
            hashlib.sha256(_canonical_json(value)).hexdigest(),
        )

    # The only patched constant is the frozen manifest pin; source identities,
    # roles, paths, per-file hashes, and loader checks stay active.
    pin_synthetic_manifest(manifest)
    anchors = budgeted.load_anchor_references(manifest, reference_root=tmp_path)
    assert tuple(anchors) == SEEDS
    for seed in SEEDS:
        assert anchors[seed].seed == seed
        assert anchors[seed].C.shape == anchors[seed].D.shape == anchors[seed].U32.shape == (3, 3)
        assert anchors[seed].C.dtype == anchors[seed].D.dtype == anchors[seed].U32.dtype == np.dtype(np.float32)
        assert not anchors[seed].C.flags.writeable
        assert not anchors[seed].D.flags.writeable
        assert not anchors[seed].U32.flags.writeable
        if seed in RECOVERED_SEEDS:
            assert not np.array_equal(anchors[seed].U32, anchors[seed].D)
        else:
            assert np.array_equal(anchors[seed].U32, anchors[seed].D)

    # Prepared identity is a three-file byte contract, not just a seed label.
    bad_identity = copy.deepcopy(manifest)
    bad_identity["cases"][0]["input_identity"]["files"]["actor.bin"]["bytes"] += 1
    pin_synthetic_manifest(bad_identity)
    with pytest.raises(ValueError, match="input identity differs"):
        budgeted.load_anchor_references(bad_identity, reference_root=tmp_path)

    bad_raw_hash = copy.deepcopy(manifest)
    bad_raw_hash["cases"][0]["C"]["raw_sha256"] = "f" * 64
    pin_synthetic_manifest(bad_raw_hash)
    with pytest.raises(ValueError, match="raw candidate SHA"):
        budgeted.load_anchor_references(bad_raw_hash, reference_root=tmp_path)

    bad_path = copy.deepcopy(manifest)
    bad_path["cases"][0]["prefix_files"]["actor.bin"] = "../outside/actor.bin"
    pin_synthetic_manifest(bad_path)
    with pytest.raises(ValueError, match="cannot traverse"):
        budgeted.load_anchor_references(bad_path, reference_root=tmp_path)

    first_entry = manifest["entries"][0]["path"]
    outside = tmp_path.parent / "e165-outside-reference.bin"
    outside.write_bytes((tmp_path.joinpath(*first_entry.split("/")).read_bytes()))
    pin_synthetic_manifest(manifest)
    with pytest.raises(ValueError, match="escapes authorized reference_root"):
        budgeted.load_anchor_references(
            manifest, reference_root=tmp_path, path_map={first_entry: outside},
        )


def test_constraint_switch_changes_only_the_first_accepted_halving(monkeypatch):
    view = _view()
    anchors = _anchor(view)
    table = _table_for({
        0.0: _metrics(20.0, 3.0, 17.0),
        1.0: _metrics(12.0, 2.0, 10.0),
        4.0: _metrics(10.0, 3.0, 7.0),
        3.0: _metrics(8.0, 4.0, 4.0),       # S0 accepts; S1 continues.
        2.0: _metrics(9.0, 3.0, 6.0),       # S1 accepts at alpha = 1/2.
    })
    calls = _patch_oracle(monkeypatch, table)

    trace = budgeted.search_budgeted_pair(view, anchors)
    locked = budgeted.lock_budgeted_pair(view, trace)

    assert trace["trials"][0]["j"] == 0
    assert trace["trials"][0]["alpha"] == 1.0
    assert trace["trials"][0]["S0_first_accepted"] is True
    assert trace["trials"][0]["S1_first_accepted"] is False
    assert trace["trials"][1]["j"] == 1
    assert trace["trials"][1]["alpha"] == 0.5
    assert trace["trials"][1]["S1_first_accepted"] is True
    assert np.array_equal(locked["probes"]["S0"]["candidate"], _coeff(3.0))
    assert np.array_equal(locked["probes"]["S1"]["candidate"], _coeff(2.0))
    assert trace["counters"]["actual_oracle_evaluations"] <= 21
    assert len(calls) == trace["counters"]["actual_oracle_evaluations"]


def test_strict_tie_and_no_better_trial_keep_the_D_incumbent(monkeypatch):
    view = _view()
    anchors = _anchor(view)
    tie_candidate = _candidate_key(np.full((3, 3), 3.0, dtype=np.float64))
    table = _table_for({
        0.0: _metrics(20.0, 3.0, 17.0),
        1.0: _metrics(12.0, 2.0, 10.0),
        4.0: _metrics(10.0, 3.0, 7.0),
        3.0: _metrics(10.0, 3.0, 7.0),  # Exact tie with D is not an improvement.
    })
    assert tie_candidate in table
    calls = _patch_oracle(monkeypatch, table, default=_metrics(11.0, 3.0, 7.0))

    trace = budgeted.search_budgeted_pair(view, anchors)
    locked = budgeted.lock_budgeted_pair(view, trace)

    assert trace["trials"][0]["fullL"] == trace["D_fullL"]
    assert trace["trials"][0]["better_than_D"] is False
    assert all(record["better_than_D"] is False for record in trace["trials"])
    assert trace["search_stop"] == {
        "j": 15, "reason": "fixed_trial_schedule_exhausted",
    }
    for label in ("S0", "S1"):
        assert trace["selections"][label]["origin"] == "D"
        assert trace["selections"][label]["j"] is None
        assert np.array_equal(locked["probes"][label]["candidate"], _coeff(4.0))
    assert trace["counters"]["actual_oracle_evaluations"] <= 21
    assert len(calls) == trace["counters"]["actual_oracle_evaluations"]


def test_float32_cast_failure_falls_back_to_C0_without_trying_a_later_step(monkeypatch):
    view = _view()
    next_f32 = np.nextafter(np.float32(1.0), np.float32(2.0))
    expected = _expected_training(
        c0=(20.0, 3.0, 17.0, 0.0),
        c=(12.0, 4.0, 8.0, 0.0),
        d=(10.0, 3.0, 7.0, 0.0),
    )
    anchors = _anchor(view, c=1.0, d=2.0, u=next_f32, expected_training=expected)
    midpoint = 1.0 + (float(next_f32) - 1.0) / 2.0
    table = _table_for({
        0.0: _metrics(20.0, 3.0, 17.0),
        1.0: _metrics(12.0, 4.0, 8.0),
        2.0: _metrics(10.0, 3.0, 7.0),
        float(next_f32): _metrics(8.0, 4.0, 4.0),
        midpoint: _metrics(9.0, 2.0, 7.0),
    })
    calls = _patch_oracle(monkeypatch, table)

    trace = budgeted.search_budgeted_pair(view, anchors)
    locked = budgeted.lock_budgeted_pair(view, trace)

    assert trace["selections"]["S1"]["j"] == 1
    assert len(trace["trials"]) == 2  # The cast failure does not resume the halving loop.
    for label in ("S0", "S1"):
        probe = locked["probes"][label]
        assert probe["fallback_applied"] is True
        assert np.array_equal(probe["candidate"], _coeff(0.0))
    assert np.array_equal(locked["probes"]["S1"]["raw_cast_candidate"], _coeff(1.0))
    assert len(calls) == trace["counters"]["actual_oracle_evaluations"]


def test_nonfinite_oracle_value_raises_instead_of_becoming_a_fallback(monkeypatch):
    view = _view()
    anchors = _anchor(view)
    table = _table_for({
        0.0: _metrics(20.0, 3.0, 17.0),
        1.0: _metrics(12.0, 2.0, 10.0),
        4.0: _metrics(10.0, 3.0, 7.0),
        3.0: _metrics(float("nan"), 4.0, 4.0),
    })
    calls = _patch_oracle(monkeypatch, table)

    with pytest.raises(FloatingPointError, match="fullL must be finite"):
        budgeted.search_budgeted_pair(view, anchors)
    assert len(calls) == 4  # Three fixed baselines, then the failing first trial.


def test_repeated_trial_requests_hit_cache_and_stay_under_the_21_miss_cap(monkeypatch):
    view = _view()
    anchors = _anchor(view, c=1.0, u=1.0)
    table = _table_for({
        0.0: _metrics(20.0, 3.0, 17.0),
        1.0: _metrics(12.0, 2.0, 10.0),
        4.0: _metrics(10.0, 3.0, 7.0),
    })
    calls = _patch_oracle(monkeypatch, table)

    trace = budgeted.search_budgeted_pair(view, anchors)
    counts = trace["counters"]

    assert counts["trial_requests"] == 16
    assert counts["trial_unique"] == 1
    assert counts["cache_hits"] >= 16
    assert counts["actual_oracle_evaluations"] == 3
    assert counts["actual_oracle_evaluations"] <= 21
    assert len(calls) == 3


def _v1_score_metrics(seed: int, prepared_id: str, candidate_sha: str) -> dict:
    return {
        "format": "h036_score_metrics_v1",
        "seed": seed,
        "prepared_id": prepared_id,
        "lambda_value": 1,
        "binding": "wildcard",
        "candidate_sha256": candidate_sha,
        "pre_codec_fresh": {"new_train": 3.0},
        "encoded_fresh": {"new_train": 3.125, "new_holdout": 0.4},
        "W0_fresh": {"new_holdout": 0.6},
        "K_old": 0.5,
        "K_new": 0.4,
        "K_new_W0": 0.6,
        "G_new": 0.2,
    }


def test_cohort_permit_and_scoring_adapter_bind_D_then_report_only_locked_probes(monkeypatch):
    locked_pairs, by_seed = _cohort_fixture(monkeypatch)
    source_sha = "d" * 64
    manifest_sha = budgeted._ANCHOR_MANIFEST_SHA256

    with pytest.raises(ValueError, match="16"):
        budgeted.seal_budgeted_cohort(
            locked_pairs[:-1],
            source_identity_sha256=source_sha,
            anchor_manifest_sha256=manifest_sha,
        )
    changed_lock = copy.deepcopy(locked_pairs)
    changed_lock[0]["trace_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        budgeted.seal_budgeted_cohort(
            changed_lock,
            source_identity_sha256=source_sha,
            anchor_manifest_sha256=manifest_sha,
        )

    # C0, C, D, U32, every scheduled trial, and both post-cast probes share
    # the same exact bytes. The three logical baselines must consume one miss.
    degenerate_view = _view(26218)
    zero_expected = _expected_training(
        c0=(0.0, 0.0, 0.0, 0.0),
        c=(0.0, 0.0, 0.0, 0.0),
        d=(0.0, 0.0, 0.0, 0.0),
    )
    degenerate_anchors = _anchor(
        degenerate_view, c=0.0, d=0.0, u=0.0,
        expected_training=zero_expected,
    )
    zero_candidate = np.zeros((3, 3), dtype=np.float64)
    degenerate_calls = _patch_oracle(
        monkeypatch,
        {_candidate_key(zero_candidate): _metrics(0.0, 0.0, 0.0)},
    )
    degenerate_trace = budgeted.search_budgeted_pair(
        degenerate_view, degenerate_anchors,
    )
    assert degenerate_trace["counters"]["actual_oracle_evaluations"] == 1
    assert degenerate_trace["counters"]["cache_misses"] == 1
    assert len(degenerate_calls) == 1
    degenerate_locked = budgeted.lock_budgeted_pair(
        degenerate_view, degenerate_trace,
    )
    assert degenerate_locked["seed"] == 26218
    assert degenerate_locked["counters"]["actual_oracle_evaluations"] == 1
    locked_pairs[-1] = degenerate_locked

    permit = budgeted.seal_budgeted_cohort(
        locked_pairs,
        source_identity_sha256=source_sha,
        anchor_manifest_sha256=manifest_sha,
    )

    view, anchors, _trace, locked = by_seed[26201]
    prepared = {
        "format": "h036_prepared_v1",
        "seed": view.seed,
        "prepared_id": view.prepared_id,
        "input_identity": json.loads(view.input_identity_json),
        "mean64": np.zeros((3, 3), dtype=np.float64),
    }
    d_sha = _sci_sha(anchors.D)
    expected = _v1_score_metrics(view.seed, view.prepared_id, d_sha)
    actor_bytes = bytes(range(89))
    actor_sha = hashlib.sha256(actor_bytes).hexdigest()
    calls: list[tuple[dict, dict, str]] = []

    def fake_v1_scorer(prepared_arg, fit_result, binding):
        _CURRENT_CALLS["mock_stage0_scorer_calls"] += 1
        calls.append((prepared_arg, fit_result, binding))
        candidate_sha = fit_result["diagnostics"]["candidate_sha256"]
        metrics = _v1_score_metrics(
            prepared_arg["seed"], prepared_arg["prepared_id"], candidate_sha,
        )
        metrics["artifacts"] = {"actor_sha256": actor_sha}
        return {
            "format": "h036_stage0_score_v1",
            "metrics": metrics,
            "deployment": {"actor_bytes": actor_bytes},
        }

    monkeypatch.setattr(inputs, "encode_and_score_stage0", fake_v1_scorer)
    with pytest.raises((PermissionError, TypeError), match="cohort"):
        budgeted.score_budgeted_anchor(
            prepared, anchors, cohort_permit=None,
            expected_metrics=expected, expected_actor_sha256=actor_sha,
        )
    with pytest.raises((PermissionError, TypeError), match="anchor"):
        budgeted.score_locked_budgeted_probe(
            prepared, locked, "S0", cohort_permit=permit, anchor_receipt=None,
        )
    assert calls == []

    anchor_result = budgeted.score_budgeted_anchor(
        prepared, anchors, cohort_permit=permit,
        expected_metrics=expected, expected_actor_sha256=actor_sha,
    )
    probe_result = budgeted.score_locked_budgeted_probe(
        prepared, locked, "S0", cohort_permit=permit,
        anchor_receipt=anchor_result["receipt"],
    )

    assert len(calls) == 2
    anchor_fit = calls[0][1]
    probe_fit = calls[1][1]
    assert calls[0][2] == calls[1][2] == "wildcard"
    assert anchor_fit["format"] == probe_fit["format"] == "h036_fit_v1"
    assert anchor_fit["lambda_value"] == probe_fit["lambda_value"] == 1
    assert np.array_equal(anchor_fit["candidate"], anchors.D)
    assert np.array_equal(probe_fit["candidate"], locked["probes"]["S0"]["candidate"])
    c0_sha = _sci_sha(np.zeros((3, 3), dtype=np.float32))
    assert locked["probes"]["S0"]["fallback_applied"] is True
    assert np.array_equal(probe_fit["candidate"], np.zeros((3, 3), dtype=np.float32))
    assert [call[1]["diagnostics"]["candidate_sha256"] for call in calls] == [d_sha, c0_sha]
    budget = probe_result["budget_diagnostic"]
    assert budget["encoded_new_train_kl"] == pytest.approx(3.125)
    assert budget["b"] == pytest.approx(3.0)
    assert budget["signed_excess"] == pytest.approx(0.125)
    assert budget["within_budget"] is False

