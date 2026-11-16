"""Training-only replay helpers for the accepted H036 diagnostic.

This module deliberately uses only the Python standard library. It consumes
scalar diagnostics and candidate identities; it never loads model state or
performs fitting or scoring.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FORMAT = "h036_acceptance_replay_v1"
COHORT_SEEDS = (26201, 26202, 26203, 26204, 26205, 26206, 26207, 26208,
                26209, 26210, 26211, 26212, 26213, 26215, 26216, 26218)
RECOVERY_SEEDS = (26204, 26206, 26209, 26212, 26213)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_TRAIN_FIELDS = ("fullL", "newKL", "oldKL", "ridge")
_CLOSURE_FIELDS = (
    "index", "fullL", "newKL", "oldKL", "ridge", "candidate_hash",
    "accepted", "reject_reason",
)
_CLOSURE_COMPARE_FIELDS = _CLOSURE_FIELDS
_F64_FIELDS = ("index", "fullL", "newKL", "oldKL", "ridge", "candidate_hash")
_CAST_IDENTITY_FIELDS = (
    "fallback_applied", "selected_float32_candidate_sha256",
    "candidate_sha256", "prepared_id", "seed",
)
_TERMINATION_FIELDS = (
    "iterations", "termination_reason", "optimizer_max_evaluations",
    "optimizer_max_iterations", "optimizer_history_size",
    "optimizer_learning_rate", "optimizer_tolerance_grad",
    "optimizer_tolerance_change",
)
_CLOSURE_COUNT_KEYS = (
    "closure_objective_evaluations", "closure_kl_metric_evaluations",
)


class ReplayMismatchError(ValueError):
    """Raised when a replay trace differs from its frozen reference."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite int or float")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _sha256(value: Any, name: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _required(mapping: Mapping[str, Any], key: str, name: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{name} is missing required field {key!r}")
    return mapping[key]


def _normalize_initial(initial_train: Mapping[str, Any]) -> dict[str, Any]:
    initial = _mapping(initial_train, "initial_train")
    normalized: dict[str, Any] = {"index": 0}
    for key in _TRAIN_FIELDS:
        normalized[key] = _number(_required(initial, key, "initial_train"), f"initial_train.{key}")
    normalized["candidate_hash"] = _sha256(
        initial.get("candidate_hash"), "initial_train.candidate_hash", required=False
    )
    return normalized


def _normalize_closures(closure_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(closure_records, (str, bytes)) or not isinstance(closure_records, Sequence):
        raise TypeError("closure_records must be an ordered sequence of mappings")
    normalized: list[dict[str, Any]] = []
    for expected_index, raw_record in enumerate(closure_records, start=1):
        record = _mapping(raw_record, f"closure_records[{expected_index - 1}]")
        index = _integer(_required(record, "index", f"closure_records[{expected_index - 1}]"),
                         f"closure_records[{expected_index - 1}].index")
        if index != expected_index:
            raise ValueError(
                f"closure indices must be continuous from 1 in input order; "
                f"expected {expected_index}, got {index}"
            )
        item: dict[str, Any] = {"index": index}
        for key in _TRAIN_FIELDS:
            item[key] = _number(
                _required(record, key, f"closure_records[{expected_index - 1}]"),
                f"closure_records[{expected_index - 1}].{key}",
            )
        item["candidate_hash"] = _sha256(
            _required(record, "candidate_hash", f"closure_records[{expected_index - 1}]"),
            f"closure_records[{expected_index - 1}].candidate_hash",
        )
        accepted = _required(record, "accepted", f"closure_records[{expected_index - 1}]")
        if not isinstance(accepted, bool):
            raise TypeError(f"closure_records[{expected_index - 1}].accepted must be bool")
        reject_reason = _required(record, "reject_reason", f"closure_records[{expected_index - 1}]")
        if reject_reason is not None and not isinstance(reject_reason, str):
            raise TypeError(
                f"closure_records[{expected_index - 1}].reject_reason must be str or None"
            )
        item["accepted"] = accepted
        item["reject_reason"] = reject_reason
        normalized.append(item)
    return normalized


def _normalize_cast_status(original_cast_status: Mapping[str, Any]) -> dict[str, Any]:
    status = _mapping(original_cast_status, "original_cast_status")
    fallback = _required(status, "fallback_applied", "original_cast_status")
    if not isinstance(fallback, bool):
        raise TypeError("original_cast_status.fallback_applied must be bool")
    selected_hash = _sha256(
        _required(status, "selected_float32_candidate_sha256", "original_cast_status"),
        "original_cast_status.selected_float32_candidate_sha256",
    )
    candidate_hash = _sha256(
        _required(status, "candidate_sha256", "original_cast_status"),
        "original_cast_status.candidate_sha256",
    )
    prepared_id = _required(status, "prepared_id", "original_cast_status")
    if not isinstance(prepared_id, str) or not prepared_id:
        raise ValueError("original_cast_status.prepared_id must be a non-empty string")
    seed = _integer(_required(status, "seed", "original_cast_status"), "original_cast_status.seed")
    if not fallback and candidate_hash != selected_hash:
        raise ValueError(
            "original_cast_status candidate_sha256 differs from selected float32 identity without fallback"
        )
    return {
        "fallback_applied": fallback,
        "selected_float32_candidate_sha256": selected_hash,
        "candidate_sha256": candidate_hash,
        "prepared_id": prepared_id,
        "seed": seed,
    }


def replay_existing_trace(
    initial_train: Mapping[str, Any],
    closure_records: Sequence[Mapping[str, Any]],
    recorded_best64: Mapping[str, Any],
    original_cast_status: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay F64/U64 selection from finite training scalars and candidate hashes.

    C0 is index 0. F64 admits a point only when its new-train KL is at most
    C0's value plus 1e-12; U64 admits every finite closure point. Both use the
    exact full objective and replace on ``<=``, so the later point wins an
    exact tie. Recorded acceptance flags are preserved for provenance but do
    not participate in either selection rule.

    The result contains training-only records and hashes. It does not accept
    holdout data, score candidates, or approximate candidate identity by loss.
    """
    initial = _normalize_initial(initial_train)
    closures = _normalize_closures(closure_records)
    best_recorded = _mapping(recorded_best64, "recorded_best64")
    recorded_full_l = _number(
        _required(best_recorded, "fullL", "recorded_best64"), "recorded_best64.fullL"
    )
    recorded_hash = _sha256(
        _required(best_recorded, "candidate_hash", "recorded_best64"),
        "recorded_best64.candidate_hash",
    )
    cast_status = _normalize_cast_status(original_cast_status)

    selection_initial = dict(initial)
    if selection_initial["candidate_hash"] is None:
        if closures:
            first = closures[0]
            if all(first[key] == initial[key] for key in _TRAIN_FIELDS):
                selection_initial["candidate_hash"] = first["candidate_hash"]
        else:
            # With no closure calls, the recorded F64 candidate is the untouched C0.
            selection_initial["candidate_hash"] = recorded_hash

    filtered = dict(selection_initial)
    unrestricted = dict(selection_initial)
    for record in closures:
        if record["newKL"] <= initial["newKL"] + 1e-12 and record["fullL"] <= filtered["fullL"]:
            filtered = {key: record[key] for key in ("index", *_TRAIN_FIELDS, "candidate_hash")}
        if record["fullL"] <= unrestricted["fullL"]:
            unrestricted = {key: record[key] for key in ("index", *_TRAIN_FIELDS, "candidate_hash")}

    if filtered["fullL"] != recorded_full_l or filtered["candidate_hash"] != recorded_hash:
        raise ValueError("recorded_best64 does not match the replayed F64 objective and hash")

    same_candidate = (
        filtered["candidate_hash"] is not None
        and unrestricted["candidate_hash"] is not None
        and filtered["candidate_hash"] == unrestricted["candidate_hash"]
    )
    reuse_eligible = same_candidate and not cast_status["fallback_applied"]

    path_record = {
        "initial_train": initial,
        "closure_records": closures,
    }
    path_sha256 = hashlib.sha256(_canonical_bytes(path_record)).hexdigest()
    return {
        "format": FORMAT,
        "F64": filtered,
        "U64": unrestricted,
        "same_float64_candidate": same_candidate,
        "reuse_eligible": reuse_eligible,
        "recovery_required": not reuse_eligible,
        "original_float32_identity": cast_status,
        "path_sha256": path_sha256,
        "closure_count": len(closures),
    }


def training_path_sha256(
    initial_train: Mapping[str, Any], closure_records: Sequence[Mapping[str, Any]]
) -> str:
    """Return the canonical ordered hash used for original training-path identity."""
    path_record = {
        "initial_train": _normalize_initial(initial_train),
        "closure_records": _normalize_closures(closure_records),
    }
    return hashlib.sha256(_canonical_bytes(path_record)).hexdigest()


def _diagnostics_mapping(diagnostics: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an original-compatible diagnostics mapping without reading holdout fields."""
    outer = _mapping(diagnostics, "diagnostics")
    nested = outer.get("diagnostics")
    if nested is not None:
        return _mapping(nested, "diagnostics['diagnostics']")
    return outer


def _initial_train_from_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    initial_metrics = _mapping(
        _required(diagnostics, "initial_metrics", "diagnostics"), "diagnostics.initial_metrics"
    )
    # Deliberately read only training metrics; no holdout, gate, or resource field
    # is allowed to enter the replay decision.
    return {
        "fullL": _number(_required(diagnostics, "initial_fullL", "diagnostics"),
                          "diagnostics.initial_fullL"),
        "newKL": _number(_required(initial_metrics, "new_train_kl", "initial_metrics"),
                          "diagnostics.initial_metrics.new_train_kl"),
        "oldKL": _number(_required(initial_metrics, "old_train_kl", "initial_metrics"),
                          "diagnostics.initial_metrics.old_train_kl"),
        "ridge": _number(_required(initial_metrics, "ridge", "initial_metrics"),
                          "diagnostics.initial_metrics.ridge"),
    }


def replay_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Bridge the original H036 D diagnostics schema into scalar F/U replay.

    Only ``initial_fullL``, initial training metrics, closure training fields,
    the recorded F64 identity, and F32 identity fields are consumed. Existing
    holdout metrics, gate outcomes, C0 booleans, and resource measurements are
    intentionally ignored.
    """
    diag = _diagnostics_mapping(diagnostics)
    closures = _required(diag, "closure_records", "diagnostics")
    if not isinstance(closures, Sequence) or isinstance(closures, (str, bytes)):
        raise TypeError("diagnostics.closure_records must be an ordered sequence")
    if "closure_count" in diag and _integer(diag["closure_count"], "diagnostics.closure_count") != len(closures):
        raise ValueError("diagnostics.closure_count does not equal closure_records length")
    cast_status = {key: _required(diag, key, "diagnostics") for key in _CAST_IDENTITY_FIELDS}
    return replay_existing_trace(
        _initial_train_from_diagnostics(diag),
        closures,
        {
            "fullL": _required(diag, "best_float64_fullL", "diagnostics"),
            "candidate_hash": _required(diag, "best_float64_candidate_sha256", "diagnostics"),
        },
        cast_status,
    )


def _replay_diagnostics_view(diag: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return normalized train/path and F32 identity records for strict comparison."""
    initial = _initial_train_from_diagnostics(diag)
    closures = _normalize_closures(_required(diag, "closure_records", "diagnostics"))
    replay = replay_diagnostics(diag)
    return {"initial_train": initial, "closure_records": closures}, replay


def _c0_candidate_hash(
    diagnostics: Mapping[str, Any],
    training: Mapping[str, Any],
    replay: Mapping[str, Any],
    *,
    require_explicit: bool,
) -> str:
    """Resolve C0 identity without substituting equal objective values for identity."""
    explicit = diagnostics.get("initial_candidate_sha256")
    if explicit is not None:
        c0_hash = _sha256(explicit, "diagnostics.initial_candidate_sha256")
    elif require_explicit:
        raise ReplayMismatchError("actual trace is missing initial_candidate_sha256 for C0")
    else:
        closures = training["closure_records"]
        if closures:
            first = closures[0]
            initial = training["initial_train"]
            for key in _TRAIN_FIELDS:
                if first[key] != initial[key]:
                    raise ReplayMismatchError(
                        f"closure[1].{key} differs from C0; old diagnostics cannot establish C0 identity"
                    )
            c0_hash = first["candidate_hash"]
        else:
            f64 = _mapping(replay["F64"], "replay.F64")
            if f64["index"] != 0:
                raise ReplayMismatchError("zero-closure trace does not retain the initial F64 candidate")
            c0_hash = _sha256(f64["candidate_hash"], "replay.F64.candidate_hash")

    closures = training["closure_records"]
    if closures:
        first = closures[0]
        initial = training["initial_train"]
        for key in _TRAIN_FIELDS:
            if first[key] != initial[key]:
                raise ReplayMismatchError(
                    f"closure[1].{key} differs from C0; first closure must evaluate the unmodified initialization"
                )
        if first["candidate_hash"] != c0_hash:
            raise ReplayMismatchError("closure[1] candidate hash differs from C0 identity")
    else:
        f64 = _mapping(replay["F64"], "replay.F64")
        if f64["index"] != 0 or f64["candidate_hash"] != c0_hash:
            raise ReplayMismatchError("zero-closure F64 identity differs from C0")
    return c0_hash


def _trace_view(actual_trace: Mapping[str, Any]) -> Mapping[str, Any]:
    outer = _mapping(actual_trace, "actual_trace")
    if "diagnostics" in outer:
        return _mapping(outer["diagnostics"], "actual_trace['diagnostics']")
    return outer


def _strict_field(expected: Mapping[str, Any], actual: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in expected or key not in actual:
        raise ReplayMismatchError(f"{where}.{key} is missing from expected or actual trace")
    left, right = expected[key], actual[key]
    if left != right:
        raise ReplayMismatchError(f"{where}.{key} differs: expected {left!r}, got {right!r}")
    return left


def assert_replay_matches(expected_diagnostics: Mapping[str, Any], actual_trace: Mapping[str, Any]) -> None:
    """Raise ``ReplayMismatchError`` unless an actual trace exactly matches D.

    The check compares C0 training scalars, ordered closure training scalars and
    hashes, F64/F32 identities, termination, and configured/legal optimizer
    budgets. It does not claim to compare gradient values because the original
    diagnostics do not store a per-closure gradient trace. Holdout outcomes and
    wall/CPU/memory measurements are excluded.
    """
    expected = _diagnostics_mapping(expected_diagnostics)
    actual = _trace_view(actual_trace)
    expected_training, expected_replay = _replay_diagnostics_view(expected)
    actual_training, actual_replay = _replay_diagnostics_view(actual)

    expected_c0_hash = _c0_candidate_hash(
        expected, expected_training, expected_replay, require_explicit=False
    )
    actual_c0_hash = _c0_candidate_hash(
        actual, actual_training, actual_replay, require_explicit=True
    )
    if expected_c0_hash != actual_c0_hash:
        raise ReplayMismatchError(
            f"C0 candidate identity differs: expected {expected_c0_hash}, got {actual_c0_hash}"
        )

    for key in _TRAIN_FIELDS:
        _strict_field(expected_training["initial_train"], actual_training["initial_train"], key, "C0")

    expected_closures = expected_training["closure_records"]
    actual_closures = actual_training["closure_records"]
    if len(expected_closures) != len(actual_closures):
        raise ReplayMismatchError(
            f"closure count differs: expected {len(expected_closures)}, got {len(actual_closures)}"
        )
    for index, (expected_record, actual_record) in enumerate(
        zip(expected_closures, actual_closures), start=1
    ):
        for key in _CLOSURE_COMPARE_FIELDS:
            _strict_field(expected_record, actual_record, key, f"closure[{index}]")

    if "closure_count" in expected and "closure_count" in actual:
        _strict_field(expected, actual, "closure_count", "diagnostics")
    elif len(expected_closures) != len(actual_closures):
        raise ReplayMismatchError("closure_count is missing or differs")

    for key in _F64_FIELDS:
        _strict_field(expected_replay["F64"], actual_replay["F64"], key, "F64")
    for key in _CAST_IDENTITY_FIELDS:
        _strict_field(
            expected_replay["original_float32_identity"],
            actual_replay["original_float32_identity"],
            key,
            "float32_identity",
        )

    # These original return fields identify whether F64 was cast, gated, or
    # replaced by C0. They are checked as evidence but are never selection inputs.
    for key in ("returned_is_C0", "fallback_reason"):
        if key in expected or key in actual:
            _strict_field(expected, actual, key, "cast_status")
    for key in ("selected_float32_new_train_kl", "final_float32_fullL", "final_float32_ridge"):
        if key in expected or key in actual:
            _strict_field(expected, actual, key, "float32_training")

    for key in _TERMINATION_FIELDS:
        _strict_field(expected, actual, key, "termination")
    expected_max_evals = _integer(expected["optimizer_max_evaluations"], "expected.optimizer_max_evaluations")
    expected_max_iterations = _integer(expected["optimizer_max_iterations"], "expected.optimizer_max_iterations")
    actual_max_evals = _integer(actual["optimizer_max_evaluations"], "actual.optimizer_max_evaluations")
    actual_max_iterations = _integer(actual["optimizer_max_iterations"], "actual.optimizer_max_iterations")
    actual_iterations = _integer(actual["iterations"], "actual.iterations")
    if len(actual_closures) > actual_max_evals or actual_iterations > actual_max_iterations:
        raise ReplayMismatchError("actual trace exceeds its configured optimizer budget")
    if expected_max_evals <= 0 or expected_max_iterations <= 0:
        raise ReplayMismatchError("expected optimizer budget is not positive")

    expected_counts = expected.get("computation_counts")
    actual_counts = actual.get("computation_counts")
    if expected_counts is not None or actual_counts is not None:
        if not isinstance(expected_counts, Mapping) or not isinstance(actual_counts, Mapping):
            raise ReplayMismatchError("computation_counts must be present as mappings in both traces")
        for key in _CLOSURE_COUNT_KEYS:
            _strict_field(expected_counts, actual_counts, key, "computation_counts")


def _load_config(config: Any) -> tuple[Mapping[str, Any], str]:
    if config is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "h036_acceptance_diagnostic_v1.json"
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        return _mapping(raw, "default config"), str(config_path)
    if isinstance(config, Mapping):
        return config, "<mapping>"
    config_path = Path(config)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return _mapping(raw, "config"), str(config_path)


def _seed(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer seed")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    raise TypeError(f"{name} must be an integer seed")


def _config_seed_list(config: Mapping[str, Any], names: tuple[str, ...], default: tuple[int, ...]) -> tuple[int, ...]:
    value = next((config[name] for name in names if name in config), None)
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"config.{names[0]} must be a sequence")
    return tuple(_seed(item, f"config.{names[0]}[{index}]") for index, item in enumerate(value))


def _normalize_seed_records(records_by_seed: Mapping[Any, Any]) -> dict[int, Mapping[str, Any]]:
    records = _mapping(records_by_seed, "records_by_seed")
    normalized: dict[int, Mapping[str, Any]] = {}
    for raw_seed, diagnostics in records.items():
        seed = _seed(raw_seed, "records_by_seed key")
        if seed in normalized:
            raise ValueError(f"duplicate seed key after normalization: {seed}")
        diag = _diagnostics_mapping(_mapping(diagnostics, f"records_by_seed[{seed}]"))
        diag_seed = _integer(_required(diag, "seed", f"records_by_seed[{seed}]"), f"records_by_seed[{seed}].seed")
        if diag_seed != seed:
            raise ValueError(f"diagnostics seed {diag_seed} does not match mapping key {seed}")
        normalized[seed] = diag
    return normalized


def build_recovery_manifest(
    records_by_seed: Mapping[Any, Mapping[str, Any]], *, config: Any = None
) -> dict[str, Any]:
    """Build the fixed full-cohort replay/recovery manifest from original D diagnostics.

    The cohort and five recovery seeds are fixed by the accepted H036 protocol.
    Classification uses only exact F64/U64 candidate hashes and the original
    fallback flag; gate, C0, holdout, and metric outcomes do not choose seeds.
    """
    config_data, _config_source = _load_config(config)
    if config_data.get("format") != "h036_acceptance_diagnostic_v1":
        raise ValueError("config format is not h036_acceptance_diagnostic_v1")
    configured_seeds = _config_seed_list(
        config_data, ("seeds", "cohort_seeds", "fixed_seeds"), COHORT_SEEDS
    )
    configured_recovery = _config_seed_list(
        config_data, ("recovery_seeds", "recovery_seed_ids", "restore_seeds"), RECOVERY_SEEDS
    )
    if configured_seeds != COHORT_SEEDS:
        raise ValueError("config cohort seeds differ from the fixed H036 acceptance cohort")
    if configured_recovery != RECOVERY_SEEDS:
        raise ValueError("config recovery seeds differ from the fixed H036 recovery set")
    if _integer(config_data.get("reuse_count"), "config.reuse_count") != 11:
        raise ValueError("config reuse_count must be 11")
    if _number(config_data.get("lambda_value"), "config.lambda_value") != 1.0:
        raise ValueError("config lambda_value must remain 1")
    labels = config_data.get("labels")
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)) or tuple(labels) != ("V", "UG", "UR"):
        raise ValueError("config labels must be exactly V, UG, UR in order")
    if _number(config_data.get("new_kl_slack"), "config.new_kl_slack") != 1e-12:
        raise ValueError("config new_kl_slack must remain 1e-12")
    if config_data.get("incumbent_tie") != "latest_exact_fullL_tie":
        raise ValueError("config incumbent_tie must select the latest exact fullL tie")
    if config_data.get("binding") != "wildcard":
        raise ValueError("config binding must remain wildcard")
    if config_data.get("holdout_selection") is not False:
        raise ValueError("config must prohibit holdout-based selection")
    if _integer(config_data.get("maximum_new_fit_calls"), "config.maximum_new_fit_calls") != 5:
        raise ValueError("config maximum_new_fit_calls must be 5")
    if _integer(config_data.get("maximum_new_scoring_packages"), "config.maximum_new_scoring_packages") != 15:
        raise ValueError("config maximum_new_scoring_packages must be 15")
    v1_source_identity = _sha256(
        config_data.get("v1_source_identity_sha256"), "config.v1_source_identity_sha256"
    )

    records = _normalize_seed_records(records_by_seed)
    if set(records) != set(COHORT_SEEDS):
        missing = sorted(set(COHORT_SEEDS) - set(records))
        extra = sorted(set(records) - set(COHORT_SEEDS))
        raise ValueError(f"records_by_seed must contain exactly the 16 fixed seeds; missing={missing}, extra={extra}")

    replays: dict[str, dict[str, Any]] = {}
    for seed in COHORT_SEEDS:
        diag = records[seed]
        replay = replay_diagnostics(diag)
        if replay["original_float32_identity"]["fallback_applied"]:
            raise ValueError(f"seed {seed} had an original float32 fallback; reuse is not valid")
        replay["seed"] = seed
        replays[str(seed)] = replay

    recovery_seeds = tuple(
        seed for seed in COHORT_SEEDS if replays[str(seed)]["recovery_required"]
    )
    reuse_seeds = tuple(seed for seed in COHORT_SEEDS if replays[str(seed)]["reuse_eligible"])
    if len(reuse_seeds) != 11 or len(recovery_seeds) != 5:
        raise ValueError(
            f"hash replay must yield exactly 11 reusable and 5 recovery prefixes; "
            f"got {len(reuse_seeds)} reusable and {len(recovery_seeds)} recovery"
        )
    if recovery_seeds != RECOVERY_SEEDS:
        raise ValueError(
            f"hash-derived recovery seeds {recovery_seeds!r} differ from fixed set {RECOVERY_SEEDS!r}"
        )

    body = {
        "format": FORMAT,
        "config_sha256": hashlib.sha256(_canonical_bytes(config_data)).hexdigest(),
        "v1_source_identity_sha256": v1_source_identity,
        "cohort_seeds": list(COHORT_SEEDS),
        "recovery_seeds": list(recovery_seeds),
        "reuse_seeds": list(reuse_seeds),
        "replays": replays,
    }
    body["manifest_sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    return body


def lock_cohort(
    manifest: Mapping[str, Any],
    recovered_locks: Mapping[Any, Any],
    reuse_bindings: Mapping[Any, Any],
) -> dict[str, Any]:
    """Freeze the full cohort after validating reuse bindings and recovery locks.

    Reused prefixes bind their original C/D prepared inputs, wildcard binding,
    foundation, v1 source, candidate/deployment/metrics artifacts, and run002
    provenance. Recovered prefixes bind the replay path and three scalar-only
    V/UG/UR lock descriptions. No candidate arrays or held-out values are read.
    """
    manifest_data = _json_object(manifest, "manifest")
    _verify_digest_field(manifest_data, "manifest_sha256", "manifest")
    if manifest_data.get("format") != FORMAT:
        raise ValueError("manifest format is not the accepted H036 replay format")
    if tuple(manifest_data.get("cohort_seeds", ())) != COHORT_SEEDS:
        raise ValueError("manifest cohort does not match the fixed 16-seed cohort")
    if tuple(manifest_data.get("recovery_seeds", ())) != RECOVERY_SEEDS:
        raise ValueError("manifest recovery set does not match the fixed five seeds")
    expected_reuse_seeds = tuple(seed for seed in COHORT_SEEDS if seed not in RECOVERY_SEEDS)
    if tuple(manifest_data.get("reuse_seeds", ())) != expected_reuse_seeds:
        raise ValueError("manifest reuse set does not match the fixed eleven seeds")
    replays = _mapping(_required(manifest_data, "replays", "manifest"), "manifest.replays")
    if set(replays) != {str(seed) for seed in COHORT_SEEDS}:
        raise ValueError("manifest replays must contain exactly the fixed 16 seeds")
    for seed in COHORT_SEEDS:
        replay = _mapping(replays[str(seed)], f"manifest.replays[{seed}]")
        if replay.get("seed", seed) != seed:
            raise ValueError(f"manifest replay seed mismatch for {seed}")
        f64 = _mapping(replay.get("F64"), f"manifest.replays[{seed}].F64")
        u64 = _mapping(replay.get("U64"), f"manifest.replays[{seed}].U64")
        cast_identity = _mapping(
            replay.get("original_float32_identity"),
            f"manifest.replays[{seed}].original_float32_identity",
        )
        same_identity = (
            f64.get("candidate_hash") is not None
            and f64.get("candidate_hash") == u64.get("candidate_hash")
        )
        if replay.get("format") != FORMAT or replay.get("same_float64_candidate") is not same_identity:
            raise ValueError(f"manifest replay candidate identity is invalid for seed {seed}")
        if cast_identity.get("fallback_applied") is not False:
            raise ValueError(f"manifest replay fallback status is invalid for seed {seed}")
        expected_reuse = same_identity and not cast_identity["fallback_applied"]
        if replay.get("reuse_eligible") is not expected_reuse:
            raise ValueError(f"manifest replay reuse flag is invalid for seed {seed}")
        if replay.get("recovery_required") is not (not expected_reuse):
            raise ValueError(f"manifest replay classification is invalid for seed {seed}")
        if expected_reuse is not (seed not in RECOVERY_SEEDS):
            raise ValueError(f"manifest replay reuse flag is invalid for seed {seed}")

    bindings = _json_object(reuse_bindings, "reuse_bindings")
    if bindings.get("format") != "h036_acceptance_bindings_v1":
        raise ValueError("reuse_bindings format must be h036_acceptance_bindings_v1")
    expected_source = _sha256(
        manifest_data.get("v1_source_identity_sha256"), "manifest.v1_source_identity_sha256"
    )
    source_identity = _sha256(
        _required(bindings, "v1_source_identity_sha256", "reuse_bindings"),
        "reuse_bindings.v1_source_identity_sha256",
    )
    if source_identity != expected_source:
        raise ValueError("reuse_bindings v1 source identity differs from the accepted config")
    run002 = _validate_reference(
        _required(bindings, "run002_provenance", "reuse_bindings"),
        "reuse_bindings.run002_provenance",
        require_path=True,
    )
    if "run002" not in (run002["key"] + " " + run002["path"]).lower():
        raise ValueError("run002_provenance must identify run002")
    foundation = _validate_reference(
        _required(bindings, "foundation", "reuse_bindings"),
        "reuse_bindings.foundation",
        require_key=False,
    )
    run002_mirror = _validate_reference(
        _required(bindings, "run002_mirror_manifest", "reuse_bindings"),
        "reuse_bindings.run002_mirror_manifest",
        require_path=True,
    )
    v1_manifest = _validate_reference(
        _required(bindings, "v1_manifest", "reuse_bindings"),
        "reuse_bindings.v1_manifest",
        require_path=True,
    )
    original_refs_raw = _required(bindings, "original_scientific_files", "reuse_bindings")
    if not isinstance(original_refs_raw, Sequence) or isinstance(original_refs_raw, (str, bytes)):
        raise TypeError("reuse_bindings.original_scientific_files must be a sequence")
    original_refs = [
        _validate_reference(
            ref, f"reuse_bindings.original_scientific_files[{index}]",
            require_key=False, require_path=True,
        )
        for index, ref in enumerate(original_refs_raw)
    ]
    if len({ref["path"] for ref in original_refs}) != len(original_refs):
        raise ValueError("original_scientific_files paths must be unique")

    raw_records = _mapping(_required(bindings, "records", "reuse_bindings"), "reuse_bindings.records")
    expected_record_keys = {str(seed) for seed in COHORT_SEEDS}
    if set(raw_records) != expected_record_keys:
        raise ValueError("reuse_bindings.records must contain the exact 16 string seed keys")
    normalized_binding_records: dict[int, dict[str, Any]] = {}
    for seed in COHORT_SEEDS:
        normalized_binding_records[seed] = _validate_binding_record(
            raw_records[str(seed)], seed, replays[str(seed)], foundation,
        )

    recovery_locks = _normalize_seed_records(recovered_locks)
    if set(recovery_locks) != set(RECOVERY_SEEDS):
        missing = sorted(set(RECOVERY_SEEDS) - set(recovery_locks))
        extra = sorted(set(recovery_locks) - set(RECOVERY_SEEDS))
        raise ValueError(f"recovered_locks must contain exactly the five recovery seeds; missing={missing}, extra={extra}")

    cases: dict[str, dict[str, Any]] = {}
    for seed in COHORT_SEEDS:
        replay = replays[str(seed)]
        binding_record = normalized_binding_records[seed]
        if seed in RECOVERY_SEEDS:
            case = _recovery_case(
                seed, replay, binding_record, recovery_locks[seed], foundation, source_identity
            )
        else:
            case = _reuse_case(seed, replay, binding_record, foundation, source_identity)
        cases[str(seed)] = case

    body: dict[str, Any] = {
        "format": "h036_acceptance_cohort_v1",
        "seeds": list(COHORT_SEEDS),
        "recovery_seeds": list(RECOVERY_SEEDS),
        "cases": cases,
        "manifest_sha256": manifest_data["manifest_sha256"],
        "bindings_sha256": hashlib.sha256(_canonical_bytes(bindings)).hexdigest(),
        "v1_source_identity_sha256": source_identity,
        "run002_provenance": run002,
        "run002_mirror_manifest": run002_mirror,
        "v1_manifest": v1_manifest,
        "foundation": foundation,
        "original_scientific_files": original_refs,
    }
    body["cohort_lock_sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    return body


def make_anchor_permit(cohort: Mapping[str, Any]) -> dict[str, Any]:
    """Create a digest-protected permit that allows only recovered-seed V anchors."""
    frozen_cohort = _json_object(cohort, "cohort")
    _verify_cohort_lock(frozen_cohort)
    body = {
        "format": "h036_acceptance_stage_permit_v1",
        "stage": "anchor",
        "cohort": frozen_cohort,
    }
    body["permit_sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    return body


def _make_counterfactual_permit(
    cohort: Mapping[str, Any], anchors_pass_token: Mapping[str, Any]
) -> dict[str, Any]:
    """Seal the counterfactual-stage permit after all five V anchors pass.

    The token must contain one exact metrics/actor identity for each recovery
    seed and be bound to this cohort lock digest. The scoring worker performs
    the scientific metric comparison before calling this helper.
    """
    frozen_cohort = _json_object(cohort, "cohort")
    _verify_cohort_lock(frozen_cohort)
    token = _validate_anchors_pass_token(anchors_pass_token, frozen_cohort)
    body = {
        "format": "h036_acceptance_stage_permit_v1",
        "stage": "counterfactual",
        "cohort": frozen_cohort,
        "anchors_pass_token": token,
    }
    body["permit_sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    return body


def confirm_anchors(
    cohort: Mapping[str, Any],
    anchor_scores: Mapping[Any, Any],
    expected_metrics_bytes: Mapping[Any, bytes],
    reuse_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact V-anchor reproduction before issuing the UG/UR permit.

    For each of the five recovered prefixes, this verifies the old metrics file
    bytes against the bound D artifact, compares every pre-codec, encoded, and
    W0 train/holdout metric exactly, and verifies the raw 89-byte actor identity
    and package-capacity constraints. CPU, wall time, memory, and other resource
    fields are intentionally ignored. The returned permit records an
    all-five pass token and is an ordering barrier, not resource authorization.
    """
    frozen_cohort = _json_object(cohort, "cohort")
    _verify_cohort_lock(frozen_cohort)
    bindings = _json_object(reuse_bindings, "reuse_bindings")
    expected_bindings_sha = _sha256(
        frozen_cohort.get("bindings_sha256"), "cohort.bindings_sha256"
    )
    if hashlib.sha256(_canonical_bytes(bindings)).hexdigest() != expected_bindings_sha:
        raise ReplayMismatchError("reuse_bindings differ from the binding snapshot in the cohort lock")

    scores = _normalize_score_records(anchor_scores)
    old_metrics = _normalize_bytes_records(expected_metrics_bytes, "expected_metrics_bytes")
    if set(scores) != set(RECOVERY_SEEDS) or set(old_metrics) != set(RECOVERY_SEEDS):
        raise ValueError("anchor scores and expected metrics bytes must contain exactly the five recovery seeds")

    records: dict[str, dict[str, str]] = {}
    cases = _mapping(frozen_cohort["cases"], "cohort.cases")
    for seed in RECOVERY_SEEDS:
        case = _mapping(cases[str(seed)], f"cohort.cases[{seed}]")
        arms = _mapping(case.get("arms"), f"cohort.cases[{seed}].arms")
        d_arm = _mapping(_required(arms, "D", f"cohort.cases[{seed}].arms"), f"cohort.cases[{seed}].arms.D")
        d_artifacts = _mapping(d_arm.get("artifacts"), f"cohort.cases[{seed}].arms.D.artifacts")
        metrics_ref = _mapping(
            _required(d_artifacts, "metrics", f"cohort.cases[{seed}].arms.D.artifacts"),
            f"cohort.cases[{seed}].arms.D.artifacts.metrics",
        )
        actor_ref = _mapping(
            _required(d_artifacts, "actor", f"cohort.cases[{seed}].arms.D.artifacts"),
            f"cohort.cases[{seed}].arms.D.artifacts.actor",
        )
        original_metrics_bytes = old_metrics[seed]
        expected_metrics_sha = hashlib.sha256(original_metrics_bytes).hexdigest()
        if (
            len(original_metrics_bytes) != metrics_ref.get("bytes")
            or expected_metrics_sha != metrics_ref.get("sha256")
        ):
            raise ReplayMismatchError(f"seed {seed} original metrics bytes differ from bound D artifact")
        if actor_ref.get("bytes") != 89:
            raise ReplayMismatchError(f"seed {seed} bound D actor is not exactly 89 bytes")
        expected_metrics = _strict_json_bytes(original_metrics_bytes, f"seed {seed} original metrics")

        score = _mapping(scores[seed], f"anchor_scores[{seed}]")
        actual_metrics = _json_object(
            _required(score, "metrics", f"anchor_scores[{seed}]"),
            f"anchor_scores[{seed}].metrics",
        )
        deployment = _required(score, "deployment", f"anchor_scores[{seed}]")
        if isinstance(deployment, bytes):
            actor_bytes = deployment
        else:
            deployment_map = _mapping(deployment, f"anchor_scores[{seed}].deployment")
            actor_bytes = _required(deployment_map, "actor_bytes", f"anchor_scores[{seed}].deployment")
            if not isinstance(actor_bytes, bytes):
                raise TypeError(f"anchor_scores[{seed}].deployment.actor_bytes must be raw bytes")
        actual_actor_sha = hashlib.sha256(actor_bytes).hexdigest()
        if len(actor_bytes) != 89:
            raise ReplayMismatchError(f"seed {seed} V anchor actor is not exactly 89 bytes")

        locked_probes = _mapping(case.get("probes"), f"cohort.cases[{seed}].probes")
        v_probe = _mapping(_required(locked_probes, "V", f"cohort.cases[{seed}].probes"),
                           f"cohort.cases[{seed}].probes.V")
        expected_candidate_sha = _sha256(v_probe.get("candidate_sha256"), f"cohort.cases[{seed}].probes.V.candidate_sha256")
        _validate_anchor_metrics(
            seed=seed,
            metrics=expected_metrics,
            actor_sha=actor_ref.get("sha256"),
            candidate_sha=expected_candidate_sha,
            prepared_id=case.get("prepared_id"),
            foundation_bytes=_mapping(frozen_cohort.get("foundation"), "cohort.foundation").get("bytes"),
            name=f"seed {seed} original D metrics",
        )
        _validate_anchor_metrics(
            seed=seed,
            metrics=actual_metrics,
            actor_sha=actual_actor_sha,
            candidate_sha=expected_candidate_sha,
            prepared_id=case.get("prepared_id"),
            foundation_bytes=_mapping(frozen_cohort.get("foundation"), "cohort.foundation").get("bytes"),
            name=f"seed {seed} new V metrics",
        )
        if actual_actor_sha != actor_ref.get("sha256"):
            raise ReplayMismatchError(f"seed {seed} V actor raw bytes differ from original D actor")
        for phase in ("pre_codec_fresh", "encoded_fresh", "W0_fresh"):
            expected_phase = _json_object(
                _mapping(_required(expected_metrics, phase, f"seed {seed} original metrics"), phase),
                f"seed {seed} original metrics.{phase}",
            )
            actual_phase = _json_object(
                _mapping(_required(actual_metrics, phase, f"seed {seed} new metrics"), phase),
                f"seed {seed} new metrics.{phase}",
            )
            if _canonical_bytes(expected_phase) != _canonical_bytes(actual_phase):
                raise ReplayMismatchError(
                    f"seed {seed} V {phase} train/holdout values differ from original D"
                )
        for key in ("G_new", "K_new", "K_new_W0", "K_old"):
            if _canonical_bytes(_required(expected_metrics, key, f"seed {seed} original metrics")) != _canonical_bytes(
                _required(actual_metrics, key, f"seed {seed} new metrics")
            ):
                raise ReplayMismatchError(f"seed {seed} V summary training field {key} differs")

        records[str(seed)] = {
            "expected_metrics_sha": expected_metrics_sha,
            "actual_metrics_canonical_sha": hashlib.sha256(_canonical_bytes(actual_metrics)).hexdigest(),
            "actor_sha": actual_actor_sha,
        }

    pass_token: dict[str, Any] = {
        "cohort_lock_sha256": frozen_cohort["cohort_lock_sha256"],
        "records": records,
    }
    pass_token["token_sha256"] = hashlib.sha256(_canonical_bytes(pass_token)).hexdigest()
    return _make_counterfactual_permit(frozen_cohort, pass_token)


def validate_stage_permit(
    permit: Mapping[str, Any], locked: Mapping[str, Any], label: str
) -> None:
    """Validate a stage permit and the exact locked case before scoring.

    Anchor permits allow only V for the five recovery seeds. Counterfactual
    permits allow only UG/UR for those seeds and require an all-five anchor
    pass token. This is ordering evidence only, not resource authorization.
    """
    stage_permit = _json_object(permit, "stage_permit")
    _verify_digest_field(stage_permit, "permit_sha256", "stage_permit")
    if stage_permit.get("format") != "h036_acceptance_stage_permit_v1":
        raise ValueError("stage permit format is invalid")
    cohort = _json_object(_required(stage_permit, "cohort", "stage_permit"), "stage_permit.cohort")
    _verify_cohort_lock(cohort)
    actual_lock = _json_object(locked, "locked identity")
    seed = _integer(_required(actual_lock, "seed", "locked identity"), "locked identity.seed")
    expected_case = _mapping(cohort["cases"], "cohort.cases").get(str(seed))
    if expected_case is None:
        raise ReplayMismatchError(f"seed {seed} is not present in the cohort lock")
    expected_case = _mapping(expected_case, f"cohort.cases[{seed}]")
    for identity_field in ("prepared_id", "path_sha256", "lock_sha256", "logical_case", "binding"):
        got = _required(actual_lock, identity_field, "locked identity")
        if got != expected_case[identity_field]:
            raise ReplayMismatchError(
                f"locked identity seed {seed} {identity_field} differs from cohort lock"
            )
    if label not in ("V", "UG", "UR"):
        raise ValueError("label must be V, UG, or UR")
    stage = stage_permit.get("stage")
    if seed not in RECOVERY_SEEDS:
        raise ValueError("stage permits do not authorize rescoring of reused prefixes")
    if stage == "anchor":
        if label != "V":
            raise ValueError("anchor permit authorizes only V probes")
    elif stage == "counterfactual":
        if label not in ("UG", "UR"):
            raise ValueError("counterfactual permit authorizes only UG/UR probes")
        _validate_anchors_pass_token(
            _required(stage_permit, "anchors_pass_token", "stage_permit"), cohort
        )
    else:
        raise ValueError("stage must be anchor or counterfactual")
    actual_probes = actual_lock.get("probes")
    if isinstance(actual_probes, Mapping):
        actual_probe_value = _required(actual_probes, label, "locked identity.probes")
    elif isinstance(actual_lock.get("probe"), Mapping):
        actual_probe_value = actual_lock["probe"]
    else:
        actual_probe_value = actual_lock
    actual_probe = _flatten_locked_probe(actual_probe_value, f"locked identity.{label}")
    expected_probes = _mapping(expected_case["probes"], f"cohort.cases[{seed}].probes")
    expected_probe = _mapping(expected_probes[label], f"cohort.cases[{seed}].probes.{label}")
    for key in (
        "selector", "origin_index", "float64_candidate_sha256", "candidate_origin_index",
        "raw_cast_candidate_sha256", "raw_cast_sha256", "post_cast_gate_enabled",
        "post_cast_gate_pass", "fallback_applied", "candidate_sha256", "raw_sha256",
    ):
        if key not in actual_probe or actual_probe[key] != expected_probe.get(key):
            raise ReplayMismatchError(f"locked seed {seed} {label}.{key} differs from cohort lock")
    if "label" in actual_lock and actual_lock["label"] != label:
        raise ReplayMismatchError(f"locked identity label does not match requested {label}")


def _flatten_locked_probe(value: Any, name: str) -> dict[str, Any]:
    probe = _json_object(_mapping(value, name), name)
    source = probe.pop("source", None)
    if source is not None:
        source_map = _json_object(_mapping(source, f"{name}.source"), f"{name}.source")
        merged = dict(source_map)
        merged.update(probe)
        probe = merged
    return probe


def _json_value(value: Any, name: str) -> Any:
    """Copy a JSON-safe tree, rejecting arrays, custom objects, and nonfinite floats."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a nonfinite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} has a non-string object key")
            result[key] = _json_value(item, f"{name}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item, f"{name}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{name} is not JSON-safe (got {type(value).__name__})")


def _json_object(value: Any, name: str) -> dict[str, Any]:
    copied = _json_value(_mapping(value, name), name)
    return copied


def _normalize_score_records(records: Mapping[Any, Any]) -> dict[int, Mapping[str, Any]]:
    scores = _mapping(records, "anchor_scores")
    normalized: dict[int, Mapping[str, Any]] = {}
    for raw_seed, score in scores.items():
        seed = _seed(raw_seed, "anchor_scores key")
        if seed in normalized:
            raise ValueError(f"duplicate anchor score seed after normalization: {seed}")
        score_map = _mapping(score, f"anchor_scores[{seed}]")
        metrics = _mapping(_required(score_map, "metrics", f"anchor_scores[{seed}]"),
                           f"anchor_scores[{seed}].metrics")
        metrics_seed = _integer(_required(metrics, "seed", f"anchor_scores[{seed}].metrics"),
                                f"anchor_scores[{seed}].metrics.seed")
        if metrics_seed != seed:
            raise ValueError(f"anchor score metrics seed {metrics_seed} does not match key {seed}")
        normalized[seed] = score_map
    return normalized


def _normalize_bytes_records(records: Mapping[Any, Any], name: str) -> dict[int, bytes]:
    values = _mapping(records, name)
    normalized: dict[int, bytes] = {}
    for raw_seed, value in values.items():
        seed = _seed(raw_seed, f"{name} key")
        if seed in normalized:
            raise ValueError(f"duplicate {name} seed after normalization: {seed}")
        if not isinstance(value, bytes):
            raise TypeError(f"{name}[{seed}] must be bytes")
        normalized[seed] = value
    return normalized


def _strict_json_bytes(value: bytes, name: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"{name} contains non-standard JSON constant {token}")

    parsed = json.loads(value.decode("utf-8"), parse_constant=reject_constant)
    return _json_object(parsed, name)


def _validate_anchor_metrics(
    *,
    seed: int,
    metrics: Mapping[str, Any],
    actor_sha: Any,
    candidate_sha: Any,
    prepared_id: Any,
    foundation_bytes: Any,
    name: str,
) -> None:
    if metrics.get("format") != "h036_score_metrics_v1":
        raise ReplayMismatchError(f"{name}.format is not h036_score_metrics_v1")
    if _integer(_required(metrics, "seed", name), f"{name}.seed") != seed:
        raise ReplayMismatchError(f"{name}.seed differs")
    if metrics.get("prepared_id") != prepared_id:
        raise ReplayMismatchError(f"{name}.prepared_id differs")
    if metrics.get("binding") != "wildcard":
        raise ReplayMismatchError(f"{name}.binding is not wildcard")
    if _number(_required(metrics, "lambda_value", name), f"{name}.lambda_value") != 1.0:
        raise ReplayMismatchError(f"{name}.lambda_value is not 1")
    if _sha256(_required(metrics, "candidate_sha256", name), f"{name}.candidate_sha256") != _sha256(
        candidate_sha, f"{name} locked candidate_sha256"
    ):
        raise ReplayMismatchError(f"{name}.candidate_sha256 differs from locked V")
    for flag in ("fit_result_unchanged", "prepared_unchanged", "fresh_local_is_zero"):
        if _required(metrics, flag, name) is not True:
            raise ReplayMismatchError(f"{name}.{flag} is not true")
    artifacts = _mapping(_required(metrics, "artifacts", name), f"{name}.artifacts")
    if _sha256(_required(artifacts, "actor_sha256", f"{name}.artifacts"),
               f"{name}.artifacts.actor_sha256") != _sha256(actor_sha, f"{name} actor identity"):
        raise ReplayMismatchError(f"{name}.artifacts.actor_sha256 differs")
    capacity = _mapping(_required(metrics, "capacity", name), f"{name}.capacity")
    actor_bytes = _integer(_required(capacity, "actor_bytes", f"{name}.capacity"),
                           f"{name}.capacity.actor_bytes")
    budget_bytes = _integer(_required(capacity, "package_budget_bytes", f"{name}.capacity"),
                            f"{name}.capacity.package_budget_bytes")
    full_bytes = _integer(_required(capacity, "full_persistent_bytes", f"{name}.capacity"),
                          f"{name}.capacity.full_persistent_bytes")
    if actor_bytes != 89 or budget_bytes != 262144 or not actor_bytes <= full_bytes <= budget_bytes:
        raise ReplayMismatchError(f"{name}.capacity violates the accepted 89 B / 262144 B limits")
    if foundation_bytes is not None:
        if _integer(_required(capacity, "foundation_bytes", f"{name}.capacity"), f"{name}.capacity.foundation_bytes") != _integer(
            foundation_bytes, "cohort.foundation.bytes"
        ):
            raise ReplayMismatchError(f"{name}.capacity.foundation_bytes differs from the bound foundation")


def _verify_digest_field(value: Mapping[str, Any], field: str, name: str) -> None:
    digest = _sha256(_required(value, field, name), f"{name}.{field}")
    body = dict(value)
    del body[field]
    expected = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    if digest != expected:
        raise ReplayMismatchError(f"{name}.{field} does not match its canonical content")


def _verify_cohort_lock(cohort: Mapping[str, Any]) -> None:
    _verify_digest_field(cohort, "cohort_lock_sha256", "cohort")
    if cohort.get("format") != "h036_acceptance_cohort_v1":
        raise ValueError("cohort format is invalid")
    if tuple(cohort.get("seeds", ())) != COHORT_SEEDS:
        raise ValueError("cohort seed order is invalid")
    if tuple(cohort.get("recovery_seeds", ())) != RECOVERY_SEEDS:
        raise ValueError("cohort recovery seed order is invalid")
    cases = _mapping(_required(cohort, "cases", "cohort"), "cohort.cases")
    if set(cases) != {str(seed) for seed in COHORT_SEEDS}:
        raise ValueError("cohort cases do not contain the fixed 16 prefixes")
    _sha256(cohort.get("manifest_sha256"), "cohort.manifest_sha256")
    _sha256(cohort.get("bindings_sha256"), "cohort.bindings_sha256")
    for seed in COHORT_SEEDS:
        case = _mapping(cases[str(seed)], f"cohort.cases[{seed}]")
        if _integer(case.get("seed"), f"cohort.cases[{seed}].seed") != seed:
            raise ValueError(f"cohort case seed mismatch for {seed}")
        if (seed in RECOVERY_SEEDS) == bool(case.get("reuse")):
            raise ValueError(f"cohort case reuse classification mismatch for {seed}")
        if case.get("reuse"):
            reusable_case = dict(case)
            reusable_case.pop("lock_sha256", None)
            if _sha256(case.get("lock_sha256"), f"cohort.cases[{seed}].lock_sha256") != hashlib.sha256(
                _canonical_bytes(reusable_case)
            ).hexdigest():
                raise ReplayMismatchError(f"reused case {seed} lock digest differs from its bindings")
        _sha256(case.get("path_sha256"), f"cohort.cases[{seed}].path_sha256")
        _sha256(case.get("lock_sha256"), f"cohort.cases[{seed}].lock_sha256")
        if not isinstance(case.get("prepared_id"), str) or not case["prepared_id"]:
            raise ValueError(f"cohort case {seed} has no prepared_id")
        probes = _mapping(case.get("probes"), f"cohort.cases[{seed}].probes")
        if set(probes) != {"V", "UG", "UR"}:
            raise ValueError(f"cohort case {seed} does not contain all three probe labels")
        for label, probe in probes.items():
            probe_map = _mapping(probe, f"cohort.cases[{seed}].probes.{label}")
            _sha256(probe_map.get("candidate_sha256"), f"cohort.cases[{seed}].probes.{label}.candidate_sha256")
            _sha256(probe_map.get("raw_sha256"), f"cohort.cases[{seed}].probes.{label}.raw_sha256")


def _validate_anchors_pass_token(token_value: Any, cohort: Mapping[str, Any]) -> dict[str, Any]:
    token = _json_object(token_value, "anchors_pass_token")
    _verify_digest_field(token, "token_sha256", "anchors_pass_token")
    expected_cohort_hash = _sha256(cohort.get("cohort_lock_sha256"), "cohort.cohort_lock_sha256")
    if _sha256(token.get("cohort_lock_sha256"), "anchors_pass_token.cohort_lock_sha256") != expected_cohort_hash:
        raise ReplayMismatchError("anchor pass token belongs to a different cohort lock")
    records = _mapping(_required(token, "records", "anchors_pass_token"), "anchors_pass_token.records")
    if set(records) != {str(seed) for seed in RECOVERY_SEEDS}:
        raise ValueError("anchor pass token must contain exactly all five recovery seeds")
    for seed in RECOVERY_SEEDS:
        record = _mapping(records[str(seed)], f"anchors_pass_token.records[{seed}]")
        for key in ("expected_metrics_sha", "actual_metrics_canonical_sha", "actor_sha"):
            _sha256(_required(record, key, f"anchors_pass_token.records[{seed}]"),
                    f"anchors_pass_token.records[{seed}].{key}")
    return token


def _validate_reference(
    reference: Any,
    name: str,
    *,
    require_path: bool = False,
    require_key: bool = True,
) -> dict[str, Any]:
    ref = _json_object(_mapping(reference, name), name)
    if require_key or "key" in ref:
        key = _required(ref, "key", name)
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name}.key must be a non-empty string")
    if require_path or "path" in ref:
        path = _required(ref, "path", name)
        if not isinstance(path, str) or not path:
            raise ValueError(f"{name}.path must be a non-empty string")
    if "remote_path" in ref and (not isinstance(ref["remote_path"], str) or not ref["remote_path"]):
        raise ValueError(f"{name}.remote_path must be a non-empty string")
    size = _integer(_required(ref, "bytes", name), f"{name}.bytes")
    if size < 0:
        raise ValueError(f"{name}.bytes must be nonnegative")
    ref["sha256"] = _sha256(_required(ref, "sha256", name), f"{name}.sha256")
    return ref


def _validate_binding_record(
    value: Any, seed: int, replay: Mapping[str, Any], foundation_identity: Mapping[str, Any]
) -> dict[str, Any]:
    name = f"reuse_bindings.records[{seed}]"
    record = _json_object(_mapping(value, name), name)
    if _integer(_required(record, "seed", name), f"{name}.seed") != seed:
        raise ValueError(f"{name}.seed does not match its key")
    logical_case = _required(record, "logical_case", name)
    if not isinstance(logical_case, str) or not logical_case:
        raise ValueError(f"{name}.logical_case must be a non-empty string")
    prepared_id = _required(record, "prepared_id", name)
    if not isinstance(prepared_id, str) or not prepared_id:
        raise ValueError(f"{name}.prepared_id must be a non-empty string")
    if record.get("binding") != "wildcard":
        raise ValueError(f"{name}.binding must be wildcard")
    record_foundation = _validate_reference(
        _required(record, "foundation", name), f"{name}.foundation", require_key=False
    )
    if (
        record_foundation["bytes"] != foundation_identity["bytes"]
        or record_foundation["sha256"] != foundation_identity["sha256"]
    ):
        raise ReplayMismatchError(f"{name}.foundation differs from cohort foundation")
    preparation_reference = _validate_reference(
        _required(record, "preparation_reference", name), f"{name}.preparation_reference",
        require_path=True,
    )
    raw_inputs = _mapping(_required(record, "inputs", name), f"{name}.inputs")
    if len(raw_inputs) != 3:
        raise ValueError(f"{name}.inputs must bind exactly the three scientific input files")
    inputs: dict[str, Any] = {}
    for basename, ref in raw_inputs.items():
        if not isinstance(basename, str) or not basename:
            raise ValueError(f"{name}.inputs has an invalid basename")
        inputs[basename] = _validate_reference(ref, f"{name}.inputs.{basename}")
    arms_raw = _mapping(_required(record, "arms", name), f"{name}.arms")
    if set(arms_raw) != {"C", "D"}:
        raise ValueError(f"{name}.arms must contain exactly C and D")
    arms: dict[str, Any] = {}
    required_artifacts = {
        "candidate_f32", "candidate_json", "metrics", "diagnostics", "actor",
        "checkpoint", "persistent_references",
    }
    for arm_name, expected_lambda in (("C", 0.0), ("D", 1.0)):
        arm = _json_object(_mapping(arms_raw[arm_name], f"{name}.arms.{arm_name}"),
                           f"{name}.arms.{arm_name}")
        if _number(_required(arm, "lambda_value", f"{name}.arms.{arm_name}"),
                   f"{name}.arms.{arm_name}.lambda_value") != expected_lambda:
            raise ValueError(f"{name}.arms.{arm_name} lambda value is invalid")
        if arm.get("prepared_id") != prepared_id or arm.get("binding") != "wildcard":
            raise ValueError(f"{name}.arms.{arm_name} prepared/binding identity differs")
        for field in (
            "helper_candidate_sha256", "raw_candidate_sha256", "selected32_helper_sha256",
            "best64_helper_sha256",
        ):
            arm[field] = _sha256(
                _required(arm, field, f"{name}.arms.{arm_name}"),
                f"{name}.arms.{arm_name}.{field}",
            )
        fallback = _required(arm, "fallback_applied", f"{name}.arms.{arm_name}")
        if not isinstance(fallback, bool):
            raise TypeError(f"{name}.arms.{arm_name}.fallback_applied must be bool")
        artifacts_raw = _mapping(
            _required(arm, "artifacts", f"{name}.arms.{arm_name}"),
            f"{name}.arms.{arm_name}.artifacts",
        )
        if set(artifacts_raw) != required_artifacts:
            raise ValueError(f"{name}.arms.{arm_name}.artifacts does not bind all required files")
        artifacts = {
            artifact_name: _validate_reference(
                artifact_ref, f"{name}.arms.{arm_name}.artifacts.{artifact_name}"
            )
            for artifact_name, artifact_ref in artifacts_raw.items()
        }
        if arm["raw_candidate_sha256"] != artifacts["candidate_f32"]["sha256"]:
            raise ValueError(f"{name}.arms.{arm_name} raw candidate SHA differs from candidate_f32 bytes")
        arm["fallback_applied"] = fallback
        arm["artifacts"] = artifacts
        arms[arm_name] = arm

    expected_f64 = _mapping(replay.get("F64"), f"manifest.replays[{seed}].F64")
    expected_f32 = _mapping(
        replay.get("original_float32_identity"),
        f"manifest.replays[{seed}].original_float32_identity",
    )
    d_arm = arms["D"]
    if d_arm["best64_helper_sha256"] != expected_f64.get("candidate_hash"):
        raise ReplayMismatchError(f"seed {seed} original D best64 hash differs from replay F64")
    if d_arm["helper_candidate_sha256"] != expected_f32.get("candidate_sha256"):
        raise ReplayMismatchError(f"seed {seed} original D helper candidate differs from replay")
    if d_arm["selected32_helper_sha256"] != expected_f32.get("selected_float32_candidate_sha256"):
        raise ReplayMismatchError(f"seed {seed} original D selected float32 hash differs from replay")
    if d_arm["fallback_applied"] is not expected_f32.get("fallback_applied"):
        raise ReplayMismatchError(f"seed {seed} original D fallback flag differs from replay")
    if d_arm["fallback_applied"]:
        raise ValueError(f"seed {seed} original D fallback prevents cohort reuse/replay")
    return {
        "seed": seed,
        "logical_case": logical_case,
        "prepared_id": prepared_id,
        "binding": "wildcard",
        "foundation": record_foundation,
        "preparation_reference": preparation_reference,
        "inputs": inputs,
        "arms": arms,
    }


def _probe_source(
    value: Any,
    *,
    seed: int,
    label: str,
    replay: Mapping[str, Any],
    probe: Mapping[str, Any],
    original_d: Mapping[str, Any] | None,
) -> dict[str, Any]:
    name = f"seed {seed} probe {label}"
    raw_source = _json_object(_mapping(value, name), name)
    source = raw_source.get("source", raw_source)
    source = _json_object(_mapping(source, f"{name}.source"), f"{name}.source")
    selector = "F64" if label == "V" else "U64"
    source_record = _mapping(replay[selector], f"replay.{selector}")
    if source.get("selector") != selector:
        raise ReplayMismatchError(f"{name}.selector must be {selector}")
    if _integer(_required(source, "origin_index", name), f"{name}.origin_index") != source_record["index"]:
        raise ReplayMismatchError(f"{name}.origin_index differs from replay {selector}")
    if _sha256(_required(source, "float64_candidate_sha256", name), f"{name}.float64_candidate_sha256") != source_record["candidate_hash"]:
        raise ReplayMismatchError(f"{name}.float64_candidate_sha256 differs from replay {selector}")
    origin_index = source_record["index"]
    candidate_origin = _integer(
        _required(source, "candidate_origin_index", name), f"{name}.candidate_origin_index"
    )
    gate_enabled = _required(source, "post_cast_gate_enabled", name)
    gate_pass = _required(source, "post_cast_gate_pass", name)
    fallback = _required(source, "fallback_applied", name)
    if not all(isinstance(flag, bool) for flag in (gate_enabled, gate_pass, fallback)):
        raise TypeError(f"{name} gate and fallback fields must be bool")
    expected_gate_enabled = label != "UR"
    if gate_enabled is not expected_gate_enabled:
        raise ReplayMismatchError(f"{name}.post_cast_gate_enabled differs from protocol")
    if gate_enabled and fallback is not (not gate_pass):
        raise ReplayMismatchError(f"{name} fallback flag disagrees with post-cast gate result")
    if not gate_enabled and fallback:
        raise ReplayMismatchError(f"{name} UR must not apply post-cast fallback")
    expected_origin = 0 if fallback else origin_index
    if candidate_origin != expected_origin:
        raise ReplayMismatchError(f"{name}.candidate_origin_index is inconsistent with fallback")
    source["raw_cast_candidate_sha256"] = _sha256(
        _required(source, "raw_cast_candidate_sha256", name), f"{name}.raw_cast_candidate_sha256"
    )
    source["raw_cast_sha256"] = _sha256(
        _required(source, "raw_cast_sha256", name), f"{name}.raw_cast_sha256"
    )
    candidate_hash = _sha256(
        _required(probe, "candidate_sha256", f"seed {seed} probe {label}"),
        f"seed {seed} probe {label}.candidate_sha256",
    )
    raw_hash = _sha256(
        _required(probe, "raw_sha256", f"seed {seed} probe {label}"),
        f"seed {seed} probe {label}.raw_sha256",
    )
    if not fallback and (
        candidate_hash != source["raw_cast_candidate_sha256"]
        or raw_hash != source["raw_cast_sha256"]
    ):
        raise ReplayMismatchError(f"seed {seed} probe {label} candidate differs from its raw cast")
    if label == "V" and original_d is not None:
        expected_helper = original_d["selected32_helper_sha256"]
        expected_raw = original_d["artifacts"]["candidate_f32"]["sha256"]
        if candidate_hash != expected_helper or raw_hash != expected_raw:
            raise ReplayMismatchError(f"seed {seed} V identity differs from the original D candidate")
        if fallback or not gate_pass:
            raise ReplayMismatchError(f"seed {seed} V does not reproduce original no-fallback D behavior")
    return {
        **source,
        "candidate_sha256": candidate_hash,
        "raw_sha256": raw_hash,
    }


def _recovery_case(
    seed: int,
    replay: Mapping[str, Any],
    binding_record: Mapping[str, Any],
    lock_value: Mapping[str, Any],
    foundation: Mapping[str, Any],
    source_identity: str,
) -> dict[str, Any]:
    name = f"recovered_locks[{seed}]"
    lock = _json_object(lock_value, name)
    if _integer(_required(lock, "seed", name), f"{name}.seed") != seed:
        raise ValueError(f"{name}.seed mismatch")
    if lock.get("prepared_id") != binding_record["prepared_id"]:
        raise ReplayMismatchError(f"{name}.prepared_id differs from the original prepared input")
    path_sha = _sha256(_required(lock, "path_sha256", name), f"{name}.path_sha256")
    if path_sha != replay["path_sha256"]:
        raise ReplayMismatchError(f"{name}.path_sha256 differs from original scalar replay")
    lock_sha = _sha256(_required(lock, "lock_sha256", name), f"{name}.lock_sha256")
    raw_probes = _mapping(_required(lock, "probes", name), f"{name}.probes")
    if set(raw_probes) != {"V", "UG", "UR"}:
        raise ValueError(f"{name}.probes must contain exactly V, UG, UR")
    original_d = binding_record["arms"]["D"]
    probes: dict[str, Any] = {}
    for label in ("V", "UG", "UR"):
        raw_probe = _json_object(
            _mapping(raw_probes[label], f"{name}.probes.{label}"),
            f"{name}.probes.{label}",
        )
        probes[label] = _probe_source(
            _required(raw_probe, "source", f"{name}.probes.{label}"),
            seed=seed,
            label=label,
            replay=replay,
            probe=raw_probe,
            original_d=original_d,
        )
    return {
        "seed": seed,
        "reuse": False,
        "logical_case": binding_record["logical_case"],
        "prepared_id": lock["prepared_id"],
        "path_sha256": path_sha,
        "lock_sha256": lock_sha,
        "binding": binding_record["binding"],
        "inputs": binding_record["inputs"],
        "foundation": foundation,
        "v1_source_identity_sha256": source_identity,
        "arms": binding_record["arms"],
        "probes": probes,
    }


def _reuse_case(
    seed: int,
    replay: Mapping[str, Any],
    binding_record: Mapping[str, Any],
    foundation: Mapping[str, Any],
    source_identity: str,
) -> dict[str, Any]:
    original_d = binding_record["arms"]["D"]
    f64 = _mapping(replay["F64"], f"replay[{seed}].F64")
    u64 = _mapping(replay["U64"], f"replay[{seed}].U64")
    f32_identity = _mapping(replay["original_float32_identity"], f"replay[{seed}].original_float32_identity")
    if f64["candidate_hash"] != u64["candidate_hash"]:
        raise ValueError(f"seed {seed} is marked reusable but F64/U64 hashes differ")
    if f32_identity["fallback_applied"]:
        raise ValueError(f"seed {seed} is marked reusable but original D had fallback")
    helper_hash = original_d["selected32_helper_sha256"]
    raw_hash = original_d["artifacts"]["candidate_f32"]["sha256"]
    if helper_hash != f32_identity["selected_float32_candidate_sha256"]:
        raise ReplayMismatchError(f"seed {seed} reused D float32 helper identity differs")
    probes: dict[str, Any] = {}
    for label, selector in (("V", "F64"), ("UG", "U64"), ("UR", "U64")):
        origin = _mapping(replay[selector], f"replay[{seed}].{selector}")
        source = {
            "selector": selector,
            "origin_index": origin["index"],
            "float64_candidate_sha256": origin["candidate_hash"],
            "candidate_origin_index": origin["index"],
            "raw_cast_candidate_sha256": helper_hash,
            "raw_cast_sha256": raw_hash,
            "post_cast_gate_enabled": label != "UR",
            "post_cast_gate_pass": True,
            "fallback_applied": False,
        }
        probes[label] = {
            **source,
            "candidate_sha256": helper_hash,
            "raw_sha256": raw_hash,
            "score_ref": original_d["artifacts"]["metrics"],
        }
    case: dict[str, Any] = {
        "seed": seed,
        "reuse": True,
        "logical_case": binding_record["logical_case"],
        "prepared_id": binding_record["prepared_id"],
        "path_sha256": replay["path_sha256"],
        "binding": binding_record["binding"],
        "inputs": binding_record["inputs"],
        "foundation": foundation,
        "v1_source_identity_sha256": source_identity,
        "arms": binding_record["arms"],
        "probes": probes,
    }
    case["lock_sha256"] = hashlib.sha256(_canonical_bytes(case)).hexdigest()
    return case
