"""Frozen H036 finite, training-only budgeted candidate search.

This module reuses the H036 training-only view and v1 scoring/codec path. It
does not fit parameters, read holdout values during selection, or add a
separate C0 reference arm.
Search and float32 post-cast checks are completed before a cohort can be sealed
for scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
import numbers
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

import numpy as np

from . import retention_acceptance_diagnostic as diagnostic
from . import retention_behavioral_candidate as behavioral
from . import retention_protected_candidate as protected


TrainingView = diagnostic.TrainingView
make_training_view = diagnostic.make_training_view
_validate_training_view = diagnostic._validate_training_view

_FORMAT = "h036_budgeted_search_v1"
_ANCHOR_FORMAT = "h036_budgeted_anchor_manifest_v1"
_LOCK_FORMAT = "h036_budgeted_locked_v1"
_COHORT_FORMAT = "h036_budgeted_cohort_v1"
_ANCHOR_RECEIPT_FORMAT = "h036_budgeted_anchor_receipt_v1"
_OBJECTIVE_VERSION = "h036_L1_lambda1_ridge1e-4_v1"
_SEEDS = tuple(range(26201, 26214)) + (26215, 26216, 26218)
_SHAPE = (3, 3)
_LAMBDA = 1
_RIDGE = 1.0e-4
_KL_TOLERANCE = 1.0e-12
_WITNESS_TOLERANCE = 1.0e-8
_MAX_TRIALS = 16
_MAX_ORACLE_EVALUATIONS = 21
_MAX_SCORE_PACKAGES = 33
_ANCHOR_SEED = 26201
_ANCHOR_MANIFEST_PATH = "research/appendix/H036/delivery_budgeted_v1/anchor_manifest.json"
_ANCHOR_MANIFEST_SHA256 = "f56cfc6ba74774ea316bbe5b3941303059a74266f74a2374787d93a3ba9c73df"
_V1_SOURCE_IDENTITY_SHA256 = "0685e20c095b4c49c12d04041e3857f19b584c062dbe3eea456af1561fa661ea"
_E164_SOURCE_IDENTITY_SHA256 = "baae9a6a766518105a6d5b2e9bc64fef3488ff581870931cf83ff2d7b23a4c9a"
_E164_COHORT_MANIFEST_SHA256 = "bf9cf10cea79e299b09cf0478dfda65c28407b0fbae72a560687e4a7e2ac2469"
_BINDING = "wildcard"
_ACTOR_BUDGET_BYTES = 89
_PACKAGE_BUDGET_BYTES = 262144
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_HELPER_COUNT_KEYS = (
    "protected_objective_and_gradient",
    "behavioral_objective_and_gradient",
    "gradient_evaluations",
    "new_kl",
    "old_kl",
    "ridge",
)


@dataclass(frozen=True)
class AnchorSet:
    """One prefix's exact float32 anchors and training-only reference identity."""

    seed: int
    prepared_id: str
    input_identity: dict[str, Any]
    C: np.ndarray
    D: np.ndarray
    U32: np.ndarray
    expected_training: dict[str, Any]
    source_references: tuple[dict[str, Any], ...]
    v1_source_identity_sha256: str
    e164_source_identity_sha256: str
    e164_cohort_manifest_sha256: str
    manifest_sha256: str


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HEX_64.fullmatch(value) is not None


def _identity_json(value: Any, name: str) -> str:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a plain JSON object")
    try:
        return _json_bytes(value).decode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON data") from exc


def _array_sha256(value: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    return _sha256(canonical.tobytes(order="C"))


def _candidate_sha256(value: np.ndarray, dtype: Any) -> str:
    array = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(repr(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _copy_readonly(value: Any, dtype: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(dtype) or array.shape != shape:
        raise ValueError(f"{name} must have dtype {np.dtype(dtype)} and shape {shape}")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} must be finite")
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _f64_candidate(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64) or array.shape != _SHAPE:
        raise ValueError(f"{name} must be float64 with shape (3, 3)")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} must be finite")
    return np.ascontiguousarray(array, dtype=np.float64)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} must be finite")
    return result


def _validate_training_metrics(value: Any, name: str) -> dict[str, float]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a plain dict")
    required = {"fullL", "new_train_kl", "old_train_kl", "ridge"}
    if set(value) != required:
        raise ValueError(f"{name} must contain exactly {sorted(required)}")
    return {key: _finite_number(value[key], f"{name}.{key}") for key in sorted(required)}


def _validate_expected_training(value: Any) -> dict[str, dict[str, float]]:
    if type(value) is not dict or set(value) != {"C0", "C", "D"}:
        raise ValueError("expected_training must contain exactly C0, C, and D")
    result: dict[str, dict[str, float]] = {}
    for label in ("C0", "D"):
        result[label] = _validate_training_metrics(value[label], f"expected_training.{label}")
    c_value = value["C"]
    if type(c_value) is not dict or set(c_value) != {"new_train_kl", "old_train_kl", "ridge"}:
        raise ValueError("expected_training.C must contain new_train_kl, old_train_kl, and ridge")
    result["C"] = {
        key: _finite_number(c_value[key], f"expected_training.C.{key}")
        for key in ("new_train_kl", "old_train_kl", "ridge")
    }
    return result


def _validate_anchor_set(anchors: Any) -> AnchorSet:
    if not isinstance(anchors, AnchorSet):
        raise TypeError("anchors must be an AnchorSet returned by load_anchor_references")
    if isinstance(anchors.seed, bool) or not isinstance(anchors.seed, numbers.Integral):
        raise ValueError("anchor seed must be an integer")
    if int(anchors.seed) not in _SEEDS:
        raise ValueError("anchor seed is outside the frozen 16-prefix cohort")
    if not isinstance(anchors.prepared_id, str) or not anchors.prepared_id:
        raise ValueError("anchor prepared_id must be nonempty")
    _identity_json(anchors.input_identity, "anchor input_identity")
    for name in ("C", "D", "U32"):
        value = getattr(anchors, name)
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float32) or value.shape != _SHAPE:
            raise ValueError(f"anchor {name} must be float32 with shape (3, 3)")
        if value.flags.writeable:
            raise ValueError(f"anchor {name} must be read-only")
        if not np.isfinite(value).all():
            raise FloatingPointError(f"anchor {name} must be finite")
    if not isinstance(anchors.source_references, tuple) or not anchors.source_references:
        raise ValueError("anchor source_references must be a nonempty tuple")
    for index, ref in enumerate(anchors.source_references):
        if type(ref) is not dict:
            raise TypeError(f"anchor source_references[{index}] must be a plain dict")
        if set(ref) != {"path", "bytes", "sha256", "role"}:
            raise ValueError(f"anchor source_references[{index}] has invalid fields")
        if not isinstance(ref["path"], str) or not ref["path"]:
            raise ValueError("anchor source reference path must be nonempty")
        if isinstance(ref["bytes"], bool) or not isinstance(ref["bytes"], numbers.Integral) or ref["bytes"] < 0:
            raise ValueError("anchor source reference bytes must be a nonnegative integer")
        if not _is_sha256(ref["sha256"]) or not isinstance(ref["role"], str) or not ref["role"]:
            raise ValueError("anchor source reference identity is invalid")
    if (anchors.v1_source_identity_sha256 != _V1_SOURCE_IDENTITY_SHA256
            or anchors.e164_source_identity_sha256 != _E164_SOURCE_IDENTITY_SHA256
            or anchors.e164_cohort_manifest_sha256 != _E164_COHORT_MANIFEST_SHA256
            or anchors.manifest_sha256 != _ANCHOR_MANIFEST_SHA256):
        raise ValueError("anchor legacy source and manifest identities must be SHA-256 strings")
    _validate_expected_training(anchors.expected_training)
    return anchors


def _validate_config(config: Mapping[str, Any] | None, anchors: AnchorSet) -> dict[str, Any]:
    """Validate the fixed search contract; caller values cannot relax constants."""
    expected = {
        "format": _FORMAT,
        "seeds": list(_SEEDS),
        "alpha_schedule": [
            {"j": j, "alpha": 2.0 ** (-j)} for j in range(_MAX_TRIALS)
        ],
        "lambda_value": float(_LAMBDA),
        "ridge": _RIDGE,
        "new_kl_slack": _KL_TOLERANCE,
        "oracle_evaluations_per_case": _MAX_ORACLE_EVALUATIONS,
        "oracle_evaluations_cohort_max": _MAX_ORACLE_EVALUATIONS * len(_SEEDS),
        "score_packages_max": _MAX_SCORE_PACKAGES,
        "score_anchor_seed": _ANCHOR_SEED,
        "binding": _BINDING,
        "codec": "int8",
        "actor_bytes": _ACTOR_BUDGET_BYTES,
        "package_budget_bytes": _PACKAGE_BUDGET_BYTES,
        "witness_tol": _WITNESS_TOLERANCE,
        "v1_source_identity_sha256": anchors.v1_source_identity_sha256,
        "e164_source_identity_sha256": anchors.e164_source_identity_sha256,
        "e164_cohort_manifest_sha256": anchors.e164_cohort_manifest_sha256,
        "anchor_manifest_path": _ANCHOR_MANIFEST_PATH,
        "anchor_manifest_sha256": anchors.manifest_sha256,
    }
    if config is None:
        return copy.deepcopy(expected)
    if type(config) is not dict:
        raise TypeError("config must be a plain JSON dict")
    if set(config) != set(expected):
        raise ValueError(f"config fields must equal {sorted(expected)}")
    if _json_bytes(config) != _json_bytes(expected):
        raise ValueError("config differs from fixed h036_budgeted_search_v1 constants")
    return copy.deepcopy(expected)


def _compute_training_metrics(view: TrainingView, candidate64: np.ndarray) -> dict[str, Any]:
    """One full training-only L1 evaluation and its real helper-call counts.

    This module-level function is the narrow monkeypatch seam for remote
    synthetic validation. Production calls the frozen v1 protected helpers.
    """
    candidate = _f64_candidate(candidate64, "candidate64")
    if float(protected._RIDGE) != _RIDGE:
        raise RuntimeError("v1 protected ridge differs from the fixed budgeted-search ridge")
    full_l, gradient = protected.protected_objective_and_gradient(
        candidate, view.new_features, view.new_teacher, view.new_weights,
        view.old_features, view.old_teacher, view.old_weights, view.mean64, _LAMBDA,
    )
    if not np.isfinite(gradient).all():
        raise FloatingPointError("training objective gradient is nonfinite")
    new_kl = protected._owner_kl_diagnostics(
        view.new_features, view.new_teacher, view.new_owners, candidate,
    )[1]
    old_kl = protected._weighted_forward_kl(
        view.old_features, view.old_teacher, view.old_weights, candidate,
    )
    ridge = protected._ridge_value(candidate, view.mean64)
    metrics = {
        "fullL": _finite_number(full_l, "fullL"),
        "new_train_kl": _finite_number(new_kl, "new_train_kl"),
        "old_train_kl": _finite_number(old_kl, "old_train_kl"),
        "ridge": _finite_number(ridge, "ridge"),
    }
    # The protected full objective invokes the behavioral objective and
    # gradient once per source (new and old); the metric helpers are separate.
    return {
        **metrics,
        "helper_counts": {
            "protected_objective_and_gradient": 1,
            "behavioral_objective_and_gradient": protected._objective_call_count(_LAMBDA),
            "gradient_evaluations": protected._objective_call_count(_LAMBDA),
            "new_kl": 1,
            "old_kl": 1,
            "ridge": 1,
        },
    }


def _logical_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or chr(92) in value:
        raise ValueError(f"{name} must be a nonempty canonical POSIX path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"{name} must be relative and cannot traverse directories")
    if ":" in parsed.parts[0]:
        raise ValueError(f"{name} cannot contain a drive prefix")
    if parsed.as_posix() != value:
        raise ValueError(f"{name} must use its canonical POSIX spelling")
    return parsed.as_posix()


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink path component is not allowed: {current}")


def _read_manifest_entry(
    root: Path, logical_path: str, path_map: Mapping[str, Any], expected_bytes: int,
    expected_sha256: str,
) -> bytes:
    mapped = path_map.get(logical_path, logical_path)
    if not isinstance(mapped, (str, Path)):
        raise TypeError(f"path_map[{logical_path!r}] must be a path string")
    mapped_path = Path(mapped)
    if not mapped_path.is_absolute():
        if any(part == ".." for part in mapped_path.parts):
            raise ValueError("path_map paths cannot traverse directories")
        mapped_path = root / mapped_path
    _reject_symlink_components(mapped_path)
    resolved_root = root.resolve(strict=True)
    resolved_file = mapped_path.resolve(strict=True)
    try:
        resolved_file.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"manifest entry escapes authorized reference_root: {logical_path}") from exc
    if not resolved_file.is_file():
        raise ValueError(f"manifest entry is not a regular file: {logical_path}")
    data = resolved_file.read_bytes()
    if len(data) != expected_bytes:
        raise ValueError(f"manifest entry byte count mismatch: {logical_path}")
    if _sha256(data) != expected_sha256:
        raise ValueError(f"manifest entry SHA-256 mismatch: {logical_path}")
    return data


def _entry_lookup(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        result.setdefault(entry["path"], []).append(entry)
    return result


def _require_entry(
    lookup: Mapping[str, list[dict[str, Any]]], path: Any, allowed_roles: set[str],
    case_name: str,
) -> dict[str, Any]:
    logical = _logical_path(path, f"{case_name} path")
    candidates = lookup.get(logical)
    if not candidates:
        raise ValueError(f"{case_name} path is absent from exact manifest entries: {logical}")
    matches = [entry for entry in candidates if entry["role"] in allowed_roles]
    if not matches:
        raise ValueError(f"{case_name} path has no permitted manifest role: {logical}")
    first = matches[0]
    if any((entry["bytes"], entry["sha256"]) != (first["bytes"], first["sha256"]) for entry in matches):
        raise ValueError(f"duplicate manifest entries disagree for {logical}")
    return first


def _json_entry(
    file_bytes: Mapping[str, bytes], lookup: Mapping[str, list[dict[str, Any]]],
    path: Any, allowed_roles: set[str], case_name: str,
) -> dict[str, Any]:
    entry = _require_entry(lookup, path, allowed_roles, case_name)
    try:
        value = json.loads(file_bytes[entry["path"]].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{case_name} must be a UTF-8 JSON object") from exc
    if type(value) is not dict:
        raise ValueError(f"{case_name} must be a JSON object")
    return value


def _load_candidate(
    reference: Any, file_bytes: Mapping[str, bytes],
    lookup: Mapping[str, list[dict[str, Any]]], case_name: str, prepared_id: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if type(reference) is not dict:
        raise TypeError(f"{case_name} reference must be a plain dict")
    required = {
        "candidate_path", "metadata_path", "metrics_path", "diagnostics_path",
        "persistent_references_path", "candidate_sha256", "raw_sha256",
    }
    if set(reference) != required:
        raise ValueError(f"{case_name} reference fields must equal {sorted(required)}")
    candidate_entry = _require_entry(lookup, reference["candidate_path"], {f"{case_name}_candidate"}, case_name)
    metadata = _json_entry(
        file_bytes, lookup, reference["metadata_path"], {f"{case_name}_metadata"}, f"{case_name} metadata",
    )
    _require_entry(lookup, reference["metrics_path"], {f"{case_name}_metrics"}, f"{case_name} metrics")
    diagnostics = _json_entry(
        file_bytes, lookup, reference["diagnostics_path"], {f"{case_name}_diagnostics"},
        f"{case_name} diagnostics",
    )
    _require_entry(
        lookup, reference["persistent_references_path"], {f"{case_name}_persistent_references"},
        f"{case_name} persistent references",
    )
    raw = file_bytes[candidate_entry["path"]]
    if len(raw) != 9 * np.dtype("<f4").itemsize:
        raise ValueError(f"{case_name} candidate bytes must encode exactly nine float32 values")
    raw_digest = _sha256(raw)
    if raw_digest != reference["raw_sha256"] or raw_digest != candidate_entry["sha256"]:
        raise ValueError(f"{case_name} raw candidate SHA differs from its exact source reference")
    if not _is_sha256(reference["candidate_sha256"]):
        raise ValueError(f"{case_name} scientific candidate SHA is invalid")
    if metadata.get("scientific_candidate_sha256") != reference["candidate_sha256"]:
        raise ValueError(f"{case_name} metadata scientific candidate SHA mismatch")
    if metadata.get("raw_file_sha256") != reference["raw_sha256"]:
        raise ValueError(f"{case_name} metadata raw candidate SHA mismatch")
    if metadata.get("prepared_id") != prepared_id:
        raise ValueError(f"{case_name} candidate metadata prepared_id differs from its prefix")
    expected_lambda = 0 if case_name == "C" else 1
    if metadata.get("lambda_value") != expected_lambda:
        raise ValueError(f"{case_name} candidate metadata lambda differs from its frozen arm")
    if (metadata.get("dtype_str") != "<f4" or metadata.get("dtype") != "float32"
            or metadata.get("shape") != [3, 3] or metadata.get("order") != "C"
            or metadata.get("byteorder") != "little"):
        raise ValueError(f"{case_name} candidate metadata layout differs from the exact float32 format")
    array = np.frombuffer(raw, dtype="<f4").reshape(_SHAPE).astype(np.float32, copy=True)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{case_name} candidate contains nonfinite float32 values")
    if _candidate_sha256(array, np.float32) != reference["candidate_sha256"]:
        raise ValueError(f"{case_name} scientific candidate SHA mismatch")
    array.setflags(write=False)
    return array, diagnostics


def _validate_manifest(exact_manifest: Any) -> tuple[dict[str, Any], str]:
    if type(exact_manifest) is not dict:
        raise TypeError("exact_manifest must be a plain JSON dict")
    expected_keys = {
        "format", "seeds", "v1_source_identity_sha256", "e164_source_identity_sha256",
        "e164_cohort_manifest_sha256", "foundation_identity", "entries", "cases",
    }
    if set(exact_manifest) != expected_keys:
        raise ValueError(f"anchor manifest fields must equal {sorted(expected_keys)}")
    if exact_manifest.get("format") != _ANCHOR_FORMAT:
        raise ValueError("unsupported anchor manifest format")
    if type(exact_manifest.get("seeds")) is not list or exact_manifest["seeds"] != list(_SEEDS):
        raise ValueError("anchor manifest must contain the exact ascending 16-prefix seed list")
    expected_source_ids = {
        "v1_source_identity_sha256": _V1_SOURCE_IDENTITY_SHA256,
        "e164_source_identity_sha256": _E164_SOURCE_IDENTITY_SHA256,
        "e164_cohort_manifest_sha256": _E164_COHORT_MANIFEST_SHA256,
    }
    for key, expected in expected_source_ids.items():
        if exact_manifest.get(key) != expected:
            raise ValueError(f"anchor manifest {key} differs from the accepted exact source identity")
    try:
        canonical = _json_bytes(exact_manifest)
    except (TypeError, ValueError) as exc:
        raise ValueError("anchor manifest must contain finite JSON data") from exc
    manifest_sha = _sha256(canonical)
    if manifest_sha != _ANCHOR_MANIFEST_SHA256:
        raise ValueError("anchor manifest file identity differs from the frozen E165 manifest")
    return exact_manifest, manifest_sha


def _validate_expected_training_from_diagnostics(
    expected: Any, c_diagnostics: Mapping[str, Any], d_diagnostics: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    if type(expected) is not dict or set(expected) != {"C0", "C", "D"}:
        raise ValueError("expected_training must contain exactly C0, C, and D")
    result = _validate_expected_training(expected)
    for label, diagnostics, metrics_name in (
        ("C0", d_diagnostics, "initial_metrics"),
        ("C", c_diagnostics, "final_metrics"),
        ("D", d_diagnostics, "final_metrics"),
    ):
        metrics = diagnostics.get(metrics_name)
        if type(metrics) is not dict:
            raise ValueError(f"{label} historical diagnostics are missing {metrics_name}")
        for key in ("new_train_kl", "old_train_kl"):
            if _finite_number(metrics.get(key), f"{label}.{key}") != result[label][key]:
                raise ValueError(f"{label} expected training {key} differs from historical diagnostics")
        if label == "C0":
            if _finite_number(metrics.get("ridge"), "C0.ridge") != result["C0"]["ridge"]:
                raise ValueError("C0 expected training ridge differs from historical diagnostics")
            if _finite_number(d_diagnostics.get("initial_fullL"), "C0.fullL") != result["C0"]["fullL"]:
                raise ValueError("C0 expected fullL differs from historical diagnostics")
        elif label == "C":
            if _finite_number(c_diagnostics.get("final_float32_ridge"), "C.ridge") != result["C"]["ridge"]:
                raise ValueError("C expected training ridge differs from historical diagnostics")
        elif label == "D":
            if _finite_number(d_diagnostics.get("final_float32_ridge"), "D.ridge") != result["D"]["ridge"]:
                raise ValueError("D expected training ridge differs from historical diagnostics")
            if _finite_number(d_diagnostics.get("final_float32_fullL"), "D.fullL") != result["D"]["fullL"]:
                raise ValueError("D expected fullL differs from historical diagnostics")
    return result


def load_anchor_references(
    exact_manifest: dict[str, Any], *, reference_root: str | Path,
    path_map: Mapping[str, str | Path] | None = None,
) -> dict[int, AnchorSet]:
    """Verify all exact manifest entries and load only C, D, and U32 arrays.

    The path map is a transport relocation layer. Its physical destinations do
    not enter source identities or the returned anchor manifest digest.
    """
    manifest, manifest_sha = _validate_manifest(exact_manifest)
    root_input = Path(reference_root)
    _reject_symlink_components(root_input)
    root = root_input.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("reference_root must be an existing directory")
    if path_map is None:
        path_map = {}
    if not isinstance(path_map, Mapping):
        raise TypeError("path_map must be a mapping")

    raw_entries = manifest.get("entries")
    raw_cases = manifest.get("cases")
    if type(raw_entries) is not list or not raw_entries or type(raw_cases) is not list:
        raise ValueError("anchor manifest entries and cases must be nonempty lists")
    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, entry in enumerate(raw_entries):
        if type(entry) is not dict or set(entry) != {"path", "bytes", "sha256", "role"}:
            raise ValueError(f"manifest entries[{index}] must have path, bytes, sha256, and role")
        logical = _logical_path(entry["path"], f"entries[{index}].path")
        if logical in seen_paths:
            raise ValueError(f"manifest entry paths must be unique: {logical}")
        seen_paths.add(logical)
        byte_count = entry["bytes"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, numbers.Integral) or byte_count < 0:
            raise ValueError(f"manifest entries[{index}].bytes must be a nonnegative integer")
        if not _is_sha256(entry["sha256"]) or not isinstance(entry["role"], str) or not entry["role"]:
            raise ValueError(f"manifest entries[{index}] has invalid SHA or role")
        entries.append({
            "path": logical, "bytes": int(byte_count),
            "sha256": entry["sha256"], "role": entry["role"],
        })
    if any(key not in seen_paths for key in path_map):
        raise ValueError("path_map keys must be exact manifest entry paths")

    allowed_roles = {
        "foundation_source_manifest", "v1_acceptance_bindings", "v1_delivery_manifest",
        "e163_mirror_manifest", "e163_run002_provenance", "e164_remote_manifest", "e164_cohort",
        "e164_recovery_manifest", "e164_environment",
        "prefix_checkpoint", "prefix_actor", "prefix_evidence",
        "C_candidate", "C_metadata", "C_metrics", "C_diagnostics", "C_persistent_references",
        "D_candidate", "D_metadata", "D_metrics", "D_diagnostics", "D_persistent_references",
        "U32_candidate", "U32_lock", "U32_trace_artifacts", "U32_trace",
    }
    if any(entry["role"] not in allowed_roles for entry in entries):
        raise ValueError("anchor manifest contains an unrecognized entry role")
    lookup = _entry_lookup(entries)
    file_bytes: dict[str, bytes] = {}
    for entry in entries:
        file_bytes[entry["path"]] = _read_manifest_entry(
            root, entry["path"], path_map, entry["bytes"], entry["sha256"],
        )

    foundation = manifest.get("foundation_identity")
    expected_foundation = {
        "logical_path": "data/foundation.pt",
        "source_manifest_path": "experiments/exp161/preparation/source-manifest.json",
        "entry": "data/foundation.pt",
        "bytes": 55477,
        "sha256": "e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222",
    }
    if type(foundation) is not dict or foundation != expected_foundation:
        raise ValueError("foundation metadata differs from the frozen H036 source")
    _require_entry(
        lookup, foundation["source_manifest_path"], {"foundation_source_manifest"},
        "foundation source manifest",
    )
    environment = _json_entry(
        file_bytes, lookup, "experiments/exp164/runs/run001/remote/environment.json",
        {"e164_environment"}, "E164 environment source identity",
    )
    if environment.get("source_identity_sha256") != _E164_SOURCE_IDENTITY_SHA256:
        raise ValueError("E164 environment source identity differs from the accepted E164 snapshot")

    if len(raw_cases) != len(_SEEDS):
        raise ValueError("anchor manifest must contain exactly 16 cases")
    recovered_seeds = {26204, 26206, 26209, 26212, 26213}
    all_source_refs = tuple(
        {
            "path": entry["path"], "bytes": entry["bytes"],
            "sha256": entry["sha256"], "role": entry["role"],
        }
        for entry in entries
    )
    output: dict[int, AnchorSet] = {}
    seen_seeds: list[int] = []
    for case_index, case in enumerate(raw_cases):
        case_fields = {
            "seed", "prepared_id", "input_identity", "prefix_files",
            "C", "D", "U32", "expected_training",
        }
        if type(case) is not dict or set(case) != case_fields:
            raise ValueError(f"manifest cases[{case_index}] has invalid fields")
        seed = case["seed"]
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or int(seed) not in _SEEDS:
            raise ValueError(f"manifest cases[{case_index}].seed is invalid")
        seed = int(seed)
        seen_seeds.append(seed)
        prepared_id = case["prepared_id"]
        if not isinstance(prepared_id, str) or not prepared_id:
            raise ValueError(f"case {seed} prepared_id must be nonempty")
        input_identity = case["input_identity"]
        identity_text = _identity_json(input_identity, f"case {seed} input_identity")
        if input_identity.get("logical_case") != f"s{seed}_dynamics_late":
            raise ValueError(f"case {seed} input_identity logical_case differs")
        basenames = ("checkpoint.pt", "actor.bin", "evidence.npz")
        prefix_files = case["prefix_files"]
        identity_files = input_identity.get("files")
        if type(prefix_files) is not dict or set(prefix_files) != set(basenames):
            raise ValueError(f"case {seed} prefix_files differ from the exact E161 prefix")
        if type(identity_files) is not dict or set(identity_files) != set(basenames):
            raise ValueError(f"case {seed} input_identity.files differ from the exact E161 prefix")
        for basename, role in zip(basenames, ("prefix_checkpoint", "prefix_actor", "prefix_evidence")):
            expected_path = (
                f"experiments/exp161/runs/run001/producer/cases/"
                f"s{seed}_dynamics_late/prefix/{basename}"
            )
            if _logical_path(prefix_files[basename], f"case {seed} prefix_files.{basename}") != expected_path:
                raise ValueError(f"case {seed} prefix path differs from its exact E161 source")
            prefix_entry = _require_entry(lookup, expected_path, {role}, f"case {seed} {basename}")
            identity_ref = identity_files[basename]
            if type(identity_ref) is not dict or set(identity_ref) != {"bytes", "sha256"}:
                raise ValueError(f"case {seed} input identity for {basename} is invalid")
            if (identity_ref["bytes"], identity_ref["sha256"]) != (
                prefix_entry["bytes"], prefix_entry["sha256"],
            ):
                raise ValueError(f"case {seed} input identity differs from exact prefix bytes")

        expected_c = {
            "candidate_path": f"experiments/exp163/runs/run002/remote/run002/cases/s{seed}/c/candidate.f32",
            "metadata_path": f"experiments/exp163/runs/run002/remote/run002/cases/s{seed}/c/candidate.json",
            "metrics_path": f"experiments/exp163/runs/run002/remote/run002/cases/s{seed}/c/metrics.json",
            "diagnostics_path": f"experiments/exp163/runs/run002/remote/run002/cases/s{seed}/c/diagnostics.json",
            "persistent_references_path": f"experiments/exp163/runs/run002/remote/run002/cases/s{seed}/c/persistent_references.json",
        }
        expected_d = {
            "candidate_path": f"experiments/exp163/runs/run002/remote/run002/cases/s{seed}/d/candidate.f32",
            "metadata_path": f"experiments/exp163/runs/run002/remote/run002/cases/s{seed}/d/candidate.json",
            "metrics_path": f"experiments/exp163/runs/run002/remote/run002/cases/s{seed}/d/metrics.json",
            "diagnostics_path": f"experiments/exp163/runs/run002/remote/run002/cases/s{seed}/d/diagnostics.json",
            "persistent_references_path": f"experiments/exp163/runs/run002/remote/run002/cases/s{seed}/d/persistent_references.json",
        }
        for label, expected_paths in (("C", expected_c), ("D", expected_d)):
            reference = case[label]
            if type(reference) is not dict or not set(expected_paths) | {"candidate_sha256", "raw_sha256"} == set(reference):
                raise ValueError(f"case {seed} {label} artifact reference fields are invalid")
            for field, expected_path in expected_paths.items():
                if _logical_path(reference[field], f"case {seed} {label}.{field}") != expected_path:
                    raise ValueError(f"case {seed} {label}.{field} differs from exact E163 source")

        c_array, c_diagnostics = _load_candidate(case["C"], file_bytes, lookup, "C", prepared_id)
        d_array, d_diagnostics = _load_candidate(case["D"], file_bytes, lookup, "D", prepared_id)

        u_ref = case["U32"]
        if type(u_ref) is not dict or set(u_ref) != {"candidate_path", "source_kind", "references"}:
            raise ValueError(f"case {seed} U32 reference fields are invalid")
        references = u_ref["references"]
        if type(references) is not dict:
            raise TypeError(f"case {seed} U32 references must be a plain dict")
        u_path = _logical_path(u_ref["candidate_path"], f"case {seed} U32 candidate_path")
        if seed in recovered_seeds:
            expected_u_path = f"experiments/exp164/runs/run001/remote/cases/s{seed}/probes/UR/candidate.f32"
            expected_reference_keys = {
                "lock_path", "trace_artifacts_path", "trace_path", "cohort_path",
                "cohort_seed", "cohort_probe", "manifest_path", "candidate_sha256", "raw_sha256",
            }
            expected_u_kind = "e164_recovery_ur"
            if set(references) != expected_reference_keys or u_ref["source_kind"] != expected_u_kind:
                raise ValueError(f"case {seed} recovered U32 provenance fields are invalid")
            if u_path != expected_u_path:
                raise ValueError(f"case {seed} recovered U32 path differs from exact E164 UR")
            if references["lock_path"] != f"experiments/exp164/runs/run001/remote/cases/s{seed}/lock.json":
                raise ValueError(f"case {seed} UR lock path differs from exact E164 source")
            if references["trace_artifacts_path"] != f"experiments/exp164/runs/run001/remote/cases/s{seed}/trace_artifacts.json":
                raise ValueError(f"case {seed} UR trace-artifacts path differs from exact E164 source")
            if references["trace_path"] != f"experiments/exp164/runs/run001/remote/cases/s{seed}/trace.json":
                raise ValueError(f"case {seed} UR trace path differs from exact E164 source")
        else:
            expected_reference_keys = {
                "cohort_path", "cohort_seed", "cohort_probe", "manifest_path",
                "candidate_sha256", "raw_sha256", "score_ref",
                "D_candidate_path", "D_candidate_sha256", "D_raw_sha256",
            }
            expected_u_kind = "e164_alias_d"
            if set(references) != expected_reference_keys or u_ref["source_kind"] != expected_u_kind:
                raise ValueError(f"case {seed} U32 alias provenance fields are invalid")
            if u_path != expected_d["candidate_path"]:
                raise ValueError(f"case {seed} U32 alias must point at the exact D candidate")
            if references["D_candidate_path"] != expected_d["candidate_path"]:
                raise ValueError(f"case {seed} U32 alias D path differs")
            if references["D_candidate_sha256"] != case["D"]["candidate_sha256"] or references["D_raw_sha256"] != case["D"]["raw_sha256"]:
                raise ValueError(f"case {seed} U32 alias D identity differs")

        u_entry = _require_entry(lookup, u_path, {"U32_candidate", "D_candidate"}, f"case {seed} U32 candidate")
        u_raw = file_bytes[u_path]
        if len(u_raw) != 9 * np.dtype("<f4").itemsize or _sha256(u_raw) != u_entry["sha256"]:
            raise ValueError(f"case {seed} U32 bytes differ from exact candidate entry")
        u_array = np.frombuffer(u_raw, dtype="<f4").reshape(_SHAPE).astype(np.float32, copy=True)
        if not np.isfinite(u_array).all():
            raise FloatingPointError(f"case {seed} U32 contains nonfinite values")
        u_array.setflags(write=False)
        if references["cohort_seed"] != seed or references["cohort_probe"] != "UR":
            raise ValueError(f"case {seed} U32 E164 cohort pointer is invalid")
        if references["cohort_path"] != "experiments/exp164/runs/run001/remote/cohort.json":
            raise ValueError(f"case {seed} U32 E164 cohort path differs")
        if references["manifest_path"] != "experiments/exp164/runs/run001/remote_manifest.json":
            raise ValueError(f"case {seed} E164 remote manifest path differs")
        _require_entry(lookup, references["manifest_path"], {"e164_remote_manifest"}, f"case {seed} E164 manifest")
        cohort = _json_entry(file_bytes, lookup, references["cohort_path"], {"e164_cohort"}, f"case {seed} E164 cohort")
        cohort_manifest_sha = cohort.get("manifest_sha256")
        if cohort_manifest_sha != _E164_COHORT_MANIFEST_SHA256:
            raise ValueError("E164 cohort manifest SHA differs from the frozen UR source identity")
        cohort_case = cohort.get("cases", {}).get(str(seed)) if type(cohort.get("cases")) is dict else None
        cohort_probe = cohort_case.get("probes", {}).get("UR") if type(cohort_case) is dict and type(cohort_case.get("probes")) is dict else None
        if type(cohort_probe) is not dict:
            raise ValueError(f"case {seed} E164 cohort lacks its UR identity")
        if cohort_probe.get("candidate_sha256") != references["candidate_sha256"] or cohort_probe.get("raw_sha256") != references["raw_sha256"]:
            raise ValueError(f"case {seed} E164 cohort UR hashes differ from U32 reference")
        if _candidate_sha256(u_array, np.float32) != references["candidate_sha256"] or _sha256(u_raw) != references["raw_sha256"]:
            raise ValueError(f"case {seed} U32 candidate bytes differ from E164 source identity")
        if seed in recovered_seeds:
            for field, role in (
                ("lock_path", "U32_lock"), ("trace_artifacts_path", "U32_trace_artifacts"),
                ("trace_path", "U32_trace"),
            ):
                _require_entry(lookup, references[field], {role}, f"case {seed} U32 {field}")
            lock = _json_entry(file_bytes, lookup, references["lock_path"], {"U32_lock"}, f"case {seed} UR lock")
            if lock.get("seed") != seed or lock.get("prepared_id") != prepared_id:
                raise ValueError(f"case {seed} recovered UR lock belongs to a different prefix")
            lock_probe = lock.get("probes", {}).get("UR") if type(lock.get("probes")) is dict else None
            if (type(lock_probe) is not dict
                    or lock_probe.get("candidate_sha256") != references["candidate_sha256"]
                    or lock_probe.get("raw_sha256") != references["raw_sha256"]):
                raise ValueError(f"case {seed} recovered UR lock does not bind U32 bytes")
        else:
            if not np.array_equal(u_array, d_array):
                raise ValueError(f"case {seed} U32 alias bytes differ from D")
            if references["candidate_sha256"] != case["D"]["candidate_sha256"] or references["raw_sha256"] != case["D"]["raw_sha256"]:
                raise ValueError(f"case {seed} E164 UR alias identity differs from D")
            score_ref = references["score_ref"]
            if type(score_ref) is not dict or set(score_ref) != {"path", "bytes", "sha256"}:
                raise ValueError(f"case {seed} U32 alias score_ref is invalid")
            score_entry = _require_entry(lookup, score_ref["path"], {"D_metrics"}, f"case {seed} UR score reference")
            if score_ref["path"] != expected_d["metrics_path"]:
                raise ValueError(f"case {seed} U32 alias score_ref is not the exact D metrics path")
            if (score_ref["bytes"], score_ref["sha256"]) != (score_entry["bytes"], score_entry["sha256"]):
                raise ValueError(f"case {seed} U32 alias score_ref differs from exact cohort file")
            cohort_score_ref = cohort_probe.get("score_ref")
            normalized_cohort_score_ref = (
                {key: cohort_score_ref.get(key) for key in ("path", "bytes", "sha256")}
                if type(cohort_score_ref) is dict else None
            )
            if normalized_cohort_score_ref != score_ref:
                raise ValueError(f"case {seed} E164 UR cohort score reference differs")

        expected_training = _validate_expected_training_from_diagnostics(
            case["expected_training"], c_diagnostics, d_diagnostics,
        )
        output[seed] = AnchorSet(
            seed=seed,
            prepared_id=prepared_id,
            input_identity=json.loads(identity_text),
            C=c_array,
            D=d_array,
            U32=u_array,
            expected_training=expected_training,
            source_references=all_source_refs,
            v1_source_identity_sha256=manifest["v1_source_identity_sha256"],
            e164_source_identity_sha256=manifest["e164_source_identity_sha256"],
            e164_cohort_manifest_sha256=manifest["e164_cohort_manifest_sha256"],
            manifest_sha256=manifest_sha,
        )
        _validate_anchor_set(output[seed])
    if seen_seeds != list(_SEEDS):
        raise ValueError("manifest cases must be in exact ascending frozen seed order")
    return output


class _TrainingOracle:
    """Prepared/training/objective-bound evaluation cache with a hard miss cap."""

    def __init__(self, view: TrainingView) -> None:
        self.view = view
        self.cache: dict[tuple[str, str, str, str, bytes], dict[str, Any]] = {}
        self.evaluations: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.counts: dict[str, Any] = {
            "logical_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "actual_oracle_evaluations": 0,
            "baseline_requests": 0,
            "trial_requests": 0,
            "trial_unique": 0,
            "postcast_requests": 0,
            "postcast_unique": 0,
            "unique_candidate_keys_by_phase": {"baseline": [], "trial": [], "postcast": []},
            "helper_counts": {key: 0 for key in _HELPER_COUNT_KEYS},
        }
        self._phase_keys: dict[str, set[tuple[str, str, str, str, bytes]]] = {
            "baseline": set(), "trial": set(), "postcast": set(),
        }

    def evaluate(self, candidate64: np.ndarray, phase: str) -> dict[str, Any]:
        if phase not in self._phase_keys:
            raise ValueError("oracle phase must be baseline, trial, or postcast")
        candidate = _f64_candidate(candidate64, "oracle candidate")
        key = (
            self.view.prepared_id,
            self.view.training_hash,
            _OBJECTIVE_VERSION,
            candidate.dtype.str,
            np.ascontiguousarray(candidate, dtype="<f8").tobytes(order="C"),
        )
        candidate_bytes = key[-1]
        candidate_sha = _candidate_sha256(candidate, np.float64)
        candidate_raw_sha = _sha256(candidate_bytes)
        self.counts["logical_requests"] += 1
        self.counts[f"{phase}_requests"] += 1
        if key not in self._phase_keys[phase]:
            self._phase_keys[phase].add(key)
            self.counts["unique_candidate_keys_by_phase"][phase].append({
                "candidate_sha256": candidate_sha,
                "candidate_raw_sha256": candidate_raw_sha,
            })
            if phase == "trial":
                self.counts["trial_unique"] += 1
            elif phase == "postcast":
                self.counts["postcast_unique"] += 1
        if key in self.cache:
            self.counts["cache_hits"] += 1
            self.events.append({
                "event": len(self.events),
                "phase": phase,
                "cache": "hit",
                "candidate_sha256": candidate_sha,
                "candidate_raw_sha256": candidate_raw_sha,
                "evaluation_index": next(
                    index for index, item in enumerate(self.evaluations)
                    if item["candidate_raw_sha256"] == candidate_raw_sha
                ),
            })
            return copy.deepcopy(self.cache[key])
        if self.counts["actual_oracle_evaluations"] >= _MAX_ORACLE_EVALUATIONS:
            raise RuntimeError("training oracle budget exhausted before an additional cache miss")
        # Charge a miss before entering scientific helpers so a failed call is
        # never reported as free. No retry is performed by this module.
        self.counts["cache_misses"] += 1
        self.counts["actual_oracle_evaluations"] += 1
        self.events.append({
            "event": len(self.events),
            "phase": phase,
            "cache": "miss",
            "candidate_sha256": candidate_sha,
            "candidate_raw_sha256": candidate_raw_sha,
            "evaluation_index": len(self.evaluations),
        })
        result = _compute_training_metrics(self.view, candidate)
        metrics = {
            key_name: _finite_number(result.get(key_name), f"oracle {key_name}")
            for key_name in ("fullL", "new_train_kl", "old_train_kl", "ridge")
        }
        helper_counts = result.get("helper_counts")
        if type(helper_counts) is not dict or set(helper_counts) != set(_HELPER_COUNT_KEYS):
            raise ValueError("oracle helper_counts have an invalid schema")
        for key_name in _HELPER_COUNT_KEYS:
            count = helper_counts[key_name]
            if isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 0:
                raise ValueError(f"oracle helper count {key_name} must be a nonnegative integer")
            self.counts["helper_counts"][key_name] += int(count)
        self.cache[key] = copy.deepcopy(metrics)
        self.evaluations.append({
            "evaluation_index": len(self.evaluations),
            "first_phase": phase,
            "candidate_sha256": candidate_sha,
            "candidate_raw_sha256": candidate_raw_sha,
            "candidate_f64_raw_hex": candidate_bytes.hex(),
            "metrics": copy.deepcopy(metrics),
            "helper_counts": {key: int(helper_counts[key]) for key in _HELPER_COUNT_KEYS},
        })
        return copy.deepcopy(metrics)

    def finalized_counts(self) -> dict[str, Any]:
        result = copy.deepcopy(self.counts)
        result["unique_candidate_keys_by_phase"] = {
            phase: sorted(entries, key=lambda item: (item["candidate_sha256"], item["candidate_raw_sha256"]))
            for phase, entries in result["unique_candidate_keys_by_phase"].items()
        }
        return result

    def evaluation_manifest(self) -> dict[str, Any]:
        return {
            "cache_key": {
                "prepared_id": self.view.prepared_id,
                "training_hash": self.view.training_hash,
                "objective_version": _OBJECTIVE_VERSION,
                "dtype": np.dtype(np.float64).str,
                "candidate_bytes": "contiguous little-endian float64 raw bytes",
            },
            "evaluations": copy.deepcopy(self.evaluations),
            "events": copy.deepcopy(self.events),
        }


def _jsonable_trace_value(value: Any, *, include_array_bytes: bool) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == np.dtype(np.float64) and value.shape == _SHAPE:
            raw_dtype = "<f8"
        elif value.dtype == np.dtype(np.float32) and value.shape == _SHAPE:
            raw_dtype = "<f4"
        else:
            raise ValueError("trace contains an unsupported array")
        result = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "raw_sha256": _array_sha256(value, raw_dtype),
            "scientific_sha256": _candidate_sha256(value, value.dtype),
        }
        if include_array_bytes:
            result["raw_bytes_hex"] = np.ascontiguousarray(value, dtype=np.dtype(raw_dtype)).tobytes().hex()
        return result
    if type(value) is dict:
        return {str(key): _jsonable_trace_value(item, include_array_bytes=include_array_bytes)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_trace_value(item, include_array_bytes=include_array_bytes) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _budgeted_trace_digest(trace: Mapping[str, Any]) -> str:
    identity = {key: value for key, value in trace.items() if key != "trace_sha256"}
    digest = hashlib.sha256(b"h036_budgeted_search_v1\0")
    digest.update(_json_bytes(_jsonable_trace_value(identity, include_array_bytes=False)))
    return digest.hexdigest()


def _validate_budgeted_trace(view: TrainingView, trace: Any) -> dict[str, Any]:
    if type(trace) is not dict or trace.get("format") != _FORMAT:
        raise TypeError("trace must be an h036_budgeted_search_v1 dict")
    if trace.get("seed") != view.seed or trace.get("prepared_id") != view.prepared_id:
        raise ValueError("trace does not belong to this training view")
    if trace.get("training_hash") != view.training_hash:
        raise ValueError("trace training hash differs from this training view")
    if trace.get("trace_sha256") != _budgeted_trace_digest(trace):
        raise ValueError("budgeted search trace SHA-256 does not match its contents")
    if trace.get("binding") != _BINDING or trace.get("lambda_value") != _LAMBDA:
        raise ValueError("trace fixed binding or lambda differs from the H036 contract")
    if trace.get("anchor_manifest_sha256") is None or not _is_sha256(trace["anchor_manifest_sha256"]):
        raise ValueError("trace anchor manifest SHA-256 is invalid")
    if not _is_sha256(trace.get("v1_source_identity_sha256")) or not _is_sha256(trace.get("e164_source_identity_sha256")):
        raise ValueError("trace legacy source identities are invalid")
    if not _is_sha256(trace.get("e164_cohort_manifest_sha256")):
        raise ValueError("trace E164 cohort manifest identity is invalid")
    if (trace.get("ridge") != _RIDGE or trace.get("new_kl_slack") != _KL_TOLERANCE
            or trace.get("witness_tolerance") != _WITNESS_TOLERANCE):
        raise ValueError("trace scientific constants differ from the fixed contract")
    if (trace.get("anchor_manifest_sha256") != _ANCHOR_MANIFEST_SHA256
            or trace.get("v1_source_identity_sha256") != _V1_SOURCE_IDENTITY_SHA256
            or trace.get("e164_source_identity_sha256") != _E164_SOURCE_IDENTITY_SHA256
            or trace.get("e164_cohort_manifest_sha256") != _E164_COHORT_MANIFEST_SHA256):
        raise ValueError("trace source identities differ from the frozen reference bundle")
    anchor_arrays = trace.get("anchor_arrays")
    anchor_ids = trace.get("anchor_identities")
    if type(anchor_arrays) is not dict or set(anchor_arrays) != {"C", "D", "U32"}:
        raise ValueError("trace is missing exact source anchor arrays")
    if type(anchor_ids) is not dict or set(anchor_ids) != {"C0", "C", "D", "U32"}:
        raise ValueError("trace anchor identities are incomplete")
    for label in ("C", "D", "U32"):
        array = anchor_arrays[label]
        if not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.float32) or array.shape != _SHAPE:
            raise ValueError(f"trace anchor {label} must be float32 with shape (3, 3)")
        if not np.isfinite(array).all():
            raise FloatingPointError(f"trace anchor {label} is nonfinite")
        if (anchor_ids[label].get("scientific_sha256") != _candidate_sha256(array, np.float32)
                or anchor_ids[label].get("raw_sha256") != _array_sha256(array, "<f4")):
            raise ValueError(f"trace anchor {label} hashes differ from its exact bytes")
    c0_expected = np.ascontiguousarray(view.c0.astype(np.float32), dtype=np.float32)
    c0_trace = trace.get("c0_float32")
    if not isinstance(c0_trace, np.ndarray) or c0_trace.dtype != np.dtype(np.float32) or not np.array_equal(c0_trace, c0_expected):
        raise ValueError("trace C0 bytes differ from the training-view float32 initialization")
    if (anchor_ids["C0"].get("scientific_sha256") != _candidate_sha256(c0_expected, np.float32)
            or anchor_ids["C0"].get("raw_sha256") != _array_sha256(c0_expected, "<f4")):
        raise ValueError("trace C0 hashes differ from the training-view initialization")
    baseline = trace.get("baseline")
    if type(baseline) is not dict or set(baseline) != {"C0", "C", "D"}:
        raise ValueError("trace must contain C0, C, and D training baselines")
    baseline = {
        label: _validate_training_metrics(baseline[label], f"trace baseline {label}")
        for label in ("C0", "C", "D")
    }
    if trace.get("b") != baseline["C0"]["new_train_kl"] or trace.get("D_fullL") != baseline["D"]["fullL"]:
        raise ValueError("trace C0 budget or D objective differs from its baseline metrics")
    if baseline["D"]["new_train_kl"] > trace["b"] + _KL_TOLERANCE:
        raise ValueError("trace D baseline violates the fixed C0 new-KL bound")
    selections = trace.get("selections")
    postcast = trace.get("postcast")
    if type(selections) is not dict or set(selections) != {"S0", "S1"}:
        raise ValueError("trace must contain S0 and S1 selections")
    if type(postcast) is not dict or set(postcast) != {"S0", "S1"}:
        raise ValueError("trace must contain S0 and S1 post-cast results")
    for label in ("S0", "S1"):
        selection = selections[label]
        if type(selection) is not dict:
            raise ValueError(f"trace {label} selection is invalid")
        selected = selection.get("selected_float64")
        _f64_candidate(selected, f"trace {label} selected_float64")
        selected_metrics = _validate_training_metrics(selection.get("metrics"), f"trace {label} metrics")
        if selected_metrics != selection.get("metrics"):
            raise ValueError(f"trace {label} training metrics are not normalized")
        cast = postcast[label]
        if type(cast) is not dict:
            raise ValueError(f"trace {label} postcast is invalid")
        for key in ("raw_cast_candidate", "candidate"):
            value = cast.get(key)
            if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float32) or value.shape != _SHAPE:
                raise ValueError(f"trace {label} {key} must be float32 with shape (3, 3)")
            if not np.isfinite(value).all():
                raise FloatingPointError(f"trace {label} {key} is nonfinite")
        _validate_training_metrics(cast.get("raw_cast_metrics"), f"trace {label} raw_cast_metrics")
        _validate_training_metrics(cast.get("final_metrics"), f"trace {label} final_metrics")
        if not isinstance(cast.get("post_cast_gate_pass"), bool) or not isinstance(cast.get("fallback_applied"), bool):
            raise ValueError(f"trace {label} gate and fallback fields must be booleans")
        if cast["fallback_applied"] is cast["post_cast_gate_pass"]:
            raise ValueError(f"trace {label} fallback must be the negation of the cast gate")
        if cast["fallback_applied"] and not np.array_equal(cast["candidate"], trace["c0_float32"]):
            raise ValueError(f"trace {label} fallback does not equal C0")
        if not cast["fallback_applied"] and not np.array_equal(cast["candidate"], cast["raw_cast_candidate"]):
            raise ValueError(f"trace {label} final candidate differs from successful cast")
    counters = trace.get("counters")
    if type(counters) is not dict or counters.get("actual_oracle_evaluations", 0) > _MAX_ORACLE_EVALUATIONS:
        raise ValueError("trace exceeds the fixed per-prefix training oracle budget")
    if counters.get("cache_misses") != counters.get("actual_oracle_evaluations"):
        raise ValueError("trace oracle cache miss and actual evaluation counts differ")
    oracle = trace.get("oracle")
    if type(oracle) is not dict or type(oracle.get("evaluations")) is not list or type(oracle.get("events")) is not list:
        raise ValueError("trace oracle events and unique evaluations are missing")
    if len(oracle["evaluations"]) != counters.get("actual_oracle_evaluations"):
        raise ValueError("trace unique oracle evaluation records differ from counted misses")
    if len(oracle["events"]) != counters.get("logical_requests"):
        raise ValueError("trace oracle request events differ from logical requests")
    cache_key = oracle.get("cache_key")
    if (type(cache_key) is not dict
            or cache_key.get("prepared_id") != view.prepared_id
            or cache_key.get("training_hash") != view.training_hash
            or cache_key.get("objective_version") != _OBJECTIVE_VERSION
            or cache_key.get("dtype") != np.dtype(np.float64).str
            or cache_key.get("candidate_bytes") != "contiguous little-endian float64 raw bytes"):
        raise ValueError("trace oracle cache key does not bind this prepared training objective")
    evaluation_by_raw: dict[str, dict[str, Any]] = {}
    for index, evaluation in enumerate(oracle["evaluations"]):
        if type(evaluation) is not dict or evaluation.get("evaluation_index") != index:
            raise ValueError("trace oracle evaluation indices are not contiguous")
        try:
            candidate_bytes = bytes.fromhex(evaluation["candidate_f64_raw_hex"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("trace oracle candidate raw bytes are malformed") from exc
        if len(candidate_bytes) != 9 * np.dtype("<f8").itemsize:
            raise ValueError("trace oracle candidate bytes must encode one 3 by 3 float64 array")
        candidate = np.frombuffer(candidate_bytes, dtype="<f8").reshape(_SHAPE)
        if (evaluation.get("candidate_raw_sha256") != _sha256(candidate_bytes)
                or evaluation.get("candidate_sha256") != _candidate_sha256(candidate, np.float64)):
            raise ValueError("trace oracle candidate hashes do not match stored float64 bytes")
        _validate_training_metrics(evaluation.get("metrics"), f"trace oracle evaluation {index} metrics")
        if evaluation["candidate_raw_sha256"] in evaluation_by_raw:
            raise ValueError("trace oracle contains duplicate unique evaluation bytes")
        evaluation_by_raw[evaluation["candidate_raw_sha256"]] = evaluation
    expected_baseline_candidates = {
        "C0": view.c0,
        "C": anchor_arrays["C"].astype(np.float64),
        "D": anchor_arrays["D"].astype(np.float64),
    }
    for label, candidate in expected_baseline_candidates.items():
        raw_sha = _array_sha256(candidate, "<f8")
        evaluation = evaluation_by_raw.get(raw_sha)
        if evaluation is None or evaluation.get("first_phase") != "baseline":
            raise ValueError(f"trace oracle lacks exact baseline evaluation for {label}")
        if evaluation.get("metrics") != baseline[label]:
            raise ValueError(f"trace baseline {label} differs from its cached oracle row")
    seen_event_bytes: dict[str, int] = {}
    phase_requests = {"baseline": 0, "trial": 0, "postcast": 0}
    for index, event in enumerate(oracle["events"]):
        if type(event) is not dict or event.get("event") != index:
            raise ValueError("trace oracle request event indices are not contiguous")
        evaluation_index = event.get("evaluation_index")
        if (isinstance(evaluation_index, bool) or not isinstance(evaluation_index, numbers.Integral)
                or not 0 <= evaluation_index < len(oracle["evaluations"])):
            raise ValueError("trace oracle request event points outside unique evaluations")
        target = oracle["evaluations"][int(evaluation_index)]
        if (event.get("candidate_raw_sha256") != target["candidate_raw_sha256"]
                or event.get("candidate_sha256") != target["candidate_sha256"]
                or event.get("cache") not in {"hit", "miss"}
                or event.get("phase") not in phase_requests):
            raise ValueError("trace oracle request event identity is inconsistent")
        phase_requests[event["phase"]] += 1
        prior_index = seen_event_bytes.get(event["candidate_raw_sha256"])
        if prior_index is None:
            if (event["cache"] != "miss" or int(evaluation_index) != len(seen_event_bytes)
                    or target.get("first_phase") != event["phase"]):
                raise ValueError("trace first-seen oracle event is not a matching cache miss")
            seen_event_bytes[event["candidate_raw_sha256"]] = int(evaluation_index)
        elif event["cache"] != "hit" or int(evaluation_index) != prior_index:
            raise ValueError("trace repeated oracle request is not an exact cache hit")
    if (phase_requests["baseline"] != counters.get("baseline_requests")
            or phase_requests["trial"] != counters.get("trial_requests")
            or phase_requests["postcast"] != counters.get("postcast_requests")
            or counters.get("baseline_requests") != 3
            or counters.get("trial_unique") > _MAX_TRIALS
            or counters.get("postcast_requests") != 2):
        raise ValueError("trace oracle phase counts differ from the fixed call plan")
    helper_totals = {key: 0 for key in _HELPER_COUNT_KEYS}
    for evaluation in oracle["evaluations"]:
        helper_counts = evaluation.get("helper_counts")
        if type(helper_counts) is not dict or set(helper_counts) != set(_HELPER_COUNT_KEYS):
            raise ValueError("trace evaluation helper counts have an invalid schema")
        for key in _HELPER_COUNT_KEYS:
            count = helper_counts[key]
            if isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 0:
                raise ValueError(f"trace helper count {key} is invalid")
            helper_totals[key] += int(count)
    if helper_totals != counters.get("helper_counts"):
        raise ValueError("trace aggregate helper counts differ from actual unique oracle rows")
    expected_trial_indices: dict[str, int | None] = {"S0": None, "S1": None}
    trials = trace.get("trials")
    if type(trials) is not list or len(trials) > _MAX_TRIALS:
        raise ValueError("trace trial path exceeds the fixed 16-point schedule")
    c64 = anchor_arrays["C"].astype(np.float64)
    d64 = anchor_arrays["D"].astype(np.float64)
    u64 = anchor_arrays["U32"].astype(np.float64)
    for index, trial in enumerate(trials):
        if type(trial) is not dict or trial.get("j") != index or trial.get("alpha") != 2.0 ** (-index):
            raise ValueError("trace trial indices/alphas differ from the frozen ordered schedule")
        candidate = c64 + (2.0 ** (-index)) * (u64 - c64)
        raw_sha = _array_sha256(candidate, "<f8")
        if (trial.get("candidate_raw_sha256") != raw_sha
                or trial.get("candidate_sha256") != _candidate_sha256(candidate, np.float64)):
            raise ValueError(f"trace trial X_{index} differs from exact C-to-U32 interpolation")
        evaluation = evaluation_by_raw.get(raw_sha)
        has_trial_request = any(
            event.get("phase") == "trial" and event.get("candidate_raw_sha256") == raw_sha
            for event in oracle["events"]
        )
        if evaluation is None or not has_trial_request:
            raise ValueError(f"trace trial X_{index} is absent from the training oracle records")
        metrics = evaluation["metrics"]
        for field in ("fullL", "new_train_kl", "old_train_kl", "ridge"):
            if trial.get(field) != metrics[field]:
                raise ValueError(f"trace trial X_{index} {field} differs from oracle record")
        better = metrics["fullL"] < baseline["D"]["fullL"]
        within = metrics["new_train_kl"] <= trace["b"] + _KL_TOLERANCE
        if trial.get("better_than_D") is not better or trial.get("within_C0_bound") is not within:
            raise ValueError(f"trace trial X_{index} eligibility flags are inconsistent")
        s0_first = expected_trial_indices["S0"] is None and better
        s1_first = expected_trial_indices["S1"] is None and better and within
        if trial.get("S0_first_accepted") is not s0_first or trial.get("S1_first_accepted") is not s1_first:
            raise ValueError(f"trace trial X_{index} first-acceptance flags are inconsistent")
        if s0_first:
            expected_trial_indices["S0"] = index
        if s1_first:
            expected_trial_indices["S1"] = index
    if trials:
        if expected_trial_indices["S0"] is not None and expected_trial_indices["S1"] is not None:
            if len(trials) != max(expected_trial_indices.values()) + 1:
                raise ValueError("trace trial path continued after both arms selected")
        elif len(trials) != _MAX_TRIALS:
            raise ValueError("trace trial path stopped before both arms selected or schedule exhaustion")
    for label in ("S0", "S1"):
        expected_index = expected_trial_indices[label]
        selection = selections[label]
        if expected_index is None:
            if (selection.get("origin") != "D" or selection.get("j") is not None
                    or not np.array_equal(selection["selected_float64"], d64)):
                raise ValueError(f"trace {label} must retain D when no point is accepted")
            selected_metrics = baseline["D"]
        else:
            trial = trials[expected_index]
            expected_candidate = c64 + trial["alpha"] * (u64 - c64)
            if (selection.get("origin") != "trial" or selection.get("j") != expected_index
                    or selection.get("alpha") != trial["alpha"]
                    or not np.array_equal(selection["selected_float64"], expected_candidate)):
                raise ValueError(f"trace {label} does not select its first eligible trial")
            selected_raw = _array_sha256(expected_candidate, "<f8")
            selected_metrics = evaluation_by_raw[selected_raw]["metrics"]
        if selection.get("metrics") != selected_metrics:
            raise ValueError(f"trace {label} selected metrics differ from its cached training oracle row")
        raw_cast = postcast[label]["raw_cast_candidate"]
        expected_cast = np.ascontiguousarray(selection["selected_float64"].astype(np.float32), dtype=np.float32)
        if not np.array_equal(raw_cast, expected_cast):
            raise ValueError(f"trace {label} float32 raw cast differs from its selected float64 array")
        raw_cast64 = raw_cast.astype(np.float64)
        raw_cast_raw_sha = _array_sha256(raw_cast64, "<f8")
        raw_cast_evaluation = evaluation_by_raw.get(raw_cast_raw_sha)
        has_postcast_request = any(
            event.get("phase") == "postcast" and event.get("candidate_raw_sha256") == raw_cast_raw_sha
            for event in oracle["events"]
        )
        if raw_cast_evaluation is None or not has_postcast_request:
            raise ValueError(f"trace {label} cast metrics are absent from the training oracle cache")
        if postcast[label]["raw_cast_metrics"] != raw_cast_evaluation["metrics"]:
            raise ValueError(f"trace {label} raw-cast metrics differ from their cached oracle row")
        gate_pass = raw_cast_evaluation["metrics"]["new_train_kl"] <= trace["b"] + _KL_TOLERANCE
        if postcast[label]["post_cast_gate_pass"] is not gate_pass:
            raise ValueError(f"trace {label} post-cast gate result is inconsistent")
        expected_final = c0_expected if not gate_pass else raw_cast
        expected_final_metrics = baseline["C0"] if not gate_pass else raw_cast_evaluation["metrics"]
        if not np.array_equal(postcast[label]["candidate"], expected_final):
            raise ValueError(f"trace {label} final candidate differs from its gate/fallback result")
        if postcast[label]["final_metrics"] != expected_final_metrics:
            raise ValueError(f"trace {label} final metrics differ from the cached final candidate")
        expected_witness = bool(
            _array_sha256(expected_final, "<f4") != anchor_ids["D"]["raw_sha256"]
            and expected_final_metrics["new_train_kl"] <= trace["b"] + _KL_TOLERANCE
            and expected_final_metrics["fullL"] < baseline["D"]["fullL"] - _WITNESS_TOLERANCE
            and expected_final_metrics["old_train_kl"] < baseline["D"]["old_train_kl"] - _WITNESS_TOLERANCE
        )
        if postcast[label].get("witness") is not expected_witness:
            raise ValueError(f"trace {label} witness flag is inconsistent")
        if (postcast[label].get("candidate_sha256") != _candidate_sha256(expected_final, np.float32)
                or postcast[label].get("raw_sha256") != _array_sha256(expected_final, "<f4")
                or postcast[label].get("raw_cast_candidate_sha256") != _candidate_sha256(raw_cast, np.float32)
                or postcast[label].get("raw_cast_sha256") != _array_sha256(raw_cast, "<f4")):
            raise ValueError(f"trace {label} float32 candidate hashes differ from exact bytes")
    if counters.get("logical_requests") > _MAX_ORACLE_EVALUATIONS:
        raise ValueError("trace logical oracle requests exceed the 21-call per-prefix allowance")
    stop = trace.get("search_stop")
    expected_stop_reason = (
        "both_arms_selected"
        if expected_trial_indices["S0"] is not None and expected_trial_indices["S1"] is not None
        else "fixed_trial_schedule_exhausted"
        if len(trials) == _MAX_TRIALS
        else "one_or_both_arms_retained_D"
    )
    if type(stop) is not dict or stop != {
        "j": trials[-1]["j"] if trials else None,
        "reason": expected_stop_reason,
    }:
        raise ValueError("trace search stop metadata differs from the fixed first-acceptance path")
    return trace


def search_budgeted_pair(
    training_view: TrainingView, anchors: AnchorSet, config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Search the fixed C-to-U32 line for first better S0/S1 candidates."""
    view = _validate_training_view(training_view)
    anchor_set = _validate_anchor_set(anchors)
    fixed = _validate_config(config, anchor_set)
    if view.seed != anchor_set.seed or view.prepared_id != anchor_set.prepared_id:
        raise ValueError("training view and anchors have different seed or prepared identity")
    if _identity_json(json.loads(view.input_identity_json), "training input_identity") != _identity_json(
        anchor_set.input_identity, "anchor input_identity",
    ):
        raise ValueError("training view and anchors have different input identity")
    if float(protected._RIDGE) != _RIDGE:
        raise RuntimeError("v1 protected ridge differs from fixed budgeted-search ridge")
    oracle = _TrainingOracle(view)
    c0_64 = np.asarray(view.c0, dtype=np.float64).copy()
    c64 = np.asarray(anchor_set.C, dtype=np.float32).astype(np.float64)
    d64 = np.asarray(anchor_set.D, dtype=np.float32).astype(np.float64)
    u64 = np.asarray(anchor_set.U32, dtype=np.float32).astype(np.float64)
    baseline = {
        "C0": oracle.evaluate(c0_64, "baseline"),
        "C": oracle.evaluate(c64, "baseline"),
        "D": oracle.evaluate(d64, "baseline"),
    }
    expected = _validate_expected_training(anchor_set.expected_training)
    for label, metrics in baseline.items():
        expected_fields = (
            ("fullL", "new_train_kl", "old_train_kl", "ridge")
            if label in {"C0", "D"} else ("new_train_kl", "old_train_kl", "ridge")
        )
        for field in expected_fields:
            if metrics[field] != expected[label][field]:
                raise ValueError(f"{label} training {field} differs exactly from frozen historical value")
    b = baseline["C0"]["new_train_kl"]
    d_metrics = baseline["D"]
    if d_metrics["new_train_kl"] > b + _KL_TOLERANCE:
        raise ValueError("historical D candidate violates the original C0 training new-KL bound")

    selected: dict[str, dict[str, Any] | None] = {"S0": None, "S1": None}
    trial_records: list[dict[str, Any]] = []
    c_to_u = u64 - c64
    for exponent in range(_MAX_TRIALS):
        alpha = 2.0 ** (-exponent)
        candidate = c64 + alpha * c_to_u
        if not np.isfinite(candidate).all():
            raise FloatingPointError(f"trial X_{exponent} is nonfinite")
        candidate = np.ascontiguousarray(candidate, dtype=np.float64)
        metrics = oracle.evaluate(candidate, "trial")
        better = metrics["fullL"] < d_metrics["fullL"]
        within_bound = metrics["new_train_kl"] <= b + _KL_TOLERANCE
        s0_first = selected["S0"] is None and better
        s1_first = selected["S1"] is None and better and within_bound
        record = {
            "j": exponent,
            "alpha": alpha,
            "candidate_sha256": _candidate_sha256(candidate, np.float64),
            "candidate_raw_sha256": _array_sha256(candidate, "<f8"),
            "fullL": metrics["fullL"],
            "new_train_kl": metrics["new_train_kl"],
            "old_train_kl": metrics["old_train_kl"],
            "ridge": metrics["ridge"],
            "better_than_D": bool(better),
            "within_C0_bound": bool(within_bound),
            "S0_first_accepted": bool(s0_first),
            "S1_first_accepted": bool(s1_first),
        }
        trial_records.append(record)
        if s0_first:
            selected_candidate = candidate.copy()
            selected_candidate.setflags(write=False)
            selected["S0"] = {
                "origin": "trial", "j": exponent, "alpha": alpha,
                "selected_float64": selected_candidate, "metrics": copy.deepcopy(metrics),
            }
        if s1_first:
            selected_candidate = candidate.copy()
            selected_candidate.setflags(write=False)
            selected["S1"] = {
                "origin": "trial", "j": exponent, "alpha": alpha,
                "selected_float64": selected_candidate, "metrics": copy.deepcopy(metrics),
            }
        if selected["S0"] is not None and selected["S1"] is not None:
            break
    for label in ("S0", "S1"):
        if selected[label] is None:
            selected_candidate = d64.copy()
            selected_candidate.setflags(write=False)
            selected[label] = {
                "origin": "D", "j": None, "alpha": None,
                "selected_float64": selected_candidate, "metrics": copy.deepcopy(d_metrics),
            }

    c0_float32 = np.asarray(view.c0, dtype=np.float64).astype(np.float32)
    c0_float32 = np.ascontiguousarray(c0_float32, dtype=np.float32)
    c0_float32.setflags(write=False)
    postcast: dict[str, dict[str, Any]] = {}
    for label in ("S0", "S1"):
        selection = selected[label]
        if selection is None:
            raise RuntimeError("internal search selection was not initialized")
        raw_cast = np.ascontiguousarray(selection["selected_float64"].astype(np.float32), dtype=np.float32)
        if not np.isfinite(raw_cast).all():
            raise FloatingPointError(f"{label} float32 cast is nonfinite")
        raw_cast.setflags(write=False)
        raw_cast_metrics = oracle.evaluate(raw_cast.astype(np.float64), "postcast")
        gate_pass = bool(raw_cast_metrics["new_train_kl"] <= b + _KL_TOLERANCE)
        fallback = not gate_pass
        final_candidate = c0_float32.copy() if fallback else raw_cast.copy()
        final_candidate.setflags(write=False)
        if fallback:
            final_metrics = baseline["C0"]
        else:
            final_metrics = raw_cast_metrics
        witness = bool(
            _array_sha256(final_candidate, "<f4") != _array_sha256(anchor_set.D, "<f4")
            and final_metrics["new_train_kl"] <= b + _KL_TOLERANCE
            and final_metrics["fullL"] < d_metrics["fullL"] - _WITNESS_TOLERANCE
            and final_metrics["old_train_kl"] < d_metrics["old_train_kl"] - _WITNESS_TOLERANCE
        )
        postcast[label] = {
            "raw_cast_candidate": raw_cast,
            "raw_cast_metrics": copy.deepcopy(raw_cast_metrics),
            "post_cast_gate_pass": gate_pass,
            "gate_tolerance": _KL_TOLERANCE,
            "fallback_applied": fallback,
            "candidate": final_candidate,
            "final_metrics": copy.deepcopy(final_metrics),
            "witness": witness,
            "candidate_sha256": _candidate_sha256(final_candidate, np.float32),
            "raw_sha256": _array_sha256(final_candidate, "<f4"),
            "raw_cast_candidate_sha256": _candidate_sha256(raw_cast, np.float32),
            "raw_cast_sha256": _array_sha256(raw_cast, "<f4"),
        }

    trace: dict[str, Any] = {
        "format": _FORMAT,
        "seed": int(view.seed),
        "prepared_id": view.prepared_id,
        "training_hash": view.training_hash,
        "input_identity": json.loads(view.input_identity_json),
        "binding": _BINDING,
        "lambda_value": _LAMBDA,
        "ridge": _RIDGE,
        "new_kl_slack": _KL_TOLERANCE,
        "witness_tolerance": _WITNESS_TOLERANCE,
        "anchor_manifest_sha256": anchor_set.manifest_sha256,
        "v1_source_identity_sha256": anchor_set.v1_source_identity_sha256,
        "e164_source_identity_sha256": anchor_set.e164_source_identity_sha256,
        "e164_cohort_manifest_sha256": anchor_set.e164_cohort_manifest_sha256,
        "legacy_source_references": copy.deepcopy(anchor_set.source_references),
        "config": copy.deepcopy(fixed),
        "anchor_identities": {
            "C": {"scientific_sha256": _candidate_sha256(anchor_set.C, np.float32),
                  "raw_sha256": _array_sha256(anchor_set.C, "<f4")},
            "D": {"scientific_sha256": _candidate_sha256(anchor_set.D, np.float32),
                  "raw_sha256": _array_sha256(anchor_set.D, "<f4")},
            "U32": {"scientific_sha256": _candidate_sha256(anchor_set.U32, np.float32),
                    "raw_sha256": _array_sha256(anchor_set.U32, "<f4")},
            "C0": {"scientific_sha256": _candidate_sha256(c0_float32, np.float32),
                   "raw_sha256": _array_sha256(c0_float32, "<f4")},
        },
        "anchor_arrays": {
            "C": anchor_set.C,
            "D": anchor_set.D,
            "U32": anchor_set.U32,
        },
        "c0_float32": c0_float32,
        "baseline": copy.deepcopy(baseline),
        "b": float(b),
        "D_fullL": float(d_metrics["fullL"]),
        "trials": trial_records,
        "search_stop": {
            "j": trial_records[-1]["j"] if trial_records else None,
            "reason": (
                "both_arms_selected" if all(selected[label]["origin"] == "trial" for label in ("S0", "S1"))
                else "fixed_trial_schedule_exhausted"
                if len(trial_records) == _MAX_TRIALS
                else "one_or_both_arms_retained_D"
            ),
        },
        "selections": selected,
        "postcast": postcast,
        "counters": oracle.finalized_counts(),
        "oracle": oracle.evaluation_manifest(),
    }
    trace["trace_sha256"] = _budgeted_trace_digest(trace)
    return _validate_budgeted_trace(view, trace)


def budgeted_trace_manifest(trace: dict[str, Any]) -> dict[str, Any]:
    """Export the complete scalar/hash and unique-evaluation evidence as JSON."""
    if type(trace) is not dict or trace.get("format") != _FORMAT:
        raise TypeError("trace must be an h036_budgeted_search_v1 dict")
    # The view is not needed to export; digest and selected identities are
    # checked below using the immutable trace's own training identity.
    if trace.get("trace_sha256") != _budgeted_trace_digest(trace):
        raise ValueError("budgeted trace SHA-256 does not match its contents")
    result = _jsonable_trace_value(trace, include_array_bytes=True)
    _json_bytes(result)
    return result


def _budgeted_lock_digest(locked: Mapping[str, Any]) -> str:
    body = {key: value for key, value in locked.items() if key != "lock_sha256"}
    digest = hashlib.sha256(b"h036_budgeted_locked_v1\0")
    digest.update(_json_bytes(_jsonable_trace_value(body, include_array_bytes=False)))
    return digest.hexdigest()


def _validate_budgeted_locked(locked: Any) -> dict[str, Any]:
    if type(locked) is not dict or locked.get("format") != _LOCK_FORMAT:
        raise TypeError("locked must be an h036_budgeted_locked_v1 dict")
    if locked.get("diagnostic_only") is not True or locked.get("binding") != _BINDING:
        raise ValueError("budgeted lock must be diagnostic-only and wildcard-bound")
    if locked.get("lock_sha256") != _budgeted_lock_digest(locked):
        raise ValueError("budgeted lock SHA-256 does not match its contents")
    if not _is_sha256(locked.get("anchor_manifest_sha256")):
        raise ValueError("locked anchor manifest SHA-256 is invalid")
    for key in ("v1_source_identity_sha256", "e164_source_identity_sha256", "e164_cohort_manifest_sha256"):
        if not _is_sha256(locked.get(key)):
            raise ValueError(f"locked {key} is invalid")
    if (locked["v1_source_identity_sha256"] != _V1_SOURCE_IDENTITY_SHA256
            or locked["e164_source_identity_sha256"] != _E164_SOURCE_IDENTITY_SHA256
            or locked["e164_cohort_manifest_sha256"] != _E164_COHORT_MANIFEST_SHA256
            or locked["anchor_manifest_sha256"] != _ANCHOR_MANIFEST_SHA256):
        raise ValueError("locked legacy source or anchor manifest identity differs from the frozen source")
    if not _is_sha256(locked.get("trace_sha256")) or not _is_sha256(locked.get("training_hash")):
        raise ValueError("locked trace or training identity is invalid")
    if isinstance(locked.get("seed"), bool) or locked.get("seed") not in _SEEDS:
        raise ValueError("locked seed is outside the fixed 16-prefix cohort")
    if not isinstance(locked.get("prepared_id"), str) or not locked["prepared_id"]:
        raise ValueError("locked prepared_id is invalid")
    _identity_json(locked.get("input_identity"), "locked input_identity")
    if locked.get("lambda_value") != _LAMBDA or locked.get("ridge") != _RIDGE:
        raise ValueError("locked objective constants differ from the frozen contract")
    b = _finite_number(locked.get("b"), "locked b")
    _validate_training_metrics(locked.get("baseline", {}).get("C0"), "locked C0 baseline")
    _validate_training_metrics(locked.get("baseline", {}).get("C"), "locked C baseline")
    _validate_training_metrics(locked.get("baseline", {}).get("D"), "locked D baseline")
    identities = locked.get("anchor_identities")
    if type(identities) is not dict or set(identities) != {"C0", "C", "D", "U32"}:
        raise ValueError("locked anchor identities are incomplete")
    for label in ("C0", "C", "D", "U32"):
        entry = identities[label]
        if type(entry) is not dict or set(entry) != {"scientific_sha256", "raw_sha256"}:
            raise ValueError(f"locked {label} anchor identity has invalid fields")
        if not all(_is_sha256(entry[key]) for key in ("scientific_sha256", "raw_sha256")):
            raise ValueError(f"locked {label} anchor hashes are invalid")
    c0_identity = identities["C0"]
    if locked.get("c0_candidate_sha256") != c0_identity["scientific_sha256"] or locked.get("c0_raw_sha256") != c0_identity["raw_sha256"]:
        raise ValueError("locked C0 hashes differ from anchor identities")
    probes = locked.get("probes")
    if type(probes) is not dict or set(probes) != {"S0", "S1"}:
        raise ValueError("locked probes must contain exactly S0 and S1")
    for label, entry in probes.items():
        if type(entry) is not dict:
            raise TypeError(f"locked {label} probe must be a plain dict")
        required_arrays = ("candidate", "raw_cast_candidate")
        for field in required_arrays:
            array = entry.get(field)
            if not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.float32) or array.shape != _SHAPE:
                raise ValueError(f"locked {label}.{field} must be float32 with shape (3, 3)")
            if not np.isfinite(array).all():
                raise FloatingPointError(f"locked {label}.{field} is nonfinite")
        if entry["candidate_sha256"] != _candidate_sha256(entry["candidate"], np.float32):
            raise ValueError(f"locked {label} scientific candidate SHA mismatch")
        if entry["raw_sha256"] != _array_sha256(entry["candidate"], "<f4"):
            raise ValueError(f"locked {label} raw candidate SHA mismatch")
        if entry["raw_cast_candidate_sha256"] != _candidate_sha256(entry["raw_cast_candidate"], np.float32):
            raise ValueError(f"locked {label} raw-cast scientific SHA mismatch")
        if entry["raw_cast_sha256"] != _array_sha256(entry["raw_cast_candidate"], "<f4"):
            raise ValueError(f"locked {label} raw-cast byte SHA mismatch")
        if not isinstance(entry.get("post_cast_gate_pass"), bool) or not isinstance(entry.get("fallback_applied"), bool):
            raise ValueError(f"locked {label} gate fields must be boolean")
        if entry["fallback_applied"] is entry["post_cast_gate_pass"]:
            raise ValueError(f"locked {label} fallback must be the negation of the post-cast gate")
        if entry["fallback_applied"]:
            if entry["candidate_sha256"] != locked["c0_candidate_sha256"] or entry["raw_sha256"] != locked["c0_raw_sha256"]:
                raise ValueError(f"locked {label} fallback candidate differs from C0")
        elif not np.array_equal(entry["candidate"], entry["raw_cast_candidate"]):
            raise ValueError(f"locked {label} candidate differs from raw cast without fallback")
        _validate_training_metrics(entry.get("selected_metrics"), f"locked {label} selected metrics")
        _validate_training_metrics(entry.get("raw_cast_metrics"), f"locked {label} raw cast metrics")
        _validate_training_metrics(entry.get("final_metrics"), f"locked {label} final metrics")
        if _finite_number(entry.get("new_kl_bound"), f"locked {label} new KL bound") != b:
            raise ValueError(f"locked {label} new KL bound differs from C0")
        if entry.get("gate_tolerance") != _KL_TOLERANCE:
            raise ValueError(f"locked {label} gate tolerance differs from fixed value")
        if not isinstance(entry.get("witness"), bool):
            raise ValueError(f"locked {label} witness field must be boolean")
        for field in ("selected_float64_sha256", "selected_float64_raw_sha256"):
            if not _is_sha256(entry.get(field)):
                raise ValueError(f"locked {label} selected float64 identity is invalid")
    counters = locked.get("counters")
    if type(counters) is not dict:
        raise ValueError("locked counters are missing")
    actual = counters.get("actual_oracle_evaluations")
    if isinstance(actual, bool) or not isinstance(actual, numbers.Integral) or not 1 <= actual <= _MAX_ORACLE_EVALUATIONS:
        raise ValueError("locked actual oracle evaluation count is outside the frozen limit")
    if counters.get("cache_misses") != actual or counters.get("cache_hits") + actual != counters.get("logical_requests"):
        raise ValueError("locked oracle cache counters are inconsistent")
    if counters.get("trial_requests", 0) > _MAX_TRIALS or counters.get("postcast_requests") != 2:
        raise ValueError("locked trial or post-cast request count is outside the fixed procedure")
    return locked


def lock_budgeted_pair(training_view: TrainingView, trace: dict[str, Any]) -> dict[str, Any]:
    """Freeze selected float64 paths and their already-computed float32 checks."""
    view = _validate_training_view(training_view)
    search_trace = _validate_budgeted_trace(view, trace)
    c0 = search_trace["c0_float32"]
    probes: dict[str, dict[str, Any]] = {}
    for label in ("S0", "S1"):
        selection = search_trace["selections"][label]
        cast = search_trace["postcast"][label]
        selected64 = selection["selected_float64"]
        probes[label] = {
            "origin": selection["origin"],
            "j": selection["j"],
            "alpha": selection["alpha"],
            "selected_float64_sha256": _candidate_sha256(selected64, np.float64),
            "selected_float64_raw_sha256": _array_sha256(selected64, "<f8"),
            "candidate": cast["candidate"],
            "candidate_sha256": cast["candidate_sha256"],
            "raw_sha256": cast["raw_sha256"],
            "raw_cast_candidate": cast["raw_cast_candidate"],
            "raw_cast_candidate_sha256": cast["raw_cast_candidate_sha256"],
            "raw_cast_sha256": cast["raw_cast_sha256"],
            "selected_metrics": copy.deepcopy(selection["metrics"]),
            "raw_cast_metrics": copy.deepcopy(cast["raw_cast_metrics"]),
            "final_metrics": copy.deepcopy(cast["final_metrics"]),
            "new_kl_bound": float(search_trace["b"]),
            "gate_tolerance": _KL_TOLERANCE,
            "post_cast_gate_pass": bool(cast["post_cast_gate_pass"]),
            "fallback_applied": bool(cast["fallback_applied"]),
            "witness": bool(cast["witness"]),
        }
    for label, entry in probes.items():
        entry["aliases"] = sorted(
            other for other, other_entry in probes.items()
            if other_entry["raw_sha256"] == entry["raw_sha256"]
        )
    locked: dict[str, Any] = {
        "format": _LOCK_FORMAT,
        "seed": search_trace["seed"],
        "prepared_id": search_trace["prepared_id"],
        "training_hash": search_trace["training_hash"],
        "input_identity": copy.deepcopy(search_trace["input_identity"]),
        "logical_case": search_trace["input_identity"].get("logical_case"),
        "binding": _BINDING,
        "lambda_value": _LAMBDA,
        "ridge": _RIDGE,
        "diagnostic_only": True,
        "anchor_manifest_sha256": search_trace["anchor_manifest_sha256"],
        "v1_source_identity_sha256": search_trace["v1_source_identity_sha256"],
        "e164_source_identity_sha256": search_trace["e164_source_identity_sha256"],
        "e164_cohort_manifest_sha256": search_trace["e164_cohort_manifest_sha256"],
        "legacy_source_references": copy.deepcopy(search_trace["legacy_source_references"]),
        "trace_sha256": search_trace["trace_sha256"],
        "b": float(search_trace["b"]),
        "D_fullL": float(search_trace["D_fullL"]),
        "baseline": copy.deepcopy(search_trace["baseline"]),
        "anchor_identities": copy.deepcopy(search_trace["anchor_identities"]),
        "c0_candidate_sha256": _candidate_sha256(c0, np.float32),
        "c0_raw_sha256": _array_sha256(c0, "<f4"),
        "counters": copy.deepcopy(search_trace["counters"]),
        "oracle_evaluation_sha256": _sha256(_json_bytes(search_trace["oracle"])),
        "probes": probes,
    }
    locked["lock_sha256"] = _budgeted_lock_digest(locked)
    return _validate_budgeted_locked(locked)


def locked_budgeted_manifest(locked: dict[str, Any]) -> dict[str, Any]:
    """Return the array-free JSON identity for a verified lock."""
    value = _validate_budgeted_locked(locked)
    result = _jsonable_trace_value(value, include_array_bytes=False)
    _json_bytes(result)
    return result


def _budgeted_cohort_digest(permit: Mapping[str, Any]) -> str:
    body = {key: value for key, value in permit.items() if key != "cohort_lock_sha256"}
    digest = hashlib.sha256(b"h036_budgeted_cohort_v1\0")
    digest.update(_json_bytes(body))
    return digest.hexdigest()


def _validate_budgeted_cohort(permit: Any) -> dict[str, Any]:
    if type(permit) is not dict or permit.get("format") != _COHORT_FORMAT:
        raise TypeError("cohort_permit must be an h036_budgeted_cohort_v1 dict")
    required = {
        "format", "seeds", "source_identity_sha256", "anchor_manifest_sha256",
        "score_packages_max", "cases", "cohort_lock_sha256",
    }
    if set(permit) != required:
        raise ValueError(f"cohort permit fields must equal {sorted(required)}")
    if permit["seeds"] != list(_SEEDS) or type(permit["seeds"]) is not list:
        raise ValueError("cohort permit must bind the exact frozen 16-prefix seed list")
    if not _is_sha256(permit["source_identity_sha256"]) or not _is_sha256(permit["anchor_manifest_sha256"]):
        raise ValueError("cohort permit source or anchor manifest SHA is invalid")
    if permit["score_packages_max"] != _MAX_SCORE_PACKAGES:
        raise ValueError("cohort permit score package cap differs from the fixed limit")
    if type(permit["cases"]) is not dict or set(permit["cases"]) != {str(seed) for seed in _SEEDS}:
        raise ValueError("cohort permit cases differ from the exact fixed seed set")
    total_oracles = 0
    for seed in _SEEDS:
        case = permit["cases"][str(seed)]
        case_fields = {
            "seed", "prepared_id", "training_hash", "input_identity", "logical_case",
            "binding", "anchor_manifest_sha256", "v1_source_identity_sha256",
            "e164_source_identity_sha256", "e164_cohort_manifest_sha256",
            "lock_sha256", "trace_sha256", "actual_oracle_evaluations",
            "c0_candidate_sha256", "c0_raw_sha256", "d_candidate_sha256", "d_raw_sha256",
            "probes",
        }
        if type(case) is not dict or set(case) != case_fields:
            raise ValueError(f"cohort case {seed} has invalid fields")
        if case["seed"] != seed or case["binding"] != _BINDING:
            raise ValueError(f"cohort case {seed} seed or binding is invalid")
        if not isinstance(case["prepared_id"], str) or not case["prepared_id"]:
            raise ValueError(f"cohort case {seed} prepared_id is invalid")
        _identity_json(case["input_identity"], f"cohort case {seed} input_identity")
        for key in (
            "training_hash", "anchor_manifest_sha256", "v1_source_identity_sha256",
            "e164_source_identity_sha256", "e164_cohort_manifest_sha256",
            "lock_sha256", "trace_sha256", "c0_candidate_sha256", "c0_raw_sha256",
            "d_candidate_sha256", "d_raw_sha256",
        ):
            if not _is_sha256(case[key]):
                raise ValueError(f"cohort case {seed} {key} is invalid")
        if (case["v1_source_identity_sha256"] != _V1_SOURCE_IDENTITY_SHA256
                or case["e164_source_identity_sha256"] != _E164_SOURCE_IDENTITY_SHA256
                or case["e164_cohort_manifest_sha256"] != _E164_COHORT_MANIFEST_SHA256
                or case["anchor_manifest_sha256"] != _ANCHOR_MANIFEST_SHA256):
            raise ValueError(f"cohort case {seed} legacy source identity differs from the frozen source")
        if case["anchor_manifest_sha256"] != permit["anchor_manifest_sha256"]:
            raise ValueError(f"cohort case {seed} uses a different anchor manifest")
        count = case["actual_oracle_evaluations"]
        # C0, C, and D are three logical baseline requests, but their
        # candidate bytes can alias. Count only distinct cache misses here;
        # the minimum is therefore one real oracle evaluation, not three.
        if isinstance(count, bool) or not isinstance(count, numbers.Integral) or not 1 <= count <= _MAX_ORACLE_EVALUATIONS:
            raise ValueError(f"cohort case {seed} actual oracle count is outside the fixed limit")
        total_oracles += int(count)
        probes = case["probes"]
        if type(probes) is not dict or set(probes) != {"S0", "S1"}:
            raise ValueError(f"cohort case {seed} must contain S0 and S1 probe identities")
        for label, probe in probes.items():
            if type(probe) is not dict or set(probe) != {"candidate_sha256", "raw_sha256", "fallback_applied", "witness"}:
                raise ValueError(f"cohort case {seed} {label} identity is invalid")
            if not _is_sha256(probe["candidate_sha256"]) or not _is_sha256(probe["raw_sha256"]):
                raise ValueError(f"cohort case {seed} {label} candidate hashes are invalid")
            if not isinstance(probe["fallback_applied"], bool) or not isinstance(probe["witness"], bool):
                raise ValueError(f"cohort case {seed} {label} selection booleans are invalid")
    if total_oracles > _MAX_ORACLE_EVALUATIONS * len(_SEEDS):
        raise ValueError("cohort permit exceeds the fixed 336 actual training oracle cap")
    if permit["cohort_lock_sha256"] != _budgeted_cohort_digest(permit):
        raise ValueError("cohort permit SHA-256 does not match its contents")
    return permit


def seal_budgeted_cohort(
    locked_pairs: list[dict[str, Any]], *,
    source_identity_sha256: str, anchor_manifest_sha256: str,
) -> dict[str, Any]:
    """Issue the immutable scoring permit only after all 16 pairs are locked."""
    if type(locked_pairs) is not list or len(locked_pairs) != len(_SEEDS):
        raise ValueError("locked_pairs must be an exact 16-element list")
    if not _is_sha256(source_identity_sha256):
        raise ValueError("source_identity_sha256 must identify the frozen E165 source snapshot")
    if not _is_sha256(anchor_manifest_sha256):
        raise ValueError("anchor_manifest_sha256 must be a SHA-256 string")
    validated = [_validate_budgeted_locked(item) for item in locked_pairs]
    seeds = [item["seed"] for item in validated]
    if seeds != list(_SEEDS):
        raise ValueError("locked_pairs must be ordered by the exact frozen 16 seeds")
    cases: dict[str, Any] = {}
    for locked in validated:
        if locked["anchor_manifest_sha256"] != anchor_manifest_sha256:
            raise ValueError("locked pair anchor manifest differs from the cohort request")
        probes = locked["probes"]
        cases[str(locked["seed"])] = {
            "seed": locked["seed"],
            "prepared_id": locked["prepared_id"],
            "training_hash": locked["training_hash"],
            "input_identity": copy.deepcopy(locked["input_identity"]),
            "logical_case": locked["logical_case"],
            "binding": locked["binding"],
            "anchor_manifest_sha256": locked["anchor_manifest_sha256"],
            "v1_source_identity_sha256": locked["v1_source_identity_sha256"],
            "e164_source_identity_sha256": locked["e164_source_identity_sha256"],
            "e164_cohort_manifest_sha256": locked["e164_cohort_manifest_sha256"],
            "lock_sha256": locked["lock_sha256"],
            "trace_sha256": locked["trace_sha256"],
            "actual_oracle_evaluations": locked["counters"]["actual_oracle_evaluations"],
            "c0_candidate_sha256": locked["c0_candidate_sha256"],
            "c0_raw_sha256": locked["c0_raw_sha256"],
            "d_candidate_sha256": locked["anchor_identities"]["D"]["scientific_sha256"],
            "d_raw_sha256": locked["anchor_identities"]["D"]["raw_sha256"],
            "probes": {
                label: {
                    "candidate_sha256": probes[label]["candidate_sha256"],
                    "raw_sha256": probes[label]["raw_sha256"],
                    "fallback_applied": probes[label]["fallback_applied"],
                    "witness": probes[label]["witness"],
                }
                for label in ("S0", "S1")
            },
        }
    permit: dict[str, Any] = {
        "format": _COHORT_FORMAT,
        "seeds": list(_SEEDS),
        "source_identity_sha256": source_identity_sha256,
        "anchor_manifest_sha256": anchor_manifest_sha256,
        "score_packages_max": _MAX_SCORE_PACKAGES,
        "cases": cases,
    }
    permit["cohort_lock_sha256"] = _budgeted_cohort_digest(permit)
    _json_bytes(permit)
    return _validate_budgeted_cohort(permit)


def _prepared_identity(
    prepared: Any, *, seed: int, prepared_id: str, input_identity: dict[str, Any],
) -> dict[str, Any]:
    if type(prepared) is not dict or prepared.get("format") != "h036_prepared_v1":
        raise TypeError("prepared must be an h036_prepared_v1 plain dict")
    actual_seed = prepared.get("seed")
    if isinstance(actual_seed, (bool, np.bool_)) or not isinstance(actual_seed, numbers.Integral) or int(actual_seed) != seed:
        raise ValueError("prepared seed differs from the locked cohort case")
    if prepared.get("prepared_id") != prepared_id:
        raise ValueError("prepared_id differs from the locked cohort case")
    actual_identity = prepared.get("input_identity")
    if _identity_json(actual_identity, "prepared input_identity") != _identity_json(input_identity, "locked input_identity"):
        raise ValueError("prepared input identity differs from the locked cohort case")
    return prepared


def _prepared_c0_raw_sha(prepared: dict[str, Any]) -> str:
    mean = prepared.get("mean64")
    if not isinstance(mean, np.ndarray) or mean.dtype != np.dtype(np.float64) or mean.shape != _SHAPE:
        raise ValueError("prepared.mean64 is required to bind the locked C0 identity")
    if not np.isfinite(mean).all():
        raise FloatingPointError("prepared.mean64 is nonfinite")
    c0 = np.ascontiguousarray(mean.astype(np.float32), dtype=np.float32)
    return _array_sha256(c0, "<f4")


def _fit_adapter(prepared: dict[str, Any], candidate: np.ndarray) -> dict[str, Any]:
    value = _copy_readonly(candidate, np.float32, _SHAPE, "score candidate")
    candidate_hash = _candidate_sha256(value, np.float32)
    return {
        "format": "h036_fit_v1",
        "prepared_id": prepared["prepared_id"],
        "lambda_value": _LAMBDA,
        "candidate": np.array(value, dtype=np.float32, order="C", copy=True),
        "diagnostics": {"candidate_sha256": candidate_hash},
    }


_EXPECTED_SCORE_FIELDS = (
    "pre_codec_fresh", "encoded_fresh", "W0_fresh",
    "K_old", "K_new", "K_new_W0", "G_new", "candidate_sha256", "binding",
)


def _score_metrics_projection(value: Any, name: str) -> dict[str, Any]:
    if type(value) is dict and value.get("format") == "h036_score_metrics_v1" and type(value.get("metrics")) is dict:
        value = value["metrics"]
    if type(value) is not dict or not set(_EXPECTED_SCORE_FIELDS) <= set(value):
        raise ValueError(f"{name} lacks required v1 scientific score fields")
    projection = {key: value[key] for key in _EXPECTED_SCORE_FIELDS}
    try:
        _json_bytes(projection)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} scientific fields must be finite JSON data") from exc
    if projection["binding"] != _BINDING or not _is_sha256(projection["candidate_sha256"]):
        raise ValueError(f"{name} candidate or binding identity is invalid")
    return projection


def _score_locked_candidate(prepared: dict[str, Any], candidate: np.ndarray) -> dict[str, Any]:
    from . import retention_protected_inputs as inputs

    fit_result = _fit_adapter(prepared, candidate)
    return inputs.encode_and_score_stage0(prepared, fit_result, _BINDING)


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    return _sha256(b"h036_budgeted_anchor_receipt_v1\0" + _json_bytes(body))


def _validate_anchor_receipt(receipt: Any, permit: dict[str, Any]) -> dict[str, Any]:
    if type(receipt) is not dict or receipt.get("format") != _ANCHOR_RECEIPT_FORMAT:
        raise TypeError("anchor_receipt must be a budgeted D anchor receipt")
    expected_fields = {
        "format", "seed", "prepared_id", "cohort_lock_sha256",
        "source_identity_sha256", "anchor_manifest_sha256",
        "legacy_source_identities", "candidate_sha256", "actor_sha256",
        "anchor_scientific_sha256", "score_scientific_sha256", "receipt_sha256",
    }
    if set(receipt) != expected_fields:
        raise ValueError("anchor receipt fields are invalid")
    if receipt.get("receipt_sha256") != _receipt_digest(receipt):
        raise ValueError("anchor receipt SHA-256 does not match its contents")
    if receipt.get("seed") != _ANCHOR_SEED:
        raise ValueError("anchor receipt must refer to fixed seed 26201")
    if receipt["cohort_lock_sha256"] != permit["cohort_lock_sha256"]:
        raise ValueError("anchor receipt belongs to a different sealed cohort")
    if receipt["source_identity_sha256"] != permit["source_identity_sha256"]:
        raise ValueError("anchor receipt belongs to a different E165 source snapshot")
    if receipt["anchor_manifest_sha256"] != permit["anchor_manifest_sha256"]:
        raise ValueError("anchor receipt belongs to a different anchor manifest")
    for key in ("candidate_sha256", "actor_sha256", "anchor_scientific_sha256", "score_scientific_sha256"):
        if not _is_sha256(receipt[key]):
            raise ValueError(f"anchor receipt {key} is invalid")
    case = permit["cases"][str(_ANCHOR_SEED)]
    if receipt["prepared_id"] != case["prepared_id"] or receipt["candidate_sha256"] != case["d_candidate_sha256"]:
        raise ValueError("anchor receipt D candidate or prepared identity differs from sealed cohort")
    expected_legacy = {
        "v1_source_identity_sha256": case["v1_source_identity_sha256"],
        "e164_source_identity_sha256": case["e164_source_identity_sha256"],
        "e164_cohort_manifest_sha256": case["e164_cohort_manifest_sha256"],
    }
    if receipt["legacy_source_identities"] != expected_legacy:
        raise ValueError("anchor receipt legacy source identities differ from the sealed cohort")
    return receipt


def score_budgeted_anchor(
    prepared: dict[str, Any], anchors: AnchorSet, *,
    cohort_permit: dict[str, Any], expected_metrics: dict[str, Any],
    expected_actor_sha256: str,
) -> dict[str, Any]:
    """Score only the fixed seed-26201 D anchor and issue its verified receipt."""
    permit = _validate_budgeted_cohort(cohort_permit)
    anchor_set = _validate_anchor_set(anchors)
    if anchor_set.seed != _ANCHOR_SEED:
        raise ValueError("score anchor must use seed 26201")
    case = permit["cases"][str(_ANCHOR_SEED)]
    _prepared_identity(
        prepared, seed=_ANCHOR_SEED, prepared_id=anchor_set.prepared_id,
        input_identity=anchor_set.input_identity,
    )
    if (case["prepared_id"] != anchor_set.prepared_id
            or case["anchor_manifest_sha256"] != anchor_set.manifest_sha256
            or case["d_candidate_sha256"] != _candidate_sha256(anchor_set.D, np.float32)
            or case["d_raw_sha256"] != _array_sha256(anchor_set.D, "<f4")):
        raise ValueError("seed-26201 D anchor differs from the sealed cohort")
    if not _is_sha256(expected_actor_sha256):
        raise ValueError("expected_actor_sha256 must be a SHA-256 string")
    expected_projection = _score_metrics_projection(expected_metrics, "expected_metrics")
    d_sha = _candidate_sha256(anchor_set.D, np.float32)
    if expected_projection["candidate_sha256"] != d_sha:
        raise ValueError("historical expected score candidate does not match D")
    if expected_projection["binding"] != _BINDING:
        raise ValueError("historical expected score binding differs from wildcard")
    score = _score_locked_candidate(prepared, anchor_set.D)
    metrics = _score_metrics_projection(score.get("metrics"), "anchor score")
    if _json_bytes(metrics) != _json_bytes(expected_projection):
        raise ValueError("fresh seed-26201 D scientific score fields differ from historical values")
    deployment = score.get("deployment")
    actor_bytes = deployment.get("actor_bytes") if type(deployment) is dict else None
    if not isinstance(actor_bytes, bytes):
        raise ValueError("anchor score lacks encoded actor bytes")
    actor_sha = _sha256(actor_bytes)
    if actor_sha != expected_actor_sha256:
        raise ValueError("seed-26201 D encoded actor SHA differs from historical actor bytes")
    if score["metrics"].get("artifacts", {}).get("actor_sha256") != actor_sha:
        raise ValueError("anchor score actor SHA differs from its metrics identity")
    anchor_scientific_sha = _sha256(_json_bytes({
        "candidate_sha256": d_sha,
        "raw_sha256": _array_sha256(anchor_set.D, "<f4"),
        "binding": _BINDING,
        "actor_sha256": actor_sha,
        "metrics": metrics,
    }))
    receipt: dict[str, Any] = {
        "format": _ANCHOR_RECEIPT_FORMAT,
        "seed": _ANCHOR_SEED,
        "prepared_id": anchor_set.prepared_id,
        "cohort_lock_sha256": permit["cohort_lock_sha256"],
        "source_identity_sha256": permit["source_identity_sha256"],
        "anchor_manifest_sha256": permit["anchor_manifest_sha256"],
        "legacy_source_identities": {
            "v1_source_identity_sha256": anchor_set.v1_source_identity_sha256,
            "e164_source_identity_sha256": anchor_set.e164_source_identity_sha256,
            "e164_cohort_manifest_sha256": anchor_set.e164_cohort_manifest_sha256,
        },
        "candidate_sha256": d_sha,
        "actor_sha256": actor_sha,
        "anchor_scientific_sha256": anchor_scientific_sha,
        "score_scientific_sha256": _sha256(_json_bytes(metrics)),
    }
    receipt["receipt_sha256"] = _receipt_digest(receipt)
    _json_bytes(receipt)
    return {"score": score, "receipt": receipt}


def score_locked_budgeted_probe(
    prepared: dict[str, Any], locked_pair: dict[str, Any], label: str, *,
    cohort_permit: dict[str, Any], anchor_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Score one already locked arm through the original diagnostic-only v1 codec."""
    permit = _validate_budgeted_cohort(cohort_permit)
    locked = _validate_budgeted_locked(locked_pair)
    receipt = _validate_anchor_receipt(anchor_receipt, permit)
    if label not in {"S0", "S1"}:
        raise ValueError("label must be S0 or S1")
    case = permit["cases"].get(str(locked["seed"]))
    if type(case) is not dict:
        raise ValueError("locked pair seed is absent from the sealed cohort")
    if locked["lock_sha256"] != case["lock_sha256"]:
        raise ValueError("locked pair hash differs from the sealed cohort case")
    if (locked["prepared_id"] != case["prepared_id"]
            or locked["training_hash"] != case["training_hash"]
            or locked["anchor_manifest_sha256"] != permit["anchor_manifest_sha256"]):
        raise ValueError("locked pair identity differs from the sealed cohort")
    _prepared_identity(
        prepared, seed=locked["seed"], prepared_id=locked["prepared_id"],
        input_identity=locked["input_identity"],
    )
    if _prepared_c0_raw_sha(prepared) != locked["c0_raw_sha256"]:
        raise ValueError("prepared C0 identity differs from the sealed training view")
    cohort_probe = case["probes"][label]
    entry = locked["probes"][label]
    if (entry["candidate_sha256"] != cohort_probe["candidate_sha256"]
            or entry["raw_sha256"] != cohort_probe["raw_sha256"]):
        raise ValueError("locked probe candidate differs from the sealed cohort")
    score = _score_locked_candidate(prepared, entry["candidate"])
    score_metrics = score.get("metrics")
    if type(score_metrics) is not dict or score_metrics.get("candidate_sha256") != entry["candidate_sha256"]:
        raise ValueError("scorer candidate hash differs from the locked probe")
    if score_metrics.get("binding") != _BINDING:
        raise ValueError("scorer binding differs from wildcard")
    encoded = score_metrics.get("encoded_fresh")
    if type(encoded) is not dict:
        raise ValueError("score result lacks encoded_fresh training metrics")
    encoded_new_train_kl = _finite_number(encoded.get("new_train"), "encoded_new_train_kl")
    b = float(locked["b"])
    signed_excess = encoded_new_train_kl - b
    budget_diagnostic = {
        "encoded_new_train_kl": encoded_new_train_kl,
        "b": b,
        "signed_excess": signed_excess,
        "violation": max(0.0, signed_excess),
        "gate_tolerance": _KL_TOLERANCE,
        "within_budget": bool(encoded_new_train_kl <= b + _KL_TOLERANCE),
    }
    selection = {
        "label": label,
        "origin": entry["origin"],
        "j": entry["j"],
        "alpha": entry["alpha"],
        "selected_float64_sha256": entry["selected_float64_sha256"],
        "selected_float64_raw_sha256": entry["selected_float64_raw_sha256"],
        "candidate_sha256": entry["candidate_sha256"],
        "raw_sha256": entry["raw_sha256"],
        "post_cast_gate_pass": entry["post_cast_gate_pass"],
        "fallback_applied": entry["fallback_applied"],
        "witness": entry["witness"],
        "trace_sha256": locked["trace_sha256"],
        "lock_sha256": locked["lock_sha256"],
        "cohort_lock_sha256": permit["cohort_lock_sha256"],
        "anchor_receipt_sha256": receipt["receipt_sha256"],
    }
    score["diagnostic_only"] = True
    score["selection"] = selection
    score["budget_diagnostic"] = budget_diagnostic
    return score


__all__ = [
    "AnchorSet",
    "TrainingView",
    "make_training_view",
    "load_anchor_references",
    "search_budgeted_pair",
    "budgeted_trace_manifest",
    "lock_budgeted_pair",
    "locked_budgeted_manifest",
    "seal_budgeted_cohort",
    "score_budgeted_anchor",
    "score_locked_budgeted_probe",
]

