"""Prepare frozen H036 inputs from the three allowlisted prefix files.

This module only decodes fixed learner, actor, and evidence payloads. It does
not fit a candidate, read outcomes, or inspect observer or simulator files.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import numbers
from pathlib import Path
import pickle
import struct
import time
from typing import Any, Mapping

import numpy as np
import psutil
import torch

from . import h033_inputs
from . import retention_behavioral_candidate as behavioral
from .storage_decision import effective_matrices, pair_mapping, task_name
from .write_decision_policy import decode_actor


_FORMAT = "h036_protected_config_v1"
_PREPARED_FORMAT = "h036_prepared_v1"
_PREFIX_SEEDS = tuple(range(26201, 26214)) + (26215, 26216, 26218)
_SIDECAR_KEYS = {
    "observations",
    "owner_indices",
    "source_slots",
    "generations",
    "bouts",
    "global_steps",
}
_PREFIX_FILES = ("checkpoint.pt", "actor.bin", "evidence.npz")
_FOUNDATION_SHA256 = "e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222"
_FOUNDATION_BYTES = 55477
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "h036_protected_v1.json"


def _expected_config() -> dict[str, Any]:
    return {
        "format": _FORMAT,
        "lambda_values": [0, 1],
        "seeds": list(_PREFIX_SEEDS),
        "rows_per_source": 128,
        "old_rows": 128,
        "train_rows": 96,
        "holdout_rows": 32,
        "temperature": 2.0,
        "epsilon": 0.05,
        "ridge": 1.0e-4,
        "max_iter": 64,
        "max_eval": 256,
        "history_size": 10,
        "lr": 1.0,
        "tolerance_grad": 1.0e-9,
        "tolerance_change": 1.0e-12,
        "new_kl_slack": 1.0e-12,
        "codec": "int8",
        "actor_budget_bytes": 89,
        "package_budget_bytes": 262144,
        "foundation": {"bytes": _FOUNDATION_BYTES, "sha256": _FOUNDATION_SHA256},
        "screen": {
            "guard": 1.0e-10,
            "old_kl_ratio_max": 0.75,
            "new_gain_ratio_min": 0.5,
            "min_prefixes": 12,
            "expected_prefixes": 16,
        },
    }


def validate_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load or validate the one frozen H036 input/fit configuration."""
    if config is None:
        try:
            loaded = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load frozen H036 config at {_CONFIG_PATH}") from exc
        if not isinstance(loaded, Mapping):
            raise ValueError("H036 config must be a JSON object")
        config = loaded
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    expected = _expected_config()
    if set(config) != set(expected):
        raise ValueError(f"H036 config keys must equal {sorted(expected)}")
    if config.get("format") != _FORMAT:
        raise ValueError("unsupported H036 config format")

    for name in ("lambda_values", "seeds"):
        value = config.get(name)
        if not isinstance(value, (list, tuple)) or list(value) != expected[name]:
            raise ValueError(f"H036 config {name} must equal the frozen value")
    for name in (
        "rows_per_source", "old_rows", "train_rows", "holdout_rows", "max_iter",
        "max_eval", "history_size", "actor_budget_bytes", "package_budget_bytes",
    ):
        value = config.get(name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            raise ValueError(f"H036 config {name} must be an integer")
        if int(value) != expected[name]:
            raise ValueError(f"H036 config {name} must equal {expected[name]}")
    for name in (
        "temperature", "epsilon", "ridge", "lr", "tolerance_grad", "tolerance_change",
        "new_kl_slack",
    ):
        value = config.get(name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"H036 config {name} must be numeric")
        if not math.isfinite(float(value)) or float(value) != expected[name]:
            raise ValueError(f"H036 config {name} must equal {expected[name]}")
    if config.get("codec") != expected["codec"]:
        raise ValueError("H036 config codec must be int8")

    foundation = config.get("foundation")
    if not isinstance(foundation, Mapping) or set(foundation) != {"bytes", "sha256"}:
        raise ValueError("H036 config foundation must contain only bytes and sha256")
    if foundation.get("bytes") != _FOUNDATION_BYTES or foundation.get("sha256") != _FOUNDATION_SHA256:
        raise ValueError("H036 config foundation identity differs from the frozen E161 source manifest")
    screen = config.get("screen")
    if not isinstance(screen, Mapping) or set(screen) != set(expected["screen"]):
        raise ValueError("H036 config screen fields differ from the frozen contract")
    for name, expected_value in expected["screen"].items():
        actual = screen.get(name)
        if isinstance(expected_value, int):
            if isinstance(actual, (bool, np.bool_)) or not isinstance(actual, numbers.Integral) or int(actual) != expected_value:
                raise ValueError(f"H036 config screen.{name} must equal {expected_value}")
        elif isinstance(actual, (bool, np.bool_)) or not isinstance(actual, numbers.Real) or float(actual) != expected_value:
            raise ValueError(f"H036 config screen.{name} must equal {expected_value}")
    return copy.deepcopy(dict(config))


def _seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (numbers.Integral, np.integer)):
        raise ValueError("checkpoint seed must be an integer")
    result = int(value)
    if result not in _PREFIX_SEEDS:
        raise ValueError("checkpoint seed is not one of the frozen E163 prefixes")
    return result


def _readonly_array(value: Any, dtype: np.dtype[Any] | type | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _value_digest(value: Any) -> str:
    """Hash a trusted in-memory checkpoint for synthetic provenance."""
    digest = hashlib.sha256()

    def add(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"torch:")
            digest.update(str(item.dtype).encode("ascii"))
            digest.update(repr(tuple(item.shape)).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        elif isinstance(item, np.ndarray):
            digest.update(b"numpy:")
            digest.update(_array_sha256(item).encode("ascii"))
        elif isinstance(item, np.generic):
            add(item.item())
        elif isinstance(item, Mapping):
            digest.update(b"mapping:")
            for key in sorted(item, key=lambda entry: repr(entry)):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"list:" if isinstance(item, list) else b"tuple:")
            digest.update(str(len(item)).encode("ascii"))
            for child in item:
                add(child)
        elif isinstance(item, bytes):
            digest.update(b"bytes:")
            digest.update(item)
        elif isinstance(item, (str, int, float, bool, type(None))):
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b":")
            digest.update(repr(item).encode("utf-8"))
        else:
            digest.update(b"pickle:")
            digest.update(pickle.dumps(item, protocol=5))

    add(value)
    return digest.hexdigest()


def _copy_features(features: behavioral.FrozenPolicyFeatures) -> behavioral.FrozenPolicyFeatures:
    return behavioral.FrozenPolicyFeatures(
        base_logits=_readonly_array(features.base_logits, np.float64),
        source_psi=_readonly_array(features.source_psi, np.float64),
        contrast_basis=_readonly_array(features.contrast_basis, np.float64),
        temperature=float(features.temperature),
        epsilon=float(features.epsilon),
    )


def _mapping_entry(mapping: Mapping[Any, Any], key: int) -> Any:
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key))


def _checkpoint_matrix(core_state: Mapping[str, Any], name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = core_state.get(name)
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != shape:
        raise ValueError(f"core_state.{name} must be float32 with shape {shape}")
    result = value.detach().cpu().numpy().copy()
    if not np.isfinite(result).all():
        raise ValueError(f"core_state.{name} must contain only finite values")
    return result


def _validate_actor(checkpoint: Mapping[str, Any], actor_data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(actor_data, bytes):
        raise TypeError("actor_data must be bytes")
    if len(actor_data) != 89:
        raise ValueError("actor.bin must be exactly 89 bytes")
    try:
        magic, version, codec, slots, groups, width = struct.unpack("<4sBBBBI", actor_data[:12])
    except struct.error as exc:
        raise ValueError("actor.bin has an invalid header") from exc
    if (magic, version, codec, slots, groups, width) != (b"WDK1", 1, 1, 2, 1, 27):
        raise ValueError("actor.bin must use the 89-byte int8 two-slot H036 layout")

    core_state = checkpoint.get("core_state")
    if not isinstance(core_state, Mapping):
        raise ValueError("checkpoint core_state must be a mapping")
    local = np.stack([
        _checkpoint_matrix(core_state, f"_nodes.{slot + 1}.A", (3, 3, 3)).reshape(27)
        for slot in range(2)
    ]).astype(np.float32, copy=False)
    shared_matrix = _checkpoint_matrix(core_state, "_shared_coefficients.0.C", (3, 3))
    shared = shared_matrix.reshape(1, 9).astype(np.float32, copy=False)
    groups_tensor = core_state.get("slot_groups")
    if not isinstance(groups_tensor, torch.Tensor) or groups_tensor.dtype != torch.int64 or tuple(groups_tensor.shape) != (2,):
        raise ValueError("core_state.slot_groups must be int64 with shape (2,)")
    groups_raw = groups_tensor.detach().cpu().numpy()
    if not np.array_equal(groups_raw, np.asarray([-1, -1], dtype=np.int64)):
        raise ValueError("checkpoint physical slot groups must be [-1, -1]")
    physical_groups = groups_raw.astype(np.int8, copy=True)
    if checkpoint.get("shared_frozen") is not True:
        raise ValueError("checkpoint shared_frozen must be true")
    parent_group = checkpoint.get("parent_group")
    if isinstance(parent_group, (bool, np.bool_)) or not isinstance(parent_group, numbers.Integral) or int(parent_group) != 0:
        raise ValueError("checkpoint parent_group must be zero")

    try:
        decoded_local, decoded_shared, decoded_groups = decode_actor(actor_data)
    except (TypeError, ValueError) as exc:
        raise ValueError("actor.bin cannot be decoded") from exc
    if not np.array_equal(decoded_local, local) or not np.array_equal(decoded_shared, shared):
        raise ValueError("actor.bin matrices differ from checkpoint core_state")
    if not np.array_equal(decoded_groups, physical_groups):
        raise ValueError("actor.bin slot groups differ from checkpoint core_state")

    generations = np.asarray(checkpoint.get("slot_generation"))
    if generations.shape != (2,) or generations.dtype != np.int64:
        raise ValueError("checkpoint slot_generation must be int64 with shape (2,)")
    logical_tags = np.asarray(checkpoint.get("slot_group_tags"))
    if logical_tags.shape != (2,) or logical_tags.dtype != np.int8 or not np.array_equal(logical_tags, [1, 1]):
        raise ValueError("checkpoint slot_group_tags must be int8 [1, 1]")
    if not np.isfinite(local).all() or not np.isfinite(shared).all():
        raise ValueError("decoded actor matrices must be finite")
    return local, shared, physical_groups, generations.copy(), logical_tags.copy()


def _sidecar_arrays(value: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or set(value) != _SIDECAR_KEYS:
        raise ValueError(f"evidence.npz keys must equal {sorted(_SIDECAR_KEYS)}")
    arrays = {name: np.asarray(value[name]) for name in _SIDECAR_KEYS}
    observations = arrays["observations"]
    if observations.dtype != np.float32 or observations.shape != (256, 13) or not np.isfinite(observations).all():
        raise ValueError("sidecar observations must be 256 finite float32 rows by 13")
    for name, dtype in (
        ("owner_indices", np.int8),
        ("source_slots", np.int8),
        ("generations", np.int64),
        ("bouts", np.int64),
        ("global_steps", np.int64),
    ):
        if arrays[name].dtype != np.dtype(dtype) or arrays[name].shape != (256,):
            raise ValueError(f"sidecar {name} must be {np.dtype(dtype)} with shape (256,)")
    owners = arrays["owner_indices"]
    slots = arrays["source_slots"]
    if np.any((owners < 0) | (owners > 1)) or not np.array_equal(np.bincount(owners.astype(np.int64), minlength=2), [128, 128]):
        raise ValueError("sidecar must contain exactly 128 rows per source owner")
    if np.any((slots < 0) | (slots > 1)):
        raise ValueError("sidecar source_slots must contain only zero and one")

    owner_slots: list[int] = []
    for owner in (0, 1):
        rows = np.flatnonzero(owners == owner)
        owner_slot_values = np.unique(slots[rows])
        if len(owner_slot_values) != 1:
            raise ValueError(f"owner {owner} must map to exactly one resident slot")
        slot = int(owner_slot_values[0])
        owner_slots.append(slot)
        if not np.array_equal(observations[rows, -1], np.full(128, slot + 1, dtype=np.float32)):
            raise ValueError(f"owner {owner} observations do not use route slot plus one")
        if np.any(np.diff(arrays["global_steps"][rows]) <= 0):
            raise ValueError(f"owner {owner} global_steps must strictly increase in source row order")
        if np.any(np.diff(arrays["bouts"][rows]) < 0):
            raise ValueError(f"owner {owner} bouts must be nondecreasing in source row order")
    if owner_slots[0] == owner_slots[1]:
        raise ValueError("the two source owners must map to distinct resident slots")
    return arrays


def _validate_cache(checkpoint: Mapping[str, Any], seed: int, owner_slots: list[int]) -> list[str]:
    cache = checkpoint.get("cache")
    if not isinstance(cache, Mapping):
        raise ValueError("checkpoint cache must be a mapping")
    capacity = cache.get("capacity")
    if isinstance(capacity, (bool, np.bool_)) or not isinstance(capacity, numbers.Integral) or int(capacity) != 2:
        raise ValueError("checkpoint cache capacity must be two")
    slots = cache.get("slot_to_task")
    mapping = cache.get("task_to_slot")
    if not isinstance(slots, (list, tuple)) or len(slots) != 2 or not isinstance(mapping, Mapping):
        raise ValueError("checkpoint cache must contain two slot tasks and a reverse mapping")
    if any(not isinstance(task, str) or not task for task in slots):
        raise ValueError("both current cache slots must contain task identities")
    if len(set(slots)) != 2 or len(mapping) != 2:
        raise ValueError("the two current cache slots must hold distinct tasks")
    for slot, task in enumerate(slots):
        if mapping.get(task) != slot:
            raise ValueError("checkpoint cache forward and reverse task mappings disagree")

    pair = pair_mapping(seed, "dynamics")
    allowed_new_tasks = {task_name(pair, 1, variant) for variant in (0, 1, 2)}
    if any(task not in allowed_new_tasks for task in slots):
        raise ValueError("resident source tasks do not match the seed's new-family task mapping")
    return [str(slots[slot]) for slot in owner_slots]


def _old_rows(checkpoint: Mapping[str, Any], seed: int) -> np.ndarray:
    evidence = checkpoint.get("evidence")
    if not isinstance(evidence, Mapping) or not isinstance(evidence.get("groups"), Mapping):
        raise ValueError("checkpoint evidence.groups must be a mapping")
    capacity = evidence.get("capacity")
    if isinstance(capacity, (bool, np.bool_)) or not isinstance(capacity, numbers.Integral) or not 128 <= int(capacity) <= 256:
        raise ValueError("checkpoint evidence capacity must be between 128 and 256")
    old = _mapping_entry(evidence["groups"], 0)
    if not isinstance(old, Mapping):
        raise ValueError("checkpoint evidence is missing old-group FIFO rows")
    try:
        observations = np.asarray(old["obs"])
        bouts = np.asarray(old["bout"])
        tasks = np.asarray(old["task"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("old-group FIFO observations, bouts, and tasks are required") from exc
    count = len(observations) if observations.ndim else 0
    if observations.dtype != np.float32 or observations.ndim != 2 or observations.shape[1] != 13:
        raise ValueError("old-group observations must be float32 N by 13")
    if count < 128 or count > int(capacity) or count > 256 or not np.isfinite(observations).all():
        raise ValueError("old-group FIFO must contain 128 to capacity finite observations")
    if bouts.shape != (count,) or bouts.dtype.kind not in "iu":
        raise ValueError("old-group FIFO bouts must be an integer vector matching observations")
    if tasks.shape != (count,) or tasks.dtype.kind not in "US":
        raise ValueError("old-group FIFO tasks must be a string vector matching observations")

    selected = slice(count - 128, count)
    selected_bouts = bouts[selected].astype(np.int64, copy=False)
    if np.any(np.diff(selected_bouts) < 0):
        raise ValueError("latest old-group FIFO bouts must be nondecreasing in original row order")
    pair = pair_mapping(seed, "dynamics")
    allowed_old_tasks = {task_name(pair, 0, variant) for variant in (0, 1, 2)}
    selected_tasks = [str(task) for task in tasks[selected].tolist()]
    if any(task not in allowed_old_tasks for task in selected_tasks):
        raise ValueError("latest old-group FIFO tasks do not match the seed's old-family mapping")
    return np.array(observations[selected], dtype=np.float32, copy=True, order="C")


def _serialize_sidecar(value: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{name: np.asarray(value[name]) for name in sorted(_SIDECAR_KEYS)})
    return buffer.getvalue()


def _rss_bytes() -> int | None:
    try:
        return int(psutil.Process().memory_info().rss)
    except (OSError, psutil.Error):
        return None


def _feature_array_bytes(features: behavioral.FrozenPolicyFeatures) -> int:
    return int(features.base_logits.nbytes + features.source_psi.nbytes + features.contrast_basis.nbytes)


def _identity_for_hash(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Keep relocation-stable logical identity; omit path-only provenance."""
    files = identity.get("files")
    if isinstance(files, Mapping):
        normalized_files: dict[str, str] = {}
        for name in _PREFIX_FILES:
            details = files.get(name)
            if isinstance(details, Mapping):
                file_digest = details.get("sha256")
            else:
                file_digest = details
            if not isinstance(file_digest, str):
                raise ValueError(f"input_identity is missing the {name} SHA-256")
            normalized_files[name] = file_digest
        return {"logical_case": identity.get("logical_case"), "files": normalized_files}
    return {
        key: value for key, value in identity.items()
        if key not in {"prefix_path", "path", "absolute_path"}
    }


def _prepare_from_payload(
    checkpoint: Mapping[str, Any],
    actor_data: bytes,
    evidence_npz: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    foundation_bytes: int,
    foundation_sha256: str | None,
    input_identity: Mapping[str, Any] | None,
    evidence_bytes: bytes | None,
    checkpoint_bytes: int | None,
    started_cpu: float,
    started_wall: float,
    rss_start: int | None,
) -> dict[str, Any]:
    config = validate_config(config)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("format") != "storage_decision_learner_v1":
        raise ValueError("checkpoint has an unsupported learner format")
    seed = _seed(checkpoint.get("seed"))
    if seed not in config["seeds"]:
        raise ValueError("checkpoint seed is not listed in the frozen H036 config")
    if isinstance(foundation_bytes, (bool, np.bool_)) or not isinstance(foundation_bytes, numbers.Integral):
        raise ValueError("foundation_bytes must be an integer")
    if int(foundation_bytes) != config["foundation"]["bytes"]:
        raise ValueError("foundation_bytes differs from the frozen H036 foundation metadata")
    expected_foundation_sha = str(config["foundation"]["sha256"])
    checkpoint_foundation_sha = checkpoint.get("foundation_sha256")
    if checkpoint_foundation_sha != expected_foundation_sha:
        raise ValueError("checkpoint foundation_sha256 differs from the frozen H036 foundation metadata")
    if foundation_sha256 is not None and foundation_sha256 != expected_foundation_sha:
        raise ValueError("provided foundation_sha256 differs from the frozen H036 foundation metadata")

    local, shared, physical_groups, generations, logical_tags = _validate_actor(checkpoint, actor_data)
    sidecar = _sidecar_arrays(evidence_npz)
    owners_in = sidecar["owner_indices"].astype(np.int64, copy=False)
    owner_slots = [int(sidecar["source_slots"][np.flatnonzero(owners_in == owner)[0]]) for owner in (0, 1)]
    for owner, slot in enumerate(owner_slots):
        rows = np.flatnonzero(owners_in == owner)
        if not np.array_equal(sidecar["generations"][rows], np.full(128, generations[slot], dtype=np.int64)):
            raise ValueError(f"owner {owner} sidecar generation differs from slot_generation")
    owner_tasks = _validate_cache(checkpoint, seed, owner_slots)

    grouped_indices = [np.flatnonzero(owners_in == owner) for owner in (0, 1)]
    order = np.concatenate(grouped_indices).astype(np.int64, copy=False)
    new_observations = np.array(sidecar["observations"][order], dtype=np.float32, copy=True, order="C")
    new_owners = np.concatenate([
        np.full(128, 0, dtype=np.int64),
        np.full(128, 1, dtype=np.int64),
    ])
    new_slots = np.asarray(owner_slots, dtype=np.int64)

    old_observations = _old_rows(checkpoint, seed)
    core_state = checkpoint["core_state"]
    decoded_local, decoded_shared, decoded_groups = decode_actor(actor_data)
    by_slot = effective_matrices(decoded_local, decoded_shared, decoded_groups)
    new_effective = np.array(by_slot[new_slots], dtype=np.float32, copy=True, order="C")
    new_features_raw = h033_inputs._frozen_features(core_state, new_observations)
    old_features_raw = h033_inputs._frozen_features(core_state, old_observations)
    new_features = _copy_features(new_features_raw)
    old_features = _copy_features(old_features_raw)
    bands = behavioral._bands_from_observations(new_observations)
    new_teacher = behavioral._teacher_probabilities(new_features, new_effective, new_owners, bands)
    old_parent = np.array(shared.reshape(3, 3), dtype=np.float32, copy=True)
    # The old keep teacher uses the shared parent alone; resident local A never
    # enters this prior, and old source routing is irrelevant to W0 features.
    old_teacher = behavioral.candidate_probabilities(old_features, old_parent)
    if new_teacher.shape != (256, 4) or old_teacher.shape != (128, 4):
        raise ValueError("teacher probability arrays have unexpected shapes")
    if not np.isfinite(new_teacher).all() or not np.isfinite(old_teacher).all():
        raise ValueError("teacher probabilities must be finite")
    if np.any(new_teacher <= 0.0) or np.any(old_teacher <= 0.0):
        raise ValueError("teacher probabilities must be positive")
    if not np.allclose(new_teacher.sum(axis=1), 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("new teacher probability rows must sum to one")
    if not np.allclose(old_teacher.sum(axis=1), 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("old teacher probability rows must sum to one")

    new_train_indices = np.concatenate((np.arange(96), np.arange(128, 224))).astype(np.int64)
    new_holdout_indices = np.concatenate((np.arange(96, 128), np.arange(224, 256))).astype(np.int64)
    old_train_indices = np.arange(96, dtype=np.int64)
    old_holdout_indices = np.arange(96, 128, dtype=np.int64)
    mean64 = new_effective.mean(axis=(0, 1), dtype=np.float64)
    if not np.isfinite(mean64).all():
        raise ValueError("new effective mean must be finite")

    raw_evidence = evidence_bytes if evidence_bytes is not None else _serialize_sidecar(sidecar)
    if not isinstance(raw_evidence, bytes):
        raise TypeError("prefix evidence bytes must be bytes")
    prefix_actor = bytes(actor_data)
    prefix_checkpoint = copy.deepcopy(dict(checkpoint))
    if checkpoint_bytes is None:
        checkpoint_buffer = io.BytesIO()
        torch.save(prefix_checkpoint, checkpoint_buffer)
        checkpoint_bytes = len(checkpoint_buffer.getvalue())
    if isinstance(checkpoint_bytes, bool) or int(checkpoint_bytes) < 1:
        raise ValueError("serialized checkpoint byte count must be positive")

    if input_identity is None:
        source_identity: dict[str, Any] = {
            "kind": "in_memory_payload",
            "checkpoint_sha256": _value_digest(checkpoint),
            "actor_sha256": hashlib.sha256(prefix_actor).hexdigest(),
            "evidence_sha256": hashlib.sha256(raw_evidence).hexdigest(),
        }
    else:
        if not isinstance(input_identity, Mapping):
            raise TypeError("input_identity must be a mapping")
        source_identity = copy.deepcopy(dict(input_identity))
        try:
            json.dumps(source_identity, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError) as exc:
            raise ValueError("input_identity must be serializable") from exc
        if not isinstance(source_identity.get("files"), Mapping):
            source_identity.setdefault(
                "memory_source",
                {
                    "checkpoint_sha256": _value_digest(checkpoint),
                    "actor_sha256": hashlib.sha256(prefix_actor).hexdigest(),
                    "evidence_sha256": hashlib.sha256(raw_evidence).hexdigest(),
                },
            )

    hashed_arrays = {
        "new_observations": new_observations,
        "new_owners": new_owners,
        "new_effective": new_effective,
        "new_base_logits": new_features.base_logits,
        "new_source_psi": new_features.source_psi,
        "new_contrast_basis": new_features.contrast_basis,
        "new_teacher": new_teacher,
        "old_observations": old_observations,
        "old_base_logits": old_features.base_logits,
        "old_source_psi": old_features.source_psi,
        "old_contrast_basis": old_features.contrast_basis,
        "old_teacher": old_teacher,
        "new_train_indices": new_train_indices,
        "new_holdout_indices": new_holdout_indices,
        "old_train_indices": old_train_indices,
        "old_holdout_indices": old_holdout_indices,
        "mean64": mean64,
        "local": local,
        "shared": shared,
        "physical_groups": physical_groups,
        "logical_tags": logical_tags,
        "old_parent": old_parent,
    }
    identity_for_hash = _identity_for_hash(source_identity)
    identity_text = json.dumps(identity_for_hash, sort_keys=True, separators=(",", ":"), default=str)
    prepared_hash = hashlib.sha256(identity_text.encode("utf-8"))
    for name in sorted(hashed_arrays):
        prepared_hash.update(name.encode("utf-8"))
        prepared_hash.update(_array_sha256(hashed_arrays[name]).encode("ascii"))
    prepared_id = prepared_hash.hexdigest()

    array_bytes = int(sum(array.nbytes for array in hashed_arrays.values()))
    rss_end = _rss_bytes()
    return {
        "format": _PREPARED_FORMAT,
        "seed": seed,
        "new_observations": _readonly_array(new_observations, np.float32),
        "new_owners": _readonly_array(new_owners, np.int64),
        "new_effective": _readonly_array(new_effective, np.float32),
        "new_features": new_features,
        "new_teacher": _readonly_array(new_teacher, np.float64),
        "old_observations": _readonly_array(old_observations, np.float32),
        "old_features": old_features,
        "old_teacher": _readonly_array(old_teacher, np.float64),
        "new_train_indices": _readonly_array(new_train_indices, np.int64),
        "new_holdout_indices": _readonly_array(new_holdout_indices, np.int64),
        "old_train_indices": _readonly_array(old_train_indices, np.int64),
        "old_holdout_indices": _readonly_array(old_holdout_indices, np.int64),
        "mean64": _readonly_array(mean64, np.float64),
        "local": _readonly_array(local, np.float32),
        "shared": _readonly_array(shared, np.float32),
        "physical_groups": _readonly_array(physical_groups, np.int8),
        "logical_tags": _readonly_array(logical_tags, np.int8),
        "old_parent": _readonly_array(old_parent, np.float32),
        "owner_slots": _readonly_array(owner_slots, np.int64),
        "owner_tasks": tuple(owner_tasks),
        "prefix_actor_bytes": prefix_actor,
        "prefix_checkpoint": prefix_checkpoint,
        "prefix_checkpoint_serialized_bytes": int(checkpoint_bytes),
        "prefix_evidence_bytes": bytes(raw_evidence),
        "foundation_bytes": int(foundation_bytes),
        "foundation_sha256": expected_foundation_sha,
        "input_identity": source_identity,
        "prepared_id": prepared_id,
        "memory_cpu_wall": {
            "cpu_s": max(0.0, time.process_time() - started_cpu),
            "wall_s": max(0.0, time.perf_counter() - started_wall),
            "rss_start_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "materialized_array_bytes": array_bytes,
            "actor_bytes": len(prefix_actor),
            "checkpoint_serialized_bytes": int(checkpoint_bytes),
            "evidence_npz_bytes": len(raw_evidence),
            "torch_threads": int(torch.get_num_threads()),
        },
    }


def prepare_from_payload(
    checkpoint: Mapping[str, Any],
    actor_data: bytes,
    evidence_npz: Mapping[str, Any],
    *,
    foundation_bytes: int,
    foundation_sha256: str | None = None,
    input_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare inputs from in-memory synthetic or already decoded payloads."""
    return _prepare_from_payload(
        checkpoint,
        actor_data,
        evidence_npz,
        config=validate_config(),
        foundation_bytes=foundation_bytes,
        foundation_sha256=foundation_sha256,
        input_identity=input_identity,
        evidence_bytes=None,
        checkpoint_bytes=None,
        started_cpu=time.process_time(),
        started_wall=time.perf_counter(),
        rss_start=_rss_bytes(),
    )


def _read_prefix(
    prefix: Path,
) -> tuple[dict[str, Any], bytes, dict[str, np.ndarray], bytes, int, dict[str, Any]]:
    prefix = Path(prefix)
    h033_inputs._reject_symlink_components(prefix)
    if prefix.is_symlink() or not prefix.is_dir():
        raise ValueError("prefix must be an existing, non-symlink directory")
    raw: dict[str, bytes] = {}
    for name in _PREFIX_FILES:
        path = prefix / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"prefix is missing regular file {name}")
        raw[name] = path.read_bytes()
    try:
        checkpoint = torch.load(io.BytesIO(raw["checkpoint.pt"]), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError("checkpoint.pt cannot be decoded") from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint.pt must contain a mapping")
    try:
        with np.load(io.BytesIO(raw["evidence.npz"]), allow_pickle=False) as archive:
            if set(archive.files) != _SIDECAR_KEYS:
                raise ValueError(f"evidence.npz keys must equal {sorted(_SIDECAR_KEYS)}")
            sidecar = {name: archive[name].copy() for name in _SIDECAR_KEYS}
    except (OSError, ValueError) as exc:
        raise ValueError("evidence.npz cannot be decoded with the expected schema") from exc
    identity = {
        "kind": "three_file_prefix",
        "prefix_path": str(prefix.resolve()),
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in raw.items()
        },
    }
    return dict(checkpoint), raw["actor.bin"], sidecar, raw["evidence.npz"], len(raw["checkpoint.pt"]), identity


def prepare_inputs(
    prefix: Path,
    *,
    config: Mapping[str, Any] | None = None,
    expected_seed: int | None = None,
    expected_identity: Mapping[str, str] | None = None,
    logical_case: str | None = None,
) -> dict[str, Any]:
    """Read three prefix files, requiring their frozen SHA-256 identities."""
    canonical = validate_config(config)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    rss_start = _rss_bytes()
    checkpoint, actor_data, evidence_npz, evidence_bytes, checkpoint_bytes, identity = _read_prefix(prefix)
    seed = _seed(checkpoint.get("seed"))
    if expected_seed is not None:
        if isinstance(expected_seed, (bool, np.bool_)) or not isinstance(expected_seed, numbers.Integral):
            raise ValueError("expected_seed must be an integer")
        if int(expected_seed) not in canonical["seeds"]:
            raise ValueError("expected_seed is not one of the frozen E163 prefixes")
        if seed != int(expected_seed):
            raise ValueError("checkpoint seed does not match expected_seed")
    actual_hashes = {
        name: str(details["sha256"])
        for name, details in identity["files"].items()
    }
    if not isinstance(expected_identity, Mapping) or set(expected_identity) != set(_PREFIX_FILES):
        raise ValueError(f"expected_identity must map exactly {_PREFIX_FILES} to SHA-256 strings")
    for name in _PREFIX_FILES:
        expected = expected_identity[name]
        if not isinstance(expected, str) or len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected.lower()):
            raise ValueError(f"expected_identity[{name}] must be a SHA-256 hex digest")
        if actual_hashes[name] != expected.lower():
            raise ValueError(f"prefix {name} SHA-256 differs from expected_identity")
    canonical_case = f"s{seed}_dynamics_late"
    if logical_case is not None and logical_case != canonical_case:
        raise ValueError(f"logical_case must match the checkpoint seed: {canonical_case}")
    identity["logical_case"] = canonical_case
    return _prepare_from_payload(
        checkpoint,
        actor_data,
        evidence_npz,
        config=canonical,
        foundation_bytes=canonical["foundation"]["bytes"],
        foundation_sha256=canonical["foundation"]["sha256"],
        input_identity=identity,
        evidence_bytes=evidence_bytes,
        checkpoint_bytes=checkpoint_bytes,
        started_cpu=started_cpu,
        started_wall=started_wall,
        rss_start=rss_start,
    )


def prepare_install(
    local: np.ndarray,
    shared: np.ndarray,
    physical_groups: np.ndarray,
    logical_tags: np.ndarray,
    candidate: np.ndarray,
    binding: str,
) -> dict[str, Any]:
    """Compensate a candidate-only install, without encoding or mutating inputs."""
    from .storage_decision import compensated_single_parent_install

    if binding not in ("new", "wildcard"):
        raise ValueError("binding must be 'new' or 'wildcard'")
    candidate = np.asarray(candidate)
    if candidate.dtype != np.float32 or candidate.shape != (3, 3) or not np.isfinite(candidate).all():
        raise ValueError("candidate must be a finite float32 3 by 3 matrix")
    before = effective_matrices(local, shared, physical_groups)
    parent_group = 1 if binding == "new" else -2
    installed_local, installed_shared, installed_groups, diagnostic = compensated_single_parent_install(
        local, shared, physical_groups, logical_tags,
        parent=candidate, parent_group=parent_group,
    )
    after = effective_matrices(installed_local, installed_shared, installed_groups)
    if not np.allclose(before, after, rtol=1e-6, atol=1e-6):
        raise RuntimeError("resident effective matrices changed before codec beyond compensation tolerance")
    return {
        "local": installed_local,
        "shared": installed_shared,
        "physical_groups": installed_groups,
        "parent_group": parent_group,
        "diagnostics": diagnostic,
    }


def _score_kl_rows(teacher: np.ndarray, student: np.ndarray) -> np.ndarray:
    teacher = np.asarray(teacher, dtype=np.float64)
    student = np.asarray(student, dtype=np.float64)
    if teacher.shape != student.shape or teacher.ndim != 2 or teacher.shape[1] != 4:
        raise ValueError("score teacher and student must have matching N by 4 shapes")
    for value in (teacher, student):
        if not np.isfinite(value).all() or np.any(value <= 0):
            raise FloatingPointError("score probabilities must be finite and positive")
        if not np.allclose(value.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("score probabilities must sum to one")
    rows = np.sum(teacher * (np.log(teacher) - np.log(student)), axis=1)
    if not np.isfinite(rows).all() or np.any(rows < -1e-12):
        raise FloatingPointError("invalid forward KL in score")
    # Only remove negative floating-point cancellation at numerical zero.
    return np.maximum(rows, 0.0)


def _prior_score(
    prepared: dict[str, Any], new_coefficient: np.ndarray, old_coefficient: np.ndarray,
) -> dict[str, Any]:
    new_probabilities = behavioral.candidate_probabilities(prepared["new_features"], new_coefficient)
    old_probabilities = behavioral.candidate_probabilities(prepared["old_features"], old_coefficient)
    new_rows = _score_kl_rows(prepared["new_teacher"], new_probabilities)
    old_rows = _score_kl_rows(prepared["old_teacher"], old_probabilities)
    result: dict[str, Any] = {}
    for split in ("train", "holdout"):
        indices = prepared[f"new_{split}_indices"]
        means = [float(np.mean(new_rows[indices][prepared["new_owners"][indices] == owner])) for owner in (0, 1)]
        result[f"new_{split}_by_owner"] = means
        result[f"new_{split}"] = float((means[0] + means[1]) / 2.0)
        result[f"old_{split}"] = float(np.mean(old_rows[prepared[f"old_{split}_indices"]]))
    return result


def _clear_actor_optimizer_copy(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Apply E161's local/shared state removal to an already cloned checkpoint."""
    optimizer = checkpoint.get("optimizer_state")
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
        raise ValueError("complete checkpoint must contain an optimizer state mapping")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 3:
        raise ValueError("E161 optimizer must contain named local, shared, and critic groups")
    by_name = {group.get("name"): group for group in groups if isinstance(group, dict)}
    if set(by_name) != {"local", "shared", "critic"}:
        raise ValueError("E161 optimizer group names are not local/shared/critic")
    all_ids: list[int] = []
    for name, expected_count in (("local", 2), ("shared", 1), ("critic", 6)):
        ids = by_name[name].get("params")
        if not isinstance(ids, list) or len(ids) != expected_count:
            raise ValueError(f"E161 optimizer {name} parameter layout differs")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
            raise ValueError("optimizer parameter identities must be integers")
        all_ids.extend(ids)
    if len(set(all_ids)) != len(all_ids) or not set(optimizer["state"]) <= set(all_ids):
        raise ValueError("optimizer state contains inconsistent parameter identities")
    critic_ids = by_name["critic"]["params"]
    critic_before = _value_digest({key: optimizer["state"][key] for key in critic_ids if key in optimizer["state"]})
    actor_ids = by_name["local"]["params"] + by_name["shared"]["params"]
    removed = [key for key in actor_ids if key in optimizer["state"]]
    for key in actor_ids:
        optimizer["state"].pop(key, None)
    critic_after = _value_digest({key: optimizer["state"][key] for key in critic_ids if key in optimizer["state"]})
    if critic_before != critic_after:
        raise RuntimeError("critic optimizer state changed during actor-state removal")
    return {"actor_parameter_ids": actor_ids, "removed_state_ids": removed,
            "critic_state_sha256": critic_after, "critic_state_preserved": True}


def encode_and_score_stage0(
    prepared: dict[str, Any], fit_result: dict[str, Any], binding: str,
) -> dict[str, Any]:
    """Install, encode the full actor, and score fresh-slot priors on copies.

    Returns ``metrics`` (JSON serializable) separately from ``deployment`` raw
    bytes. A caller persisting a package must write those bytes verbatim.
    No environment, PPO, future reward, file output, or original-state mutation
    occurs here. The experimental wrapper owns the campaign and output paths.
    """
    from . import retention_protected_candidate as protected
    from .write_decision_policy import encode_actor

    cpu, wall = time.process_time(), time.perf_counter()
    memory_before = protected.process_memory_snapshot()
    protected._validate_prepared(prepared)
    if not isinstance(fit_result, dict) or fit_result.get("format") != "h036_fit_v1":
        raise ValueError("fit_result must have format h036_fit_v1")
    if fit_result.get("prepared_id") != prepared["prepared_id"]:
        raise ValueError("fit_result does not belong to these prepared inputs")
    lam = protected._validate_lambda(fit_result.get("lambda_value"))
    candidate = np.asarray(fit_result.get("candidate"))
    if candidate.dtype != np.float32 or candidate.shape != (3, 3) or not np.isfinite(candidate).all():
        raise ValueError("fit_result candidate must be finite float32 3 by 3")
    expected_hash = fit_result.get("diagnostics", {}).get("candidate_sha256")
    if expected_hash != protected._candidate_hash(candidate, np.dtype(np.float32)):
        raise ValueError("fit_result candidate hash does not match its frozen diagnostics")
    before = _value_digest(prepared)
    fit_before = _value_digest(fit_result)

    installed = prepare_install(prepared["local"], prepared["shared"], prepared["physical_groups"],
                                prepared["logical_tags"], candidate, binding)
    actor = encode_actor(installed["local"], installed["shared"], installed["physical_groups"],
                         codec="int8", budget_bytes=89).data
    if len(actor) != 89:
        raise RuntimeError("encoded actor does not have the fixed 89-byte layout")
    decoded_local, decoded_shared, decoded_groups = decode_actor(actor)

    # These evaluation arrays represent two independently reallocated slots:
    # old logical group 0 and new logical group 1. They are never serialized.
    fresh_local = np.zeros((2, 27), dtype=np.float32)
    fresh_groups = np.asarray([
        0 if installed["parent_group"] in (-2, group) else -1 for group in (0, 1)
    ], dtype=np.int8)
    fresh_effective = effective_matrices(fresh_local, decoded_shared, fresh_groups)
    old_decoded_prior, new_decoded_prior = fresh_effective[0, 0], fresh_effective[1, 0]
    zero = np.zeros((3, 3), dtype=np.float32)
    old_pre_prior = candidate if installed["parent_group"] == -2 else zero
    pre_codec = _prior_score(prepared, candidate, old_pre_prior)
    encoded = _prior_score(prepared, new_decoded_prior, old_decoded_prior)
    w0 = _prior_score(prepared, zero, zero)

    prefix_effective = effective_matrices(prepared["local"], prepared["shared"], prepared["physical_groups"])
    decoded_effective = effective_matrices(decoded_local, decoded_shared, decoded_groups)
    source_effective = decoded_effective[prepared["owner_slots"]]
    encoded_resident_probabilities = behavioral._teacher_probabilities(
        prepared["new_features"], source_effective, prepared["new_owners"],
        behavioral._bands_from_observations(prepared["new_observations"]),
    )
    resident_rows = _score_kl_rows(prepared["new_teacher"], encoded_resident_probabilities)
    resident_kl = [float(np.mean(resident_rows[prepared["new_owners"] == owner])) for owner in (0, 1)]

    checkpoint = copy.deepcopy(prepared["prefix_checkpoint"])
    required = {"format", "seed", "foundation_sha256", "core_state", "optimizer_state", "optimizer_step_count",
                "shared_frozen", "counts", "runtime_counts", "slot_generation", "fitted_steps", "cache", "evidence",
                "recent_groups", "parent_group", "slot_group_tags", "action_rng", "behavior_rng", "torch_rng",
                "numpy_rng", "python_rng"}
    if not required <= set(checkpoint):
        raise ValueError("complete persistent checkpoint is missing learner fields")
    core_state = checkpoint["core_state"]
    for slot in range(2):
        core_state[f"_nodes.{slot + 1}.A"] = torch.from_numpy(decoded_local[slot].reshape(3, 3, 3).copy())
    core_state["_shared_coefficients.0.C"] = torch.from_numpy(decoded_shared.reshape(3, 3).copy())
    core_state["slot_groups"] = torch.from_numpy(decoded_groups.astype(np.int64, copy=True))
    checkpoint["parent_group"] = int(installed["parent_group"])
    optimizer_diagnostic = _clear_actor_optimizer_copy(checkpoint)
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    checkpoint_bytes = stream.getvalue()
    evidence_bytes = prepared["prefix_evidence_bytes"]
    foundation_bytes = int(prepared["foundation_bytes"])
    if (foundation_bytes != _FOUNDATION_BYTES or prepared["foundation_sha256"] != _FOUNDATION_SHA256
            or checkpoint["foundation_sha256"] != _FOUNDATION_SHA256):
        raise ValueError("persistent foundation identity differs from the frozen configuration")
    package_bytes = foundation_bytes + len(checkpoint_bytes) + len(actor) + len(evidence_bytes)
    if package_bytes > 262144:
        raise ValueError(f"complete persistent package {package_bytes} exceeds 262144 bytes")
    if before != _value_digest(prepared) or fit_before != _value_digest(fit_result):
        raise RuntimeError("installation/scoring mutated prepared inputs or fit result")

    memory_after = protected.process_memory_snapshot()
    metrics = {
        "format": "h036_score_metrics_v1", "seed": int(prepared["seed"]), "prepared_id": prepared["prepared_id"],
        "lambda_value": lam, "binding": binding, "parent_group": installed["parent_group"],
        "candidate_sha256": expected_hash,
        "decoded_parent_sha256": _array_sha256(decoded_shared.reshape(3, 3)),
        "decoded_parent": decoded_shared.reshape(3, 3).astype(float).tolist(),
        "K_old": encoded["old_holdout"], "K_new": encoded["new_holdout"],
        "K_new_W0": w0["new_holdout"], "G_new": w0["new_holdout"] - encoded["new_holdout"],
        "pre_codec_fresh": pre_codec, "encoded_fresh": encoded, "W0_fresh": w0,
        "fresh_local_is_zero": True, "fresh_logical_groups": [0, 1], "fresh_physical_groups": fresh_groups.tolist(),
        "compensation": installed["diagnostics"],
        "resident_drift": {
            "effective_max_abs_by_slot": np.max(np.abs(decoded_effective - prefix_effective), axis=(1, 2, 3)).astype(float).tolist(),
            "probability_max_abs": float(np.max(np.abs(encoded_resident_probabilities - prepared["new_teacher"]))),
            "KL_by_owner_all128": resident_kl,
        },
        "optimizer": optimizer_diagnostic,
        "capacity": {"foundation_bytes": foundation_bytes, "checkpoint_bytes": len(checkpoint_bytes),
                     "actor_bytes": len(actor), "evidence_npz_bytes": len(evidence_bytes),
                     "full_persistent_bytes": package_bytes, "package_budget_bytes": 262144,
                     "checkpoint_serialization": "torch.save to BytesIO, verbatim returned ZIP bytes"},
        "artifacts": {"actor_sha256": hashlib.sha256(actor).hexdigest(),
                      "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                      "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest()},
        "computation_counts": {"prior_probability_passes": 6, "resident_probability_passes": 1, "kl_row_passes": 7},
        "resources": {"cpu_s": max(0.0, time.process_time() - cpu), "wall_s": max(0.0, time.perf_counter() - wall),
                      "temporary_checkpoint_tensor_bytes": protected._tensor_bytes(checkpoint),
                      "temporary_serialized_bytes": len(checkpoint_bytes) + len(actor),
                      "memory_before": memory_before, "memory_after": memory_after,
                      "process_peak_scope": "interpreter lifetime high-water mark, not isolated score workspace"},
        "prepared_unchanged": True, "fit_result_unchanged": True,
    }
    json.dumps(metrics, allow_nan=False)
    return {"format": "h036_stage0_score_v1", "metrics": metrics,
            "deployment": {"actor_bytes": actor, "checkpoint_bytes": checkpoint_bytes, "evidence_bytes": evidence_bytes}}


__all__ = ["prepare_from_payload", "prepare_inputs", "validate_config", "prepare_install", "encode_and_score_stage0"]
