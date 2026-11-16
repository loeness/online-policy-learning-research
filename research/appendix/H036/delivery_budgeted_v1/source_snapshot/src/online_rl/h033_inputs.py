"""Bounded, pre-action inputs and explicitly separated E161 outcome labels.

This module never inspects prefix directory contents.  Prefix extraction reads
only ``checkpoint.pt``, ``actor.bin``, and ``evidence.npz``.  The outcome adapter
is a separate entry point and projects the saved report onto its frozen label
fields before returning a packet.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import math
import numbers
from pathlib import Path
import time
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from . import h033_diagnostic as diagnostic
from . import retention_behavioral_candidate as behavioral
from .c1_policy import _contrast_basis
from .storage_decision import effective_matrices, pair_mapping
from .write_decision_policy import decode_actor


_CHECKPOINT_FILES = ("checkpoint.pt", "actor.bin", "evidence.npz")
_SIDE_CAR_KEYS = {
    "observations",
    "owner_indices",
    "source_slots",
    "generations",
    "bouts",
    "global_steps",
}
_FUTURES = ("old_major", "new_major", "balanced")
_ARMS = ("keep", "replace_behavioral")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"input path contains a symlink component: {cursor}")


def _array_hash(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (numbers.Real, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (numbers.Integral, np.integer)):
        raise ValueError("seed must be an integer")
    return int(value)


def _tensor_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"core_state.{name} must be a tensor")
    if value.dtype != torch.float32 or tuple(value.shape) != shape:
        raise ValueError(f"core_state.{name} must be float32 with shape {shape}")
    result = value.detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError(f"core_state.{name} must contain only finite values")
    return result


def _checkpoint_array(
    payload: Mapping[str, Any], name: str, shape: tuple[int, ...], dtype: np.dtype[Any]
) -> np.ndarray:
    try:
        value = np.asarray(payload[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint is missing valid {name}") from exc
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(f"checkpoint {name} must have dtype {dtype} and shape {shape}")
    return value


def _state_digest(
    payload: Mapping[str, Any], actor_data: bytes, sidecar: Mapping[str, np.ndarray]
) -> str:
    core_state = payload.get("core_state")
    if not isinstance(core_state, Mapping):
        raise ValueError("checkpoint core_state must be a mapping")
    relevant = (
        "source_basis",
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.policy_net.2.weight",
        "mlp_extractor.policy_net.2.bias",
        "action_net.weight",
        "action_net.bias",
        "_nodes.1.A",
        "_nodes.2.A",
        "_shared_coefficients.0.C",
        "slot_groups",
    )
    digest = hashlib.sha256()
    for name in relevant:
        try:
            tensor = core_state[name]
        except KeyError as exc:
            raise ValueError(f"checkpoint core_state is missing {name}") from exc
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"core_state.{name} must be a tensor")
        digest.update(name.encode("utf-8"))
        digest.update(_array_hash(tensor.detach().cpu().numpy()).encode("ascii"))
    for name in ("slot_generation", "fitted_steps", "slot_group_tags"):
        digest.update(name.encode("ascii"))
        digest.update(_array_hash(payload.get(name)).encode("ascii"))
    digest.update(repr(payload.get("recent_groups")).encode("ascii"))
    digest.update(repr(payload.get("parent_group")).encode("ascii"))
    digest.update(repr(payload.get("shared_frozen")).encode("ascii"))
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping) or not isinstance(evidence.get("groups"), Mapping):
        raise ValueError("checkpoint evidence.groups must be a mapping")
    groups = evidence["groups"]
    old_group = groups.get(0, groups.get("0"))
    if not isinstance(old_group, Mapping) or "obs" not in old_group:
        raise ValueError("checkpoint evidence is missing old-group observations")
    digest.update(_array_hash(old_group["obs"]).encode("ascii"))
    digest.update(_sha256(actor_data).encode("ascii"))
    for name in sorted(_SIDE_CAR_KEYS):
        digest.update(name.encode("ascii"))
        digest.update(_array_hash(sidecar[name]).encode("ascii"))
    return digest.hexdigest()


def _sidecar_arrays(value: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if set(value) != _SIDE_CAR_KEYS:
        raise ValueError(f"evidence.npz keys must equal {sorted(_SIDE_CAR_KEYS)}")
    arrays = {name: np.asarray(value[name]) for name in _SIDE_CAR_KEYS}
    observations = arrays["observations"]
    if observations.dtype != np.float32 or observations.ndim != 2 or observations.shape[1] != 13:
        raise ValueError("sidecar observations must be float32 N by 13")
    count = len(observations)
    expected = {
        "owner_indices": (np.dtype(np.int8), (count,)),
        "source_slots": (np.dtype(np.int8), (count,)),
        "generations": (np.dtype(np.int64), (count,)),
        "bouts": (np.dtype(np.int64), (count,)),
        "global_steps": (np.dtype(np.int64), (count,)),
    }
    for name, (dtype, shape) in expected.items():
        if arrays[name].dtype != dtype or arrays[name].shape != shape:
            raise ValueError(f"sidecar {name} must have dtype {dtype} and shape {shape}")
    if count != 256 or not np.isfinite(observations).all():
        raise ValueError("sidecar must contain 256 finite observations")
    owners = arrays["owner_indices"]
    if np.any((owners < 0) | (owners > 1)):
        raise ValueError("sidecar owner_indices must contain only zero and one")
    if not np.array_equal(np.bincount(owners.astype(np.int64), minlength=2), [128, 128]):
        raise ValueError("sidecar must contain exactly 128 rows for each owner")
    slots = arrays["source_slots"]
    if np.any((slots < 0) | (slots > 1)):
        raise ValueError("sidecar source_slots must contain only zero and one")
    owner_slots: list[int] = []
    for owner in (0, 1):
        rows = np.flatnonzero(owners == owner)
        owner_slot_values = np.unique(slots[rows])
        if len(owner_slot_values) != 1:
            raise ValueError(f"owner {owner} must map to exactly one resident slot")
        owner_slots.append(int(owner_slot_values[0]))
        if not np.array_equal(observations[rows, -1], np.full(len(rows), owner_slots[-1] + 1, dtype=np.float32)):
            raise ValueError(f"owner {owner} observations do not use their resident route")
    if owner_slots[0] == owner_slots[1]:
        raise ValueError("the two source owners must map to distinct resident slots")
    return arrays


def _frozen_features(core_state: Mapping[str, Any], observations: np.ndarray) -> behavioral.FrozenPolicyFeatures:
    obs = np.asarray(observations)
    if obs.dtype != np.float32 or obs.ndim != 2 or obs.shape[1] != 13:
        raise ValueError("observations must be float32 N by 13")
    if not np.isfinite(obs).all():
        raise ValueError("observations must contain only finite values")

    # E161's W0 actor feature input is the first four observation coordinates.
    # Keep every network operation float32, matching the saved actor and H026
    # feature extractor before it converts frozen feature arrays to float64.
    x = torch.as_tensor(obs[:, :4].copy(), dtype=torch.float32, device="cpu")
    w0 = _tensor_array(core_state["mlp_extractor.policy_net.0.weight"], "mlp_extractor.policy_net.0.weight", (64, 4))
    b0 = _tensor_array(core_state["mlp_extractor.policy_net.0.bias"], "mlp_extractor.policy_net.0.bias", (64,))
    w2 = _tensor_array(core_state["mlp_extractor.policy_net.2.weight"], "mlp_extractor.policy_net.2.weight", (64, 64))
    b2 = _tensor_array(core_state["mlp_extractor.policy_net.2.bias"], "mlp_extractor.policy_net.2.bias", (64,))
    wa = _tensor_array(core_state["action_net.weight"], "action_net.weight", (4, 64))
    ba = _tensor_array(core_state["action_net.bias"], "action_net.bias", (4,))
    source_basis = _tensor_array(core_state["source_basis"], "source_basis", (3, 65))
    with torch.no_grad():
        latent = torch.tanh(F.linear(x, torch.from_numpy(w0), torch.from_numpy(b0)))
        latent = torch.tanh(F.linear(latent, torch.from_numpy(w2), torch.from_numpy(b2)))
        base_logits = F.linear(latent, torch.from_numpy(wa), torch.from_numpy(ba))
        phi = torch.cat((latent, torch.ones_like(latent[:, :1])), dim=1)
        source_psi = F.linear(phi, torch.from_numpy(source_basis))
    return behavioral.FrozenPolicyFeatures(
        base_logits=base_logits.detach().cpu().numpy().astype(np.float64, copy=True),
        source_psi=source_psi.detach().cpu().numpy().astype(np.float64, copy=True),
        contrast_basis=_contrast_basis(dtype=torch.float32, device="cpu").cpu().numpy().astype(np.float64, copy=True),
        temperature=2.0,
        epsilon=0.05,
    )


def _probability_rows(value: Any, count: int, label: str) -> np.ndarray:
    probabilities = np.asarray(value, dtype=np.float64)
    if probabilities.shape != (count, 4) or not np.isfinite(probabilities).all():
        raise ValueError(f"{label} must be finite N by 4 probabilities")
    if np.any(probabilities <= 0.0) or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{label} must contain positive probability distributions")
    return probabilities


def _kl_rows(p: np.ndarray, q: np.ndarray, label: str) -> np.ndarray:
    left = _probability_rows(p, len(p), f"{label} reference")
    right = _probability_rows(q, len(left), f"{label} candidate")
    if left.shape != right.shape:
        raise ValueError(f"{label} probability shapes differ")
    values = np.sum(left * (np.log(left) - np.log(right)), axis=1)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} KL values must be finite")
    return np.maximum(values, 0.0)


def _owner_equal_mean(values: np.ndarray, owners: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    owners = np.asarray(owners)
    if values.ndim != 1 or owners.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("owner-weighted values must be finite matching vectors")
    means: list[float] = []
    for owner in (0, 1):
        selected = values[owners == owner]
        if len(selected) == 0:
            raise ValueError(f"owner {owner} has no values")
        means.append(float(np.mean(selected, dtype=np.float64)))
    result = (means[0] + means[1]) / 2.0
    if not math.isfinite(result):
        raise ValueError("owner-equal mean is not finite")
    return result


def _selected_evidence_group(evidence: Mapping[str, Any], group: int) -> np.ndarray:
    groups = evidence.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError("checkpoint evidence.groups must be a mapping")
    state = groups.get(group, groups.get(str(group)))
    if not isinstance(state, Mapping) or "obs" not in state:
        raise ValueError(f"checkpoint evidence is missing group {group} observations")
    observations = np.asarray(state["obs"])
    if observations.dtype != np.float32 or observations.ndim != 2 or observations.shape[1] != 13:
        raise ValueError(f"checkpoint evidence group {group} obs must be float32 N by 13")
    if not 1 <= len(observations) <= 256 or not np.isfinite(observations).all():
        raise ValueError(f"checkpoint evidence group {group} must contain 1 to 256 finite observations")
    return observations


@contextmanager
def _single_thread_cpu() -> Iterator[None]:
    previous = torch.get_num_threads()
    if previous != 1:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if previous != 1:
            torch.set_num_threads(previous)


def extract_from_payload(
    checkpoint: Mapping[str, Any],
    actor_data: bytes,
    evidence_npz: Mapping[str, Any],
    seed: int,
    *,
    input_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Extract the three fixed features from already bounded prefix payloads.

    This pure-data entry point is also used by synthetic tests.  It does not
    inspect observer, simulator, schedule, or reward fields.
    """
    seed_id = _seed(seed)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("format") != "storage_decision_learner_v1":
        raise ValueError("checkpoint has an unsupported learner format")
    checkpoint_seed = checkpoint.get("seed")
    if (
        isinstance(checkpoint_seed, (bool, np.bool_))
        or not isinstance(checkpoint_seed, (numbers.Integral, np.integer))
        or int(checkpoint_seed) != seed_id
    ):
        raise ValueError("checkpoint seed does not match the requested seed")
    if not isinstance(actor_data, bytes):
        raise TypeError("actor_data must be bytes")
    if not isinstance(evidence_npz, Mapping):
        raise TypeError("evidence_npz must be a mapping")
    sidecar = _sidecar_arrays(evidence_npz)
    before = _state_digest(checkpoint, actor_data, sidecar)
    started_cpu, started_wall = time.process_time(), time.perf_counter()

    with _single_thread_cpu():
        core_state = checkpoint.get("core_state")
        if not isinstance(core_state, Mapping):
            raise ValueError("checkpoint core_state must be a mapping")
        if checkpoint.get("shared_frozen") is not True:
            raise ValueError("checkpoint shared_frozen must be true")
        if isinstance(checkpoint.get("parent_group"), (bool, np.bool_)) or checkpoint.get("parent_group") != 0:
            raise ValueError("checkpoint parent_group must be zero")

        local = np.stack([
            _tensor_array(core_state[f"_nodes.{slot + 1}.A"], f"_nodes.{slot + 1}.A", (3, 3, 3)).reshape(27)
            for slot in range(2)
        ])
        shared_matrix = _tensor_array(core_state["_shared_coefficients.0.C"], "_shared_coefficients.0.C", (3, 3))
        shared = shared_matrix.reshape(1, 9)
        groups_tensor = core_state.get("slot_groups")
        if not isinstance(groups_tensor, torch.Tensor) or groups_tensor.dtype != torch.int64 or tuple(groups_tensor.shape) != (2,):
            raise ValueError("core_state.slot_groups must be int64 with shape (2,)")
        physical_groups_raw = groups_tensor.detach().cpu().numpy()
        if np.any((physical_groups_raw < -1) | (physical_groups_raw > 0)):
            raise ValueError("core_state.slot_groups must contain only -1 or zero")
        if not np.array_equal(physical_groups_raw, np.asarray([-1, -1], dtype=np.int64)):
            raise ValueError("E161 new-group residents must have physical slot groups [-1, -1]")
        physical_groups = physical_groups_raw.astype(np.int8, copy=True)

        decoded_local, decoded_shared, decoded_groups = decode_actor(actor_data)
        if not np.array_equal(decoded_local, local):
            raise ValueError("actor.bin local matrices differ from checkpoint core_state")
        if not np.array_equal(decoded_shared, shared):
            raise ValueError("actor.bin shared matrices differ from checkpoint core_state")
        if not np.array_equal(decoded_groups, physical_groups):
            raise ValueError("actor.bin slot groups differ from checkpoint core_state")

        generations = _checkpoint_array(checkpoint, "slot_generation", (2,), np.dtype(np.int64))
        fitted_steps = _checkpoint_array(checkpoint, "fitted_steps", (2,), np.dtype(np.uint64))
        logical_tags = _checkpoint_array(checkpoint, "slot_group_tags", (2,), np.dtype(np.int8))
        if not np.array_equal(logical_tags, np.asarray([1, 1], dtype=np.int8)):
            raise ValueError("E161 prefix logical slot_group_tags must be [1, 1]")

        owners = sidecar["owner_indices"]
        slots = sidecar["source_slots"]
        for owner in (0, 1):
            rows = np.flatnonzero(owners == owner)
            slot = int(slots[rows[0]])
            if not np.array_equal(sidecar["generations"][rows], np.full(len(rows), generations[slot], dtype=np.int64)):
                raise ValueError(f"owner {owner} sidecar generation does not match its resident slot")

        recent_groups_raw = checkpoint.get("recent_groups")
        if not isinstance(recent_groups_raw, list) or not 1 <= len(recent_groups_raw) <= 8:
            raise ValueError("checkpoint recent_groups must be a nonempty list of at most 8 groups")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (numbers.Integral, np.integer)) for value in recent_groups_raw):
            raise ValueError("checkpoint recent_groups must contain integer group IDs")
        if any(int(value) not in (0, 1) for value in recent_groups_raw):
            raise ValueError("checkpoint recent_groups must contain only old/new groups 0/1")
        recent_groups = np.asarray(recent_groups_raw, dtype=np.int8)
        recent_frequency_contrast = float(
            (np.count_nonzero(recent_groups == 1) - np.count_nonzero(recent_groups == 0)) / len(recent_groups)
        )

        evidence = checkpoint.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("checkpoint evidence must be a mapping")
        old_observations = _selected_evidence_group(evidence, 0)

        side_observations = sidecar["observations"]
        effective_by_slot = effective_matrices(local, shared, physical_groups)
        owner_slots = [int(slots[np.flatnonzero(owners == owner)[0]]) for owner in (0, 1)]
        source_effective = effective_by_slot[np.asarray(owner_slots, dtype=np.int64)].copy()

        side_features = _frozen_features(core_state, side_observations)
        old_features = _frozen_features(core_state, old_observations)
        bands = behavioral._bands_from_observations(side_observations)
        teacher = behavioral._teacher_probabilities(side_features, source_effective, owners.astype(np.int64), bands)
        teacher = _probability_rows(teacher, len(side_observations), "new-source teacher")

        fit_started_cpu, fit_started_wall = time.process_time(), time.perf_counter()
        candidate, fit_diagnostics = behavioral.fit_behavioral_candidate(
            side_observations,
            owners,
            source_effective,
            side_features,
            min_rows_per_owner=16,
        )
        fit_cpu_s = max(0.0, time.process_time() - fit_started_cpu)
        fit_wall_s = max(0.0, time.perf_counter() - fit_started_wall)
        candidate = np.asarray(candidate, dtype=np.float32)
        if candidate.shape != (3, 3) or not np.isfinite(candidate).all():
            raise ValueError("behavioral candidate must be a finite float32 3 by 3 matrix")

        p_zero = behavioral.candidate_probabilities(side_features, np.zeros((3, 3), dtype=np.float32))
        p_candidate = behavioral.candidate_probabilities(side_features, candidate)
        zero_kl = _kl_rows(teacher, _probability_rows(p_zero, len(teacher), "new W0"), "new teacher to W0")
        candidate_kl = _kl_rows(
            teacher, _probability_rows(p_candidate, len(teacher), "new candidate"), "new teacher to candidate"
        )
        new_fit_gain = _owner_equal_mean(zero_kl, owners) - _owner_equal_mean(candidate_kl, owners)

        old_parent = shared_matrix.astype(np.float32, copy=True)
        old_zero = behavioral.candidate_probabilities(old_features, np.zeros((3, 3), dtype=np.float32))
        old_parent_probabilities = behavioral.candidate_probabilities(old_features, old_parent)
        old_prior_loss = float(np.mean(
            _kl_rows(
                _probability_rows(old_parent_probabilities, len(old_observations), "old parent prior"),
                _probability_rows(old_zero, len(old_observations), "old W0"),
                "old prior to W0",
            ),
            dtype=np.float64,
        ))
        if not all(math.isfinite(value) for value in (recent_frequency_contrast, new_fit_gain, old_prior_loss)):
            raise ValueError("extracted H033 feature is not finite")

    after = _state_digest(checkpoint, actor_data, sidecar)
    if before != after:
        raise RuntimeError("feature extraction mutated its in-memory inputs")
    hashes = dict(input_sha256 or {})
    if not hashes:
        hashes = {"in_memory_sha256": before}
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in hashes.items()):
        raise ValueError("input_sha256 must map string names to digest strings")
    candidate_sha256 = _array_hash(candidate)
    total_cpu_s = max(0.0, time.process_time() - started_cpu)
    total_wall_s = max(0.0, time.perf_counter() - started_wall)
    mapping = pair_mapping(seed_id, "dynamics")
    history_regime = "old_major" if seed_id % 2 == 0 else "new_major"
    return {
        "seed": seed_id,
        "old_family": str(mapping["families"]["old"]),
        "history_regime": history_regime,
        "features": {
            "recent_frequency_contrast": recent_frequency_contrast,
            "new_fit_gain": float(new_fit_gain),
            "old_prior_loss": old_prior_loss,
        },
        "provenance": {
            "input_sha256": hashes,
            "candidate_sha256": candidate_sha256,
            "sidecar_rows_by_owner": [128, 128],
            "old_evidence_rows": int(len(old_observations)),
            "slot_generation": [int(value) for value in generations.tolist()],
            "fitted_steps": [int(value) for value in fitted_steps.tolist()],
            "candidate_fit_diagnostics": dict(fit_diagnostics),
            "timing": {
                "cpu_s": total_cpu_s,
                "wall_s": total_wall_s,
                "candidate_fit_cpu_s": fit_cpu_s,
                "candidate_fit_wall_s": fit_wall_s,
                "torch_threads": 1,
            },
        },
    }


def _read_prefix_payloads(prefix: Path) -> tuple[dict[str, Any], bytes, dict[str, np.ndarray], dict[str, str]]:
    prefix = Path(prefix)
    _reject_symlink_components(prefix)
    if prefix.is_symlink() or not prefix.is_dir():
        raise ValueError("prefix must be an existing, non-symlink directory")
    raw: dict[str, bytes] = {}
    for name in _CHECKPOINT_FILES:
        path = prefix / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"prefix is missing regular file {name}")
        raw[name] = path.read_bytes()
    hashes = {name: _sha256(data) for name, data in raw.items()}
    try:
        checkpoint = torch.load(io.BytesIO(raw["checkpoint.pt"]), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError("checkpoint.pt cannot be decoded") from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint.pt must contain a mapping")
    try:
        with np.load(io.BytesIO(raw["evidence.npz"]), allow_pickle=False) as archive:
            if set(archive.files) != _SIDE_CAR_KEYS:
                raise ValueError(f"evidence.npz keys must equal {sorted(_SIDE_CAR_KEYS)}")
            sidecar = {name: archive[name].copy() for name in _SIDE_CAR_KEYS}
    except (OSError, ValueError) as exc:
        raise ValueError("evidence.npz cannot be decoded with the expected schema") from exc
    return dict(checkpoint), raw["actor.bin"], sidecar, hashes


def extract_prefix_features(prefix: Path, seed: int) -> dict[str, Any]:
    """Read exactly three prefix files and return one pre-action feature row."""
    prefix = Path(prefix)
    checkpoint, actor_data, sidecar, hashes = _read_prefix_payloads(prefix)
    row = extract_from_payload(checkpoint, actor_data, sidecar, seed, input_sha256=hashes)
    for name, original_hash in hashes.items():
        if _sha256((prefix / name).read_bytes()) != original_hash:
            raise RuntimeError(f"prefix input {name} changed during feature extraction")
    row["provenance"]["prefix_path"] = str(prefix.resolve())
    return row


def build_feature_packet(producer: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract one row per configured seed from its fixed E161 prefix path."""
    canonical = diagnostic.validate_config(dict(config))
    producer = Path(producer)
    _reject_symlink_components(producer)
    if producer.is_symlink() or not producer.is_dir():
        raise ValueError("producer must be an existing, non-symlink directory")
    producer_resolved = producer.resolve()
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in canonical["seeds"]:
        prefix = producer / "cases" / f"s{seed}_dynamics_late" / "prefix"
        _reject_symlink_components(prefix)
        if not prefix.resolve().is_relative_to(producer_resolved):
            raise ValueError(f"seed {seed} prefix escapes the producer directory")
        rows.append(extract_prefix_features(prefix, seed))
    packet = {
        "format": "h033_features_v1",
        "rows": rows,
        "metadata": {
            "phase": "pre_action_extraction",
            "input_allowlist": list(_CHECKPOINT_FILES),
            "cpu_s": max(0.0, time.process_time() - started_cpu),
            "wall_s": max(0.0, time.perf_counter() - started_wall),
            "torch_threads": 1,
            "producer_path": str(producer_resolved),
        },
    }
    diagnostic.make_folds(rows, canonical)
    return packet


def load_outcome_packet(analysis: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Project analysis branch rows onto the frozen outcome-label schema."""
    canonical = diagnostic.validate_config(dict(config))
    analysis = Path(analysis)
    _reject_symlink_components(analysis)
    if analysis.is_symlink() or not analysis.is_file():
        raise ValueError("analysis must be an existing, non-symlink file")
    raw = analysis.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("analysis is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict) or not isinstance(document.get("branch_horizon_observations"), list):
        raise ValueError("analysis.branch_horizon_observations must be an array")

    rows: list[dict[str, Any]] = []
    configured_seeds = set(canonical["seeds"])
    for index, source in enumerate(document["branch_horizon_observations"]):
        if not isinstance(source, dict):
            raise ValueError(f"analysis branch row {index} must be an object")
        if source.get("arm") not in _ARMS or source.get("horizon") != canonical["horizon"]:
            continue
        seed = _seed(source.get("seed"))
        if seed not in configured_seeds:
            raise ValueError(f"selected outcome row has unconfigured seed {seed}")
        expected_case = f"s{seed}_dynamics_late"
        if source.get("pair_kind") != "dynamics" or source.get("state") != "late":
            raise ValueError(f"selected outcome row for seed {seed} is not dynamics / late")
        if source.get("case_id") != expected_case:
            raise ValueError(f"selected outcome row for seed {seed} has an invalid case_id")
        mapping = pair_mapping(seed, "dynamics")
        if source.get("family") != mapping["families"]["old"]:
            raise ValueError(f"selected outcome row for seed {seed} has an invalid old family")
        expected_history = "old_major" if seed % 2 == 0 else "new_major"
        if source.get("history_regime") != expected_history:
            raise ValueError(f"selected outcome row for seed {seed} has an invalid history regime")
        future = source.get("future_regime")
        if future not in _FUTURES:
            raise ValueError(f"selected outcome row for seed {seed} has an invalid future regime")
        replica = source.get("replicate")
        if isinstance(replica, (bool, np.bool_)) or not isinstance(replica, (numbers.Integral, np.integer)) or int(replica) not in (0, 1):
            raise ValueError(f"selected outcome row for seed {seed} has an invalid replica")
        rows.append(
            {
                "seed": seed,
                "future_regime": str(future),
                "replica": int(replica),
                "arm": str(source["arm"]),
                "horizon": int(source["horizon"]),
                "reward_per_step": _finite_float(source.get("reward_per_step"), f"row {index}.reward_per_step"),
            }
        )

    packet = {
        "format": "h033_outcomes_v1",
        "rows": rows,
        "metadata": {
            "source_path": str(analysis.resolve()),
            "source_sha256": _sha256(raw),
            "filter": {
                "arms": list(_ARMS),
                "horizon": canonical["horizon"],
                "pair_kind": "dynamics",
                "state": "late",
            },
        },
    }
    diagnostic.make_targets(packet, canonical)
    return packet


__all__ = [
    "build_feature_packet",
    "extract_from_payload",
    "extract_prefix_features",
    "load_outcome_packet",
]
