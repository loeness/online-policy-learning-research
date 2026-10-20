"""Independent, read-only integrity audit for retention-candidate cohorts.

The audit consumes saved checkpoint, ledger, and evidence files only. It never
steps an environment, compares candidate rewards while making a decision, or
uses held-out KL to accept or reject a behavioral candidate. Its JSON report
contains integrity errors and non-gating diagnostic warnings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from online_rl.storage_decision import effective_matrices, semantic_hash


FORMAT = "retention_candidate_audit_v1"
SUMMARY_FORMAT = "retention_candidate_summary_v1"
PRIVATE_PARTS = {"agent.md", "agents.md", "tmp"}
EXPECTED_SOURCE_FILES = {
    "configs/retention_candidate.json",
    "docs/RETENTION_CANDIDATE_PROTOCOL.md",
    "scripts/run_retention_candidate.py",
    "scripts/run_retention_mechanism.py",
    "scripts/run_storage_decision.py",
    "scripts/run_knowledge_loop.py",
    "scripts/run_write_stream.py",
    "src/online_rl/retention_consolidation.py",
    "src/online_rl/consolidation_probe.py",
    "src/online_rl/retention_behavioral_candidate.py",
    "src/online_rl/retention_stream_value.py",
    "src/online_rl/storage_decision.py",
    "src/online_rl/write_decision_policy.py",
    "src/online_rl/write_decision_envs.py",
    "src/online_rl/score_calibration.py",
    "src/online_rl/consolidation_probe.py",
    "src/online_rl/write_decision_evidence.py",
    "src/online_rl/write_fast_evidence.py",
    "src/online_rl/c1_supplement_experiment.py",
    "src/online_rl/c1_tensor_controls.py",
    "src/online_rl/c1_policy.py",
    "src/online_rl/c1_envs.py",
    "src/online_rl/experiment.py",
    "src/online_rl/policy.py",
    "src/online_rl/shared_write_policy.py",
    "src/online_rl/shared_write_storage.py",
    "src/online_rl/envs.py",
    "configs/storage_decision.json",
    "docs/STORAGE_DECISION_PROTOCOL.md",
    "configs/retention_mechanism.json",
    "docs/RETENTION_MECHANISM_PROTOCOL.md",
}
EXPECTED_EVIDENCE_KEYS = {
    "observations", "owner_indices", "source_slots", "generations", "bouts", "global_steps",
}
ARM_SPECS = {
    "keep": ("keep", "mean", "keep"),
    "replace_mean": ("replace", "mean", "replace"),
    "replace_behavioral": ("replace", "behavioral", "replace"),
    "merge_mean": ("mean", "mean", "merge"),
    "merge_behavioral": ("mean", "behavioral", "merge"),
}
_LOCAL_KEY = re.compile(r"^_nodes\.(\d+)\.A$")
_SHARED_KEY = re.compile(r"^_shared_coefficients\.(\d+)\.C$")


class Audit:
    """Accumulates deterministic checks without stopping at the first failure."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.observations: dict[str, Any] = {}

    def check(self, ok: bool, code: str, where: str, message: str, **details: Any) -> None:
        if not ok:
            item: dict[str, Any] = {"code": code, "where": where, "message": message}
            if details:
                item["details"] = details
            self.errors.append(item)

    def warn(self, code: str, where: str, message: str, **details: Any) -> None:
        item: dict[str, Any] = {"code": code, "where": where, "message": message}
        if details:
            item["details"] = details
        self.warnings.append(item)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError("checkpoint must contain a mapping")
    return value


def _relative_file(root: Path, relative: str) -> Path | None:
    """Resolve a manifest path only if it stays inside ``root``."""
    candidate = (root / Path(relative.replace("/", "\\"))).resolve()
    resolved = root.resolve()
    if candidate != resolved and resolved not in candidate.parents:
        return None
    return candidate


def _core_arrays(checkpoint: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = checkpoint["core_state"]
    local: list[tuple[int, np.ndarray]] = []
    shared: list[tuple[int, np.ndarray]] = []
    for name, value in state.items():
        lm = _LOCAL_KEY.match(str(name))
        sm = _SHARED_KEY.match(str(name))
        if lm:
            local.append((int(lm.group(1)), np.asarray(value.detach().cpu()).reshape(27).astype(np.float32)))
        elif sm:
            shared.append((int(sm.group(1)), np.asarray(value.detach().cpu()).reshape(9).astype(np.float32)))
    local.sort(key=lambda item: item[0])
    shared.sort(key=lambda item: item[0])
    if [index for index, _ in local] != list(range(1, len(local) + 1)):
        raise ValueError("checkpoint local actor keys are not contiguous")
    if [index for index, _ in shared] != list(range(len(shared))):
        raise ValueError("checkpoint shared actor keys are not contiguous")
    if not local or not shared:
        raise ValueError("checkpoint lacks local or shared actor matrices")
    groups = np.asarray(state["slot_groups"].detach().cpu(), dtype=np.int8).copy()
    return np.stack([row for _, row in local]), np.stack([row for _, row in shared]), groups


def _float_rows(value: Any, dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _json_array(value: str, shape: tuple[int, ...] | None = None, dtype: Any = np.float32) -> np.ndarray:
    result = np.asarray(json.loads(value), dtype=dtype)
    if shape is not None and result.shape != shape:
        raise ValueError(f"expected shape {shape}, got {result.shape}")
    return result


def _audit_source_snapshot(audit: Audit, summary: Mapping[str, Any]) -> int | None:
    snapshot = audit.root / "source_snapshot"
    manifest_path = snapshot / "manifest.json"
    try:
        manifest = _json(manifest_path)
    except Exception as exc:
        audit.check(False, "source_manifest_read", str(manifest_path),
                    "source manifest is missing or unreadable", error=repr(exc))
        return None
    declared = summary.get("source_manifest")
    if declared is not None:
        audit.check(manifest == declared, "source_manifest_summary", str(manifest_path),
                    "cohort summary differs from the source snapshot manifest")
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping):
        audit.check(False, "source_manifest_shape", str(manifest_path),
                    "source manifest must contain a sources mapping")
        return None
    names = {str(name).replace("\\", "/") for name in sources}
    audit.check(EXPECTED_SOURCE_FILES <= names, "source_manifest_required", str(manifest_path),
                "snapshot omits sources required to reproduce the audit", missing=sorted(EXPECTED_SOURCE_FILES - names))
    # The source list is an explicit allowlist. This catches accidental inclusion
    # of private instructions or unrelated user data even when the hash is valid.
    allowed = EXPECTED_SOURCE_FILES
    audit.check(names <= allowed, "source_manifest_allowlist", str(manifest_path),
                "source manifest includes a file outside the frozen audit allowlist",
                extra=sorted(names - allowed))
    for name, record in sources.items():
        normalized = str(name).replace("\\", "/")
        parts = PurePosixPath(normalized).parts
        audit.check(not any(part.casefold() in PRIVATE_PARTS for part in parts),
                    "private_source_listed", f"source_snapshot/{normalized}",
                    "snapshot lists a private instruction or temporary file")
        target = _relative_file(snapshot, normalized)
        if target is None or not target.is_file():
            audit.check(False, "source_copy_missing", f"source_snapshot/{normalized}",
                        "manifest source is missing or escapes the snapshot")
            continue
        try:
            audit.check(target.stat().st_size == int(record.get("bytes", -1))
                        and _sha_file(target) == record.get("sha256"),
                        "source_copy_hash", str(target), "source snapshot file differs from its digest record")
        except Exception as exc:
            audit.check(False, "source_copy_hash_error", str(target),
                        "source snapshot file could not be checked", error=repr(exc))
    for target in snapshot.rglob("*"):
        relative = target.relative_to(snapshot)
        audit.check(not any(part.casefold() in PRIVATE_PARTS for part in relative.parts),
                    "private_snapshot_path", str(target),
                    "source snapshot contains a private instruction or temporary path")
    exclusions = {str(item).casefold() for item in manifest.get("excluded_local_instructions", [])}
    audit.check(exclusions == {"agent.md", "agents.md"}, "private_exclusions", str(manifest_path),
                "manifest must declare both local instruction exclusions", exclusions=sorted(exclusions))
    foundation = manifest.get("foundation", {})
    foundation_size: int | None = None
    for label in ("foundation", "dataset"):
        record = manifest.get(label, {})
        try:
            target = Path(str(record.get("path", "")))
            audit.check(target.is_file(), f"{label}_missing", str(manifest_path),
                        f"{label} path is unavailable")
            size = target.stat().st_size
            digest = _sha_file(target)
            audit.check(size == int(record.get("bytes", -1)) and digest == record.get("sha256"),
                        f"{label}_hash", str(target), f"{label} differs from its source manifest")
            if label == "foundation":
                foundation_size = size
        except Exception as exc:
            audit.check(False, f"{label}_check", str(manifest_path),
                        f"{label} provenance could not be verified", error=repr(exc))
    source_by_name = {str(name).replace("\\", "/"): record for name, record in sources.items()}
    for summary_field, source_name in (("config_sha256", "configs/retention_candidate.json"),
                                       ("protocol_sha256", "docs/RETENTION_CANDIDATE_PROTOCOL.md")):
        expected_hash = source_by_name.get(source_name, {}).get("sha256")
        actual_hash = summary.get(summary_field)
        if actual_hash is not None:
            audit.check(actual_hash == expected_hash, "source_summary_hash", str(manifest_path),
                        "cohort summary hash differs from its frozen source file", field=summary_field,
                        recorded=actual_hash, manifest=expected_hash)
    return foundation_size


def _audit_npz(audit: Audit, path: Path, record: Mapping[str, Any] | None = None) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        audit.check(False, "evidence_missing", str(path), "bounded evidence sidecar is missing")
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            audit.check(names == EXPECTED_EVIDENCE_KEYS, "evidence_keys", str(path),
                        "evidence sidecar must contain only the six bounded observation/index arrays",
                        actual=sorted(names), expected=sorted(EXPECTED_EVIDENCE_KEYS))
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
        if record is not None:
            audit.check(int(record.get("bytes", -1)) == path.stat().st_size
                        and record.get("sha256") == _sha_file(path),
                        "evidence_record", str(path), "evidence sidecar differs from its observer record",
                        recorded_bytes=record.get("bytes"), actual_bytes=path.stat().st_size)
        obs = arrays.get("observations")
        audit.check(obs is not None and obs.dtype == np.float32 and obs.ndim == 2
                    and obs.shape[1:] == (13,) and 0 < len(obs) <= 256 and np.isfinite(obs).all(),
                    "evidence_observation_shape", str(path),
                    "evidence observations must be finite float32 rows with shape at most 256 by 13",
                    shape=None if obs is None else list(obs.shape),
                    dtype=None if obs is None else str(obs.dtype))
        expected_types = {
            "owner_indices": np.dtype(np.int8), "source_slots": np.dtype(np.int8),
            "generations": np.dtype(np.int64), "bouts": np.dtype(np.int64),
            "global_steps": np.dtype(np.int64),
        }
        if obs is not None:
            for key, dtype in expected_types.items():
                value = arrays.get(key)
                audit.check(value is not None and value.dtype == dtype and value.shape == (len(obs),),
                            "evidence_index_shape", str(path),
                            "evidence index vector has the wrong shape or dtype", key=key,
                            shape=None if value is None else list(value.shape),
                            dtype=None if value is None else str(value.dtype))
        return arrays
    except Exception as exc:
        audit.check(False, "evidence_read", str(path), "evidence sidecar could not be read safely", error=repr(exc))
        return None


def _audit_evidence_selection(
    audit: Audit,
    evidence: Mapping[str, np.ndarray],
    source_records: Sequence[Mapping[str, Any]],
    prefix_rows: Sequence[Mapping[str, str]],
    rollout_rows: Sequence[Mapping[str, str]],
    where: str,
    latest_source_global_steps: Sequence[int] | None = None,
) -> None:
    """Reconstruct exactly the final 128 observed rows of each source rollout."""
    try:
        from scripts import run_knowledge_loop as loop

        owners = np.asarray(evidence["owner_indices"], dtype=np.int64)
        if len(source_records) != 2:
            audit.check(False, "candidate_source_count", where,
                        "evidence needs exactly two ordered current candidate sources")
            return
        if not np.array_equal(np.bincount(owners, minlength=2), np.asarray([128, 128])):
            audit.check(False, "evidence_rows_per_owner", where,
                        "evidence must contain 128 observations for each current source",
                        counts=np.bincount(owners, minlength=2).tolist())
        expected_rows: list[dict[str, Any]] = []
        for owner, source in enumerate(source_records):
            slot = int(source["slot"])
            generation = int(source["generation"])
            bout = int(source.get("source_bout", source.get("bout", -1)))
            matches = [row for row in prefix_rows
                       if int(row["bout"]) == bout and int(row["slot"]) == slot
                       and int(row["generation"]) == generation]
            rollouts = [row for row in rollout_rows if int(row["bout"]) == bout]
            audit.check(bool(matches) and bool(rollouts), "source_final_rollout_exists", where,
                        "candidate source has no matching raw transitions or rollout record",
                        owner=owner, slot=slot, generation=generation, bout=bout)
            if not matches or not rollouts:
                continue
            final_rollout_rows = int(rollouts[-1]["rows"])
            expected_rollout_index = len(rollouts) - 1
            audit.check(final_rollout_rows == 256 and len(matches) >= final_rollout_rows
                        and int(rollouts[-1].get("rollout_index", -1)) == expected_rollout_index,
                        "source_final_rollout_size", where,
                        "final source rollout is not the final complete 256-row rollout",
                        owner=owner, final_rollout_rows=final_rollout_rows, source_rows=len(matches),
                        rollout_index=rollouts[-1].get("rollout_index"), expected_rollout_index=expected_rollout_index)
            expected_rows.extend(matches[-128:])
        expected_rows.sort(key=lambda row: int(row["global_step"]))
        audit.check(len(expected_rows) == len(evidence["observations"]), "evidence_selection_count", where,
                    "raw source rows selected for evidence differ from the saved sidecar row count",
                    raw=len(expected_rows), evidence=len(evidence["observations"]))
        observations = np.asarray(evidence["observations"], dtype=np.float32)
        owners = np.asarray(evidence["owner_indices"], dtype=np.int64)
        source_records = list(source_records)
        for index, row in enumerate(expected_rows[:len(observations)]):
            owner = next((owner for owner, source in enumerate(source_records)
                          if int(source["slot"]) == int(row["slot"])
                          and int(source["generation"]) == int(row["generation"])
                          and int(source.get("source_bout", source.get("bout", -1))) == int(row["bout"])), None)
            try:
                physical_obs = _json_array(row["obs"], dtype=np.float32)
                expected_obs = np.asarray(loop.augment_observation(physical_obs, int(row["slot"]) + 1), dtype=np.float32)
                audit.check(expected_obs.shape == (13,) and np.array_equal(observations[index], expected_obs),
                            "evidence_observation_origin", where,
                            "evidence observation differs from its augmented raw environment observation",
                            row=index, global_step=row.get("global_step"))
            except Exception as exc:
                audit.check(False, "evidence_observation_rebuild", where,
                            "raw observation could not be route-augmented for evidence comparison",
                            row=index, error=repr(exc))
            audit.check(owner is not None and int(owners[index]) == owner,
                        "evidence_owner_consistency", where,
                        "evidence owner index does not map to its candidate source slot/generation/bout",
                        row=index, saved_owner=int(owners[index]), expected_owner=owner)
            for key, expected in (("source_slots", int(row["slot"])),
                                  ("generations", int(row["generation"])),
                                  ("bouts", int(row["bout"])),
                                  ("global_steps", int(row["global_step"]))):
                audit.check(int(evidence[key][index]) == expected, "evidence_provenance", where,
                            "sidecar source metadata differs from the corresponding raw transition",
                            row=index, field=key, saved=int(evidence[key][index]), expected=expected)
        steps = np.asarray(evidence["global_steps"], dtype=np.int64)
        audit.check(np.all(np.diff(steps) > 0), "evidence_global_step_order", where,
                    "bounded evidence rows must retain chronological order")
        for owner in (0, 1):
            selected = steps[owners == owner]
            if len(selected):
                source = source_records[owner]
                observer_latest = (int(latest_source_global_steps[owner])
                                   if latest_source_global_steps is not None
                                   and owner < len(latest_source_global_steps) else None)
                audit.check(int(selected[-1]) == int(source.get("latest_source_global_step", -1))
                            and (observer_latest is None or int(selected[-1]) == observer_latest),
                            "evidence_latest_step", where,
                            "owner evidence does not end at its recorded latest source step",
                            owner=owner, saved=int(selected[-1]),
                            source_recorded=source.get("latest_source_global_step"),
                            observer_recorded=observer_latest)
    except Exception as exc:
        audit.check(False, "evidence_selection_audit", where,
                    "raw evidence selection could not be independently checked", error=repr(exc))


def _episode_continuity_errors(rows: Sequence[Mapping[str, str]], *, start_driver: Mapping[str, Any] | None = None) -> list[str]:
    """Return episode/observation continuity defects found in raw step rows."""
    errors: list[str] = []
    if not rows:
        return errors
    prior_id = int((start_driver or {}).get("episode_ordinal", int(rows[0]["episode_id"])))
    prior_reset = int((start_driver or {}).get("episode_reset_seed", int(rows[0]["reset_seed"])))
    prior_done = False
    prior_next: np.ndarray | None = None
    if start_driver and start_driver.get("obs") is not None:
        try:
            prior_next = np.asarray(start_driver["obs"], dtype=np.float32)
        except Exception:
            errors.append("saved start simulator observation is malformed")
    for index, row in enumerate(rows):
        try:
            episode_id = int(row["episode_id"])
            reset_seed = int(row["reset_seed"])
            done = bool(int(row["environment_done"]))
            term = bool(int(row["terminated"]))
            trunc = bool(int(row["truncated"]))
            obs = _json_array(row["obs"], dtype=np.float32)
            next_obs = _json_array(row["next_obs"], dtype=np.float32)
            if obs.shape != (4,) or next_obs.shape != (4,):
                errors.append(f"row {index}: environment observations are not four dimensional")
            if done != (term or trunc):
                errors.append(f"row {index}: environment_done disagrees with termination flags")
            if not prior_done:
                if episode_id != prior_id or reset_seed != prior_reset:
                    errors.append(f"row {index}: episode or reset seed changed without a completed episode")
                if prior_next is not None and not np.array_equal(prior_next, obs):
                    errors.append(f"row {index}: adjacent observations are discontinuous")
            else:
                if episode_id != prior_id + 1:
                    errors.append(f"row {index}: episode ordinal did not increment after completion")
            prior_id, prior_reset, prior_done, prior_next = episode_id, reset_seed, done, next_obs
        except Exception:
            errors.append(f"row {index}: malformed episode or observation fields")
    return errors


def _audit_episode_continuity(audit: Audit, rows: Sequence[Mapping[str, str]], where: str,
                              start_simulator: Mapping[str, Any] | None = None) -> None:
    start_driver = (start_simulator or {}).get("driver", {}) if isinstance(start_simulator, Mapping) else {}
    for message in _episode_continuity_errors(rows, start_driver=start_driver):
        audit.check(False, "episode_continuity", where, message)


def _audit_episode_ledger(audit: Audit, step_rows: Sequence[Mapping[str, str]],
                          episode_rows: Sequence[Mapping[str, str]], where: str) -> None:
    completed = [row for row in step_rows if bool(int(row.get("environment_done", "0")))]
    audit.check(len(completed) == len(episode_rows), "episode_ledger_count", where,
                "completed episodes in raw steps differ from episode ledger rows",
                completed_steps=len(completed), episode_rows=len(episode_rows))
    for index, (step, episode) in enumerate(zip(completed, episode_rows)):
        try:
            audit.check(int(step["episode_id"]) == int(episode["episode_id"])
                        and int(step["reset_seed"]) == int(episode["reset_seed"])
                        and int(step["terminated"]) == int(episode["terminated"])
                        and int(step["truncated"]) == int(episode["truncated"]),
                        "episode_ledger_identity", where,
                        "episode completion row differs from its terminal raw step", row=index)
            audit.check(int(episode["length"]) > 0 and math.isfinite(float(episode["return_value"])),
                        "episode_ledger_values", where,
                        "episode length or return is invalid", row=index)
        except Exception as exc:
            audit.check(False, "episode_ledger_parse", where,
                        "episode ledger row is missing or malformed", row=index, error=repr(exc))


def _audit_episode_state(audit: Audit, rows: Sequence[Mapping[str, str]], start_simulator: Mapping[str, Any] | None,
                         end_simulator: Mapping[str, Any], seed: int, where: str) -> None:
    """Check episode continuity against raw transitions and saved simulator endpoints."""
    from scripts import run_storage_decision as storage

    start_driver = (start_simulator or {}).get("driver", {}) if isinstance(start_simulator, Mapping) else {}
    end_driver = end_simulator.get("driver", {}) if isinstance(end_simulator, Mapping) else {}
    if start_driver:
        ordinal = int(start_driver["episode_ordinal"])
        reset_seed = int(start_driver["episode_reset_seed"])
        episode_length = int(start_driver["episode_length"])
        expected_obs = np.asarray(start_driver["obs"], dtype=np.float32)
    else:
        ordinal = 0
        reset_seed = int(storage._reset_seed(int(seed), ordinal))
        episode_length = 0
        expected_obs = None
    waiting_for_reset_observation = expected_obs is None
    last_done = False
    last_next: np.ndarray | None = None
    for index, row in enumerate(rows):
        try:
            episode_id = int(row["episode_id"])
            row_seed = int(row["reset_seed"])
            obs = _json_array(row["obs"], dtype=np.float32)
            next_obs = _json_array(row["next_obs"], dtype=np.float32)
            audit.check(obs.shape == (4,) and next_obs.shape == (4,)
                        and np.isfinite(obs).all() and np.isfinite(next_obs).all(),
                        "episode_observation_shape", where,
                        "raw episode observations must be finite four-dimensional vectors", row=index)
            audit.check(episode_id == ordinal and row_seed == reset_seed,
                        "episode_state_identity", where,
                        "raw step episode ordinal or reset seed differs from the live simulator state", row=index,
                        actual_ordinal=episode_id, expected_ordinal=ordinal,
                        actual_reset_seed=row_seed, expected_reset_seed=reset_seed)
            if expected_obs is not None and not waiting_for_reset_observation:
                audit.check(np.array_equal(obs, expected_obs), "episode_state_observation", where,
                            "raw observation differs from the restored or prior next observation", row=index)
            done = bool(int(row["environment_done"]))
            episode_length += 1
            if done:
                ordinal += 1
                reset_seed = int(storage._reset_seed(int(seed), ordinal))
                episode_length = 0
                expected_obs = None
                waiting_for_reset_observation = True
            else:
                expected_obs = next_obs
                waiting_for_reset_observation = False
            last_done = done
            last_next = next_obs
        except Exception as exc:
            audit.check(False, "episode_state_parse", where,
                        "raw step could not be reconciled with simulator episode state", row=index, error=repr(exc))
    if not end_driver:
        audit.check(False, "episode_end_simulator_missing", where,
                    "saved simulator endpoint has no driver state")
        return
    audit.check(int(end_driver.get("episode_ordinal", -1)) == ordinal
                and int(end_driver.get("episode_reset_seed", -1)) == reset_seed
                and int(end_driver.get("episode_length", -1)) == episode_length,
                "episode_end_simulator_state", where,
                "saved simulator episode counters differ from the raw transition sequence",
                expected={"episode_ordinal": ordinal, "episode_reset_seed": reset_seed,
                          "episode_length": episode_length},
                actual={key: end_driver.get(key) for key in
                        ("episode_ordinal", "episode_reset_seed", "episode_length")})
    try:
        end_obs = np.asarray(end_driver["obs"], dtype=np.float32)
        audit.check(end_obs.shape == (4,) and np.isfinite(end_obs).all(),
                    "episode_end_observation_shape", where,
                    "saved simulator endpoint observation is malformed")
        if rows and not last_done and last_next is not None:
            audit.check(np.array_equal(end_obs, last_next), "episode_end_observation", where,
                        "saved simulator observation differs from the last raw next observation")
    except Exception as exc:
        audit.check(False, "episode_end_observation_read", where,
                    "saved simulator endpoint observation could not be checked", error=repr(exc))


def _reseeded_state_hash(checkpoint: Mapping[str, Any], simulator: Mapping[str, Any], seed: int) -> str:
    """Hash the deterministic RNG-only transform of a restored prefix package."""
    from copy import deepcopy

    transformed = deepcopy(dict(checkpoint))
    transformed["seed"] = int(seed)
    transformed["action_rng"] = np.random.default_rng(int(seed) + 0xA17).bit_generator.state
    transformed["behavior_rng"] = np.random.default_rng(int(seed) + 0xB3E).bit_generator.state
    transformed["numpy_rng"] = np.random.RandomState(int(seed)).get_state()
    transformed["python_rng"] = random.Random(int(seed)).getstate()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    transformed["torch_rng"] = generator.get_state()
    return semantic_hash({"learner": transformed, "simulator": simulator})


def _audit_rng_restore(audit: Audit, branch: Mapping[str, Any], checkpoint: Mapping[str, Any],
                       simulator: Mapping[str, Any], config: Mapping[str, Any], where: str) -> None:
    seed = int(branch.get("seed", -1))
    replicate = int(branch.get("replicate", -1))
    expected_seed = int(config["future_seed_offset"]) + int(config["future_seed_multiplier"]) * seed + replicate
    audit.check(int(branch.get("future_seed", -1)) == expected_seed, "future_seed", where,
                "future seed differs from the frozen seed/replicate formula",
                saved=branch.get("future_seed"), expected=expected_seed)
    prefix_hash = semantic_hash({"learner": checkpoint, "simulator": simulator})
    expected_prefix_hash = branch.get("prefix_state_hash")
    audit.check(branch.get("prefix_restore_verified") is True
                and branch.get("restored_state_hash") == prefix_hash
                and expected_prefix_hash == prefix_hash,
                "full_prefix_restore", where,
                "branch did not start from the exact saved learner and simulator prefix",
                saved_prefix=expected_prefix_hash, restored=branch.get("restored_state_hash"), actual=prefix_hash)
    transform = branch.get("rng_transform", {})
    before = transform.get("before", {}) if isinstance(transform, Mapping) else {}
    after = transform.get("after", {}) if isinstance(transform, Mapping) else {}
    env = simulator.get("environment", {})
    env_hash = semantic_hash({"bit_generator": env.get("np_random_bit_generator"),
                              "state": env.get("np_random_state")})
    before_expected = {
        "action_rng_sha256": semantic_hash(checkpoint.get("action_rng")),
        "behavior_rng_sha256": semantic_hash(checkpoint.get("behavior_rng")),
        "numpy_global_rng_sha256": semantic_hash(checkpoint.get("numpy_rng")),
        "python_global_rng_sha256": semantic_hash(checkpoint.get("python_rng")),
        "torch_global_rng_sha256": semantic_hash(checkpoint.get("torch_rng")),
        "environment_rng_sha256": env_hash,
    }
    for key, expected in before_expected.items():
        audit.check(before.get(key) == expected, "rng_restore_before", where,
                    "before-transform RNG hash differs from the restored prefix", stream=key)
    after_expected = {
        "action_rng_sha256": semantic_hash(np.random.default_rng(expected_seed + 0xA17).bit_generator.state),
        "behavior_rng_sha256": semantic_hash(np.random.default_rng(expected_seed + 0xB3E).bit_generator.state),
        "numpy_global_rng_sha256": semantic_hash(np.random.RandomState(expected_seed).get_state()),
        "python_global_rng_sha256": semantic_hash(random.Random(expected_seed).getstate()),
        "torch_global_rng_sha256": semantic_hash(torch.Generator(device="cpu").manual_seed(expected_seed).get_state()),
        "environment_rng_sha256": env_hash,
    }
    for key, expected in after_expected.items():
        audit.check(after.get(key) == expected, "rng_reseed_after", where,
                    "after-transform RNG hash differs from the documented reseed", stream=key)
    driver = simulator.get("driver", {})
    for key in ("episode_reset_seed", "episode_ordinal", "episode_length"):
        audit.check(before.get(key) == driver.get(key) and after.get(key) == driver.get(key),
                    "rng_episode_preserved", where,
                    "RNG transform changed the in-flight episode state", field=key,
                    prefix=driver.get(key), before=before.get(key), after=after.get(key))
    audit.check(transform.get("environment_rng_preserved") is True
                and transform.get("current_episode_preserved") is True,
                "rng_preservation_flags", where,
                "runner did not attest to preserving the environment RNG and live episode")
    expected_hash = _reseeded_state_hash(checkpoint, simulator, expected_seed)
    audit.check(branch.get("post_rng_transform_state_hash") == expected_hash,
                "rng_transform_state_hash", where,
                "recorded post-transform hash is not the deterministic RNG-only transform",
                recorded=branch.get("post_rng_transform_state_hash"), actual=expected_hash)


def _package_components(folder: Path, foundation_bytes: int | None) -> dict[str, int]:
    if foundation_bytes is None:
        raise ValueError("foundation byte count is unavailable")
    files = {name: (folder / name).stat().st_size for name in ("checkpoint.pt", "actor.bin", "evidence.npz")}
    return {
        "foundation_bytes": int(foundation_bytes),
        "checkpoint_bytes": int(files["checkpoint.pt"]),
        "actor_bytes": int(files["actor.bin"]),
        "evidence_bytes": int(files["evidence.npz"]),
        "full_persistent_bytes": int(foundation_bytes + sum(files.values())),
    }


def _audit_package(audit: Audit, folder: Path, record: Mapping[str, Any], foundation_bytes: int | None,
                   config: Mapping[str, Any], where: str) -> int | None:
    try:
        actual = _package_components(folder, foundation_bytes)
    except Exception as exc:
        audit.check(False, "package_files", where,
                    "full deployment package components are missing or unreadable", error=repr(exc))
        return None
    for key, value in actual.items():
        audit.check(int(record.get(key, -1)) == value, "package_accounting", where,
                    "capacity metadata differs from physically saved package bytes",
                    field=key, recorded=record.get(key), actual=value)
    cap = int(config.get("package_budget_bytes", -1))
    audit.check(int(record.get("package_budget_bytes", -1)) == cap,
                "package_budget_record", where,
                "package capacity record differs from the frozen package budget",
                recorded=record.get("package_budget_bytes"), expected=cap)
    deployment_bytes = actual["foundation_bytes"] + actual["checkpoint_bytes"] + actual["actor_bytes"]
    audit.check(int(record.get("deployment_state_bytes", -1)) == deployment_bytes
                and int(record.get("package_bytes", -1)) == actual["full_persistent_bytes"]
                and int(record.get("actor_container_bytes", -1)) == actual["actor_bytes"],
                "package_alias_accounting", where,
                "deployment, package, or actor capacity aliases differ from physical files",
                deployment_state_bytes=record.get("deployment_state_bytes"),
                package_bytes=record.get("package_bytes"), actor_container_bytes=record.get("actor_container_bytes"))
    audit.check(actual["full_persistent_bytes"] <= cap, "package_cap", where,
                "full package including evidence.npz exceeds the frozen persistent cap",
                actual=actual["full_persistent_bytes"], cap=cap)
    return actual["full_persistent_bytes"]


def _audit_ledgers(audit: Audit, folder: Path, records: Mapping[str, Any] | None, where: str) -> dict[str, Any]:
    result: dict[str, Any] = {"environment_steps": 0, "reward_total": 0.0,
                              "rollouts": 0, "ppo_calls": 0, "training_rows": 0, "episodes": 0}
    for key, filename in (("steps_csv", "steps.csv"), ("episodes_csv", "episodes.csv"),
                          ("rollouts_csv", "rollouts.csv")):
        path = folder / filename
        if not path.is_file():
            audit.check(False, "raw_ledger_missing", str(path), "required raw ledger is missing")
            continue
        try:
            rows = _csv(path)
            saved = (records or {}).get(key, {}) if isinstance(records, Mapping) else {}
            if isinstance(saved, Mapping):
                actual_sha = _sha_file(path)
                saved_bytes = saved.get("bytes")
                audit.check(saved.get("sha256") == actual_sha
                            and (saved_bytes is None or int(saved_bytes) == path.stat().st_size),
                            "raw_ledger_digest", str(path),
                            "raw ledger digest or byte count differs from its record",
                            recorded=saved.get("sha256"), actual=actual_sha,
                            recorded_bytes=saved_bytes, actual_bytes=path.stat().st_size)
                count = saved.get("rows", saved.get("row_count"))
                if count is not None:
                    audit.check(int(count) == len(rows), "raw_ledger_row_count", str(path),
                                "raw ledger record row count differs from physical CSV rows",
                                recorded=count, actual=len(rows))
            if key == "steps_csv":
                rewards = [float(row["reward"]) for row in rows]
                audit.check(all(math.isfinite(value) for value in rewards), "raw_reward_finite", str(path),
                            "raw step rewards include non-finite values")
                result["environment_steps"] = len(rows)
                result["reward_total"] = float(sum(rewards))
                if isinstance(saved, Mapping) and saved.get("reward_sum") is not None:
                    audit.check(math.isclose(result["reward_total"], float(saved["reward_sum"]),
                                             rel_tol=1e-8, abs_tol=1e-6),
                                "raw_reward_record", str(path),
                                "raw reward sum differs from its ledger record",
                                recorded=saved.get("reward_sum"), actual=result["reward_total"])
            elif key == "episodes_csv":
                result["episodes"] = len(rows)
            else:
                sizes = [int(row["rows"]) for row in rows]
                updates = [int(row["updates"]) for row in rows]
                result["rollouts"] = len(rows)
                result["ppo_calls"] = int(sum(updates))
                result["training_rows"] = int(sum(sizes))
                audit.check(all(size > 0 and count >= 0 for size, count in zip(sizes, updates)),
                            "raw_rollout_counts", str(path),
                            "raw rollout rows or PPO update counts are invalid")
                if isinstance(saved, Mapping):
                    saved_rollouts = saved.get("rollout_count")
                    saved_rows = saved.get("trained_rows")
                    saved_updates = saved.get("minibatch_updates")
                    for name, value, actual in (("rollout_count", saved_rollouts, len(rows)),
                                                ("trained_rows", saved_rows, result["training_rows"]),
                                                ("minibatch_updates", saved_updates, result["ppo_calls"])):
                        if value is not None:
                            audit.check(int(value) == actual, "raw_rollout_record", str(path),
                                        "rollout count differs from raw rollout rows", field=name,
                                        recorded=value, actual=actual)
        except Exception as exc:
            audit.check(False, "raw_ledger_read", str(path),
                        "raw ledger could not be recounted independently", error=repr(exc))
    return result


def _expected_prefix_schedule(seed: int, pair_kind: str, state: str, history_regime: str,
                              config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rebuild the frozen prefix schedule from the config and deterministic map."""
    from online_rl.storage_decision import bootstrap_schedule, pair_mapping, task_name, template_schedule

    mapping = pair_mapping(seed, pair_kind)
    schedule = bootstrap_schedule(mapping, int(config["bootstrap_steps"]))
    next_bout = len(schedule)
    if state == "late":
        for cycle in range(int(config["late_warmup_cycles"])):
            for group in (1, 0):
                phase = "late_warmup_new" if group == 1 else "late_warmup_old"
                for variant in (0, 1):
                    schedule.append({
                        "phase": phase,
                        "bout": next_bout,
                        "group": group,
                        "variant": variant,
                        "task": task_name(mapping, group, variant),
                        "steps": int(config["warmup_pair_steps"]),
                        "warmup_cycle": cycle,
                        "pair_group": group,
                    })
                    next_bout += 1
    history = template_schedule(mapping, history_regime, kind="history",
                                steps=int(config["bout_steps"]), start_bout=next_bout)
    source_steps = int(config["early_source_steps"] if state == "early" else config["late_source_steps"])
    for row in history[-2:]:
        row["steps"] = source_steps
    return schedule + history


def _expected_future_schedule(seed: int, pair_kind: str, future_regime: str,
                              start_bout: int, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rebuild the predeclared minority/dominant/dominant future blocks."""
    from online_rl.storage_decision import pair_mapping, task_name

    mapping = pair_mapping(seed, pair_kind)
    dominant = 0 if future_regime == "old_major" else 1
    minority = 1 - dominant
    result: list[dict[str, Any]] = []
    for block in range(int(config["future_blocks"])):
        variant = block % 3
        entries = ((minority, variant), (dominant, variant), (dominant, (variant + 1) % 3))
        for within_block, (group, current_variant) in enumerate(entries):
            result.append({
                "phase": "future", "bout": int(start_bout) + len(result),
                "future_block": block, "within_block": within_block,
                "group": group, "variant": current_variant,
                "task": task_name(mapping, group, current_variant),
                "steps": int(config["bout_steps"]),
            })
    return result


def _check_raw_schedule(audit: Audit, rows: Sequence[Mapping[str, str]], schedule: Sequence[Mapping[str, Any]],
                        *, phase: str, action: str = "", future: str = "", global_start: int,
                        where: str) -> float:
    expected_steps = sum(int(item["steps"]) for item in schedule)
    audit.check(len(rows) == expected_steps, "scheduled_raw_row_count", where,
                "raw transition count differs from the frozen schedule",
                actual=len(rows), expected=expected_steps)
    cursor = 0
    total_reward = 0.0
    for item in schedule:
        bout_rows = int(item["steps"])
        segment = rows[cursor:min(len(rows), cursor + bout_rows)]
        expected_phase = str(item.get("phase", phase))
        if expected_phase == "history":
            expected_phase = "prefix_history"
        for offset in range(bout_rows):
            index = cursor + offset
            if index >= len(rows):
                break
            row = rows[index]
            expected_global = int(global_start) + index + 1
            expected_bout_step = offset + 1
            audit.check(int(row.get("global_step", -1)) == expected_global,
                        "scheduled_global_step", where,
                        "global step differs from its chronological position", row=index,
                        actual=row.get("global_step"), expected=expected_global)
            audit.check(int(row.get("bout", -1)) == int(item["bout"])
                        and int(row.get("bout_step", -1)) == expected_bout_step
                        and int(row.get("group", -1)) == int(item["group"])
                        and row.get("taskobserver") == item["task"],
                        "scheduled_bout_identity", where,
                        "raw transition bout, position, group, or task differs from the frozen schedule",
                        row=index, expected={key: item.get(key) for key in ("bout", "group", "task")})
            audit.check(row.get("phase") == expected_phase,
                        "scheduled_phase", where,
                        "raw transition phase differs from its schedule segment", row=index,
                        actual=row.get("phase"), expected=expected_phase)
            if expected_phase == "future":
                audit.check(row.get("action") == action and row.get("future_regime") == future,
                            "scheduled_branch_labels", where,
                            "raw action or future-regime label differs from its branch", row=index)
            if bool(int(row.get("learning_boundary", "0"))) != (offset + 1 == bout_rows):
                audit.check(False, "scheduled_learning_boundary", where,
                            "learning boundary does not occur at the expected bout end", row=index)
        total_reward += _audit_raw_step_sequence(
            audit,
            segment,
            phase=expected_phase,
            action=action,
            future=future,
            expected_start=int(global_start) + cursor,
            where=where,
        )
        cursor += bout_rows
    return total_reward


def _audit_prefix_source_state(audit: Audit, checkpoint: Mapping[str, Any], prefix_summary: Mapping[str, Any],
                               observer: Mapping[str, Any], config: Mapping[str, Any], state: str,
                               evidence: Mapping[str, np.ndarray], where: str) -> np.ndarray | None:
    try:
        local, shared, physical = _core_arrays(checkpoint)
        effective_all = effective_matrices(local, shared, physical)
        tags = np.asarray(checkpoint["slot_group_tags"], dtype=np.int8)
        generations = np.asarray(checkpoint["slot_generation"], dtype=np.int64)
        fitted_steps = np.asarray(checkpoint["fitted_steps"], dtype=np.int64)
        sources = list(observer["candidate_sources"])
        source_effective = np.asarray(observer["candidate_source_effective_matrices"], dtype=np.float32)
        audit.check(len(sources) == 2 and source_effective.shape == (2, 3, 3, 3),
                    "candidate_source_layout", where,
                    "prefix must contain two ordered current source records and effective matrices",
                    records=len(sources), shape=list(source_effective.shape))
        if len(sources) != 2 or source_effective.shape != (2, 3, 3, 3):
            return None
        audit.check(observer.get("candidate_source_effective_matrices_sha256") == semantic_hash(source_effective),
                    "candidate_source_effective_hash", where,
                    "candidate effective matrix hash differs from the saved source matrices")
        for owner, source in enumerate(sources):
            slot = int(source.get("slot", -1))
            generation = int(source.get("generation", -1))
            audit.check(0 <= slot < len(effective_all), "candidate_source_slot", where,
                        "candidate source slot is outside the saved learner", owner=owner, slot=slot)
            if not 0 <= slot < len(effective_all):
                continue
            expected_fit = int(config["early_source_steps"] if state == "early" else config["late_source_steps"])
            audit.check(generation == int(generations[slot])
                        and generation >= 0 and int(source.get("fitted_steps", -1)) == int(fitted_steps[slot])
                        and int(fitted_steps[slot]) == expected_fit,
                        "candidate_source_generation_fit", where,
                        "candidate source generation or fitted-step count differs from the checkpoint",
                        owner=owner, slot=slot, generation=generation,
                        checkpoint_generation=int(generations[slot]), fitted=int(fitted_steps[slot]),
                        expected_fit=expected_fit)
            audit.check(int(source.get("group", -1)) == 1 and int(tags[slot]) == 1
                        and int(physical[slot]) == -1,
                        "candidate_source_unbound", where,
                        "current group-one candidate sources must remain local-only and unbound to the old parent",
                        owner=owner, logical_tag=int(tags[slot]), physical_group=int(physical[slot]))
            audit.check(np.allclose(source_effective[owner], effective_all[slot], atol=0.0, rtol=0.0),
                        "candidate_source_checkpoint_matrix", where,
                        "saved candidate effective matrix differs from its live checkpoint slot",
                        owner=owner, slot=slot,
                        max_error=float(np.max(np.abs(source_effective[owner] - effective_all[slot]))))
            audit.check(np.allclose(np.asarray(source.get("effective_matrix", []), dtype=np.float32),
                                    source_effective[owner], atol=0.0, rtol=0.0),
                        "candidate_source_record_matrix", where,
                        "candidate source record differs from its ordered effective matrix", owner=owner)
            audit.check(source.get("effective_sha256") == semantic_hash(source_effective[owner]),
                        "candidate_source_record_hash", where,
                        "candidate source record hash differs from its effective matrix", owner=owner)
            for key in ("source_bout", "slot", "generation"):
                if key == "source_bout":
                    bout_value = int(source.get("source_bout", -1))
                    evidence_bouts = np.asarray(evidence["bouts"])[np.asarray(evidence["owner_indices"]) == owner]
                    audit.check(len(evidence_bouts) == 128 and np.all(evidence_bouts == bout_value),
                                "candidate_source_bout", where,
                                "evidence bout ids do not match the source’s final current bout", owner=owner,
                                source_bout=bout_value)
            sidecar_mask = np.asarray(evidence["owner_indices"]) == owner
            audit.check(np.all(np.asarray(evidence["source_slots"])[sidecar_mask] == slot)
                        and np.all(np.asarray(evidence["generations"])[sidecar_mask] == generation),
                        "candidate_source_evidence_generation", where,
                        "sidecar source slot or generation differs from the current learner source", owner=owner)
        shared_parent = np.asarray(shared[0], dtype=np.float32).reshape(3, 3)
        parent_group = int(checkpoint.get("parent_group", -999))
        audit.check(parent_group == int(observer.get("parent_group", -998)) == 0
                    and np.array_equal(shared_parent, np.asarray(observer.get("current_parent"), dtype=np.float32)),
                    "prefix_old_parent_binding", where,
                    "mature old parent or its logical binding differs from the current checkpoint",
                    checkpoint_binding=parent_group, observer_binding=observer.get("parent_group"))
        audit.check(bool(checkpoint.get("shared_frozen")) and bool(prefix_summary.get("shared_frozen")),
                    "prefix_shared_frozen", where,
                    "shared parent is not recorded frozen in the saved current learner")
        current_mean = source_effective.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        fit = observer.get("candidate_fit", {})
        audit.check(fit.get("mean_candidate_sha256") == semantic_hash(current_mean)
                    and observer.get("current_mean_candidate_sha256") == semantic_hash(current_mean),
                    "mean_candidate_hash", where,
                    "arithmetic mean candidate hash differs from current source matrices")
        audit.check(np.allclose(np.asarray(observer.get("current_parent"), dtype=np.float32), shared_parent,
                                atol=0.0, rtol=0.0),
                    "current_parent_matrix", where,
                    "observer parent matrix differs from learner shared coefficients")
        return source_effective
    except Exception as exc:
        audit.check(False, "prefix_source_state_error", where,
                    "current source state could not be reconstructed from the prefix learner", error=repr(exc))
        return None


def _audit_raw_step_sequence(audit: Audit, rows: Sequence[Mapping[str, str]], *, phase: str,
                             action: str, future: str, expected_start: int, where: str) -> float:
    total = 0.0
    for index, row in enumerate(rows):
        try:
            global_step = int(row["global_step"])
            reward = float(row["reward"])
            audit.check(global_step == expected_start + index + 1, "raw_global_step_sequence", where,
                        "raw global step sequence has a gap or duplicate", row=index,
                        actual=global_step, expected=expected_start + index + 1)
            audit.check(row.get("phase") == phase, "raw_phase", where,
                        "raw transition phase differs from its ledger location", row=index)
            if phase == "future":
                audit.check(row.get("action") == action and row.get("future_regime") == future,
                            "raw_branch_labels", where,
                            "raw transition action/future labels differ from their branch", row=index)
            audit.check(math.isfinite(reward), "raw_reward_finite", where,
                        "raw reward is non-finite", row=index, reward=reward)
            total += reward
            probabilities = _json_array(row["probabilities"], shape=(4,), dtype=np.float64)
            action_id = int(row["action_id"])
            uniform = float(row["action_uniform"])
            audit.check(np.isfinite(probabilities).all() and np.all(probabilities >= 0.0)
                        and math.isclose(float(probabilities.sum()), 1.0, abs_tol=2e-6),
                        "raw_behavior_probabilities", where,
                        "raw action probabilities are malformed", row=index)
            audit.check(0 <= action_id < 4 and 0.0 <= uniform < 1.0,
                        "raw_action_sample", where,
                        "raw action id or uniform variate is out of range", row=index)
            if 0 <= action_id < 4 and probabilities.shape == (4,):
                selected = min(int(np.searchsorted(np.cumsum(probabilities), uniform, side="right")), 3)
                audit.check(selected == action_id, "raw_action_reconstruction", where,
                            "action does not match its probabilities and recorded uniform", row=index,
                            actual=action_id, reconstructed=selected)
            done = bool(int(row["environment_done"]))
            term = bool(int(row["terminated"]))
            trunc = bool(int(row["truncated"]))
            audit.check(done == (term or trunc), "raw_done_flags", where,
                        "environment_done differs from termination/truncation flags", row=index)
        except Exception as exc:
            audit.check(False, "raw_step_parse", where,
                        "raw step row is missing or has malformed integrity fields", row=index, error=repr(exc))
    return total


def _instantiate_core(checkpoint: Mapping[str, Any], config: Mapping[str, Any], manifest: Mapping[str, Any]):
    """Recreate the fixed actor trunk from the frozen foundation and checkpoint."""
    from scripts import run_knowledge_loop as loop
    from scripts import run_storage_decision as storage

    base = loop.load_base_config()
    foundation = manifest.get("foundation", {}).get("path")
    if foundation:
        base = dict(base)
        base["foundation"] = str(foundation)
    agent = storage._new_agent(int(checkpoint["seed"]), base, config)
    agent.core.load_state_dict(checkpoint["core_state"], strict=True)
    return agent.core


def _kl_rows(teacher: np.ndarray, student: np.ndarray) -> np.ndarray:
    teacher = np.asarray(teacher, dtype=np.float64)
    student = np.asarray(student, dtype=np.float64)
    teacher_safe = np.clip(teacher, 1e-12, 1.0)
    student_safe = np.clip(student, 1e-12, 1.0)
    return np.sum(teacher_safe * (np.log(teacher_safe) - np.log(student_safe)), axis=1)


def _candidate_module_hash(value: Any) -> str:
    """Reproduce the candidate module's private float32 parameter hash."""
    array = np.ascontiguousarray(value, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(b"float32\0")
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _feature_probabilities(features: Any, coefficients: Any, *, effective: bool = False,
                           owners: np.ndarray | None = None, observations: np.ndarray | None = None) -> np.ndarray:
    """Compute policy probabilities in float64 from frozen extracted features."""
    base = np.asarray(features.base_logits, dtype=np.float64)
    psi = np.asarray(features.source_psi, dtype=np.float64)
    basis = np.asarray(features.contrast_basis, dtype=np.float64)
    coeff = np.asarray(coefficients, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 4 or psi.shape != (len(base), 3) or basis.shape != (4, 3):
        raise ValueError("frozen policy features have inconsistent shapes")
    if effective:
        if owners is None or observations is None or coeff.shape != (2, 3, 3, 3):
            raise ValueError("effective-matrix probabilities need owners, observations and a 2 by 3 by 3 by 3 matrix")
        owner_array = np.asarray(owners, dtype=np.int64)
        obs_array = np.asarray(observations, dtype=np.float32)
        if owner_array.shape != (len(base),) or obs_array.shape != (len(base), 13):
            raise ValueError("owner or observation rows do not match the frozen features")
        if np.any((owner_array < 0) | (owner_array > 1)):
            raise ValueError("owner indices must be zero or one")
        bands = np.floor(np.clip(obs_array[:, 0], 0.0, 1.0) * 3.0).astype(np.int64).clip(0, 2)
        row_coefficients = coeff[owner_array, bands]
        contrast = np.einsum("nij,nj->ni", row_coefficients, psi)
    else:
        if coeff.shape != (3, 3):
            raise ValueError("shared coefficients must be 3 by 3")
        contrast = psi @ coeff.T
    logits = base + contrast @ basis.T
    scaled = logits / float(features.temperature)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    softmax = np.exp(shifted)
    softmax /= softmax.sum(axis=1, keepdims=True)
    epsilon = float(features.epsilon)
    return (1.0 - epsilon) * softmax + epsilon / 4.0


def _feature_objective(features: Any, coefficients: np.ndarray, teacher: np.ndarray,
                       row_weights: np.ndarray, common: np.ndarray, ridge: float) -> float:
    student = _feature_probabilities(features, coefficients)
    row_kl = _kl_rows(teacher, student)
    return float(np.asarray(row_weights, dtype=np.float64) @ row_kl
                 + float(ridge) * np.mean((np.asarray(coefficients, dtype=np.float64)
                                          - np.asarray(common, dtype=np.float64)) ** 2))


def _audit_fit_math(audit: Audit, observer: Mapping[str, Any], evidence: Mapping[str, np.ndarray],
                    candidate: np.ndarray | None, checkpoint: Mapping[str, Any], config: Mapping[str, Any],
                    manifest: Mapping[str, Any], where: str) -> dict[str, Any] | None:
    """Recompute teachers, split KLs, candidate objective, and the analytic gradient."""
    try:
        from online_rl import retention_behavioral_candidate as behavioral
        from scripts import run_storage_decision as storage

        owners = np.asarray(evidence["owner_indices"], dtype=np.int64)
        observations = np.asarray(evidence["observations"], dtype=np.float32)
        effective = np.asarray(observer["candidate_source_effective_matrices"], dtype=np.float32)
        audit.check(effective.shape == (2, 3, 3, 3), "source_effective_shape", where,
                    "candidate source effective matrices must have shape 2 by 3 by 3 by 3",
                    shape=list(effective.shape))
        if effective.shape != (2, 3, 3, 3):
            return None
        if candidate is None:
            # The candidate is intentionally transient. Refit only from the
            # saved common-prefix state and bounded past evidence.
            candidate, refit_diagnostics = behavioral.fit_behavioral_candidate(
                observations,
                np.asarray(evidence["owner_indices"], dtype=np.int8),
                effective,
                behavioral.extract_frozen_policy_features(
                    _instantiate_core(checkpoint, config, manifest), observations,
                ),
                min_rows_per_owner=int(config["behavioral_fit"]["min_rows_per_owner"]),
            )
            saved_fit = observer.get("candidate_fit", {})
            saved_diagnostics = saved_fit.get("behavioral", {})
            audit.check(refit_diagnostics.get("parameter_hashes", {}).get("candidate_sha256")
                        == _candidate_module_hash(candidate),
                        "behavioral_refit_hash", where,
                        "deterministically refitted candidate differs from the saved candidate hash",
                        refit=refit_diagnostics.get("parameter_hashes", {}).get("candidate_sha256"),
                        recomputed=_candidate_module_hash(candidate))
            audit.check(saved_fit.get("behavioral_candidate_sha256") == semantic_hash(candidate)
                        and saved_fit.get("behavioral_candidate_module_sha256") == _candidate_module_hash(candidate),
                        "behavioral_refit_saved_hash", where,
                        "saved candidate semantic/module hashes differ from the deterministic refit",
                        semantic_saved=saved_fit.get("behavioral_candidate_sha256"),
                        module_saved=saved_fit.get("behavioral_candidate_module_sha256"))
            for key in ("status", "optimizer_status", "optimizer_function_evaluations", "iterations"):
                audit.check(saved_diagnostics.get(key) == refit_diagnostics.get(key),
                            "behavioral_refit_diagnostic", where,
                            "deterministic refit differs from saved optimizer diagnostics", field=key,
                            saved=saved_diagnostics.get(key), recomputed=refit_diagnostics.get(key))
        candidate = np.asarray(candidate, dtype=np.float32)
        audit.check(candidate.shape == (3, 3) and np.isfinite(candidate).all(), "behavioral_candidate_shape", where,
                    "behavioral candidate must be a finite 3 by 3 float32 matrix",
                    shape=list(candidate.shape), dtype=str(candidate.dtype))
        if candidate.shape != (3, 3) or not np.isfinite(candidate).all():
            return None
        core = _instantiate_core(checkpoint, config, manifest)
        features = behavioral.extract_frozen_policy_features(core, observations)
        teacher = _feature_probabilities(features, effective, effective=True, owners=owners,
                                         observations=observations)
        live_teacher = np.empty((len(observations), 4), dtype=np.float32)
        for owner in (0, 1):
            mask = owners == owner
            if np.any(mask):
                live_teacher[mask] = storage._coefficient_probs(core, observations[mask], effective[owner])
        live_teacher_error = float(np.max(np.abs(teacher - live_teacher.astype(np.float64)), initial=0.0))
        audit.check(live_teacher_error <= 2e-6, "current_teacher_probabilities", where,
                    "independent float64 current-source teacher probabilities disagree with the live coefficient path",
                    max_abs_error=live_teacher_error)
        # Compute diagnostics from independent float64 feature logits. The
        # persistent live actor path is float32; its small rounding error must
        # not weaken the optimizer's strict no-worse-than-start requirement.
        student = _feature_probabilities(features, candidate)
        common = effective.mean(axis=(0, 1), dtype=np.float64)
        initial = common.astype(np.float32).astype(np.float64)
        train_mask = np.zeros(len(owners), dtype=bool)
        holdout_mask = np.zeros(len(owners), dtype=bool)
        for owner in (0, 1):
            owner_rows = np.flatnonzero(owners == owner)
            split = int(math.floor(0.75 * len(owner_rows)))
            train_mask[owner_rows[:split]] = True
            holdout_mask[owner_rows[split:]] = True

        def split_metrics(mask: np.ndarray, probs: np.ndarray) -> tuple[list[float | None], float | None]:
            row_kl = _kl_rows(teacher[mask], probs[mask])
            owner_subset = owners[mask]
            by_owner = [float(np.mean(row_kl[owner_subset == owner])) if np.any(owner_subset == owner) else None
                        for owner in (0, 1)]
            overall = None if any(value is None for value in by_owner) else float(np.mean(by_owner))
            return by_owner, overall

        train_fit_by_owner, train_kl = split_metrics(train_mask, student)
        train_init_by_owner, train_init_kl = split_metrics(train_mask, _feature_probabilities(features, initial))
        holdout_fit_by_owner, holdout_kl = split_metrics(holdout_mask, student)
        holdout_init_by_owner, holdout_init_kl = split_metrics(
            holdout_mask, _feature_probabilities(features, initial))
        ridge = float(config["behavioral_fit"]["ridge"])
        objective_fit = float(train_kl + ridge * np.mean((candidate.astype(np.float64) - common) ** 2))
        objective_init = float(train_init_kl + ridge * np.mean((initial - common) ** 2))
        fit_record = observer["candidate_fit"]
        diagnostics = fit_record["behavioral"]
        tolerance = 2e-6
        for key, actual in (("objective_init", objective_init), ("objective_fit", objective_fit),
                            ("train_kl_mean_init", train_init_kl), ("train_kl_mean_fit", train_kl),
                            ("holdout_kl_mean_init", holdout_init_kl), ("holdout_kl_mean_fit", holdout_kl)):
            saved = diagnostics.get(key)
            audit.check(saved is not None and math.isclose(float(saved), actual, rel_tol=2e-5, abs_tol=tolerance),
                        "behavioral_fit_metric", where,
                        "saved fit diagnostic differs from independently recomputed values",
                        field=key, saved=saved, recomputed=actual)
        for key, actual in (("train_kl_by_owner_init", train_init_by_owner),
                            ("train_kl_by_owner_fit", train_fit_by_owner),
                            ("holdout_kl_by_owner_init", holdout_init_by_owner),
                            ("holdout_kl_by_owner_fit", holdout_fit_by_owner)):
            saved = diagnostics.get(key)
            audit.check(isinstance(saved, Sequence) and len(saved) == 2
                        and all(value is not None and math.isclose(float(value), float(expected),
                                                                  rel_tol=2e-5, abs_tol=tolerance)
                                for value, expected in zip(saved, actual)),
                        "behavioral_fit_owner_metric", where,
                        "per-owner fit diagnostic differs from independently recomputed values",
                        field=key, saved=saved, recomputed=actual)
        audit.check(objective_fit <= objective_init + 1e-12, "behavioral_training_objective", where,
                    "fitted training objective is worse than the arithmetic-mean initialization",
                    initial=objective_init, fitted=objective_fit)
        audit.check(int(diagnostics.get("optimizer_function_evaluations", 257)) <= 256
                    and int(diagnostics.get("optimizer_max_evaluations", -1)) == 256,
                    "behavioral_closure_budget", where,
                    "candidate optimizer exceeded or misreported its hard closure-evaluation budget",
                    evaluations=diagnostics.get("optimizer_function_evaluations"),
                    max=diagnostics.get("optimizer_max_evaluations"))
        audit.check(diagnostics.get("optimizer") == "torch_lbfgs_strong_wolfe"
                    and int(diagnostics.get("optimizer_max_iterations", -1)) == 64,
                    "behavioral_optimizer_config", where,
                    "candidate optimizer differs from the frozen optimizer protocol")
        audit.check(float(diagnostics.get("ridge", float("nan"))) == ridge,
                    "behavioral_ridge", where,
                    "candidate fit ridge differs from the frozen configuration")
        candidate_hash = semantic_hash(candidate)
        module_candidate_hash = _candidate_module_hash(candidate)
        audit.check(fit_record.get("behavioral_candidate_sha256") == candidate_hash
                    and observer.get("current_behavioral_candidate_sha256") == candidate_hash
                    and fit_record.get("behavioral_candidate_module_sha256") == module_candidate_hash
                    and diagnostics.get("parameter_hashes", {}).get("candidate_sha256") == module_candidate_hash,
                    "behavioral_candidate_hash", where,
                    "saved semantic or module candidate hashes do not match the independently recovered coefficients",
                    semantic_actual=candidate_hash, module_actual=module_candidate_hash)
        module_hashes = diagnostics.get("parameter_hashes", {})
        audit.check(module_hashes.get("common_mean_sha256") == _candidate_module_hash(initial)
                    and module_hashes.get("effective_matrices_sha256") == _candidate_module_hash(effective),
                    "behavioral_module_parameter_hashes", where,
                    "candidate module-specific hashes do not match its initialization or source matrices",
                    saved=module_hashes)
        for field, array in (("observations_sha256", observations),
                             ("owner_indices_sha256", np.asarray(evidence["owner_indices"], dtype=np.int8)),
                             ("effective_matrices_sha256", effective)):
            saved = fit_record.get(field)
            actual = semantic_hash(array)
            audit.check(saved == actual, "behavioral_fit_input_hash", where,
                        "candidate fit input hash differs from the audited bounded inputs",
                        field=field, saved=saved, recomputed=actual)
        # Finite-difference the public analytic gradient at the returned point.
        train_indices = np.flatnonzero(train_mask)
        train_features = behavioral.FrozenPolicyFeatures(
            base_logits=features.base_logits[train_indices],
            source_psi=features.source_psi[train_indices],
            contrast_basis=features.contrast_basis,
            temperature=features.temperature,
            epsilon=features.epsilon,
        )
        train_weights = np.asarray([
            1.0 / (2.0 * np.sum(owners[train_indices] == owner))
            for owner in owners[train_indices]
        ], dtype=np.float64)
        train_teacher = teacher[train_indices]
        _, gradient = behavioral.behavioral_objective_and_gradient(
            candidate.astype(np.float64), train_features, train_teacher, train_weights, common,
        )
        finite_gradient = np.empty_like(gradient)
        delta = 1e-6
        for row in range(3):
            for col in range(3):
                plus, minus = candidate.astype(np.float64).copy(), candidate.astype(np.float64).copy()
                plus[row, col] += delta
                minus[row, col] -= delta
                f_plus = _feature_objective(train_features, plus, train_teacher, train_weights, common, ridge)
                f_minus = _feature_objective(train_features, minus, train_teacher, train_weights, common, ridge)
                finite_gradient[row, col] = (f_plus - f_minus) / (2.0 * delta)
        max_gradient_error = float(np.max(np.abs(gradient - finite_gradient), initial=0.0))
        audit.check(max_gradient_error <= 2e-5, "behavioral_gradient", where,
                    "analytic fit gradient differs from central finite differences",
                    max_abs_error=max_gradient_error)
        # Holdout is diagnostic only. It is measured and emitted but never used as a gate.
        if holdout_kl is not None and holdout_kl > 0.5:
            audit.warn("behavioral_holdout_kl_high", where,
                       "temporally later holdout KL is high; this diagnostic does not gate candidate selection",
                       holdout_kl=holdout_kl)
        return {"candidate": candidate.copy(), "mean_candidate": common.astype(np.float32),
                "features": features,
                "train_kl": train_kl, "holdout_kl": holdout_kl,
                "objective_fit": objective_fit, "objective_init": objective_init,
                "max_gradient_error": max_gradient_error}
    except Exception as exc:
        audit.check(False, "behavioral_fit_audit_error", where,
                    "candidate fit could not be independently recomputed", error=repr(exc))
        return None


def _audit_artifact_records(audit: Audit, root: Path, artifacts: Mapping[str, Any], where: str) -> None:
    for name, record in artifacts.items():
        if not isinstance(record, Mapping) or not record.get("path"):
            audit.check(False, "artifact_record_shape", where,
                        "artifact record is missing its relative path", artifact=name)
            continue
        path = _relative_file(root, str(record["path"]))
        if path is None or not path.is_file():
            audit.check(False, "artifact_missing", where,
                        "recorded package artifact is missing or escapes the cohort root", artifact=name,
                        path=record.get("path"))
            continue
        try:
            audit.check(path.stat().st_size == int(record.get("bytes", -1))
                        and _sha_file(path) == record.get("sha256"),
                        "artifact_hash", str(path),
                        "package artifact differs from its recorded digest or byte count", artifact=name)
        except Exception as exc:
            audit.check(False, "artifact_read", str(path),
                        "package artifact could not be checked", artifact=name, error=repr(exc))


def _audit_evidence_copy(audit: Audit, source: Path, target: Path, expected: Mapping[str, Any] | None,
                         where: str) -> None:
    if not source.is_file() or not target.is_file():
        audit.check(False, "evidence_copy_missing", where,
                    "prefix and deployment evidence sidecars must both be present",
                    source_exists=source.is_file(), target_exists=target.is_file())
        return
    try:
        source_hash, target_hash = _sha_file(source), _sha_file(target)
        same = source.stat().st_size == target.stat().st_size and source_hash == target_hash
        audit.check(same, "evidence_copy_identity", where,
                    "deployment evidence sidecar is not byte-identical to its common-prefix source",
                    source_bytes=source.stat().st_size, target_bytes=target.stat().st_size,
                    source_sha256=source_hash, target_sha256=target_hash)
        if expected is not None:
            audit.check(int(expected.get("bytes", -1)) == target.stat().st_size
                        and expected.get("sha256") == target_hash,
                        "evidence_copy_record", where,
                        "deployment evidence copy differs from its recorded evidence metadata")
    except Exception as exc:
        audit.check(False, "evidence_copy_read", where,
                    "evidence copy could not be hashed", error=repr(exc))


def _audit_warmup_writes(audit: Audit, prefix: Mapping[str, Any], observer: Mapping[str, Any],
                         state: str, config: Mapping[str, Any], where: str) -> None:
    records = prefix.get("warmup_pair_records", [])
    expected_count = 4 * 2 if state == "late" else 0
    audit.check(len(records) == expected_count, "warmup_record_count", where,
                "late state must record four local new-pair phases and four old-parent writes",
                actual=len(records), expected=expected_count)
    audit.check(config.get("warmup_write_rule") == "replace_old_group_only_after_pair",
                "warmup_write_rule", where,
                "late warmup does not use the frozen old-group-only parent write rule",
                actual=config.get("warmup_write_rule"))
    last_after_hash: str | None = None
    for index, record in enumerate(records):
        expected_cycle = index // 2
        expected_group = 1 if index % 2 == 0 else 0
        expected_phase = "late_warmup_new" if expected_group == 1 else "late_warmup_old"
        expected_write = expected_group == 0
        audit.check(int(record.get("cycle", -1)) == expected_cycle
                    and int(record.get("group", -1)) == expected_group
                    and record.get("phase") == expected_phase
                    and bool(record.get("write_applied")) is expected_write
                    and record.get("write_rule") == "replace_old_group_only_after_pair",
                    "warmup_pair_order", where,
                    "late warmup pair order or parent-write provenance differs from the frozen construction",
                    pair=index, cycle=record.get("cycle"), group=record.get("group"),
                    phase=record.get("phase"), write_applied=record.get("write_applied"))
        binding = record.get("parent_binding_before_after", [])
        hashes = record.get("parent_sha256_before_after", [])
        audit.check(len(binding) == 2 and binding == [0, 0]
                    and len(hashes) == 2 and hashes[0] == (last_after_hash or hashes[0]),
                    "warmup_parent_continuity", where,
                    "old parent binding or coefficients did not persist continuously across warmup pairs",
                    pair=index, bindings=binding, hashes=hashes, previous_after=last_after_hash)
        if len(hashes) == 2:
            last_after_hash = str(hashes[1])
        matrices = np.asarray(record.get("source_effective_matrices", []), dtype=np.float32)
        audit.check(matrices.shape == (2, 3, 3, 3)
                    and record.get("source_effective_sha256") == semantic_hash(matrices),
                    "warmup_source_matrix_hash", where,
                    "warmup source matrix payload or hash is malformed", pair=index,
                    shape=list(matrices.shape), saved=record.get("source_effective_sha256"))
        source_records = list(record.get("source_records", []))
        audit.check(len(source_records) == 2, "warmup_source_records", where,
                    "each warmup pair must identify two trained source variants", pair=index,
                    records=len(source_records))
        for owner, source in enumerate(source_records[:2]):
            try:
                saved_matrix = np.asarray(source.get("effective_matrix", []), dtype=np.float32)
                audit.check(int(source.get("group", -1)) == expected_group
                            and int(source.get("variant", -1)) == owner
                            and int(source.get("fitted_steps", -1)) == int(config["warmup_pair_steps"])
                            and np.array_equal(saved_matrix, matrices[owner]),
                            "warmup_source_identity", where,
                            "warmup source record differs from the ordered trained pair", pair=index, owner=owner)
            except Exception as exc:
                audit.check(False, "warmup_source_identity_read", where,
                            "warmup source record could not be validated", pair=index, owner=owner,
                            error=repr(exc))
        if expected_write and matrices.shape == (2, 3, 3, 3):
            expected_parent = matrices.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
            audit.check(record.get("parent_sha256") == semantic_hash(expected_parent),
                        "warmup_old_mean_write", where,
                        "old parent write is not the arithmetic mean of the completed old pair",
                        pair=index, expected=semantic_hash(expected_parent), saved=record.get("parent_sha256"))
            audit.check(isinstance(record.get("formation"), Mapping)
                        and math.isfinite(float(record.get("codec_max_coefficient_error", float("nan")))),
                        "warmup_old_write_operation", where,
                        "old-group warmup is missing its parent formation or codec record", pair=index)
        else:
            audit.check(bool(record.get("parent_coefficients_unchanged"))
                        and len(hashes) == 2 and hashes[0] == hashes[1]
                        and record.get("formation") is None
                        and float(record.get("codec_max_coefficient_error", float("nan"))) == 0.0,
                        "warmup_new_local_only", where,
                        "new-group warmup must be local-only and must not write or codec the shared parent",
                        pair=index)
    if state == "early":
        audit.check(last_after_hash is None, "early_has_no_warmups", where,
                    "early prefix unexpectedly contains late warmup write records")
    elif records:
        audit.check(last_after_hash == prefix.get("current_parent_sha256"), "warmup_final_parent", where,
                    "last old-group warmup write does not match the mature prefix parent",
                    last_write=last_after_hash, prefix_parent=prefix.get("current_parent_sha256"))
    incumbent = np.asarray(observer.get("incumbent_source_effective_matrices", []), dtype=np.float32)
    if incumbent.shape == (2, 3, 3, 3):
        expected_mean = incumbent.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        parent = np.asarray(observer.get("current_parent", []), dtype=np.float32)
        error = float(np.max(np.abs(expected_mean - parent), initial=0.0)) if parent.shape == (3, 3) else math.inf
        audit.check(error <= 0.02, "mature_parent_source_relation", where,
                    "frozen old parent differs materially from the old pair mean that formed it",
                    max_abs_error=error)
    else:
        audit.check(False, "incumbent_source_effective_shape", where,
                    "prefix must record the two source-time old-group effective matrices",
                    shape=list(incumbent.shape))


def _audit_actor_payload(audit: Audit, folder: Path, checkpoint: Mapping[str, Any], config: Mapping[str, Any],
                         where: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        from online_rl.write_decision_policy import decode_actor, encode_actor

        actor_path = folder / "actor.bin"
        if not actor_path.is_file():
            audit.check(False, "actor_payload_missing", where, "actor codec payload is missing")
            return None
        blob = actor_path.read_bytes()
        local, shared, physical = _core_arrays(checkpoint)
        decoded_local, decoded_shared, decoded_groups = decode_actor(blob)
        audit.check(len(blob) == int(config["actor_budget_bytes"]), "actor_payload_budget", where,
                    "serialized actor exceeds or differs from the fixed actor budget",
                    actual=len(blob), expected=config["actor_budget_bytes"])
        audit.check(np.array_equal(decoded_local, local) and np.array_equal(decoded_shared, shared)
                    and np.array_equal(decoded_groups, physical),
                    "actor_checkpoint_agreement", where,
                    "actor codec payload does not decode to the saved checkpoint actor matrices")
        try:
            canonical = encode_actor(local, shared, physical, codec="int8",
                                     budget_bytes=int(config["actor_budget_bytes"])).data
            audit.check(blob == canonical, "actor_codec_canonical", where,
                        "actor payload is not the canonical int8 encoding of the checkpoint actor")
        except Exception as exc:
            audit.check(False, "actor_codec_encode", where,
                        "checkpoint actor could not be encoded within its fixed codec budget", error=repr(exc))
        return local, shared, physical
    except Exception as exc:
        audit.check(False, "actor_payload_audit", where,
                    "actor payload could not be decoded and compared", error=repr(exc))
        return None


def _teacher_kl_metrics_float64(features: Any, observations: np.ndarray, owners: np.ndarray,
                                effective: np.ndarray, parent: np.ndarray) -> dict[str, Any]:
    teacher = _feature_probabilities(features, effective, effective=True, owners=owners,
                                     observations=observations)
    student = _feature_probabilities(features, parent)
    row_kl = _kl_rows(teacher, student)
    all_by_owner: list[float] = []
    train_by_owner: list[float] = []
    holdout_by_owner: list[float] = []
    for owner in (0, 1):
        positions = np.flatnonzero(owners == owner)
        split = int(math.floor(0.75 * len(positions)))
        all_by_owner.append(float(np.mean(row_kl[positions])))
        train_by_owner.append(float(np.mean(row_kl[positions[:split]])))
        holdout_by_owner.append(float(np.mean(row_kl[positions[split:]])))
    return {
        "mean": float(np.mean(all_by_owner)),
        "by_owner": all_by_owner,
        "train_mean": float(np.mean(train_by_owner)),
        "train_by_owner": train_by_owner,
        "holdout_mean": float(np.mean(holdout_by_owner)),
        "holdout_by_owner": holdout_by_owner,
        "rows_by_owner": [int(np.sum(owners == owner)) for owner in (0, 1)],
    }


def _compare_kl_record(audit: Audit, recorded: Mapping[str, Any], expected: Mapping[str, Any],
                       where: str, label: str) -> None:
    for key in ("mean", "train_mean", "holdout_mean"):
        saved = recorded.get(key)
        actual = expected.get(key)
        audit.check(saved is not None and actual is not None
                    and math.isclose(float(saved), float(actual), rel_tol=2e-5, abs_tol=2e-6),
                    "teacher_kl_metric", where,
                    "saved source-teacher KL summary differs from independent float64 probabilities",
                    stage=label, field=key, saved=saved, recomputed=actual)
    for key in ("by_owner", "train_by_owner", "holdout_by_owner"):
        saved = recorded.get(key)
        actual = expected.get(key)
        audit.check(isinstance(saved, Sequence) and isinstance(actual, Sequence)
                    and len(saved) == len(actual) == 2
                    and all(math.isclose(float(left), float(right), rel_tol=2e-5, abs_tol=2e-6)
                            for left, right in zip(saved, actual)),
                    "teacher_kl_owner_metric", where,
                    "saved per-owner source-teacher KL differs from independent float64 probabilities",
                    stage=label, field=key, saved=saved, recomputed=actual)


def _audit_action_install(audit: Audit, branch: Mapping[str, Any], prefix_checkpoint: Mapping[str, Any],
                          post_checkpoint: Mapping[str, Any], prefix_actor: bytes, post_folder: Path,
                          observer: Mapping[str, Any], evidence: Mapping[str, np.ndarray],
                          fit_result: Mapping[str, Any] | None, config: Mapping[str, Any],
                          where: str) -> None:
    try:
        from online_rl.retention_stream_value import apply_retention_action
        from online_rl.write_decision_policy import decode_actor, encode_actor
        from scripts import run_storage_decision as storage

        arm = str(branch["arm"])
        if arm not in ARM_SPECS:
            audit.check(False, "branch_arm", where, "branch arm is outside the frozen five-arm design", arm=arm)
            return
        operation, candidate_kind, _ = ARM_SPECS[arm]
        action = branch.get("action_install", {})
        saved_candidate_kind = "not_used" if arm == "keep" else candidate_kind
        audit.check(action.get("arm") == arm and action.get("operation") == operation
                    and action.get("candidate_kind") == saved_candidate_kind,
                    "action_arm_mapping", where,
                    "branch action installation does not match its predeclared arm",
                    arm=arm, operation=action.get("operation"), candidate_kind=action.get("candidate_kind"))
        candidate_matrix = (np.asarray(fit_result["mean_candidate"], dtype=np.float32)
                            if candidate_kind == "mean" else
                            np.asarray(fit_result["candidate"], dtype=np.float32)
                            if fit_result is not None else None)
        if candidate_matrix is None:
            audit.check(False, "action_candidate_missing", where,
                        "candidate action cannot be replayed because the prefix fit audit failed")
            return
        expected_candidate_hash = None if arm == "keep" else semantic_hash(candidate_matrix)
        audit.check(action.get("candidate_sha256") == expected_candidate_hash,
                    "action_candidate_hash", where,
                    "action candidate hash differs from the independently reconstructed fit",
                    saved=action.get("candidate_sha256"), expected=expected_candidate_hash)

        local, shared, physical = _core_arrays(prefix_checkpoint)
        logical_tags = np.asarray(prefix_checkpoint["slot_group_tags"], dtype=np.int8)
        parent_binding = int(prefix_checkpoint["parent_group"])
        source_records = list(observer["candidate_sources"])
        candidate_slots = [int(row["slot"]) for row in source_records]
        audit.check(len(candidate_slots) == 2 and len(set(candidate_slots)) == 2,
                    "action_candidate_slots", where,
                    "candidate operation must address the two distinct prefix resident slots",
                    slots=candidate_slots)
        expected_local, expected_shared, expected_groups, details = apply_retention_action(
            local, shared, physical, logical_tags, parent_binding, candidate_matrix, 1, operation,
        )
        expected_effective = effective_matrices(expected_local, expected_shared, expected_groups)
        saved_details = action.get("details", {})
        for key in ("action", "parent_group", "incumbent_binding", "candidate_group", "bound_slots"):
            audit.check(saved_details.get(key) == details.get(key), "action_intervention_details", where,
                        "saved action intervention details differ from independent replay", field=key,
                        saved=saved_details.get(key), recomputed=details.get(key))
        for key in ("max_effective_preservation_error",):
            audit.check(saved_details.get(key) is not None
                        and math.isclose(float(saved_details[key]), float(details[key]), rel_tol=1e-6, abs_tol=1e-8),
                        "action_preservation_metric", where,
                        "saved resident-preservation error differs from replay", field=key,
                        saved=saved_details.get(key), recomputed=details.get(key))
        precodec_sources = expected_effective[np.asarray(candidate_slots, dtype=np.int64)]
        saved_precodec = np.asarray(action.get("candidate_source_effective_precodec", []), dtype=np.float32)
        audit.check(saved_precodec.shape == (2, 3, 3, 3)
                    and np.array_equal(saved_precodec, precodec_sources)
                    and action.get("candidate_source_effective_precodec_sha256") == semantic_hash(precodec_sources),
                    "action_precodec_sources", where,
                    "pre-codec candidate resident matrices differ from the independently replayed operation")
        pre_parent = np.asarray(expected_shared[0], dtype=np.float32).reshape(3, 3)
        audit.check(np.array_equal(np.asarray(action.get("prequantized_parent", []), dtype=np.float32), pre_parent)
                    and action.get("prequantized_parent_sha256") == semantic_hash(pre_parent),
                    "action_precodec_parent", where,
                    "pre-codec parent differs from the independently replayed operation")

        if operation == "keep":
            expected_blob = prefix_actor
        else:
            expected_blob = encode_actor(expected_local, expected_shared, expected_groups, codec="int8",
                                         budget_bytes=int(config["actor_budget_bytes"])).data
        actual_blob = (post_folder / "actor.bin").read_bytes()
        audit.check(actual_blob == expected_blob, "action_codec_payload", where,
                    "post-action actor bytes differ from replayed action and the specified codec",
                    expected_bytes=len(expected_blob), actual_bytes=len(actual_blob))
        decoded_local, decoded_shared, decoded_groups = decode_actor(expected_blob)
        post_local, post_shared, post_groups = _core_arrays(post_checkpoint)
        audit.check(np.array_equal(post_local, decoded_local) and np.array_equal(post_shared, decoded_shared)
                    and np.array_equal(post_groups, decoded_groups),
                    "action_checkpoint_actor", where,
                    "post-action checkpoint actor differs from its canonical serialized actor")
        post_effective = effective_matrices(post_local, post_shared, post_groups)
        post_parent = np.asarray(post_shared[0], dtype=np.float32).reshape(3, 3)
        post_sources = post_effective[np.asarray(candidate_slots, dtype=np.int64)]
        saved_post_sources = np.asarray(action.get("candidate_source_effective_postcodec", []), dtype=np.float32)
        audit.check(saved_post_sources.shape == (2, 3, 3, 3)
                    and np.array_equal(saved_post_sources, post_sources)
                    and action.get("candidate_source_effective_postcodec_sha256") == semantic_hash(post_sources),
                    "action_postcodec_sources", where,
                    "post-codec candidate source matrices differ from the serialized post-action learner")
        audit.check(np.array_equal(np.asarray(action.get("postquantized_parent", []), dtype=np.float32), post_parent)
                    and action.get("postquantized_parent_sha256") == semantic_hash(post_parent)
                    and int(post_checkpoint["parent_group"]) == int(action.get("parent_group_after_install", -999)),
                    "action_postcodec_parent", where,
                    "post-codec parent or binding differs from the serialized post-action learner")

        codec_error = max(float(np.max(np.abs(decoded_local - expected_local), initial=0.0)),
                          float(np.max(np.abs(decoded_shared - expected_shared), initial=0.0)))
        codec_effective = np.max(np.abs(post_effective - expected_effective), axis=(1, 2, 3)).astype(float)
        audit.check(math.isclose(float(action.get("codec_max_coefficient_error", -1.0)), codec_error,
                                 rel_tol=1e-6, abs_tol=1e-8)
                    and np.allclose(np.asarray(action.get("codec_effective_errors", []), dtype=np.float64),
                                    codec_effective, rtol=1e-6, atol=1e-8),
                    "action_codec_errors", where,
                    "recorded action codec errors differ from quantization of the replayed actor",
                    coefficient_error=codec_error, effective_errors=codec_effective.tolist())
        pre_error = np.max(np.abs(expected_effective - effective_matrices(local, shared, physical)), axis=(1, 2, 3))
        audit.check(np.allclose(np.asarray(action.get("prequantized_effective_errors", []), dtype=np.float64),
                                pre_error, rtol=1e-6, atol=1e-8),
                    "action_preserved_effective_matrices", where,
                    "recorded pre-codec resident preservation errors differ from replay")
        audit.check(int(post_checkpoint["parent_group"]) == int(action.get("parent_group_after_install", -999))
                    and bool(post_checkpoint["shared_frozen"]),
                    "action_binding_frozen", where,
                    "post-action shared parent binding or frozen flag is invalid")

        features = fit_result.get("features") if fit_result is not None else None
        if features is not None:
            owners = np.asarray(evidence["owner_indices"], dtype=np.int64)
            observations = np.asarray(evidence["observations"], dtype=np.float32)
            effective = np.asarray(observer["candidate_source_effective_matrices"], dtype=np.float32)
            for field, parent in (("pre_action_new_source_teacher_kl", shared[0].reshape(3, 3)),
                                  ("new_source_teacher_kl_before_codec", pre_parent),
                                  ("new_source_teacher_kl_after_codec", post_parent)):
                expected_kl = _teacher_kl_metrics_float64(features, observations, owners, effective,
                                                         np.asarray(parent, dtype=np.float32))
                _compare_kl_record(audit, action.get(field, {}), expected_kl, where, field)
    except Exception as exc:
        audit.check(False, "action_install_audit", where,
                    "branch action installation could not be independently replayed", error=repr(exc))


def _audit_branch(audit: Audit, root: Path, case_folder: Path, branch: Mapping[str, Any],
                  prefix_checkpoint: Mapping[str, Any], prefix_simulator: Mapping[str, Any],
                  prefix_actor: bytes, evidence: Mapping[str, np.ndarray], observer: Mapping[str, Any],
                  fit_result: Mapping[str, Any] | None, prefix_steps: int, prefix_schedule: Sequence[Mapping[str, Any]],
                  seed: int, pair_kind: str, state: str, history_regime: str, foundation_bytes: int | None,
                  config: Mapping[str, Any], where: str) -> dict[str, Any]:
    branch_id = str(branch.get("branch_id", ""))
    branch_root = case_folder / "branches"
    branch_folder = _relative_file(branch_root, branch_id)
    if branch_folder is None or not branch_folder.is_dir():
        audit.check(False, "branch_folder_missing", where,
                    "branch directory is missing or escapes the case branch folder", branch_id=branch_id)
        return {"branch_id": branch_id, "environment_steps": 0, "reward_total": 0.0, "ppo_calls": 0}
    summary_path = branch_folder / "summary.json"
    try:
        disk_branch = _json(summary_path)
        audit.check(disk_branch == dict(branch), "branch_summary_duplicate", where,
                    "case-level branch summary differs from its on-disk branch summary")
        branch = disk_branch
    except Exception as exc:
        audit.check(False, "branch_summary_read", str(summary_path),
                    "on-disk branch summary is missing or unreadable", error=repr(exc))
        return {"branch_id": branch_id, "environment_steps": 0, "reward_total": 0.0, "ppo_calls": 0}

    identity = ((branch.get("seed"), seed), (branch.get("pair_kind"), pair_kind),
                (branch.get("state"), state), (branch.get("history_regime"), history_regime),
                (branch.get("case_id"), case_folder.name))
    for actual, expected in identity:
        audit.check(actual == expected, "branch_identity", where,
                    "branch identity differs from its containing common-prefix case",
                    actual=actual, expected=expected)
    future = str(branch.get("future_regime", ""))
    replicate = int(branch.get("replicate", -1))
    expected_id = f"{future}__r{replicate}__{branch.get('arm', '')}"
    audit.check(branch_id == expected_id and future in config["future_regimes"]
                and replicate in [int(value) for value in config["replicates"]],
                "branch_id_contract", where,
                "branch identifier does not match the frozen future/replicate/arm tuple",
                branch_id=branch_id, expected=expected_id)
    errors = branch.get("errors", [])
    audit.check(not errors, "runner_branch_error", where,
                "runner recorded a branch error", errors=errors)

    records = branch.get("raw_ledgers", {})
    raw = _audit_ledgers(audit, branch_folder, records, where)
    step_rows = _csv(branch_folder / "steps.csv") if (branch_folder / "steps.csv").is_file() else []
    episode_rows = _csv(branch_folder / "episodes.csv") if (branch_folder / "episodes.csv").is_file() else []
    rollout_rows = _csv(branch_folder / "rollouts.csv") if (branch_folder / "rollouts.csv").is_file() else []
    audit.check(branch.get("raw_environment_steps") is not None
                and int(branch.get("raw_environment_steps", -1)) == raw["environment_steps"],
                "branch_raw_step_count", where,
                "branch step count differs from physical steps.csv rows",
                recorded=branch.get("raw_environment_steps"), actual=raw["environment_steps"])
    audit.check(branch.get("prefix_state_hash") == semantic_hash({"learner": prefix_checkpoint,
                                                                    "simulator": prefix_simulator}),
                "branch_prefix_hash", where,
                "branch prefix hash differs from the saved common-prefix payload")
    _audit_rng_restore(audit, branch, prefix_checkpoint, prefix_simulator, config, where)

    try:
        future_schedule = _expected_future_schedule(seed, pair_kind, future,
                                                   len(prefix_schedule), config)
        schedule_reward = _check_raw_schedule(
            audit, step_rows, future_schedule, phase="future", action=str(branch.get("arm", "")),
            future=future, global_start=prefix_steps, where=where,
        )
        audit.check(math.isclose(schedule_reward, raw["reward_total"], rel_tol=1e-10, abs_tol=1e-8),
                    "branch_raw_reward_sum", where,
                    "independent scheduled reward total differs from the raw ledger total")
    except Exception as exc:
        audit.check(False, "branch_schedule_audit", where,
                    "future schedule could not be reconstructed from the frozen design", error=repr(exc))
    _audit_episode_continuity(audit, step_rows, where, start_simulator=prefix_simulator)
    _audit_episode_ledger(audit, step_rows, episode_rows, where)
    _audit_per_bout(audit, step_rows, branch.get("per_bout", []), where)

    post_folder = branch_folder / "post_action"
    try:
        post_checkpoint = _checkpoint(post_folder / "checkpoint.pt")
        post_simulator = _json(post_folder / "simulator.json")
    except Exception as exc:
        audit.check(False, "post_action_package_read", str(post_folder),
                    "post-action learner or simulator package is missing or unreadable", error=repr(exc))
        return {"branch_id": branch_id, **raw}
    final_checkpoint: dict[str, Any] | None = None
    final_simulator: dict[str, Any] | None = None
    try:
        final_checkpoint = _checkpoint(branch_folder / "checkpoint.pt")
        final_simulator = _json(branch_folder / "simulator.json")
    except Exception as exc:
        audit.check(False, "final_package_read", str(branch_folder),
                    "final learner or simulator package is missing or unreadable", error=repr(exc))
    action_hash = semantic_hash({"learner": post_checkpoint, "simulator": post_simulator})
    audit.check(branch.get("post_action_state_hash") == action_hash,
                "post_action_state_hash", where,
                "post-action state hash differs from the saved post-action learner/simulator",
                recorded=branch.get("post_action_state_hash"), actual=action_hash)
    audit.check(semantic_hash(post_simulator) == semantic_hash(prefix_simulator),
                "post_action_simulator_unchanged", where,
                "action installation changed the common-prefix environment or live episode")
    _audit_action_install(audit, branch, prefix_checkpoint, post_checkpoint, prefix_actor, post_folder,
                          observer, evidence, fit_result, config, where)

    _audit_artifact_records(audit, root, branch.get("post_action_artifacts", {}), where + ":post_action")
    _audit_artifact_records(audit, root, branch.get("final_artifacts", {}), where + ":final")
    _audit_evidence_copy(audit, case_folder / "prefix" / "evidence.npz", branch_folder / "evidence.npz",
                         observer.get("evidence_buffer", {}), where + ":branch_evidence")
    _audit_evidence_copy(audit, case_folder / "prefix" / "evidence.npz", post_folder / "evidence.npz",
                         observer.get("evidence_buffer", {}), where + ":post_action_evidence")
    _audit_package(audit, post_folder, branch.get("capacity_after_action", {}) or {}, foundation_bytes,
                   config, where + ":post_action")
    _audit_package(audit, branch_folder, branch.get("capacity", {}) or {}, foundation_bytes,
                   config, where + ":final")
    _audit_actor_payload(audit, post_folder, post_checkpoint, config, where + ":post_action_actor")

    ppo = branch.get("ppo_call_counts", {})
    audit.check(int(ppo.get("rollouts_attempted", -1)) == raw["rollouts"]
                and int(ppo.get("rollouts_completed", -1)) == raw["rollouts"]
                and int(ppo.get("rollouts_failed", -1)) == 0
                and int(ppo.get("minibatch_updates", -1)) == raw["ppo_calls"],
                "branch_ppo_calls", where,
                "PPO calls or minibatch updates differ from raw rollouts.csv", saved=ppo,
                raw_rollouts=raw["rollouts"], raw_updates=raw["ppo_calls"])
    counts_delta = branch.get("counts_delta")
    audit.check(isinstance(counts_delta, Mapping)
                and int(counts_delta.get("raw_steps", -1)) == raw["environment_steps"]
                and int(counts_delta.get("environment_steps", -1)) == raw["environment_steps"]
                and int(counts_delta.get("rollouts_trained", -1)) == raw["rollouts"]
                and int(counts_delta.get("minibatch_updates", -1)) == raw["ppo_calls"]
                and int(counts_delta.get("rollouts_csv_completed", -1)) == raw["rollouts"]
                and int(counts_delta.get("minibatch_updates_from_rollout_ledger", -1)) == raw["ppo_calls"],
                "branch_counts_delta", where,
                "learner counter deltas differ from physical step and rollout ledgers",
                counts_delta=counts_delta, raw=raw)
    horizon_rows = list(branch.get("horizons", []))
    for horizon in config.get("horizons", []):
        saved = next((row for row in horizon_rows if int(row.get("steps", -1)) == int(horizon)), None)
        selected_steps = step_rows[:int(horizon)]
        reward = float(sum(float(row["reward"]) for row in selected_steps))
        rollout_count = 0
        update_count = 0
        covered = 0
        for rollout in rollout_rows:
            if covered >= int(horizon):
                break
            covered += int(rollout["rows"])
            rollout_count += 1
            update_count += int(rollout["updates"])
        audit.check(saved is not None and len(selected_steps) == int(horizon)
                    and math.isclose(float(saved.get("reward_sum", float("nan"))), reward,
                                     rel_tol=1e-9, abs_tol=1e-7)
                    and math.isclose(float(saved.get("reward_per_step", float("nan"))), reward / int(horizon),
                                     rel_tol=1e-9, abs_tol=1e-9)
                    and int(saved.get("raw_environment_steps", -1)) == int(horizon)
                    and int(saved.get("rollouts_trained", -1)) == rollout_count
                    and int(saved.get("minibatch_updates", -1)) == update_count,
                    "branch_horizon_raw_metrics", where,
                    "saved horizon reward/step/PPO metrics differ from the raw ledgers",
                    horizon=horizon, raw_reward=reward, rollouts=rollout_count, updates=update_count)

    if final_checkpoint is not None and final_simulator is not None:
        final_hash = semantic_hash({"learner": final_checkpoint, "simulator": final_simulator})
        audit.check(branch.get("final_state_hash") == final_hash,
                    "final_state_hash", where,
                    "final state hash differs from the saved final learner/simulator",
                    recorded=branch.get("final_state_hash"), actual=final_hash)
        post_local, post_shared, _ = _core_arrays(post_checkpoint)
        final_local, final_shared, _ = _core_arrays(final_checkpoint)
        parent_drift = float(np.max(np.abs(final_shared - post_shared), initial=0.0))
        saved_parent_invariant = branch.get("shared_parent_invariant", {})
        audit.check(np.array_equal(final_shared, post_shared)
                    and bool(saved_parent_invariant.get("exactly_fixed"))
                    and math.isclose(float(saved_parent_invariant.get("max_coefficient_drift", -1.0)),
                                     parent_drift, rel_tol=1e-6, abs_tol=1e-8)
                    and int(final_checkpoint.get("parent_group", -999)) == int(post_checkpoint.get("parent_group", -998))
                    and bool(final_checkpoint.get("shared_frozen")),
                    "branch_parent_fixed", where,
                    "shared parent coefficients or binding changed after action during online continuation",
                    actual_max_drift=parent_drift, saved=saved_parent_invariant)
        _audit_actor_payload(audit, branch_folder, final_checkpoint, config, where + ":final_actor")
        _audit_episode_state(audit, step_rows, prefix_simulator, final_simulator,
                             int(branch.get("future_seed", -1)), where)
        if raw["environment_steps"]:
            audit.check(int(final_checkpoint["seed"]) == int(branch.get("future_seed", -2)),
                        "branch_final_seed", where,
                        "final learner seed differs from the branch future seed")
    return {"branch_id": branch_id, **raw}


def _audit_per_bout(audit: Audit, rows: Sequence[Mapping[str, str]],
                    records: Sequence[Mapping[str, Any]], where: str) -> None:
    raw_by_bout: dict[int, list[Mapping[str, str]]] = {}
    for row in rows:
        try:
            raw_by_bout.setdefault(int(row["bout"]), []).append(row)
        except Exception:
            continue
    audit.check(len(records) == len(raw_by_bout), "per_bout_count", where,
                "per-bout summary count differs from unique raw step bout ids",
                summary=len(records), raw=len(raw_by_bout))
    for index, record in enumerate(records):
        try:
            bout = int(record["bout"])
            bout_rows = raw_by_bout.get(bout, [])
            reward = float(sum(float(row["reward"]) for row in bout_rows))
            first_256 = float(sum(float(row["reward"]) for row in bout_rows[:256]))
            first_512 = float(sum(float(row["reward"]) for row in bout_rows[:512]))
            audit.check(len(bout_rows) == int(record["steps"])
                        and len(bout_rows) > 0
                        and str(bout_rows[0]["taskobserver"]) == str(record["task"])
                        and int(bout_rows[0]["group"]) == int(record["group"])
                        and int(bout_rows[0]["slot"]) == int(record["slot"])
                        and int(bout_rows[0]["generation"]) == int(record["generation"])
                        and all(int(row["slot"]) == int(record["slot"])
                                and int(row["generation"]) == int(record["generation"])
                                for row in bout_rows)
                        and math.isclose(float(record["return_full"]), reward, rel_tol=1e-9, abs_tol=1e-7)
                        and math.isclose(float(record["return_first_256"]), first_256,
                                         rel_tol=1e-9, abs_tol=1e-7)
                        and math.isclose(float(record["return_first_512"]), first_512,
                                         rel_tol=1e-9, abs_tol=1e-7),
                        "per_bout_raw_metrics", where,
                        "per-bout identity, steps, generation or reward differs from raw transitions",
                        bout=bout, raw_steps=len(bout_rows), summary_steps=record.get("steps"),
                        raw_reward=reward, summary_reward=record.get("return_full"), index=index)
        except Exception as exc:
            audit.check(False, "per_bout_record_parse", where,
                        "per-bout summary record is malformed", index=index, error=repr(exc))


def _audit_case(audit: Audit, root: Path, case: Mapping[str, Any], config: Mapping[str, Any],
                manifest: Mapping[str, Any], foundation_bytes: int | None) -> dict[str, Any]:
    case_id = str(case.get("case_id", ""))
    case_folder = _relative_file(root / "cases", case_id)
    result: dict[str, Any] = {"case_id": case_id, "prefix_steps": 0, "prefix_ppo_calls": 0,
                              "branch_steps": 0, "branch_ppo_calls": 0, "branches": 0}
    where = f"case:{case_id}"
    if case_folder is None or not case_folder.is_dir():
        audit.check(False, "case_folder_missing", where,
                    "common-prefix case folder is missing or escapes the cases directory")
        return result
    try:
        disk_case = _json(case_folder / "summary.json")
        audit.check(disk_case == dict(case), "case_summary_duplicate", where,
                    "cohort-level case summary differs from the saved case summary")
        case = disk_case
    except Exception as exc:
        audit.check(False, "case_summary_read", where,
                    "on-disk case summary is missing or unreadable", error=repr(exc))
        return result

    seed = int(case.get("seed", -1))
    pair_kind = str(case.get("pair_kind", ""))
    state = str(case.get("state", ""))
    history_regime = str(case.get("history_regime", ""))
    expected_case_id = f"s{seed}_{pair_kind}_{state}"
    audit.check(case_id == expected_case_id
                and seed in [int(value) for value in config.get("seeds", [])]
                and pair_kind in config.get("pair_kinds", [])
                and state in config.get("states", []),
                "case_identity", where,
                "case id or seed/pair/state tuple is outside the frozen design",
                expected_case_id=expected_case_id)
    if config.get("history_regime_by_seed") == "old_major_if_even_else_new_major":
        expected_history = "old_major" if seed % 2 == 0 else "new_major"
        audit.check(history_regime == expected_history,
                    "case_history_regime", where,
                    "history regime differs from the frozen seed-parity assignment",
                    actual=history_regime, expected=expected_history)
    audit.check(history_regime in config.get("history_regimes", []),
                "case_history_regime_domain", where,
                "history regime is outside the declared design", actual=history_regime)

    prefix_folder = case_folder / "prefix"
    try:
        prefix = _json(prefix_folder / "summary.json")
        audit.check(prefix == case.get("prefix"), "prefix_summary_duplicate", where,
                    "case summary prefix section differs from prefix/summary.json")
        observer_path = _relative_file(root, str(case.get("observer_prefix_path", "")))
        expected_observer_path = prefix_folder / "observer_prefix.json"
        audit.check(observer_path == expected_observer_path.resolve(),
                    "observer_prefix_path", where,
                    "observer record path does not identify this case's prefix observer file",
                    recorded=case.get("observer_prefix_path"))
        observer = _json(expected_observer_path)
        observer_artifact = prefix.get("observer_artifact", {})
        audit.check(observer.get("format") == "retention_candidate_observer_prefix_v1"
                    and observer.get("accessible_to_deployed_selector") is False
                    and observer.get("included_in_deployment_capacity") is False,
                    "observer_prefix_role", where,
                    "observer prefix must remain an audit-only, non-deployed record")
        audit.check(observer_artifact.get("path") == case.get("observer_prefix_path")
                    and int(observer_artifact.get("bytes", -1)) == expected_observer_path.stat().st_size
                    and observer_artifact.get("sha256") == _sha_file(expected_observer_path),
                    "observer_prefix_artifact", where,
                    "observer artifact digest differs from its saved record")
        audit.check(case.get("candidate_fit") == prefix.get("candidate_fit")
                    and case.get("candidate_fit") == observer.get("candidate_fit"),
                    "candidate_fit_duplicate", where,
                    "fit provenance differs between case, prefix and observer records")
    except Exception as exc:
        audit.check(False, "prefix_observer_read", where,
                    "prefix summaries or observer record are missing or unreadable", error=repr(exc))
        return result

    try:
        checkpoint = _checkpoint(prefix_folder / "checkpoint.pt")
        simulator = _json(prefix_folder / "simulator.json")
        prefix_actor = (prefix_folder / "actor.bin").read_bytes()
    except Exception as exc:
        audit.check(False, "prefix_package_read", where,
                    "prefix learner, actor or simulator package is missing or unreadable", error=repr(exc))
        return result
    state_hash = semantic_hash({"learner": checkpoint, "simulator": simulator})
    learner_hash = semantic_hash(checkpoint)
    audit.check(prefix.get("prefix_state_hash") == state_hash
                and prefix.get("prefix_learner_hash") == learner_hash,
                "prefix_state_hash", where,
                "saved prefix state or learner hash differs from the physical checkpoint/simulator")
    _audit_artifact_records(audit, root, prefix.get("artifacts", {}), where + ":prefix")
    _audit_package(audit, prefix_folder, prefix.get("capacity", {}) or {}, foundation_bytes,
                   config, where + ":prefix")
    _audit_actor_payload(audit, prefix_folder, checkpoint, config, where + ":prefix_actor")
    audit.check(bool(case.get("foundation_unchanged")), "foundation_unchanged", where,
                "runner did not confirm the frozen foundation stayed unchanged")
    if isinstance(manifest.get("foundation"), Mapping):
        audit.check(prefix.get("foundation_sha256") == manifest["foundation"].get("sha256"),
                    "prefix_foundation_hash", where,
                    "prefix learner foundation digest differs from the source manifest")

    ledgers = _audit_ledgers(audit, prefix_folder, prefix.get("raw_ledgers", {}), where + ":prefix")
    step_rows = _csv(prefix_folder / "steps.csv") if (prefix_folder / "steps.csv").is_file() else []
    episode_rows = _csv(prefix_folder / "episodes.csv") if (prefix_folder / "episodes.csv").is_file() else []
    rollout_rows = _csv(prefix_folder / "rollouts.csv") if (prefix_folder / "rollouts.csv").is_file() else []
    prefix_steps = ledgers["environment_steps"]
    result["prefix_steps"] = prefix_steps
    result["prefix_ppo_calls"] = ledgers["ppo_calls"]
    audit.check(int(prefix.get("environment_steps", prefix.get("steps", -1))) == prefix_steps
                and int(prefix.get("steps", -1)) == prefix_steps
                and math.isclose(float(prefix.get("reward_total", float("nan"))), ledgers["reward_total"],
                                 rel_tol=1e-8, abs_tol=1e-6)
                and math.isclose(float(prefix.get("reward_per_step", float("nan"))),
                                 ledgers["reward_total"] / max(prefix_steps, 1), rel_tol=1e-8, abs_tol=1e-8)
                and int(prefix.get("rollouts_trained", -1)) == ledgers["rollouts"]
                and int(prefix.get("minibatch_updates", -1)) == ledgers["ppo_calls"],
                "prefix_raw_counts", where,
                "prefix steps, aggregate reward, rollout or minibatch counts differ from physical ledgers",
                summary_steps=prefix.get("environment_steps"), raw_steps=prefix_steps,
                summary_rollouts=prefix.get("rollouts_trained"), raw_rollouts=ledgers["rollouts"],
                summary_updates=prefix.get("minibatch_updates"), raw_updates=ledgers["ppo_calls"])
    ppo = prefix.get("ppo_call_counts", {})
    audit.check(int(ppo.get("rollouts_attempted", -1)) == ledgers["rollouts"]
                and int(ppo.get("rollouts_completed", -1)) == ledgers["rollouts"]
                and int(ppo.get("rollouts_failed", -1)) == 0
                and int(ppo.get("minibatch_updates", -1)) == ledgers["ppo_calls"],
                "prefix_ppo_calls", where,
                "prefix PPO counter record differs from raw rollouts.csv", saved=ppo, raw=ledgers)
    _audit_per_bout(audit, step_rows, prefix.get("per_bout", []), where + ":prefix")
    try:
        prefix_schedule = _expected_prefix_schedule(seed, pair_kind, state, history_regime, config)
        schedule_reward = _check_raw_schedule(audit, step_rows, prefix_schedule,
                                              phase="prefix", global_start=0, where=where + ":prefix")
        audit.check(math.isclose(schedule_reward, ledgers["reward_total"], rel_tol=1e-10, abs_tol=1e-7),
                    "prefix_raw_reward_sum", where,
                    "independent scheduled prefix reward differs from raw step ledger")
    except Exception as exc:
        prefix_schedule = []
        audit.check(False, "prefix_schedule_audit", where,
                    "prefix schedule could not be reconstructed from the frozen design", error=repr(exc))
    _audit_episode_ledger(audit, step_rows, episode_rows, where + ":prefix")
    _audit_episode_continuity(audit, step_rows, where + ":prefix")
    _audit_episode_state(audit, step_rows, None, simulator, seed, where + ":prefix")

    evidence_record = observer.get("evidence_buffer", {})
    evidence = _audit_npz(audit, prefix_folder / "evidence.npz", evidence_record)
    fit_result: dict[str, Any] | None = None
    if evidence is not None:
        _audit_evidence_copy(audit, prefix_folder / "evidence.npz", prefix_folder / "evidence.npz",
                             evidence_record, where + ":prefix_evidence")
        _audit_evidence_selection(audit, evidence, observer.get("candidate_sources", []),
                                  step_rows, rollout_rows, where + ":prefix_evidence",
                                  evidence_record.get("latest_source_global_steps"))
        _audit_prefix_source_state(audit, checkpoint, prefix, observer, config, state, evidence,
                                   where + ":prefix_sources")
        _audit_warmup_writes(audit, prefix, observer, state, config, where + ":warmup")
        fit_result = _audit_fit_math(audit, observer, evidence, None, checkpoint, config,
                                     manifest, where + ":candidate_fit")
        _audit_fit_provenance(audit, case.get("candidate_fit", {}), checkpoint, simulator, evidence,
                              evidence_record, fit_result, seed, where + ":candidate_fit")
    else:
        audit.check(False, "candidate_fit_unchecked", where,
                    "bounded evidence sidecar is unavailable, so candidate fit integrity cannot be checked")

    audit.check(not case.get("errors", []), "runner_case_error", where,
                "runner recorded a common-prefix case error", errors=case.get("errors", []))
    seen_branches: set[tuple[str, int, str]] = set()
    branches = list(case.get("branches", []))
    expected_branch_keys = {
        (str(future), int(replicate), str(arm))
        for future in config.get("future_regimes", [])
        for replicate in config.get("replicates", [])
        for arm in config.get("arms", [])
    }
    for branch in branches:
        branch_key = (str(branch.get("future_regime", "")), int(branch.get("replicate", -1)),
                      str(branch.get("arm", "")))
        audit.check(branch_key not in seen_branches, "duplicate_case_branch", where,
                    "case contains a duplicate future/replicate/arm branch tuple", branch=branch_key)
        seen_branches.add(branch_key)
        branch_result = _audit_branch(
            audit, root, case_folder, branch, checkpoint, simulator, prefix_actor, evidence or {},
            observer, fit_result, prefix_steps, prefix_schedule, seed, pair_kind, state,
            history_regime, foundation_bytes, config, f"{where}:branch:{branch.get('branch_id', '')}",
        )
        result["branch_steps"] += int(branch_result.get("environment_steps", 0))
        result["branch_ppo_calls"] += int(branch_result.get("ppo_calls", 0))
        result["branches"] += 1
    audit.check(seen_branches == expected_branch_keys,
                "case_branch_coverage", where,
                "case does not contain exactly the predeclared branch combinations",
                missing=sorted(expected_branch_keys - seen_branches), extra=sorted(seen_branches - expected_branch_keys))
    return result


def audit_cohort(root: Path, report_path: Path) -> dict[str, Any]:
    """Audit one frozen candidate cohort and write a machine-readable report."""
    root = root.resolve()
    report_path = report_path.resolve()
    audit = Audit(root)
    try:
        summary = _json(root / "summary.json")
        audit.check(summary.get("format") == SUMMARY_FORMAT, "summary_format", str(root / "summary.json"),
                    "cohort summary format is not retention_candidate_summary_v1",
                    actual=summary.get("format"))
    except Exception as exc:
        summary = {}
        audit.check(False, "summary_read", str(root / "summary.json"),
                    "cohort summary is missing or unreadable", error=repr(exc))
    config = summary.get("config", {}) if isinstance(summary, Mapping) else {}
    if not isinstance(config, Mapping):
        config = {}
        audit.check(False, "summary_config_shape", str(root / "summary.json"),
                    "cohort config must be a mapping")
    try:
        saved_config = _json(root / "config.json")
        audit.check(saved_config == dict(config), "config_duplicate", str(root / "config.json"),
                    "root config differs from the cohort summary frozen config")
    except Exception as exc:
        audit.check(False, "config_read", str(root / "config.json"),
                    "root config is missing or unreadable", error=repr(exc))
    try:
        frozen_run = _json(root / "frozen_run.json")
        audit.check(frozen_run.get("phase") == summary.get("phase")
                    and frozen_run.get("config") == dict(config),
                    "frozen_run_duplicate", str(root / "frozen_run.json"),
                    "frozen run record differs from phase or frozen config")
    except Exception as exc:
        audit.check(False, "frozen_run_read", str(root / "frozen_run.json"),
                    "frozen run record is missing or unreadable", error=repr(exc))
    foundation_bytes = _audit_source_snapshot(audit, summary)
    phase = str(summary.get("phase", ""))
    cases = list(summary.get("cases", [])) if isinstance(summary, Mapping) else []
    top_branches = list(summary.get("branches", [])) if isinstance(summary, Mapping) else []
    seeds = [int(value) for value in config.get("seeds", [])]
    pair_kinds = [str(value) for value in config.get("pair_kinds", [])]
    states = [str(value) for value in config.get("states", [])]
    expected_case_ids = {f"s{seed}_{pair_kind}_{state}"
                         for seed in seeds for pair_kind in pair_kinds for state in states}
    observed_case_ids = [str(row.get("case_id", "")) for row in cases]
    audit.check(len(observed_case_ids) == len(set(observed_case_ids))
                and set(observed_case_ids) == expected_case_ids
                and int(summary.get("case_count", -1)) == len(cases),
                "cohort_case_coverage", str(root / "summary.json"),
                "cohort cases differ from the frozen seed/pair/state Cartesian product",
                expected=sorted(expected_case_ids), actual=sorted(observed_case_ids),
                recorded_count=summary.get("case_count"))
    expected_branch_count = len(expected_case_ids) * len(config.get("future_regimes", [])) \
        * len(config.get("replicates", [])) * len(config.get("arms", []))
    audit.check(int(summary.get("branch_count", -1)) == len(top_branches) == expected_branch_count,
                "cohort_branch_count", str(root / "summary.json"),
                "cohort branch count differs from the frozen design",
                recorded=summary.get("branch_count"), rows=len(top_branches), expected=expected_branch_count)
    audit.check(not summary.get("errors", []), "runner_cohort_error", str(root / "summary.json"),
                "runner recorded a cohort-level error", errors=summary.get("errors", []))

    case_results: list[dict[str, Any]] = []
    nested_branches: dict[tuple[str, str], Mapping[str, Any]] = {}
    for case in cases:
        case_result = _audit_case(audit, root, case, config, summary.get("source_manifest", {}),
                                  foundation_bytes)
        case_results.append(case_result)
        for branch in case.get("branches", []):
            nested_branches[(str(case.get("case_id", "")), str(branch.get("branch_id", "")))] = branch
    top_branch_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    for branch in top_branches:
        key = (str(branch.get("case_id", "")), str(branch.get("branch_id", "")))
        if key in top_branch_map:
            audit.check(False, "duplicate_flat_branch", str(root / "summary.json"),
                        "flat cohort summary duplicates a branch identity", branch=key)
        top_branch_map[key] = branch
    audit.check(set(nested_branches) == set(top_branch_map)
                and all(nested_branches[key] == top_branch_map[key] for key in set(nested_branches) & set(top_branch_map)),
                "flat_branch_summary", str(root / "summary.json"),
                "flat branch index differs from case-contained branch summaries",
                missing=sorted(set(nested_branches) - set(top_branch_map)),
                extra=sorted(set(top_branch_map) - set(nested_branches)))

    raw_prefix_steps = sum(int(row["prefix_steps"]) for row in case_results)
    raw_branch_steps = sum(int(row["branch_steps"]) for row in case_results)
    raw_prefix_updates = sum(int(row["prefix_ppo_calls"]) for row in case_results)
    raw_branch_updates = sum(int(row["branch_ppo_calls"]) for row in case_results)
    actual_total_steps = raw_prefix_steps + raw_branch_steps
    costs = summary.get("costs", {}) if isinstance(summary, Mapping) else {}
    expected_per_branch = int(config.get("future_blocks", 0)) * 3 * int(config.get("bout_steps", 0))
    expected_branch_steps = expected_branch_count * expected_per_branch
    expected_prefix_steps = 0
    for seed in seeds:
        for pair_kind in pair_kinds:
            for state in states:
                history = "old_major" if seed % 2 == 0 else "new_major"
                expected_prefix_steps += sum(int(row["steps"]) for row in
                                              _expected_prefix_schedule(seed, pair_kind, state, history, config))
    expected_total_steps = expected_prefix_steps + expected_branch_steps
    audit.check(int(summary.get("branch_count", 0)) == sum(int(row["branches"]) for row in case_results)
                and raw_prefix_steps == expected_prefix_steps
                and raw_branch_steps == expected_branch_steps
                and actual_total_steps == expected_total_steps,
                "cohort_raw_step_totals", str(root / "summary.json"),
                "aggregate physical step counts differ from the frozen schedule",
                raw_prefix_steps=raw_prefix_steps, expected_prefix_steps=expected_prefix_steps,
                raw_branch_steps=raw_branch_steps, expected_branch_steps=expected_branch_steps,
                raw_total_steps=actual_total_steps, expected_total_steps=expected_total_steps)
    for key, actual in (("prefix_environment_steps", raw_prefix_steps),
                        ("branch_environment_steps", raw_branch_steps),
                        ("environment_steps_total", actual_total_steps),
                        ("expected_environment_steps_total", expected_total_steps)):
        if key in costs:
            audit.check(int(costs[key]) == actual, "cohort_cost_step_total", str(root / "summary.json"),
                        "cohort cost ledger step count differs from raw per-case ledgers",
                        field=key, recorded=costs[key], raw=actual)
    if "expected_interactions" in costs:
        expected_record = costs["expected_interactions"]
        for key, actual in (("cases", len(expected_case_ids)), ("branches", expected_branch_count),
                            ("prefix_steps", expected_prefix_steps), ("branch_steps", expected_branch_steps),
                            ("total_steps", expected_total_steps), ("steps_per_branch", expected_per_branch)):
            audit.check(int(expected_record.get(key, -1)) == actual,
                        "cohort_expected_interactions", str(root / "summary.json"),
                        "cohort expected-interaction record differs from schedule-derived counts",
                        field=key, recorded=expected_record.get(key), actual=actual)
    report: dict[str, Any] = {
        "format": FORMAT,
        "status": "passed" if not audit.errors else "failed",
        "cohort": str(root),
        "phase": phase,
        "provenance": {
            "auditor_source_path": str(Path(__file__).resolve()),
            "auditor_source_sha256": _sha_file(Path(__file__).resolve()),
            "auditor_source_bytes": Path(__file__).resolve().stat().st_size,
            "cohort_summary_sha256": _sha_file(root / "summary.json")
            if (root / "summary.json").is_file() else None,
        },
        "source_manifest_sha256": _sha_file(root / "source_snapshot" / "manifest.json")
        if (root / "source_snapshot" / "manifest.json").is_file() else None,
        "coverage": {"cases": len(cases), "expected_cases": len(expected_case_ids),
                     "branches": len(top_branches), "expected_branches": expected_branch_count},
        "raw_counts": {"prefix_environment_steps": raw_prefix_steps,
                       "branch_environment_steps": raw_branch_steps,
                       "environment_steps_total": actual_total_steps,
                       "prefix_minibatch_updates": raw_prefix_updates,
                       "branch_minibatch_updates": raw_branch_updates},
        "integrity_errors": audit.errors,
        "diagnostic_warnings": audit.warnings,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
                           encoding="utf-8")
    return report


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="frozen retention-candidate cohort folder")
    parser.add_argument("--report", type=Path, help="audit JSON output path")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    report_path = args.report or root.parent / f"{root.name}_audit.json"
    result = audit_cohort(root, report_path)
    print(json.dumps({"status": result["status"], "report": str(report_path.resolve()),
                      "errors": len(result["integrity_errors"]),
                      "warnings": len(result["diagnostic_warnings"]),
                      "raw_counts": result["raw_counts"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


def _audit_fit_provenance(audit: Audit, fit: Mapping[str, Any], checkpoint: Mapping[str, Any],
                          simulator: Mapping[str, Any], evidence: Mapping[str, np.ndarray],
                          evidence_record: Mapping[str, Any], fit_result: Mapping[str, Any] | None,
                          seed: int, where: str) -> None:
    learner_hash = semantic_hash(checkpoint)
    audit.check(fit.get("fit_calls") == 1 and fit.get("fit_state_unchanged") is True
                and fit.get("learner_hash_before") == learner_hash
                and fit.get("learner_hash_after") == learner_hash
                and fit.get("rng_hashes_before") == fit.get("rng_hashes_after"),
                "candidate_fit_state_isolation", where,
                "candidate fitting must be a single pass that leaves learner and every RNG unchanged",
                fit_calls=fit.get("fit_calls"), state_unchanged=fit.get("fit_state_unchanged"),
                learner_before=fit.get("learner_hash_before"), learner_after=fit.get("learner_hash_after"),
                prefix_learner=learner_hash)
    before = fit.get("rng_hashes_before", {})
    env = simulator.get("environment", {})
    driver = simulator.get("driver", {})
    expected_rng = {
        "agent_seed": int(seed),
        "action_rng_sha256": semantic_hash(checkpoint.get("action_rng")),
        "behavior_rng_sha256": semantic_hash(checkpoint.get("behavior_rng")),
        "numpy_global_rng_sha256": semantic_hash(checkpoint.get("numpy_rng")),
        "python_global_rng_sha256": semantic_hash(checkpoint.get("python_rng")),
        "torch_global_rng_sha256": semantic_hash(checkpoint.get("torch_rng")),
        "environment_rng_sha256": semantic_hash({"bit_generator": env.get("np_random_bit_generator"),
                                                   "state": env.get("np_random_state")}),
        "episode_reset_seed": int(driver.get("episode_reset_seed", -1)),
        "episode_ordinal": int(driver.get("episode_ordinal", -1)),
        "episode_length": int(driver.get("episode_length", -1)),
    }
    for key, expected in expected_rng.items():
        audit.check(before.get(key) == expected, "candidate_fit_rng_provenance", where,
                    "candidate fit pre-state RNG/episode record differs from the saved prefix payload",
                    stream=key, saved=before.get(key), expected=expected)
    observations = np.asarray(evidence["observations"], dtype=np.float32)
    owners = np.asarray(evidence["owner_indices"], dtype=np.int8)
    audit.check(fit.get("source_buffer_sha256") == evidence_record.get("sha256")
                and int(fit.get("source_buffer_bytes", -1)) == int(evidence_record.get("bytes", -2)),
                "candidate_fit_evidence_reference", where,
                "candidate fit evidence reference differs from the byte-hashed bounded evidence sidecar",
                fit_hash=fit.get("source_buffer_sha256"), evidence_hash=evidence_record.get("sha256"),
                fit_bytes=fit.get("source_buffer_bytes"), evidence_bytes=evidence_record.get("bytes"))
    audit.check(fit.get("observations_sha256") == semantic_hash(observations)
                and fit.get("owner_indices_sha256") == semantic_hash(owners)
                and fit.get("source_rows_by_owner") == [128, 128]
                and fit.get("feature_rows_by_owner") == [128, 128],
                "candidate_fit_observed_inputs", where,
                "fit input counts or hashes do not match the bounded observed prefix evidence",
                observations=fit.get("observations_sha256"), owners=fit.get("owner_indices_sha256"),
                source_rows=fit.get("source_rows_by_owner"))
    latest_steps = [int(np.max(np.asarray(evidence["global_steps"])[owners == owner]))
                    for owner in (0, 1)]
    audit.check(fit.get("latest_source_global_steps") == latest_steps
                and evidence_record.get("latest_source_global_steps") == latest_steps,
                "candidate_fit_latest_source_steps", where,
                "fit source endpoints differ from the independently selected evidence rows",
                fit=fit.get("latest_source_global_steps"), evidence=evidence_record.get("latest_source_global_steps"),
                actual=latest_steps)
    forbidden_keys = {"reward", "rewards", "return", "returns", "future_observations", "future_rewards"}
    present = forbidden_keys.intersection({str(key).casefold() for key in fit})
    audit.check(not present, "candidate_fit_no_reward_inputs", where,
                "candidate fit record must not expose reward or future-observation inputs",
                fields=sorted(present))
    if fit_result is not None:
        features = fit_result.get("features")
        feature_hash = semantic_hash({
            "base_logits": np.asarray(features.base_logits),
            "source_psi": np.asarray(features.source_psi),
            "contrast_basis": np.asarray(features.contrast_basis),
            "temperature": float(features.temperature),
            "epsilon": float(features.epsilon),
        })
        audit.check(fit.get("feature_sha256") == feature_hash,
                    "candidate_fit_feature_hash", where,
                    "saved frozen-feature hash differs from the checkpoint-derived logits and features",
                    saved=fit.get("feature_sha256"), recomputed=feature_hash)


if __name__ == "__main__":
    raise SystemExit(main())

