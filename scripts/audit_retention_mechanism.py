"""Read-only integrity audit for retention-mechanism diagnostic artifacts.

The audit consumes saved files only. It does not create, reset, or step a task
environment, and it does not rank action returns or train a selector.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from online_rl.storage_decision import (
    ACTIONS,
    HISTORY_TEMPLATES,
    bootstrap_schedule,
    common_mean_effective,
    effective_matrices,
    pair_mapping,
    semantic_hash,
    task_name,
    template_schedule,
)
from online_rl.retention_stream_value import apply_retention_action
from online_rl.write_decision_policy import decode_actor, encode_actor
from scripts import audit_storage_decision as storage_audit


FORMAT = "retention_mechanism_audit_v1"
SUMMARY_FORMAT = "retention_mechanism_summary_v1"
CONFIG_FORMAT = "retention_mechanism_v1"
PRIVATE_PARTS = {"agent.md", "agents.md", "tmp"}
REQUIRED_SOURCES = {
    "configs/retention_mechanism.json",
    "docs/RETENTION_MECHANISM_PROTOCOL.md",
    "scripts/run_retention_mechanism.py",
    "scripts/run_storage_decision.py",
    "scripts/run_knowledge_loop.py",
    "scripts/run_write_stream.py",
    "src/online_rl/storage_decision.py",
    "src/online_rl/retention_stream_value.py",
    "src/online_rl/write_decision_policy.py",
    "src/online_rl/write_decision_envs.py",
    "src/online_rl/envs.py",
}


class Audit:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.observations: dict[str, Any] = {}

    def check(self, condition: bool, code: str, where: str, message: str,
              **details: Any) -> None:
        if not condition:
            row: dict[str, Any] = {"code": code, "where": where, "message": message}
            if details:
                row["details"] = details
            self.errors.append(row)

    def fail(self, code: str, where: str, message: str, **details: Any) -> None:
        self.check(False, code, where, message, **details)

    def warn(self, code: str, where: str, message: str, **details: Any) -> None:
        row: dict[str, Any] = {"code": code, "where": where, "message": message}
        if details:
            row["details"] = details
        self.warnings.append(row)


def _check(audit: Audit, condition: bool, code: str, where: str,
           message: str, **details: Any) -> None:
    audit.check(condition, code, where, message, **details)


def expected_plan(
    phase: str,
    *,
    seeds: Sequence[int],
    pair_kinds: Sequence[str],
    histories: Sequence[str],
    futures: Sequence[str],
    replicates: Sequence[int],
    actions: Sequence[str] = ACTIONS,
    bootstrap_steps: int = 1024,
    bout_steps: int = 512,
    candidate_source_steps: int = 1024,
    future_blocks: int = 8,
    rollout_steps: int = 256,
) -> dict[str, int]:
    """Return the exact prefix, branch, and rollout budgets for a phase."""
    if phase not in {"pilot", "main"}:
        raise ValueError(f"unknown phase: {phase!r}")
    if len(ACTIONS) != len(tuple(actions)) or tuple(actions) != ACTIONS:
        raise ValueError(f"actions must be ordered as {ACTIONS}")
    cases = len(seeds) * len(pair_kinds) * len(histories)
    branches = cases * len(futures) * len(replicates) * len(actions)
    prefix_per_case = (
        2 * int(bootstrap_steps)
        + 6 * int(bout_steps)
        + 2 * int(candidate_source_steps)
    )
    branch_per_branch = int(future_blocks) * 3 * int(bout_steps)
    prefix_steps = cases * prefix_per_case
    branch_steps = branches * branch_per_branch
    total_steps = prefix_steps + branch_steps
    if int(rollout_steps) <= 0 or total_steps % int(rollout_steps):
        raise ValueError("phase step budget must be divisible by rollout_steps")
    return {
        "case_count": cases,
        "branch_count": branches,
        "prefix_steps_per_case": prefix_per_case,
        "branch_steps_per_branch": branch_per_branch,
        "prefix_environment_steps": prefix_steps,
        "branch_environment_steps": branch_steps,
        "environment_steps_total": total_steps,
        "ppo_rollouts_total": total_steps // int(rollout_steps),
    }


def prefix_schedule(mapping: Mapping[str, Any], config: Mapping[str, Any],
                    history: str) -> list[dict[str, Any]]:
    """Build the protocol prefix schedule, including the two 1024-step tails."""
    if history not in HISTORY_TEMPLATES:
        raise ValueError(f"unknown history regime: {history!r}")
    bootstrap_steps = int(config["bootstrap_steps"])
    bout_steps = int(config["bout_steps"])
    source_steps = int(config["candidate_source_steps"])
    rows = bootstrap_schedule(mapping, bootstrap_steps)
    history_rows = template_schedule(
        mapping, history, kind="history", steps=bout_steps, start_bout=len(rows),
    )
    for row in history_rows[-2:]:
        row["steps"] = source_steps
    return rows + history_rows


def future_schedule(mapping: Mapping[str, Any], future: str,
                    config: Mapping[str, Any], *, start_bout: int = 8) -> list[dict[str, Any]]:
    """Build the fixed eight-block future with minority/dominant task order."""
    if future not in {"old_major", "new_major"}:
        raise ValueError(f"unknown future regime: {future!r}")
    dominant = 0 if future == "old_major" else 1
    minority = 1 - dominant
    blocks = int(config["future_blocks"])
    bout_steps = int(config["bout_steps"])
    result: list[dict[str, Any]] = []
    for block in range(blocks):
        variant = block % 3
        entries = (
            (minority, variant),
            (dominant, variant),
            (dominant, (variant + 1) % 3),
        )
        for position, (group, task_variant) in enumerate(entries):
            result.append({
                "phase": "future",
                "bout": int(start_bout) + len(result),
                "future_block": block,
                "within_block": position,
                "group": group,
                "variant": task_variant,
                "task": task_name(mapping, group, task_variant),
                "steps": bout_steps,
            })
    return result


def future_seed(original_seed: int, replicate: int, config: Mapping[str, Any]) -> int:
    """Return the frozen conditional-replica stochastic seed."""
    return (
        int(config["future_seed_offset"])
        + int(config["future_seed_multiplier"]) * int(original_seed)
        + int(replicate)
    )


def horizon_totals(rows: Sequence[Mapping[str, Any]], horizons: Sequence[int]) -> dict[str, dict[str, float | int]]:
    """Calculate nested cumulative reward metrics from one chronological ledger."""
    ordered_horizons = tuple(int(value) for value in horizons)
    if not ordered_horizons or tuple(sorted(set(ordered_horizons))) != ordered_horizons:
        raise ValueError("horizons must be a nonempty strictly increasing sequence")
    if ordered_horizons[0] < 1 or ordered_horizons[-1] > len(rows):
        raise ValueError("horizons must fit within the one raw branch ledger")
    rewards = [float(row["reward"]) for row in rows]
    if not all(math.isfinite(value) for value in rewards):
        raise ValueError("raw rewards must be finite")
    prefix = np.cumsum(np.asarray(rewards, dtype=np.float64))
    return {
        str(horizon): {
            "steps": horizon,
            "reward_total": float(prefix[horizon - 1]),
            "reward_per_step": float(prefix[horizon - 1] / horizon),
        }
        for horizon in ordered_horizons
    }


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(audit: Audit, path: Path, *, required: bool = True) -> list[dict[str, str]]:
    if not path.is_file():
        if required:
            audit.fail("csv_missing", str(path), "required raw or training ledger is missing")
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    except Exception as exc:
        audit.fail("csv_unreadable", str(path), "CSV ledger is malformed or unreadable", error=repr(exc))
        return []


def _jsonl(audit: Audit, path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.is_file():
        if required:
            audit.fail("jsonl_missing", str(path), "required JSONL log is missing")
        return []
    try:
        raw = path.read_bytes()
        audit.check(not raw or raw.endswith(b"\n"), "jsonl_unflushed_tail", str(path),
                    "JSONL ends with a partial line")
        rows = []
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object")
            rows.append(value)
        return rows
    except Exception as exc:
        audit.fail("jsonl_unreadable", str(path), "JSONL is malformed or unreadable", error=repr(exc))
        return []


def _safe_path(root: Path, name: str) -> Path | None:
    normalized = str(name).replace("\\", "/")
    relative = Path(normalized)
    if relative.is_absolute() or not relative.parts:
        return None
    if any(part in {"", ".", ".."} or ":" in part for part in relative.parts):
        return None
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        return None
    return candidate


def _load_checkpoint(audit: Audit, path: Path) -> dict[str, Any] | None:
    try:
        return storage_audit._checkpoint(path)
    except Exception as exc:
        audit.fail("checkpoint_read", str(path), "checkpoint is missing or unreadable", error=repr(exc))
        return None


def _check_source_snapshot(audit: Audit, cohort: Path,
                           summary: Mapping[str, Any]) -> int | None:
    snapshot = cohort / "source_snapshot"
    manifest_path = snapshot / "manifest.json"
    frozen_path = cohort / "frozen_run.json"
    try:
        manifest = _json(manifest_path)
        frozen = _json(frozen_path)
    except Exception as exc:
        audit.fail("source_manifest_read", str(snapshot),
                   "source manifest or frozen run record is missing or unreadable", error=repr(exc))
        return None
    audit.check(manifest == summary.get("source_manifest"), "source_manifest_summary",
                str(manifest_path), "summary manifest differs from the source snapshot manifest")
    audit.check(frozen.get("source_manifest") == manifest, "source_manifest_frozen_run",
                str(frozen_path), "frozen run manifest differs from the source snapshot manifest")
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping):
        audit.fail("source_manifest_sources", str(manifest_path), "source manifest lacks a sources mapping")
        return None
    names = set(map(str, sources))
    audit.check(REQUIRED_SOURCES <= names, "source_manifest_required", str(manifest_path),
                "snapshot omits one or more sources required by this audit",
                missing=sorted(REQUIRED_SOURCES - names))
    for name, item in sources.items():
        relative = Path(str(name).replace("\\", "/"))
        private = any(part.casefold() in PRIVATE_PARTS for part in relative.parts)
        audit.check(not private, "private_source_listed", f"source_snapshot/{name}",
                    "source manifest includes a local instruction or temporary path")
        copied = _safe_path(snapshot, str(name))
        if copied is None or not copied.is_file():
            audit.fail("source_copy_missing", f"source_snapshot/{name}",
                       "manifest-listed source is missing or escapes the snapshot")
            continue
        if not isinstance(item, Mapping):
            audit.fail("source_record_invalid", f"source_snapshot/{name}", "source digest record is malformed")
            continue
        try:
            actual_size = copied.stat().st_size
            actual_sha = _digest(copied)
            audit.check(actual_size == int(item.get("bytes", -1))
                        and actual_sha == item.get("sha256"),
                        "source_copy_hash", f"source_snapshot/{name}",
                        "snapshot copy differs from its recorded size or hash",
                        expected_bytes=item.get("bytes"), actual_bytes=actual_size,
                        expected_sha256=item.get("sha256"), actual_sha256=actual_sha)
        except Exception as exc:
            audit.fail("source_copy_hash_error", f"source_snapshot/{name}",
                       "snapshot source could not be verified", error=repr(exc))
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot)
        audit.check(not any(part.casefold() in PRIVATE_PARTS for part in relative.parts),
                    "private_snapshot_path", f"source_snapshot/{relative.as_posix()}",
                    "snapshot contains a local instruction or temporary path")
    excluded = {str(value).casefold() for value in manifest.get("excluded_local_instructions", [])}
    audit.check(excluded == {"agent.md", "agents.md"}, "private_exclusions", str(manifest_path),
                "manifest must record both local instruction exclusions", excluded=sorted(excluded))

    try:
        foundation = manifest.get("foundation", {})
        foundation_path = Path(str(foundation.get("path", "")))
        if not foundation_path.is_file():
            raise FileNotFoundError(foundation_path)
        foundation_bytes = foundation_path.stat().st_size
        foundation_sha = _digest(foundation_path)
        audit.check(foundation_bytes == int(foundation.get("bytes", -1))
                    and foundation_sha == foundation.get("sha256"),
                    "foundation_hash", str(foundation_path),
                    "foundation differs from the source manifest",
                    expected_bytes=foundation.get("bytes"), actual_bytes=foundation_bytes,
                    expected_sha256=foundation.get("sha256"), actual_sha256=foundation_sha)
    except Exception as exc:
        audit.fail("foundation_unavailable", str(manifest_path),
                   "foundation provenance cannot be verified", error=repr(exc))
        foundation_bytes = None
    dataset = manifest.get("dataset", {})
    try:
        dataset_path = Path(str(dataset.get("path", "")))
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        audit.check(dataset_path.stat().st_size == int(dataset.get("bytes", -1))
                    and _digest(dataset_path) == dataset.get("sha256"),
                    "dataset_hash", str(dataset_path), "dataset differs from source manifest")
    except Exception as exc:
        audit.fail("dataset_unavailable", str(manifest_path),
                   "dataset provenance cannot be verified", error=repr(exc))

    try:
        original_config = _json(snapshot / "config_original.json")
        copied_config = _json(snapshot / "configs" / "retention_mechanism.json")
        audit.check(original_config == copied_config, "original_config_snapshot_match",
                    str(snapshot / "config_original.json"),
                    "saved original config differs from its source snapshot copy")
        normalized = _json(cohort / "config.json")
        audit.check(normalized == summary.get("config") == frozen.get("config"),
                    "normalized_config_match", str(cohort / "config.json"),
                    "cohort, summary, and frozen run configs disagree")
        source_config = sources.get("configs/retention_mechanism.json", {})
        config_copy = snapshot / "configs" / "retention_mechanism.json"
        audit.check(_digest(config_copy) == source_config.get("sha256")
                    and int(source_config.get("bytes", -1)) == config_copy.stat().st_size,
                    "config_snapshot_hash", str(config_copy),
                    "config copy differs from its source manifest")
        source_protocol = sources.get("docs/RETENTION_MECHANISM_PROTOCOL.md", {})
        protocol_copy = snapshot / "docs" / "RETENTION_MECHANISM_PROTOCOL.md"
        protocol_sha = _digest(protocol_copy)
        audit.check(protocol_sha == source_protocol.get("sha256")
                    and int(source_protocol.get("bytes", -1)) == protocol_copy.stat().st_size,
                    "protocol_snapshot_hash", str(protocol_copy),
                    "protocol copy differs from its source manifest")
        audit.check(protocol_sha == frozen.get("protocol_sha256"), "frozen_protocol_hash",
                    str(frozen_path), "frozen run protocol hash differs from the snapshot")
        config_sha = _digest(config_copy)
        audit.check(config_sha == frozen.get("config_sha256"), "frozen_config_hash",
                    str(frozen_path), "frozen run config hash differs from the source snapshot")
    except Exception as exc:
        audit.fail("frozen_inputs_unavailable", str(snapshot),
                   "frozen config or protocol copy could not be verified", error=repr(exc))
    return foundation_bytes


def _selected_grid(config: Mapping[str, Any], phase: str) -> tuple[list[int], list[str], list[str], list[str], list[int]]:
    if phase not in {"pilot", "main"}:
        raise ValueError(f"invalid phase {phase!r}")
    seeds = [int(value) for value in config.get("seeds", [])]
    pairs = [str(value) for value in config.get("pair_kinds", [])]
    histories = [str(value) for value in config.get("history_regimes", [])]
    futures = [str(value) for value in config.get("future_regimes", [])]
    replicates = [int(value) for value in config.get("replicates", [])]
    return seeds, pairs, histories, futures, replicates


def _check_config(audit: Audit, config: Mapping[str, Any], phase: str) -> dict[str, Any]:
    where = "config.json"
    _check(audit, config.get("mechanism_format") == CONFIG_FORMAT,
           "mechanism_config_format", where, "config does not declare the frozen mechanism format")
    _check(audit, config.get("phase") == phase, "phase_config", where,
           "normalized config phase differs from the cohort phase",
           config_phase=config.get("phase"), cohort_phase=phase)
    frozen_values = {
        "bootstrap_steps": 1024,
        "bout_steps": 512,
        "candidate_source_steps": 1024,
        "history_tail_source_bouts": 2,
        "future_blocks": 4 if phase == "pilot" else 8,
        "rollout_steps": 256,
        "local_slots": 2,
        "shared_groups": 1,
        "local_learning_rate": 0.003,
        "shared_learning_rate": 0.0,
        "critic_learning_rate": 0.003,
        "actor_budget_bytes": 89,
        "package_budget_bytes": 262144,
        "evidence_rows_per_group": 256,
        "recent_bouts": 8,
        "torch_threads": 1,
        "future_seed_offset": 40000000,
        "future_seed_multiplier": 1000,
        "execution_order_seed": 2026101901,
    }
    for key, expected in frozen_values.items():
        if key not in config:
            audit.fail("frozen_config_field_missing", where,
                       "normalized config omits a required frozen field", field=key)
        else:
            _check(audit, config[key] == expected, "frozen_config_value", where,
                   "normalized config changes a frozen value", field=key,
                   actual=config[key], expected=expected)
    seeds, pairs, histories, futures, replicates = _selected_grid(config, phase)
    expected_grid = {
        "main": ([25101, 25102, 25103, 25104, 25107, 25108],
                 ["dynamics", "reward_conflict"], ["old_major", "new_major"],
                 ["old_major", "new_major"], [0, 1]),
        "pilot": ([25089], ["dynamics"], ["old_major"], ["old_major"], [0, 1]),
    }[phase]
    _check(audit, (seeds, pairs, histories, futures, replicates) == expected_grid,
           "frozen_grid", where, "seed, pair, history, future, or replica grid changed",
           actual={"seeds": seeds, "pair_kinds": pairs, "histories": histories,
                   "futures": futures, "replicates": replicates},
           expected={"seeds": expected_grid[0], "pair_kinds": expected_grid[1],
                     "histories": expected_grid[2], "futures": expected_grid[3],
                     "replicates": expected_grid[4]})
    _check(audit, tuple(config.get("actions", ())) == ACTIONS, "action_grid", where,
           "physical actions differ from the frozen four-action order")
    expected_horizons = [1536, 6144, 12288] if phase == "main" else [1536, 6144]
    _check(audit, config.get("horizons") == expected_horizons, "horizon_grid", where,
           "nested outcome horizons differ from the frozen protocol",
           actual=config.get("horizons"), expected=expected_horizons)
    expected_plan_data = expected_plan(
        phase,
        seeds=seeds,
        pair_kinds=pairs,
        histories=histories,
        futures=futures,
        replicates=replicates,
        actions=ACTIONS,
        bootstrap_steps=int(config.get("bootstrap_steps", 1024)),
        bout_steps=int(config.get("bout_steps", 512)),
        candidate_source_steps=int(config.get("candidate_source_steps", 1024)),
        future_blocks=int(config.get("future_blocks", 0)),
        rollout_steps=int(config.get("rollout_steps", 256)),
    )
    expected_record = config.get("expected_pilot" if phase == "pilot" else "expected_main", {})
    mapped = {
        "case_count": "cases",
        "branch_count": "branches",
        "prefix_steps_per_case": "prefix_steps_per_case",
        "branch_steps_per_branch": "steps_per_branch",
        "prefix_environment_steps": "prefix_steps",
        "branch_environment_steps": "branch_steps",
        "environment_steps_total": "total_steps",
    }
    for plan_key, config_key in mapped.items():
        _check(audit, int(expected_record.get(config_key, -1)) == expected_plan_data[plan_key],
               "frozen_expected_budget", where,
               "normalized config's expected budget differs from the independently calculated plan",
               field=config_key, actual=expected_record.get(config_key),
               expected=expected_plan_data[plan_key])
    return expected_plan_data


def _check_seed_strata(audit: Audit, seeds: Sequence[int], pair_kind: str) -> None:
    mappings = [pair_mapping(int(seed), pair_kind) for seed in seeds]
    old_families = [str(value["families"]["old"]) for value in mappings]
    parity = [int(seed) % 2 for seed in seeds]
    _check(audit, old_families.count("A") == 3 and old_families.count("B") == 3,
           "seed_family_strata", "config.json", "main seeds do not contain three old-family A and three B",
           old_family_counts={family: old_families.count(family) for family in ("A", "B")})
    _check(audit, parity.count(0) == 3 and parity.count(1) == 3,
           "seed_parity_strata", "config.json", "main seeds do not contain three even and three odd",
           even=parity.count(0), odd=parity.count(1))


def _expected_branch_order(config: Mapping[str, Any], case_id: str) -> tuple[int, list[str]]:
    specs = [
        f"{future}__r{replicate}__{action}"
        for future in config["future_regimes"]
        for replicate in config["replicates"]
        for action in config["actions"]
    ]
    material = f"{int(config['execution_order_seed'])}|{case_id}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    permutation = np.random.default_rng(seed).permutation(len(specs)).tolist()
    return seed, [specs[int(index)] for index in permutation]


def _check_execution_order(audit: Audit, case_dir: Path, record: Mapping[str, Any],
                           config: Mapping[str, Any]) -> list[str]:
    case_id = str(record.get("case_id", ""))
    where = str(case_dir / "execution_order.json")
    expected_seed, expected_order = _expected_branch_order(config, case_id)
    try:
        saved = _json(case_dir / "execution_order.json")
    except Exception as exc:
        audit.fail("execution_order_read", where, "saved blinded order is missing or unreadable", error=repr(exc))
        saved = {}
    order = [str(value) for value in saved.get("order", [])]
    audit.check(int(saved.get("seed", -1)) == expected_seed
                and order == expected_order
                and record.get("execution_order") == expected_order
                and record.get("execution_order_seed") == expected_seed,
                "blinded_execution_order", where,
                "case order differs from the deterministic observer-only shuffle",
                recorded_seed=saved.get("seed"), expected_seed=expected_seed,
                recorded_order=order, expected_order=expected_order)
    return expected_order


def _check_artifact_record(audit: Audit, cohort: Path, artifact: Mapping[str, Any],
                           where: str) -> Path | None:
    relative = artifact.get("path")
    path = _safe_path(cohort, str(relative)) if relative is not None else None
    if path is None or not path.is_file():
        audit.fail("artifact_missing", where, "recorded artifact is missing or escapes the cohort",
                   path=relative)
        return None
    try:
        size = path.stat().st_size
        digest = _digest(path)
        audit.check(size == int(artifact.get("bytes", -1))
                    and digest == artifact.get("sha256"),
                    "artifact_hash", str(path), "artifact differs from its recorded hash or size",
                    recorded_bytes=artifact.get("bytes"), actual_bytes=size,
                    recorded_sha256=artifact.get("sha256"), actual_sha256=digest)
        return path
    except Exception as exc:
        audit.fail("artifact_hash_error", str(path), "artifact could not be hashed", error=repr(exc))
        return None


def _check_actor(audit: Audit, checkpoint: Mapping[str, Any], actor_path: Path,
                 config: Mapping[str, Any], where: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        local, shared, physical = storage_audit._core_arrays(checkpoint)
        actor_data = actor_path.read_bytes()
        decoded_local, decoded_shared, decoded_physical = decode_actor(actor_data)
        audit.check(len(actor_data) == int(config["actor_budget_bytes"]),
                    "actor_payload_bytes", str(actor_path),
                    "int8 actor payload differs from its frozen byte count",
                    actual_bytes=len(actor_data), expected_bytes=config["actor_budget_bytes"])
        audit.check(np.array_equal(decoded_local, local)
                    and np.array_equal(decoded_shared, shared)
                    and np.array_equal(decoded_physical.astype(np.int8), physical.astype(np.int8)),
                    "actor_checkpoint_match", str(actor_path),
                    "actor sidecar does not exactly encode the learner checkpoint")
        storage_audit._check_checkpoint_shape(audit, checkpoint, config, where)
        return local, shared, physical
    except Exception as exc:
        audit.fail("actor_checkpoint_audit", where,
                   "actor sidecar or learner arrays could not be checked", error=repr(exc))
        return None


def _check_package(audit: Audit, folder: Path, capacity: Mapping[str, Any],
                   foundation_bytes: int | None, config: Mapping[str, Any], where: str) -> int | None:
    try:
        checkpoint_bytes = (folder / "checkpoint.pt").stat().st_size
        actor_bytes = (folder / "actor.bin").stat().st_size
    except Exception as exc:
        audit.fail("package_files_missing", where, "checkpoint or actor file is missing", error=repr(exc))
        return None
    if foundation_bytes is None:
        return None
    actual = {
        "foundation_bytes": foundation_bytes,
        "checkpoint_bytes": checkpoint_bytes,
        "actor_bytes": actor_bytes,
        "full_persistent_bytes": foundation_bytes + checkpoint_bytes + actor_bytes,
        "package_budget_bytes": int(config["package_budget_bytes"]),
    }
    for key, value in actual.items():
        audit.check(int(capacity.get(key, -1)) == value, "package_accounting", where,
                    "recorded package component differs from physical files",
                    field=key, recorded=capacity.get(key), actual=value)
    audit.check(actual["full_persistent_bytes"] <= actual["package_budget_bytes"],
                "package_cap", where, "complete persistent package exceeds the frozen cap",
                actual_bytes=actual["full_persistent_bytes"], cap=actual["package_budget_bytes"])
    return actual["full_persistent_bytes"]


def _check_raw_ledger_record(audit: Audit, cohort: Path, folder: Path,
                             records: Mapping[str, Any], where: str) -> dict[str, dict[str, Any]]:
    """Recount the three phase ledgers and compare their saved digests."""
    result: dict[str, dict[str, Any]] = {}
    for key, filename in (("steps_csv", "steps.csv"), ("episodes_csv", "episodes.csv"),
                          ("rollouts_csv", "rollouts.csv")):
        path = folder / filename
        saved = records.get(key, {}) if isinstance(records, Mapping) else {}
        if not path.is_file():
            if isinstance(saved, Mapping) and saved.get("exists") is False:
                result[key] = {"rows": 0, "reward_sum": 0.0, "updates": 0}
                continue
            audit.fail("raw_ledger_missing", str(path), "runner did not persist a required raw ledger")
            result[key] = {"rows": 0, "reward_sum": 0.0, "updates": 0}
            continue
        try:
            actual_sha = _digest(path)
            actual_bytes = path.stat().st_size
            saved_path = _safe_path(cohort, str(saved.get("path", ""))) if isinstance(saved, Mapping) else None
            audit.check(saved_path is not None and saved_path == path.resolve(),
                        "raw_ledger_path", str(path),
                        "recorded raw-ledger path differs from its case-local file",
                        recorded=saved.get("path") if isinstance(saved, Mapping) else None)
            bytes_match = ("bytes" not in saved) or actual_bytes == int(saved.get("bytes", -1))
            audit.check(actual_sha == saved.get("sha256") and bytes_match,
                        "raw_ledger_hash", str(path), "raw ledger differs from its saved digest or size",
                        recorded_sha256=saved.get("sha256"), actual_sha256=actual_sha,
                        recorded_bytes=saved.get("bytes"), actual_bytes=actual_bytes)
            rows = _csv_rows(audit, path)
            reward_sum = 0.0
            updates = trained_rows = 0
            for index, row in enumerate(rows):
                try:
                    if key == "steps_csv":
                        reward_sum += float(row["reward"])
                    elif key == "rollouts_csv":
                        updates += int(row["updates"])
                        trained_rows += int(row["rows"])
                except Exception as exc:
                    audit.fail("raw_ledger_count_parse", str(path),
                               "raw ledger contains a malformed count or reward", row=index, error=repr(exc))
            audit.check(len(rows) == int(saved.get("rows", -1)), "raw_ledger_rows", str(path),
                        "raw ledger row count differs from its saved ledger summary",
                        recorded=saved.get("rows"), actual=len(rows))
            if key == "steps_csv":
                audit.check(math.isclose(reward_sum, float(saved.get("reward_sum", float("nan"))),
                                         abs_tol=1e-6, rel_tol=1e-8),
                            "raw_ledger_reward_sum", str(path),
                            "raw reward sum differs from its saved ledger summary",
                            recorded=saved.get("reward_sum"), actual=reward_sum)
            elif key == "rollouts_csv":
                audit.check(len(rows) == int(saved.get("rollout_count", -1))
                            and updates == int(saved.get("minibatch_updates", -1))
                            and trained_rows == int(saved.get("trained_rows", -1)),
                            "raw_ledger_rollout_counts", str(path),
                            "rollout ledger count, training rows, or update count differs from its saved summary",
                            recorded={name: saved.get(name) for name in ("rollout_count", "trained_rows", "minibatch_updates")},
                            actual={"rollout_count": len(rows), "trained_rows": trained_rows,
                                    "minibatch_updates": updates})
            result[key] = {"rows": len(rows), "reward_sum": reward_sum,
                           "updates": updates, "trained_rows": trained_rows}
        except Exception as exc:
            audit.fail("raw_ledger_audit", str(path), "raw ledger could not be independently counted", error=repr(exc))
            result[key] = {"rows": 0, "reward_sum": 0.0, "updates": 0}
    return result


def _check_episode_ledger(audit: Audit, folder: Path, step_rows: Sequence[Mapping[str, str]],
                          final_simulator: Mapping[str, Any], *, seed: int, agent_seed: int,
                          start_simulator: Mapping[str, Any] | None = None,
                          action: str = "", future: str = "", where: str) -> dict[str, int]:
    """Check episode boundaries and resets without constructing an environment."""
    episode_path = folder / "episodes.csv"
    episode_rows = _csv_rows(audit, episode_path)
    start_driver = (start_simulator or {}).get("driver", {})
    final_driver = final_simulator.get("driver", {}) if isinstance(final_simulator, Mapping) else {}
    start_ordinal = int(start_driver.get("episode_ordinal", 0))
    start_length = int(start_driver.get("episode_length", 0))
    start_reset_seed = int(start_driver.get("episode_reset_seed", storage_audit.runner._reset_seed(seed, start_ordinal)))
    environment = final_simulator.get("environment", {}) if isinstance(final_simulator, Mapping) else {}
    audit.check(int(environment.get("episode_length", -1)) == int(final_driver.get("episode_length", -2)),
                "simulator_episode_length", where,
                "physical simulator and saved driver disagree about the current episode length",
                simulator_length=environment.get("episode_length"), driver_length=final_driver.get("episode_length"))

    completions: list[dict[str, Any]] = []
    segment_id: int | None = None
    segment_reset: int | None = None
    segment_length = start_length
    segment_reward = 0.0
    first_segment = True
    previous_done = False
    previous_episode_id = start_ordinal
    previous_reset_seed = start_reset_seed
    for index, row in enumerate(step_rows):
        try:
            episode_id = int(row["episode_id"])
            reset_seed = int(row["reset_seed"])
            reward = float(row["reward"])
            done = bool(int(row["environment_done"]))
            terminated, truncated = bool(int(row["terminated"])), bool(int(row["truncated"]))
            obs = np.asarray(json.loads(row["obs"]), dtype=np.float32)
            next_obs = np.asarray(json.loads(row["next_obs"]), dtype=np.float32)
        except Exception as exc:
            audit.fail("episode_step_parse", str(folder / "steps.csv"),
                       "step row lacks valid episode, reset, reward, or observation data", row=index, error=repr(exc))
            continue
        if index == 0:
            audit.check(episode_id == start_ordinal and reset_seed == start_reset_seed,
                        "branch_episode_continuity", str(folder / "steps.csv"),
                        "first step does not continue from the saved prefix episode and reset seed",
                        episode_id=episode_id, expected_episode_id=start_ordinal,
                        reset_seed=reset_seed, expected_reset_seed=start_reset_seed)
            if start_driver and "obs" in start_driver:
                prior_obs = np.asarray(start_driver["obs"], dtype=np.float32)
                audit.check(obs.shape == prior_obs.shape and np.array_equal(obs, prior_obs),
                            "branch_observation_continuity", str(folder / "steps.csv"),
                            "first branch observation differs from the saved prefix observation")
            segment_id, segment_reset = episode_id, reset_seed
        else:
            if previous_done:
                expected_id = previous_episode_id + 1
                expected_seed = storage_audit.runner._reset_seed(agent_seed, expected_id)
                audit.check(episode_id == expected_id and reset_seed == expected_seed,
                            "episode_reset_transition", str(folder / "steps.csv"),
                            "post-completion step does not use the next ordinal and deterministic reset seed",
                            row=index, episode_id=episode_id, expected_episode_id=expected_id,
                            reset_seed=reset_seed, expected_reset_seed=expected_seed)
                segment_id, segment_reset = episode_id, reset_seed
                segment_length, segment_reward = 0, 0.0
                first_segment = False
            else:
                audit.check(episode_id == previous_episode_id and reset_seed == previous_reset_seed,
                            "episode_id_continuity", str(folder / "steps.csv"),
                            "episode or reset seed changed without termination or truncation", row=index,
                            episode_id=episode_id, previous_episode_id=previous_episode_id,
                            reset_seed=reset_seed, previous_reset_seed=previous_reset_seed)
                prior = step_rows[index - 1]
                try:
                    prior_next = np.asarray(json.loads(prior["next_obs"]), dtype=np.float32)
                    audit.check(np.array_equal(prior_next, obs), "trajectory_observation_continuity",
                                str(folder / "steps.csv"),
                                "step observation is discontinuous within a live episode", row=index)
                except Exception as exc:
                    audit.fail("trajectory_observation_parse", str(folder / "steps.csv"),
                               "adjacent episode observation cannot be parsed", row=index, error=repr(exc))
        if segment_id is None:
            segment_id, segment_reset = episode_id, reset_seed
        segment_length += 1
        segment_reward += reward
        audit.check(done == (terminated or truncated), "episode_done_flag", str(folder / "steps.csv"),
                    "environment_done differs from terminated or truncated", row=index)
        if done:
            completions.append({
                "episode_id": episode_id, "reset_seed": reset_seed,
                "next_reset_seed": storage_audit.runner._reset_seed(agent_seed, episode_id + 1),
                "length": segment_length, "reward": segment_reward,
                "terminated": terminated, "truncated": truncated,
            })
            segment_length, segment_reward = 0, 0.0
        previous_done = done
        previous_episode_id, previous_reset_seed = episode_id, reset_seed
        if index + 1 == len(step_rows) and not done:
            # This is only compared with the driver below; the returned observation is
            # from the current physical episode, so it should match exactly.
            audit.check(np.asarray(final_driver.get("obs", []), dtype=np.float32).shape == next_obs.shape
                        and np.array_equal(np.asarray(final_driver.get("obs", []), dtype=np.float32), next_obs),
                        "final_live_observation", where,
                        "last raw next observation differs from the persisted live episode observation")

    audit.check(len(episode_rows) == len(completions), "episode_completion_count", str(episode_path),
                "episode ledger rows differ from terminated and truncated transitions",
                recorded=len(episode_rows), actual=len(completions))
    for index, (record, completion) in enumerate(zip(episode_rows, completions)):
        try:
            matches = (
                int(record["episode_id"]) == completion["episode_id"]
                and int(record["reset_seed"]) == completion["reset_seed"]
                and int(record["next_reset_seed"]) == completion["next_reset_seed"]
                and int(record["length"]) == completion["length"]
                and bool(int(record["terminated"])) == completion["terminated"]
                and bool(int(record["truncated"])) == completion["truncated"]
            )
            audit.check(matches, "episode_ledger_record", str(episode_path),
                        "episode summary differs from its raw termination boundary", row=index,
                        expected=completion, recorded={key: record.get(key) for key in completion})
            if not (start_simulator is not None and index == 0
                    and int(record["episode_id"]) == start_ordinal):
                audit.check(math.isclose(float(record["return_value"]), completion["reward"],
                                         abs_tol=1e-6, rel_tol=1e-8),
                            "episode_return_rows", str(episode_path),
                            "episode return differs from its complete raw step segment", row=index,
                            recorded=record.get("return_value"), actual=completion["reward"])
            audit.check(record.get("action", "") == action and record.get("future_regime", "") == future,
                        "episode_branch_labels", str(episode_path),
                        "episode ledger action or future labels differ from the branch", row=index)
        except Exception as exc:
            audit.fail("episode_record_parse", str(episode_path),
                       "episode summary row is malformed", row=index, error=repr(exc))

    expected_ordinal = start_ordinal + len(completions)
    audit.check(int(final_driver.get("episode_ordinal", -1)) == expected_ordinal,
                "final_episode_ordinal", where,
                "persisted episode ordinal differs from completed raw episode count",
                recorded=final_driver.get("episode_ordinal"), expected=expected_ordinal)
    final_length = (start_length + len(step_rows)) if not completions else len(step_rows) - (
        max(index for index, row in enumerate(step_rows) if bool(int(row.get("environment_done", "0")))) + 1
    )
    audit.check(int(final_driver.get("episode_length", -1)) == final_length,
                "final_episode_length", where,
                "persisted live episode length differs from raw step continuity",
                recorded=final_driver.get("episode_length"), expected=final_length)
    if expected_ordinal == start_ordinal:
        expected_reset = start_reset_seed
    else:
        expected_reset = storage_audit.runner._reset_seed(agent_seed, expected_ordinal)
    audit.check(int(final_driver.get("episode_reset_seed", -1)) == expected_reset,
                "final_episode_reset_seed", where,
                "persisted live reset seed differs from the transformed seed stream",
                recorded=final_driver.get("episode_reset_seed"), expected=expected_reset)
    return {"completed_episodes": len(completions), "episode_rows": len(episode_rows),
            "final_episode_ordinal": expected_ordinal, "final_episode_length": final_length}


def _check_raw_bout_metadata(audit: Audit, step_rows: Sequence[Mapping[str, str]],
                             bout_records: Sequence[Mapping[str, Any]], where: str) -> None:
    observed: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(step_rows):
        try:
            bout = int(row["bout"])
            values = (str(row["taskobserver"]), int(row["group"]), int(row["slot"]), int(row["generation"]))
        except Exception as exc:
            audit.fail("raw_bout_parse", where, "raw step lacks task or slot-generation metadata",
                       row=index, error=repr(exc))
            continue
        state = observed.setdefault(bout, {"steps": 0, "values": set()})
        state["steps"] += 1
        state["values"].add(values)
    _check(audit, set(observed) == {int(row.get("bout", -1)) for row in bout_records},
           "raw_bout_universe", where, "raw step bouts differ from saved bout summaries",
           raw=sorted(observed), summary=sorted(int(row.get("bout", -1)) for row in bout_records))
    for row in bout_records:
        try:
            bout = int(row["bout"])
            state = observed[bout]
            expected = (str(row["task"]), int(row["group"]), int(row["slot"]), int(row["generation"]))
            _check(audit, state["steps"] == int(row["steps"]) and state["values"] == {expected},
                   "raw_bout_metadata", where,
                   "raw task, group, resident slot, or generation differs from its saved bout record",
                   bout=bout, raw_steps=state["steps"], summary_steps=row.get("steps"),
                   raw_values=sorted(state["values"]), expected=expected)
        except Exception as exc:
            audit.fail("raw_bout_summary_parse", where, "saved bout metadata is malformed", error=repr(exc))


def _check_json_state_hash(audit: Audit, checkpoint: Mapping[str, Any], simulator: Mapping[str, Any],
                           recorded: Any, where: str, code: str) -> str | None:
    try:
        actual = semantic_hash({"learner": checkpoint, "simulator": simulator})
        audit.check(actual == recorded, code, where,
                    "learner and simulator state hash differs from the recorded semantic hash",
                    recorded=recorded, actual=actual)
        return actual
    except Exception as exc:
        audit.fail(code + "_error", where, "learner and simulator state could not be hashed", error=repr(exc))
        return None


def _check_prefix_state(audit: Audit, cohort: Path, case_dir: Path,
                        case_record: Mapping[str, Any], config: Mapping[str, Any],
                        foundation_bytes: int | None, expected: Mapping[str, int]) -> dict[str, Any] | None:
    prefix_dir = case_dir / "prefix"
    where = str(prefix_dir)
    try:
        saved = _json(prefix_dir / "summary.json")
        simulator = _json(prefix_dir / "simulator.json")
    except Exception as exc:
        audit.fail("prefix_artifacts_read", where, "prefix summary or simulator snapshot is missing", error=repr(exc))
        return None
    checkpoint = _load_checkpoint(audit, prefix_dir / "checkpoint.pt")
    if checkpoint is None:
        return None
    schedule = prefix_schedule(
        case_record["mapping"], config, str(case_record["history_regime"]),
    )
    record_schedule = saved.get("per_bout", [])
    audit.check(len(record_schedule) == len(schedule), "prefix_bout_count", where,
                "prefix summary omits one or more frozen bouts",
                actual=len(record_schedule), expected=len(schedule))
    for actual, expected_row in zip(record_schedule, schedule):
        for key in ("bout", "task", "group", "steps"):
            audit.check(actual.get(key) == expected_row.get(key), "prefix_bout_schedule", where,
                        "prefix bout differs from the frozen schedule", field=key,
                        actual=actual.get(key), expected=expected_row.get(key))
    rows = _csv_rows(audit, prefix_dir / "steps.csv")
    _check_raw_ledger_record(audit, cohort, prefix_dir, saved.get("raw_ledgers", {}), where)
    raw_steps, raw_reward, _ = storage_audit._check_schedule_rows(
        audit, rows, schedule, phase="prefix", action="", future="",
        global_start=0, seed=int(case_record["seed"]), where=str(prefix_dir / "steps.csv"),
    )
    audit.check(raw_steps == expected["prefix_steps_per_case"], "prefix_raw_budget", where,
                "prefix raw rows differ from the frozen step budget",
                actual=raw_steps, expected=expected["prefix_steps_per_case"])
    _check_raw_bout_metadata(audit, rows, saved.get("per_bout", []), str(prefix_dir / "steps.csv"))
    storage_audit._check_reward_summary(audit, rows, saved,
                                        raw_reward=raw_reward, raw_steps=raw_steps,
                                        where=str(prefix_dir / "summary.json"))
    rollouts = storage_audit._check_rollouts(
        audit, prefix_dir,
        phase=("bootstrap", "prefix_history"), expected_rows=raw_steps,
        summary={**saved, "rollout_steps": int(config["rollout_steps"])}, where=where,
    )
    audit.check(rollouts["rows"] == raw_steps, "prefix_rollout_coverage", where,
                "prefix PPO ledger does not cover each fresh row once")
    observed_capacity = _check_package(audit, prefix_dir, saved.get("capacity", {}),
                                       foundation_bytes, config, where)
    arrays = _check_actor(audit, checkpoint, prefix_dir / "actor.bin", config, where)
    learner_hash = semantic_hash(checkpoint)
    simulator_hash = semantic_hash(simulator)
    prefix_hash = _check_json_state_hash(
        audit, checkpoint, simulator, saved.get("prefix_state_hash"),
        where, "prefix_state_hash",
    )
    audit.check(learner_hash == saved.get("prefix_learner_hash"), "prefix_learner_hash", where,
                "learner checkpoint differs from its saved semantic hash",
                recorded=saved.get("prefix_learner_hash"), actual=learner_hash)
    audit.check(prefix_hash == case_record.get("prefix", {}).get("prefix_state_hash"),
                "case_prefix_state_hash", where,
                "case summary prefix hash differs from the serialized prefix state")
    audit.check(bool(saved.get("shared_frozen")) and bool(checkpoint.get("shared_frozen")),
                "prefix_shared_frozen", where, "shared parent is not frozen at the common prefix")
    audit.check(saved.get("foundation_sha256") == checkpoint.get("foundation_sha256"),
                "prefix_foundation_hash", where, "prefix summary and learner foundation hashes disagree")
    source_fits = saved.get("source_fits", {})
    audit.check(bool(source_fits.get("verified")) and bool(source_fits.get("old_verified"))
                and bool(source_fits.get("candidate_verified")),
                "source_fit_flags", where, "runner did not verify all incumbent and candidate source fits",
                source_fits=source_fits)
    observer_record: dict[str, Any] | None = None
    observer_artifact = saved.get("observer_artifact")
    if isinstance(observer_artifact, Mapping):
        observer_path = _check_artifact_record(audit, cohort, observer_artifact,
                                               f"{where}/observer_artifact")
        if observer_path is not None:
            try:
                observer_record = _json(observer_path)
            except Exception as exc:
                audit.fail("observer_prefix_read", str(observer_path),
                           "observer prefix evidence is malformed", error=repr(exc))
    else:
        audit.fail("observer_prefix_missing", where,
                   "observer-only source matrices and pre-action actor arrays are required")
    if arrays is not None and observer_record is not None:
        _check_prefix_sources(audit, observer_record, source_fits, checkpoint, arrays, where,
                              per_bout=saved.get("per_bout", []), schedule=schedule,
                              mapping=case_record["mapping"])
    _check_episode_ledger(audit, prefix_dir, rows, simulator,
                          seed=int(case_record["seed"]), agent_seed=int(case_record["seed"]), where=where)
    del rows
    return {
        "summary": saved,
        "checkpoint": checkpoint,
        "simulator": simulator,
        "prefix_hash": prefix_hash,
        "learner_hash": learner_hash,
        "simulator_hash": simulator_hash,
        "arrays": arrays,
        "source_fits": source_fits,
        "observer": observer_record,
        "capacity_bytes": observed_capacity,
        "rollouts": rollouts,
        "schedule": schedule,
    }


def _check_prefix_sources(audit: Audit, observer: Mapping[str, Any],
                          source_fits: Mapping[str, Any], checkpoint: Mapping[str, Any],
                          arrays: tuple[np.ndarray, np.ndarray, np.ndarray], where: str, *,
                          per_bout: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]],
                          mapping: Mapping[str, Any]) -> None:
    local, shared, physical = arrays
    old_sources = list(source_fits.get("old_sources", []))
    candidate_sources = list(source_fits.get("candidate_sources", []))
    source_metrics: dict[str, Any] = {
        "old_sources": [],
        "candidate_sources": [],
        "candidate_current_generation_fitted_steps_max": None,
    }
    old_matrices = list(observer.get("old_source_effective_matrices_at_bootstrap", []))
    candidate_matrices = list(observer.get("candidate_current_effective_matrices_before_averaging", []))
    _check(audit, len(old_sources) == len(old_matrices) == 2, "old_source_universe", where,
           "exactly two old source-time effective matrices must be preserved",
           source_records=len(old_sources), matrices=len(old_matrices))
    _check(audit, len(candidate_sources) == len(candidate_matrices) == 2,
           "candidate_source_universe", where,
           "exactly two current candidate source matrices must be preserved",
           source_records=len(candidate_sources), matrices=len(candidate_matrices))
    for index, (record, matrix_record) in enumerate(zip(old_sources, old_matrices)):
        matrix = np.asarray(matrix_record.get("effective_matrix", []), dtype=np.float32)
        _check(audit, matrix.shape == (3, 3, 3), "old_source_matrix_shape", where,
               "bootstrap incumbent source matrix must have three 3 by 3 bands",
               index=index, shape=list(matrix.shape))
        for key in ("task", "slot", "generation"):
            _check(audit, record.get(key) == matrix_record.get(key), "old_source_provenance", where,
                   "source-fit provenance disagrees with its saved observer matrix",
                   index=index, field=key, record=record.get(key), matrix_record=matrix_record.get(key))
        _check(audit, int(record.get("source_time_fitted_steps", -1)) == int(matrix_record.get("fitted_steps", -2)),
               "old_source_fit_provenance", where,
               "bootstrap observer matrix fit count differs from source-time fit record",
               index=index, source_time_fitted_steps=record.get("source_time_fitted_steps"),
               observer_fitted_steps=matrix_record.get("fitted_steps"))
        _check(audit, int(record.get("source_time_fitted_steps", -1)) == 1024,
               "old_source_fit", where, "each incumbent source must have 1024 source-time fitted rows",
               source=record.get("task"), actual=record.get("source_time_fitted_steps"))
        source_metrics["old_sources"].append({
            key: record.get(key) for key in (
                "task", "slot", "generation", "source_time_fitted_steps",
                "later_same_generation_fitted_steps", "task_lifetime_observed_steps",
            )
        })
    old_parent = np.asarray(observer.get("incumbent_mean_effective_parent_at_bootstrap", []), dtype=np.float32)
    if len(old_matrices) == 2 and all(np.asarray(row.get("effective_matrix", [])).shape == (3, 3, 3)
                                      for row in old_matrices):
        expected_parent = np.asarray(
            np.stack([np.asarray(row["effective_matrix"], dtype=np.float32) for row in old_matrices])
            .mean(axis=(0, 1), dtype=np.float64), dtype=np.float32,
        )
        # The action parent is the common mean of both sources and all bands.
        audit.check(old_parent.shape == (3, 3) and np.allclose(old_parent, expected_parent, atol=1e-7, rtol=0.0),
                    "incumbent_parent_mean", where,
                    "saved bootstrap parent differs from the mean of the two old sources and three bands")
        try:
            quantized = encode_actor(local, old_parent.reshape(1, 9), physical,
                                     codec="int8", budget_bytes=89)
            _, quantized_shared, _ = decode_actor(quantized.data)
            audit.check(shared.shape == quantized_shared.shape
                        and np.array_equal(shared, quantized_shared),
                        "incumbent_parent_frozen", where,
                        "prefix shared parent differs from the one-time quantized bootstrap incumbent parent")
        except Exception as exc:
            audit.fail("incumbent_parent_quantization", where,
                       "bootstrap incumbent parent could not be quantized for a frozen-parent check", error=repr(exc))
    else:
        audit.fail("incumbent_source_matrices", where,
                   "cannot recompute the bootstrap incumbent parent from saved source matrices")

    final_local = np.asarray(observer.get("prefix_actor_arrays_before_action", {}).get("local", []), dtype=np.float32)
    final_shared = np.asarray(observer.get("prefix_actor_arrays_before_action", {}).get("shared", []), dtype=np.float32)
    final_physical = np.asarray(observer.get("prefix_actor_arrays_before_action", {}).get("physical_groups", []), dtype=np.int8)
    final_logical = np.asarray(observer.get("prefix_actor_arrays_before_action", {}).get("logical_group_tags", []), dtype=np.int8)
    final_binding = observer.get("prefix_actor_arrays_before_action", {}).get("parent_group")
    _check(audit, final_local.shape == local.shape and np.array_equal(final_local, local)
                and final_shared.shape == shared.shape and np.array_equal(final_shared, shared)
                and final_physical.shape == physical.shape and np.array_equal(final_physical, physical)
                and final_logical.shape == np.asarray(checkpoint.get("slot_group_tags")).shape
                and np.array_equal(final_logical, np.asarray(checkpoint.get("slot_group_tags"), dtype=np.int8))
                and final_binding == checkpoint.get("parent_group"),
           "prefix_preaction_arrays", where,
           "observer pre-action arrays differ from the common-prefix checkpoint")
    slots = [int(row.get("slot", -1)) for row in candidate_sources]
    _check(audit, len(slots) == 2 and len(set(slots)) == 2
                and all(0 <= value < len(local) for value in slots),
           "candidate_source_slots", where,
           "current candidate sources must occupy two different valid resident slots", slots=slots)
    if len(slots) == 2 and all(0 <= value < len(local) for value in slots):
        candidate = common_mean_effective(local, shared, physical, slots)
        candidate_sha = semantic_hash(candidate)
        _check(audit, candidate_sha == source_fits.get("candidate_sha256"),
               "candidate_mean_hash", where,
               "reconstructed current candidate differs from its saved provenance hash",
               recorded=source_fits.get("candidate_sha256"), actual=candidate_sha)
        candidate_matrix = np.asarray(observer.get("candidate_mean_effective", []), dtype=np.float32)
        _check(audit, candidate_matrix.shape == (3, 3)
                    and np.allclose(candidate_matrix, candidate, atol=1e-6, rtol=0.0),
               "candidate_mean", where,
               "saved current candidate is not the mean of two current tasks across all three bands")
        for index, (record, matrix_record) in enumerate(zip(candidate_sources, candidate_matrices)):
            slot = int(record.get("slot", -1))
            matrix = np.asarray(matrix_record.get("effective_matrix", []), dtype=np.float32)
            _check(audit, matrix.shape == (3, 3, 3), "candidate_source_matrix_shape", where,
                   "candidate source matrix must have three 3 by 3 bands", index=index,
                   shape=list(matrix.shape))
            for key in ("task", "slot", "generation"):
                _check(audit, record.get(key) == matrix_record.get(key), "candidate_source_provenance", where,
                       "candidate source-fit provenance disagrees with its saved observer matrix",
                       index=index, field=key, record=record.get(key), matrix_record=matrix_record.get(key))
            _check(audit, int(record.get("fitted_steps", -1)) == int(matrix_record.get("fitted_steps", -2)),
                   "candidate_source_fit_provenance", where,
                   "current candidate observer matrix fit count differs from source-fit record",
                   index=index, fitted_steps=record.get("fitted_steps"),
                   observer_fitted_steps=matrix_record.get("fitted_steps"))
            _check(audit, int(record.get("group", -1)) == 1
                        and int(record.get("source_time_fitted_steps", -1)) == 1024
                        and int(record.get("fitted_steps", -1)) == 1024,
                   "candidate_source_fit", where,
                   "each current candidate generation must be group 1 with exactly 1024 fitted rows",
                   source=record.get("task"), fitted_steps=record.get("fitted_steps"),
                   source_time_fitted_steps=record.get("source_time_fitted_steps"))
            if matrix.shape == (3, 3, 3) and 0 <= slot < len(local):
                actual_effective = effective_matrices(local, shared, physical)[slot]
                _check(audit, np.allclose(matrix, actual_effective, atol=1e-6, rtol=0.0),
                       "candidate_source_effective_matrix", where,
                       "saved candidate matrix differs from its current resident effective matrix",
                       source=record.get("task"), slot=slot)
            source_metrics["candidate_sources"].append({
                key: record.get(key) for key in (
                    "task", "slot", "generation", "source_time_fitted_steps",
                    "fitted_steps", "task_lifetime_observed_steps",
                )
            })

    # Source histories are independently recounted from the prefix bout ledger.
    observed_exposures = [
        {key: row.get(key) for key in ("bout", "slot", "generation", "steps", "group", "task")}
        for row in per_bout
    ]
    for name, rows in (("old_sources", old_sources), ("candidate_sources", candidate_sources)):
        for source_index, record in enumerate(rows):
            actual = [row for row in observed_exposures if row["task"] == record.get("task")]
            saved = list(record.get("task_exposures", []))
            expected_exposures = [
                {key: item.get(key) for key in ("bout", "slot", "generation", "steps", "group")}
                for item in actual
            ]
            actual_saved = [
                {key: item.get(key) for key in ("bout", "slot", "generation", "steps", "group")}
                for item in saved
            ]
            _check(audit, actual_saved == expected_exposures, "source_task_exposures", where,
                   "saved task lifetime exposure differs from raw prefix bout assignments",
                   source_kind=name, source=record.get("task"), saved=actual_saved,
                   actual=expected_exposures)
            lifetime = sum(int(row["steps"]) for row in actual)
            _check(audit, int(record.get("task_lifetime_observed_steps", -1)) == lifetime,
                   "source_task_lifetime", where,
                   "reported source task lifetime differs from prefix exposure rows",
                   source_kind=name, source=record.get("task"), recorded=record.get("task_lifetime_observed_steps"),
                   actual=lifetime)
            if name == "old_sources":
                source_bout = int(old_matrices[source_index].get("bout", -1)) if source_index < len(old_matrices) else -1
                source_slot = int(record.get("slot", -1))
                source_generation = int(record.get("generation", -1))
                later = sum(int(row["steps"]) for row in per_bout
                            if int(row.get("bout", -1)) > source_bout
                            and int(row.get("slot", -2)) == source_slot
                            and int(row.get("generation", -3)) == source_generation)
                _check(audit, source_bout in (0, 1)
                            and record.get("task") == schedule[source_bout].get("task")
                            and int(record.get("group", -1)) == 0
                            and int(record.get("later_same_generation_fitted_steps", -1)) == later,
                       "incumbent_lifetime_fit", where,
                       "incumbent source-time identity or later same-generation fit disagrees with prefix history",
                       source=record.get("task"), source_bout=source_bout,
                       later_recorded=record.get("later_same_generation_fitted_steps"), later_actual=later)
            else:
                expected_task = task_name(mapping, 1, source_index)
                _check(audit, record.get("task") == expected_task and int(record.get("group", -1)) == 1,
                       "candidate_tail_task", where,
                       "candidate sources are not group-1 variants 0 and 1 in the declared tail order",
                       source=record.get("task"), expected=expected_task)
                source_slot = int(record.get("slot", -1))
                source_generation = int(record.get("generation", -1))
                generation_steps = sum(int(row["steps"]) for row in per_bout
                                       if int(row.get("slot", -2)) == source_slot
                                       and int(row.get("generation", -3)) == source_generation)
                latest = per_bout[-2 + source_index] if len(per_bout) >= 2 else {}
                _check(audit, latest.get("task") == expected_task
                            and int(latest.get("slot", -1)) == source_slot
                            and int(latest.get("generation", -1)) == source_generation
                            and generation_steps == 1024
                            and int(record.get("source_time_fitted_steps", -1)) == 1024
                            and int(record.get("fitted_steps", -1)) == 1024,
                       "candidate_fresh_generation", where,
                       "candidate is not a fresh 1024-row current resident generation from the final tail",
                       source=record.get("task"), generation_steps=generation_steps,
                       latest_bout=latest, fitted_steps=record.get("fitted_steps"))
                existing = source_metrics["candidate_sources"][source_index]
                existing["raw_current_generation_observed_steps"] = generation_steps
                existing["current_generation_exceeds_1024"] = generation_steps > 1024
                old_max = source_metrics["candidate_current_generation_fitted_steps_max"]
                source_metrics["candidate_current_generation_fitted_steps_max"] = max(
                    generation_steps, int(old_max or 0),
                )
    try:
        old_parent = np.asarray(observer.get("incumbent_mean_effective_parent_at_bootstrap", []), dtype=np.float32)
        quantized = encode_actor(local, old_parent.reshape(1, 9), physical,
                                 codec="int8", budget_bytes=89)
        _, quantized_shared, _ = decode_actor(quantized.data)
        source_metrics["bootstrap_parent_precodec_to_decoded_max_abs_error"] = float(
            np.max(np.abs(old_parent.reshape(1, 9) - quantized_shared), initial=0.0)
        )
        source_metrics["decoded_parent_matches_prefix_shared"] = bool(np.array_equal(shared, quantized_shared))
    except Exception:
        source_metrics["bootstrap_parent_precodec_to_decoded_max_abs_error"] = None
        source_metrics["decoded_parent_matches_prefix_shared"] = False
    audit.observations.setdefault("source_fits_by_case", {})[where] = source_metrics


def _reseeded_checkpoint_hash(checkpoint: Mapping[str, Any], simulator: Mapping[str, Any],
                              seed: int) -> str:
    transformed = copy.deepcopy(dict(checkpoint))
    transformed["seed"] = int(seed)
    transformed["action_rng"] = np.random.default_rng(int(seed) + 0xA17).bit_generator.state
    transformed["behavior_rng"] = np.random.default_rng(int(seed) + 0xB3E).bit_generator.state
    transformed["numpy_rng"] = np.random.RandomState(int(seed)).get_state()
    transformed["python_rng"] = random.Random(int(seed)).getstate()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    transformed["torch_rng"] = generator.get_state()
    return semantic_hash({"learner": transformed, "simulator": simulator})


def _check_rng_transform(audit: Audit, branch: Mapping[str, Any], prefix_checkpoint: Mapping[str, Any],
                         prefix_simulator: Mapping[str, Any], where: str,
                         config: Mapping[str, Any]) -> int:
    seed = int(branch.get("seed", -1))
    replicate = int(branch.get("replicate", -1))
    expected_seed = future_seed(seed, replicate, config)
    actual_seed = int(branch.get("future_seed", -1))
    _check(audit, actual_seed == expected_seed, "future_seed", where,
           "branch future seed differs from the frozen case and replica formula",
           recorded=actual_seed, expected=expected_seed)
    transform = branch.get("rng_transform", {})
    before, after = transform.get("before", {}), transform.get("after", {})
    reseed = transform.get("reseed", {})
    driver = prefix_simulator.get("driver", {})
    env_snapshot = prefix_simulator.get("environment", {})
    environment_hash = semantic_hash({
        "bit_generator": env_snapshot.get("np_random_bit_generator"),
        "state": env_snapshot.get("np_random_state"),
    })
    expected_before_hashes = {
        "action_rng_sha256": semantic_hash(prefix_checkpoint.get("action_rng")),
        "behavior_rng_sha256": semantic_hash(prefix_checkpoint.get("behavior_rng")),
        "numpy_global_rng_sha256": semantic_hash(prefix_checkpoint.get("numpy_rng")),
        "python_global_rng_sha256": semantic_hash(prefix_checkpoint.get("python_rng")),
        "torch_global_rng_sha256": semantic_hash(prefix_checkpoint.get("torch_rng")),
        "environment_rng_sha256": environment_hash,
    }
    expected_after_hashes = {
        "action_rng_sha256": semantic_hash(np.random.default_rng(expected_seed + 0xA17).bit_generator.state),
        "behavior_rng_sha256": semantic_hash(np.random.default_rng(expected_seed + 0xB3E).bit_generator.state),
        "numpy_global_rng_sha256": semantic_hash(np.random.RandomState(expected_seed).get_state()),
        "python_global_rng_sha256": semantic_hash(random.Random(expected_seed).getstate()),
        "torch_global_rng_sha256": semantic_hash(torch.Generator(device="cpu").manual_seed(expected_seed).get_state()),
        "environment_rng_sha256": environment_hash,
    }
    for field, expected in expected_before_hashes.items():
        _check(audit, before.get(field) == expected, "rng_before_stream", where,
               "pre-transform RNG stream differs from the serialized prefix", stream=field,
               recorded=before.get(field), expected=expected)
    for field, expected in expected_after_hashes.items():
        _check(audit, after.get(field) == expected, "rng_after_stream", where,
               "post-transform RNG stream differs from its declared deterministic seed", stream=field,
               recorded=after.get(field), expected=expected)
    expected_episode = {
        "episode_reset_seed": int(driver.get("episode_reset_seed", -1)),
        "episode_ordinal": int(driver.get("episode_ordinal", -1)),
        "episode_length": int(driver.get("episode_length", -1)),
    }
    for field, value in expected_episode.items():
        _check(audit, before.get(field) == value and after.get(field) == value,
               "rng_preserved_episode", where,
               "RNG transformation changed the current episode or its reset seed", field=field,
               before=before.get(field), after=after.get(field), expected=value)
    _check(audit, bool(transform.get("environment_rng_preserved"))
                and bool(transform.get("current_episode_preserved")),
           "rng_transform_flags", where,
           "runner did not confirm preservation of the private environment RNG and current episode")
    expected_reseed = {
        "future_seed": expected_seed,
        "action_rng_seed": expected_seed + 0xA17,
        "action_rng_stream": 0xA17,
        "behavior_rng_seed": expected_seed + 0xB3E,
        "behavior_rng_stream": 0xB3E,
        "numpy_global_seed": expected_seed,
        "python_global_seed": expected_seed,
        "torch_global_seed": expected_seed,
    }
    for key, value in expected_reseed.items():
        _check(audit, reseed.get(key) == value, "rng_reseed_record", where,
               "reseed metadata differs from the documented future RNG transform",
               field=key, recorded=reseed.get(key), expected=value)
    _check(audit, reseed.get("environment_rng_action") == "preserved until a normal episode reset",
           "rng_no_environment_reset", where,
           "reseed metadata does not document preserving the active environment RNG")
    reconstructed = _reseeded_checkpoint_hash(prefix_checkpoint, prefix_simulator, expected_seed)
    _check(audit, branch.get("post_rng_transform_state_hash") == reconstructed,
           "post_rng_transform_hash", where,
           "deterministic RNG-only checkpoint transform does not match the recorded state hash",
           recorded=branch.get("post_rng_transform_state_hash"), actual=reconstructed)
    return expected_seed


def _check_action_install(audit: Audit, cohort: Path, branch_dir: Path,
                          branch: Mapping[str, Any], prefix: Mapping[str, Any],
                          config: Mapping[str, Any], foundation_bytes: int | None) -> tuple[dict[str, Any] | None,
                                                                                          tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    where = str(branch_dir)
    action = str(branch.get("action", ""))
    prefix_checkpoint = prefix["checkpoint"]
    prefix_arrays = prefix.get("arrays")
    record = branch.get("action_install", {})
    if prefix_arrays is None:
        audit.fail("action_prefix_arrays", where, "common-prefix actor arrays are unavailable")
        return None, None
    local, shared, physical = prefix_arrays
    tags = np.asarray(prefix_checkpoint.get("slot_group_tags", []), dtype=np.int8)
    source_fits = prefix.get("source_fits", {})
    slots = [int(row.get("slot", -1)) for row in source_fits.get("candidate_sources", [])]
    candidate = common_mean_effective(local, shared, physical, slots)
    incumbent_binding = int(prefix_checkpoint.get("parent_group", -999))
    expected_seed = int(branch.get("future_seed", -1))
    _check(audit, record.get("candidate_sha256") == source_fits.get("candidate_sha256"),
           "action_candidate_hash", where,
           "branch action did not use the common-prefix candidate")
    observation = np.asarray(record.get("fork_observation", []), dtype=np.float32)
    prefix_observation = np.asarray(prefix["simulator"].get("driver", {}).get("obs", []), dtype=np.float32)
    _check(audit, observation.shape == prefix_observation.shape and np.array_equal(observation, prefix_observation),
           "action_fork_observation", where,
           "action installation observation differs from the restored current episode state")
    pre_probs = np.asarray(record.get("pre_action_resident_probabilities", []), dtype=np.float64)
    prequant_probs = np.asarray(record.get("prequantized_resident_probabilities", []), dtype=np.float64)
    postquant_probs = np.asarray(record.get("postquantized_resident_probabilities", []), dtype=np.float64)
    for name, probabilities in (("pre_action", pre_probs), ("prequantized", prequant_probs),
                                ("postquantized", postquant_probs)):
        _check(audit, probabilities.shape == (2, 4) and np.isfinite(probabilities).all()
                    and np.all(probabilities >= 0.0)
                    and np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-6, rtol=0.0),
               "action_probability_snapshot", where,
               "saved resident behavior probabilities are malformed", stage=name,
               shape=list(probabilities.shape))
    try:
        derived_prequant_drift = storage_audit.runner._probability_drift(pre_probs, prequant_probs)
        derived_postquant_drift = storage_audit.runner._probability_drift(pre_probs, postquant_probs)
        _check(audit, storage_audit._close(derived_prequant_drift, record.get("prequantized_behavior_drift", {})),
               "prequantized_behavior_drift", where,
               "saved pre-codec behavior drift differs from its probability snapshots")
        _check(audit, storage_audit._close(derived_postquant_drift, record.get("postquantized_behavior_drift", {})),
               "postquantized_behavior_drift", where,
               "saved post-codec behavior drift differs from its probability snapshots")
    except Exception as exc:
        audit.fail("action_drift_audit", where, "behavior drift metadata could not be independently checked", error=repr(exc))
    try:
        expected_local, expected_shared, expected_groups, details = apply_retention_action(
            local, shared, physical, tags, incumbent_binding, candidate, 1, action,
        )
        _check(audit, storage_audit._close(details, record.get("details", {}), atol=1e-6, rtol=1e-6),
               "action_intervention_details", where,
               "saved generic retention action details differ from pure action replay",
               action=action)
        _check(audit, int(record.get("parent_group", -999)) == int(details["parent_group"]),
               "action_parent_binding", where,
               "recorded post-action logical binding differs from pure action replay",
               action=action, recorded=record.get("parent_group"), expected=details["parent_group"])
        actor = encode_actor(expected_local, expected_shared, expected_groups,
                             codec="int8", budget_bytes=int(config["actor_budget_bytes"]))
        decoded_local, decoded_shared, decoded_groups = decode_actor(actor.data)
    except Exception as exc:
        audit.fail("action_replay", where, "generic action or actor codec could not be independently replayed",
                   action=action, error=repr(exc))
        return None, None
    post_dir = branch_dir / "post_action"
    post_checkpoint = _load_checkpoint(audit, post_dir / "checkpoint.pt")
    if post_checkpoint is None:
        return None, None
    post_actor_path = post_dir / "actor.bin"
    post_sim_path = post_dir / "simulator.json"
    try:
        post_simulator = _json(post_sim_path)
        post_actor = post_actor_path.read_bytes()
    except Exception as exc:
        audit.fail("post_action_artifacts_read", str(post_dir), "post-action package is incomplete", error=repr(exc))
        return post_checkpoint, None
    actual_arrays = storage_audit._core_arrays(post_checkpoint)
    _check(audit, np.array_equal(actual_arrays[0], decoded_local)
                and np.array_equal(actual_arrays[1], decoded_shared)
                and np.array_equal(actual_arrays[2], decoded_groups)
                and post_actor == actor.data,
           "action_installed_actor", where,
           "post-action checkpoint and 89-byte actor do not equal the requested action after int8 encoding",
           action=action)
    if action == "keep":
        prefix_actor = (branch_dir.parent.parent / "prefix" / "actor.bin")
        _check(audit, bool(record.get("keep_avoided_extra_install_and_codec"))
                    and prefix_actor.is_file() and post_actor == prefix_actor.read_bytes(),
               "keep_no_codec", where,
               "keep must preserve the prefix actor bytes and skip an extra codec installation")
    else:
        _check(audit, record.get("keep_avoided_extra_install_and_codec") is False,
               "action_codec_flag", where, "non-keep action incorrectly claims to avoid codec installation")
    audit.check(bool(post_checkpoint.get("shared_frozen")), "post_action_parent_frozen", where,
                "shared parent was not frozen immediately after action installation")
    audit.check(post_checkpoint.get("foundation_sha256") == prefix_checkpoint.get("foundation_sha256"),
                "post_action_foundation", where, "W0 digest changed during action installation")
    audit.check(int(post_checkpoint.get("parent_group", -999)) == int(details["parent_group"]),
                "post_action_parent_group", where, "checkpoint action binding differs from replayed binding")
    audit.check(np.array_equal(np.asarray(post_checkpoint.get("slot_group_tags"), dtype=np.int8), tags),
                "post_action_logical_groups", where, "action changed logical resident group labels")
    actual_post_hash = semantic_hash({"learner": post_checkpoint, "simulator": post_simulator})
    audit.check(actual_post_hash == branch.get("post_action_state_hash"), "post_action_state_hash", where,
                "post-action checkpoint and simulator do not match their recorded semantic hash",
                recorded=branch.get("post_action_state_hash"), actual=actual_post_hash)
    artifacts = branch.get("post_action_artifacts", {})
    for key in ("checkpoint", "actor", "simulator"):
        artifact_path = _check_artifact_record(audit, cohort, artifacts.get(key, {}),
                                               f"{where}/post_action_artifacts/{key}")
        expected_path = (post_dir / {"checkpoint": "checkpoint.pt", "actor": "actor.bin",
                                     "simulator": "simulator.json"}[key]).resolve()
        audit.check(artifact_path is not None and artifact_path == expected_path,
                    "post_action_artifact_location", where,
                    "post-action artifact path does not identify the saved action-time package", artifact=key)
    post_capacity = _check_package(audit, post_dir, branch.get("capacity_after_action", {}),
                                   foundation_bytes, config, f"{where}/post_action")
    return post_checkpoint, actual_arrays


def _check_future_schedule_rows(audit: Audit, rows: Sequence[Mapping[str, str]],
                                schedule: Sequence[Mapping[str, Any]], *, action: str,
                                future: str, global_start: int, where: str) -> tuple[int, float]:
    """Check branch transitions; reset streams are checked with prefix continuity separately."""
    expected_count = sum(int(row["steps"]) for row in schedule)
    _check(audit, len(rows) == expected_count, "branch_raw_step_count", where,
           "raw future rows differ from the single continuation budget",
           actual=len(rows), expected=expected_count)
    total_reward = 0.0
    bout_starts: list[int] = []
    cursor = 0
    for item in schedule:
        bout_starts.append(cursor)
        cursor += int(item["steps"])
    for index, row in enumerate(rows):
        spec_index = next((j for j, start in enumerate(bout_starts)
                           if index < start + int(schedule[j]["steps"])), len(schedule) - 1)
        item = schedule[spec_index]
        within = index - bout_starts[spec_index] + 1
        try:
            global_step = int(row["global_step"])
            bout, bout_step = int(row["bout"]), int(row["bout_step"])
            group, action_id = int(row["group"]), int(row["action_id"])
            reward = float(row["reward"])
            probabilities = np.asarray(json.loads(row["probabilities"]), dtype=np.float64)
            uniform = float(row["action_uniform"])
            log_probability = float(row["behavior_log_prob"])
            task = str(row["taskobserver"])
        except Exception as exc:
            audit.fail("branch_raw_step_parse", where,
                       "raw future row has malformed counters, reward, or probabilities", row=index, error=repr(exc))
            continue
        _check(audit, row.get("phase") == "future" and row.get("action") == action
                    and row.get("future_regime") == future,
               "branch_raw_labels", where, "raw row phase, action, or future differs from the branch",
               row=index, phase=row.get("phase"), action=row.get("action"), future=row.get("future_regime"))
        _check(audit, global_step == global_start + index + 1,
               "branch_global_step", where, "raw branch global step has a gap or duplicate",
               row=index, actual=global_step, expected=global_start + index + 1)
        _check(audit, bout == int(item["bout"]) and bout_step == within
                    and group == int(item["group"]) and task == str(item["task"]),
               "branch_raw_schedule", where,
               "raw task, group, bout, or within-bout position differs from the frozen continuation",
               row=index, actual={"bout": bout, "bout_step": bout_step, "group": group, "task": task},
               expected={"bout": item["bout"], "bout_step": within, "group": item["group"], "task": item["task"]})
        _check(audit, probabilities.shape == (4,) and np.isfinite(probabilities).all()
                    and np.all(probabilities >= 0.0) and math.isclose(float(probabilities.sum()), 1.0, abs_tol=2e-6),
               "branch_behavior_probabilities", where, "raw action probabilities are malformed",
               row=index, probabilities=probabilities.tolist())
        _check(audit, 0.0 <= uniform < 1.0, "branch_action_uniform", where,
               "raw action uniform is outside the half-open unit interval", row=index, uniform=uniform)
        if probabilities.shape == (4,) and 0 <= action_id < 4:
            selected = min(int(np.searchsorted(np.cumsum(probabilities), uniform, side="right")), 3)
            _check(audit, selected == action_id, "branch_action_uniform_usage", where,
                   "recorded action differs from the probability vector and saved uniform",
                   row=index, recorded=action_id, reconstructed=selected)
            expected_log = math.log(max(float(probabilities[action_id]), 1e-12))
            _check(audit, math.isclose(log_probability, expected_log, abs_tol=2e-6, rel_tol=2e-6),
                   "branch_behavior_log_probability", where,
                   "behavior log probability differs from the selected action probability",
                   row=index, recorded=log_probability, expected=expected_log)
        environment_done = bool(int(row.get("environment_done", "0")))
        terminated = bool(int(row.get("terminated", "0")))
        truncated = bool(int(row.get("truncated", "0")))
        _check(audit, environment_done == (terminated or truncated), "branch_environment_done", where,
               "environment_done must match termination or truncation", row=index)
        boundary = bool(int(row.get("learning_boundary", "0")))
        _check(audit, boundary == (within == int(item["steps"])), "branch_learning_boundary", where,
               "learning boundary must occur exactly at each scheduled bout end", row=index,
               actual=boundary, expected=within == int(item["steps"]))
        _check(audit, math.isfinite(reward), "branch_reward_finite", where,
               "raw reward must be finite", row=index, reward=reward)
        total_reward += reward
    return len(rows), total_reward


def _check_branch_costs(audit: Audit, costs: Mapping[str, Any], where: str) -> None:
    cpu_parts = ("setup_cpu_s", "install_cpu_s", "simulation_cpu_s", "serialization_cpu_s")
    wall_parts = ("setup_wall_s", "install_wall_s", "simulation_wall_s", "serialization_wall_s")
    for key in cpu_parts + wall_parts + ("branch_cpu_s", "branch_wall_s"):
        try:
            value = float(costs[key])
            _check(audit, math.isfinite(value) and value >= 0.0, "branch_cost_finite", where,
                   "measured CPU or wall cost must be finite and nonnegative", field=key, value=value)
        except Exception:
            audit.fail("branch_cost_missing", where, "branch timing field is missing or malformed", field=key)
    for mode, parts, total_key, residual_key in (("CPU", cpu_parts, "branch_cpu_s", "unaccounted_cpu_s"),
                                                  ("wall", wall_parts, "branch_wall_s", "unaccounted_wall_s")):
        try:
            total = float(costs[total_key])
            accounted = sum(float(costs[key]) for key in parts)
            residual = float(costs[residual_key])
            expected_residual = max(0.0, total - accounted)
            _check(audit, accounted <= total + 1e-5 and math.isclose(residual, expected_residual,
                                                                     abs_tol=1e-5, rel_tol=1e-6),
                   "branch_cost_partition", where,
                   f"{mode} setup, install, simulation, serialization, and unaccounted time do not reconcile",
                   total=total, accounted=accounted, recorded_residual=residual,
                   expected_residual=expected_residual)
        except Exception as exc:
            audit.fail("branch_cost_partition_parse", where, f"{mode} cost partition is malformed", error=repr(exc))


def _check_branch_bout_rewards(audit: Audit, rows: Sequence[Mapping[str, str]],
                               bouts: Sequence[Mapping[str, Any]], where: str) -> None:
    rewards: dict[int, float] = {}
    steps: dict[int, int] = {}
    groups: dict[int, int] = {}
    for row in rows:
        try:
            bout = int(row["bout"])
            rewards[bout] = rewards.get(bout, 0.0) + float(row["reward"])
            steps[bout] = steps.get(bout, 0) + 1
            groups[bout] = int(row["group"])
        except Exception:
            continue
    _check(audit, set(rewards) == {int(row.get("bout", -1)) for row in bouts},
           "branch_bout_reward_universe", where,
           "raw branch bout IDs differ from saved bout summaries")
    for row in bouts:
        try:
            bout = int(row["bout"])
            _check(audit, steps[bout] == int(row["steps"])
                        and groups[bout] == int(row["group"])
                        and math.isclose(rewards[bout], float(row["return_full"]),
                                         abs_tol=1e-6, rel_tol=1e-8),
                   "branch_bout_reward", where,
                   "saved bout reward, group, or steps differ from raw transitions",
                   bout=bout, raw_steps=steps[bout], summary_steps=row.get("steps"),
                   raw_reward=rewards[bout], summary_reward=row.get("return_full"),
                   raw_group=groups[bout], summary_group=row.get("group"))
        except Exception as exc:
            audit.fail("branch_bout_summary_parse", where,
                       "saved branch bout reward record is malformed", error=repr(exc))


def _audit_branch(audit: Audit, cohort: Path, case_dir: Path, case_record: Mapping[str, Any],
                  branch_record: Mapping[str, Any], prefix: Mapping[str, Any], config: Mapping[str, Any],
                  foundation_bytes: int | None, expected: Mapping[str, int]) -> dict[str, Any]:
    case_id = str(case_record["case_id"])
    branch_id = str(branch_record.get("branch_id", ""))
    branch_dir = case_dir / "branches" / branch_id
    where = str(branch_dir)
    start_errors = len(audit.errors)
    audit.check(branch_record.get("case_id") == case_id, "branch_case_id", where,
                "branch case identity differs from its containing case", recorded=branch_record.get("case_id"),
                expected=case_id)
    action = str(branch_record.get("action", ""))
    future = str(branch_record.get("future_regime", ""))
    replicate = int(branch_record.get("replicate", -1))
    expected_id = f"{future}__r{replicate}__{action}"
    audit.check(branch_id == expected_id, "branch_id_encoding", where,
                "branch identifier does not match its future, replica, and action", expected=expected_id)
    audit.check(branch_record.get("format") == "retention_mechanism_branch_v1",
                "branch_summary_format", where, "branch summary format is unsupported")
    if branch_record.get("errors"):
        audit.fail("branch_runner_error", where, "runner recorded a failed branch",
                   errors=branch_record.get("errors"))
    audit.check(branch_record.get("prefix_restore_verified") is True
                and branch_record.get("restored_state_hash") == prefix.get("prefix_hash")
                and branch_record.get("prefix_state_hash") == prefix.get("prefix_hash"),
                "prefix_restore", where,
                "branch did not restore the exact common learner and simulator state before reseeding",
                restored=branch_record.get("restored_state_hash"), expected=prefix.get("prefix_hash"))
    future_value = _check_rng_transform(audit, branch_record, prefix["checkpoint"],
                                        prefix["simulator"], where, config)
    post_checkpoint, post_arrays = _check_action_install(audit, cohort, branch_dir, branch_record,
                                                         prefix, config, foundation_bytes)
    branch_schedule = future_schedule(case_record["mapping"], future, config,
                                      start_bout=len(prefix["schedule"]))
    recorded_schedule = branch_record.get("future_schedule", [])
    audit.check(storage_audit._close(recorded_schedule, branch_schedule, atol=0.0, rtol=0.0),
                "future_schedule", where,
                "branch future schedule differs from the frozen block template")
    raw_rows = _csv_rows(audit, branch_dir / "steps.csv")
    raw_ledgers = _check_raw_ledger_record(audit, cohort, branch_dir,
                                           branch_record.get("raw_ledgers", {}), where)
    raw_steps, raw_reward = _check_future_schedule_rows(
        audit, raw_rows, branch_schedule, action=action, future=future,
        global_start=expected["prefix_steps_per_case"], where=str(branch_dir / "steps.csv"),
    )
    expected_steps = expected["branch_steps_per_branch"]
    audit.check(raw_steps == expected_steps == int(branch_record.get("raw_environment_steps", -1)),
                "branch_raw_budget", where,
                "branch raw transitions differ from the frozen continuation budget",
                raw_steps=raw_steps, recorded=branch_record.get("raw_environment_steps"), expected=expected_steps)
    audit.check(math.isclose(raw_reward, float(branch_record.get("reward_total", raw_reward)),
                             abs_tol=1e-6, rel_tol=1e-8),
                "branch_reward_total", where, "branch summary reward differs from raw transitions",
                raw_reward=raw_reward, summary_reward=branch_record.get("reward_total"))
    _check_branch_bout_rewards(audit, raw_rows, branch_record.get("per_bout", []), where)
    _check_raw_bout_metadata(audit, raw_rows, branch_record.get("per_bout", []),
                             str(branch_dir / "steps.csv"))
    rollouts = storage_audit._check_rollouts(
        audit, branch_dir, phase="future", expected_rows=raw_steps,
        summary={**branch_record, "rollout_steps": int(config["rollout_steps"]),
                 "counts": {"environment_steps": raw_steps,
                            "rollouts_trained": raw_ledgers["rollouts_csv"]["rows"],
                            "minibatch_updates": raw_ledgers["rollouts_csv"]["updates"]}}, where=where,
    )
    horizon_data = horizon_totals(raw_rows, config["horizons"])
    horizon_rows = {str(int(row.get("steps", -1))): row for row in branch_record.get("horizons", [])}
    audit.check(set(horizon_rows) == set(horizon_data), "horizon_universe", where,
                "branch summary does not report each frozen nested horizon",
                actual=sorted(horizon_rows), expected=sorted(horizon_data))
    rollout_rows = _csv_rows(audit, branch_dir / "rollouts.csv")
    for horizon, expected_horizon in horizon_data.items():
        row = horizon_rows.get(horizon, {})
        _check(audit, int(row.get("raw_environment_steps", -1)) == int(horizon)
                    and int(row.get("steps", -1)) == int(horizon)
                    and math.isclose(float(row.get("reward_sum", float("nan"))), expected_horizon["reward_total"],
                                     abs_tol=1e-6, rel_tol=1e-8)
                    and math.isclose(float(row.get("reward_per_step", float("nan"))), expected_horizon["reward_per_step"],
                                     abs_tol=1e-7, rel_tol=1e-7),
               "nested_horizon_recount", where,
               "horizon reward or step count differs from the one chronological branch ledger",
               horizon=horizon, recorded=row, actual=expected_horizon)
        prefix_rollouts = int(horizon) // int(config["rollout_steps"])
        seen_rows = trained = updates = 0
        for rollout in rollout_rows:
            count = int(rollout["rows"])
            if seen_rows + count > int(horizon):
                break
            seen_rows += count
            trained += 1
            updates += int(rollout["updates"])
        _check(audit, seen_rows == int(horizon) and trained == prefix_rollouts
                    and int(row.get("rollouts_trained", -1)) == trained
                    and int(row.get("minibatch_updates", -1)) == updates,
               "horizon_ppo_counts", where,
               "horizon PPO rollout/update totals do not match the chronological rollout ledger",
               horizon=horizon, raw_rollout_rows=seen_rows, expected_rollout_rows=horizon,
               recorded_rollouts=row.get("rollouts_trained"), actual_rollouts=trained,
               recorded_updates=row.get("minibatch_updates"), actual_updates=updates)
        audit.check(math.isclose(float(row.get("action_install_cpu_s", float("nan"))),
                                 float(branch_record.get("costs", {}).get("install_cpu_s", float("nan"))),
                                 abs_tol=1e-8, rel_tol=1e-6),
                    "horizon_install_cost", where,
                    "horizon install CPU does not match the one action installation", horizon=horizon)
    counts = branch_record.get("counts_delta")
    if isinstance(counts, Mapping):
        expected_counts = {
            "raw_steps": raw_steps, "environment_steps": raw_steps, "fit_steps": raw_steps,
            "rollouts_trained": rollouts["rollouts"], "minibatch_updates": rollouts["updates"],
            "optimizer_step_count": rollouts["updates"], "completed_episodes": raw_ledgers["episodes_csv"]["rows"],
            "slot_allocations": sum(bool(row.get("allocated", False)) for row in branch_record.get("per_bout", [])),
            "slot_evictions": sum(row.get("evicted_task") is not None for row in branch_record.get("per_bout", [])),
            "partial_rollout_rows": 0,
        }
        for key, value in expected_counts.items():
            _check(audit, int(counts.get(key, -1)) == value, "branch_count_delta", where,
                   "learner count delta differs from raw interaction, episode, or rollout ledgers",
                   field=key, recorded=counts.get(key), expected=value)
        runtime = counts.get("runtime_counts", {})
        for key, value in (("environment_steps", raw_steps), ("rollouts_trained", rollouts["rollouts"]),
                           ("minibatch_updates", rollouts["updates"]),
                           ("completed_episodes", raw_ledgers["episodes_csv"]["rows"])):
            _check(audit, int(runtime.get(key, -1)) == value, "branch_runtime_count_delta", where,
                   "runtime count delta differs from raw ledgers", field=key,
                   recorded=runtime.get(key), expected=value)
    else:
        audit.fail("branch_count_delta_missing", where, "branch did not save learner/runtime counter deltas")

    final_checkpoint: dict[str, Any] | None = None
    final_simulator: dict[str, Any] | None = None
    final_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    artifacts = branch_record.get("final_artifacts", {})
    for key, filename in (("checkpoint", "checkpoint.pt"), ("actor", "actor.bin"), ("simulator", "simulator.json")):
        path = _check_artifact_record(audit, cohort, artifacts.get(key, {}), f"{where}/final_artifacts/{key}")
        expected_path = (branch_dir / filename).resolve()
        audit.check(path is not None and path == expected_path, "final_artifact_location", where,
                    "final package artifact path does not identify the branch's final file", artifact=key)
    final_checkpoint = _load_checkpoint(audit, branch_dir / "checkpoint.pt")
    try:
        final_simulator = _json(branch_dir / "simulator.json")
    except Exception as exc:
        audit.fail("final_simulator_read", where, "final branch simulator snapshot is unreadable", error=repr(exc))
    if final_checkpoint is not None and final_simulator is not None:
        final_arrays = _check_actor(audit, final_checkpoint, branch_dir / "actor.bin", config, where)
        _check_json_state_hash(audit, final_checkpoint, final_simulator,
                               branch_record.get("final_state_hash"), where, "final_state_hash")
        audit.check(final_checkpoint.get("foundation_sha256") == prefix["checkpoint"].get("foundation_sha256"),
                    "final_foundation_hash", where, "W0 changed during branch PPO continuation")
        _check_package(audit, branch_dir, branch_record.get("capacity", {}), foundation_bytes, config, where)
        invariant = branch_record.get("shared_parent_invariant", {})
        if post_arrays is not None and final_arrays is not None:
            drift = float(np.max(np.abs(post_arrays[1] - final_arrays[1]), initial=0.0))
            _check(audit, drift == 0.0 and invariant.get("exactly_fixed") is True
                        and math.isclose(float(invariant.get("max_coefficient_drift", float("nan"))), 0.0,
                                         abs_tol=0.0, rel_tol=0.0),
                   "frozen_parent_invariant", where,
                   "shared parent changed after action installation during local PPO continuation",
                   actual_drift=drift, recorded=invariant)
        action_parent = {"keep": int(prefix["checkpoint"].get("parent_group", -1)),
                         "replace": 1, "mean": -2, "none": -1}.get(action)
        if action_parent is not None:
            try:
                reconstructed_cache = storage_audit._check_cache_and_inheritance(
                    audit, branch_record.get("per_bout", []), prefix["checkpoint"]["cache"],
                    prefix["checkpoint"]["slot_generation"],
                    post_arrays[2] if post_arrays is not None else storage_audit._core_arrays(post_checkpoint)[2],
                    prefix["checkpoint"]["slot_group_tags"], parent_group=action_parent,
                    config=config, where=where,
                )
                expected_cache, expected_generation, expected_physical, expected_logical = reconstructed_cache
                final_cache = final_checkpoint["cache"]
                fields = ("capacity", "eviction", "task_to_slot", "slot_to_task", "last_used", "frequency", "clock")
                _check(audit, all(storage_audit._close(expected_cache.get(key), final_cache.get(key), atol=0.0, rtol=0.0)
                                  for key in fields)
                            and [int(value) for value in final_checkpoint["slot_generation"]] == expected_generation
                            and [int(value) for value in final_checkpoint["slot_group_tags"]] == expected_logical
                            and np.array_equal(final_arrays[2], np.asarray(expected_physical, dtype=np.int8)),
                       "final_lru_reconstruction", where,
                       "final natural-LRU resident slots, generations, or parent bindings differ from bout ledger")
            except Exception as exc:
                audit.fail("lru_reconstruction_error", where,
                           "branch resident cache could not be reconstructed from the raw bout schedule", error=repr(exc))
        episodes = _check_episode_ledger(audit, branch_dir, raw_rows, final_simulator,
                                         seed=int(branch_record.get("seed", -1)), agent_seed=future_value,
                                         start_simulator=prefix["simulator"], action=action, future=future,
                                         where=where)
    else:
        episodes = {"completed_episodes": raw_ledgers["episodes_csv"]["rows"]}
    _check_branch_costs(audit, branch_record.get("costs", {}), where)
    del raw_rows, rollout_rows
    return {"case_id": case_id, "branch_id": branch_id,
            "environment_steps": raw_steps, "reward_total": raw_reward,
            "rollouts": rollouts["rollouts"], "minibatch_updates": rollouts["updates"],
            "errors": len(audit.errors) - start_errors, "episodes": episodes}


def _partial_ledger_counts(audit: Audit, folder: Path, where: str) -> dict[str, Any]:
    counts = {"environment_steps": 0, "reward_total": 0.0, "rollouts": 0,
              "minibatch_updates": 0, "episodes": 0}
    for name, key in (("steps.csv", "environment_steps"), ("rollouts.csv", "rollouts.csv"),
                      ("episodes.csv", "episodes")):
        path = folder / name
        if not path.is_file():
            continue
        rows = _csv_rows(audit, path, required=False)
        if name == "steps.csv":
            counts["environment_steps"] = len(rows)
            try:
                counts["reward_total"] = sum(float(row["reward"]) for row in rows)
            except Exception as exc:
                audit.fail("partial_reward_parse", str(path), "partial raw rewards could not be summed", error=repr(exc))
        elif name == "rollouts.csv":
            counts["rollouts"] = len(rows)
            for index, row in enumerate(rows):
                try:
                    counts["minibatch_updates"] += int(row["updates"])
                except Exception as exc:
                    audit.fail("partial_rollout_parse", str(path), "partial rollout row is malformed",
                               row=index, error=repr(exc))
        else:
            counts["episodes"] = len(rows)
    return counts


def _audit_failed_case_ledgers(audit: Audit, cohort: Path, case_dir: Path,
                               case_record: Mapping[str, Any], where: str) -> dict[str, Any]:
    """Count persisted interactions from a preserved partial run without treating it as evidence."""
    result = {"environment_steps": 0, "reward_total": 0.0, "rollouts": 0,
              "minibatch_updates": 0, "episodes": 0, "branches": []}
    prefix = _partial_ledger_counts(audit, case_dir / "prefix", str(case_dir / "prefix"))
    result.update(prefix)
    declared = None
    for error in case_record.get("errors", []):
        if isinstance(error.get("persisted_interactions"), Mapping):
            declared = error["persisted_interactions"]
            break
    if declared is not None:
        for key, actual in (("prefix_environment_steps", prefix["environment_steps"]),
                            ("prefix_rollouts", prefix["rollouts"]),
                            ("prefix_minibatch_updates", prefix["minibatch_updates"])):
            audit.check(int(declared.get(key, -1)) == actual, "partial_persisted_count", where,
                        "runner failure ledger differs from physical prefix CSV rows",
                        field=key, recorded=declared.get(key), actual=actual)
    branch_root = case_dir / "branches"
    if branch_root.is_dir():
        for branch_dir in sorted(path for path in branch_root.iterdir() if path.is_dir()):
            counts = _partial_ledger_counts(audit, branch_dir, str(branch_dir))
            result["environment_steps"] += counts["environment_steps"]
            result["reward_total"] += counts["reward_total"]
            result["rollouts"] += counts["rollouts"]
            result["minibatch_updates"] += counts["minibatch_updates"]
            result["episodes"] += counts["episodes"]
            result["branches"].append({"branch_id": branch_dir.name, **counts})
    case_errors = case_record.get("errors", [])
    if case_errors or len(case_record.get("branches", [])) == 0:
        audit.fail("incomplete_case_preserved", where,
                   "case failed before completing its frozen branch grid; persisted work is counted separately",
                   errors=case_errors, persisted=result)
    return result


def audit_cohort(cohort: Path) -> dict[str, Any]:
    """Audit one saved pilot or main cohort without making environment calls."""
    cohort = cohort.resolve()
    audit = Audit(cohort)
    summary_path = cohort / "summary.json"
    if not cohort.is_dir():
        audit.fail("cohort_missing", str(cohort), "cohort directory does not exist")
        return {"format": FORMAT, "path": str(cohort), "passed": False,
                "errors": audit.errors, "warnings": audit.warnings}
    try:
        summary = _json(summary_path)
    except Exception as exc:
        audit.fail("cohort_summary_missing", str(summary_path), "cohort summary is missing or unreadable", error=repr(exc))
        return {"format": FORMAT, "path": str(cohort), "passed": False,
                "errors": audit.errors, "warnings": audit.warnings}
    phase = str(summary.get("phase", ""))
    audit.check(summary.get("format") == SUMMARY_FORMAT, "cohort_summary_format", str(summary_path),
                "runner summary format is unsupported")
    expected = _check_config(audit, summary.get("config", {}), phase)
    foundation_bytes = _check_source_snapshot(audit, cohort, summary)
    seeds, pairs, histories, futures, replicates = _selected_grid(summary.get("config", {}), phase)
    actions = [str(value) for value in summary.get("config", {}).get("actions", [])]
    expected_cases = {
        f"s{seed}_{pair}_{history}" for seed in seeds for pair in pairs for history in histories
    }
    case_records = summary.get("cases", [])
    ids = [str(row.get("case_id", "")) for row in case_records]
    audit.check(len(ids) == len(set(ids)) and set(ids) == expected_cases,
                "case_universe", str(summary_path),
                "case records differ from the frozen case grid", actual=sorted(ids), expected=sorted(expected_cases))
    audit.check(int(summary.get("case_count", -1)) == len(expected_cases), "case_count", str(summary_path),
                "saved case count differs from the frozen grid",
                recorded=summary.get("case_count"), expected=len(expected_cases))
    case_by_id = {str(row.get("case_id")): row for row in case_records}
    for case_id, record in case_by_id.items():
        seed = int(record.get("seed", -1))
        pair = str(record.get("pair_kind", ""))
        history = str(record.get("history_regime", ""))
        mapping = pair_mapping(seed, pair)
        audit.check(record.get("mapping") == mapping, "case_mapping", case_id,
                    "case task-family mapping differs from its deterministic seed mapping",
                    recorded=record.get("mapping"), expected=mapping)
        audit.check(seed in seeds and pair in pairs and history in histories,
                    "case_stratum", case_id, "case seed, relation, or history is outside the frozen grid")
        if phase == "main":
            _check_seed_strata(audit, seeds, pair)

    nested_branches: dict[tuple[str, str], Mapping[str, Any]] = {}
    for case_id, record in case_by_id.items():
        for branch in record.get("branches", []):
            key = (case_id, str(branch.get("branch_id", "")))
            audit.check(key not in nested_branches, "duplicate_case_branch", case_id,
                        "case repeats a branch identifier", branch_id=key[1])
            nested_branches[key] = branch
    flat_branches: dict[tuple[str, str], Mapping[str, Any]] = {}
    for branch in summary.get("branches", []):
        key = (str(branch.get("case_id", "")), str(branch.get("branch_id", "")))
        audit.check(key not in flat_branches, "duplicate_flat_branch", str(summary_path),
                    "flattened summary repeats a case-scoped branch identifier", case_id=key[0], branch_id=key[1])
        flat_branches[key] = branch
    audit.check(set(flat_branches) == set(nested_branches), "flat_branch_index", str(summary_path),
                "flattened branch index differs from case-local branch summaries",
                missing=sorted(set(nested_branches) - set(flat_branches)),
                extra=sorted(set(flat_branches) - set(nested_branches)))
    for key in set(flat_branches) & set(nested_branches):
        audit.check(storage_audit._close(flat_branches[key], nested_branches[key], atol=0.0, rtol=0.0),
                    "flat_branch_copy", str(summary_path),
                    "flattened branch record differs from its case-local summary",
                    case_id=key[0], branch_id=key[1])
    expected_keys = {
        (case_id, f"{future}__r{replicate}__{action}")
        for case_id in expected_cases for future in futures for replicate in replicates for action in actions
    }
    audit.check(set(nested_branches) == expected_keys, "branch_universe", str(summary_path),
                "branch summaries differ from the frozen case/future/replica/action grid",
                missing=sorted(expected_keys - set(nested_branches)),
                extra=sorted(set(nested_branches) - expected_keys))
    audit.check(len(summary.get("errors", [])) == 0, "runner_errors", str(summary_path),
                "runner summary records failed or incomplete work", errors=summary.get("errors", []))
    for error in summary.get("errors", []):
        audit.fail("runner_error", str(summary_path), "runner failure is retained in the audit",
                   error_record=error)

    case_results: list[dict[str, Any]] = []
    branch_results: list[dict[str, Any]] = []
    actual_prefix_steps = actual_branch_steps = 0
    actual_prefix_rollouts = actual_branch_rollouts = 0
    actual_prefix_updates = actual_branch_updates = 0
    actual_total_reward = 0.0
    for case_id in sorted(expected_cases):
        record = case_by_id.get(case_id)
        case_dir = cohort / "cases" / case_id
        if record is None:
            audit.fail("case_record_missing", case_id, "expected case has no summary record")
            continue
        case_error_count_before = len(audit.errors)
        prefix_summary_path = case_dir / "prefix" / "summary.json"
        if not prefix_summary_path.is_file():
            partial = _audit_failed_case_ledgers(audit, cohort, case_dir, record, case_id)
            prefix_steps = min(partial["environment_steps"], expected["prefix_steps_per_case"])
            branch_steps = max(0, partial["environment_steps"] - prefix_steps)
            prefix_rollouts = min(partial["rollouts"], prefix_steps // int(summary["config"]["rollout_steps"]))
            branch_rollouts = max(0, partial["rollouts"] - prefix_rollouts)
            prefix_updates = min(partial["minibatch_updates"], prefix_rollouts * 10**9)
            branch_updates = max(0, partial["minibatch_updates"] - prefix_updates)
            actual_prefix_steps += prefix_steps
            actual_branch_steps += branch_steps
            actual_prefix_rollouts += prefix_rollouts
            actual_branch_rollouts += branch_rollouts
            actual_prefix_updates += prefix_updates
            actual_branch_updates += branch_updates
            actual_total_reward += partial["reward_total"]
            case_results.append({"case_id": case_id, "passed": False, "partial": True,
                                 "environment_steps": partial["environment_steps"],
                                 "rollouts": partial["rollouts"],
                                 "minibatch_updates": partial["minibatch_updates"],
                                 "errors": len(audit.errors) - case_error_count_before})
            continue
        prefix = _check_prefix_state(audit, cohort, case_dir, record, summary["config"],
                                     foundation_bytes, expected)
        if prefix is None:
            audit.fail("prefix_audit_failed", case_id, "prefix could not be fully audited")
            continue
        prefix_steps = int(prefix["summary"].get("steps", 0))
        prefix_rollouts = int(prefix["rollouts"]["rollouts"])
        prefix_updates = int(prefix["rollouts"]["updates"])
        actual_prefix_steps += prefix_steps
        actual_prefix_rollouts += prefix_rollouts
        actual_prefix_updates += prefix_updates
        actual_total_reward += float(prefix["summary"].get("reward_total", 0.0))
        execution_order = _check_execution_order(audit, case_dir, record, summary["config"])
        case_keys = [(case_id, branch_id) for branch_id in execution_order]
        for key in case_keys:
            branch = nested_branches.get(key)
            if branch is None:
                continue
            result = _audit_branch(audit, cohort, case_dir, record, branch, prefix,
                                   summary["config"], foundation_bytes, expected)
            branch_results.append(result)
            actual_branch_steps += int(result["environment_steps"])
            actual_branch_rollouts += int(result["rollouts"])
            actual_branch_updates += int(result["minibatch_updates"])
            actual_total_reward += float(result["reward_total"])
        case_results.append({"case_id": case_id,
                             "passed": len(audit.errors) == case_error_count_before,
                             "partial": False, "prefix_environment_steps": prefix_steps,
                             "branch_environment_steps": sum(r["environment_steps"] for r in branch_results
                                                             if r["case_id"] == case_id),
                             "branch_count_audited": sum(r["case_id"] == case_id for r in branch_results),
                             "errors": len(audit.errors) - case_error_count_before})
    expected_steps = int(expected["environment_steps_total"])
    actual_steps = actual_prefix_steps + actual_branch_steps
    audit.check(actual_prefix_steps == expected["prefix_environment_steps"], "cohort_prefix_steps",
                str(summary_path), "independently counted prefix rows differ from the frozen budget",
                actual=actual_prefix_steps, expected=expected["prefix_environment_steps"])
    audit.check(actual_branch_steps == expected["branch_environment_steps"], "cohort_branch_steps",
                str(summary_path), "independently counted branch rows differ from the frozen budget",
                actual=actual_branch_steps, expected=expected["branch_environment_steps"])
    audit.check(actual_steps == expected_steps, "cohort_total_steps", str(summary_path),
                "independently counted raw rows differ from the frozen campaign budget",
                actual=actual_steps, expected=expected_steps)
    campaign_cost = summary.get("costs", {})
    for key, value in (("prefix_environment_steps", actual_prefix_steps),
                       ("branch_environment_steps", actual_branch_steps),
                       ("environment_steps_total", actual_steps)):
        audit.check(int(campaign_cost.get(key, -1)) == value, "campaign_count_record", str(summary_path),
                    "campaign count differs from physical raw CSV row counts", field=key,
                    recorded=campaign_cost.get(key), actual=value)
    expected_rollouts = expected_steps // int(summary["config"]["rollout_steps"])
    actual_rollouts = actual_prefix_rollouts + actual_branch_rollouts
    actual_updates = actual_prefix_updates + actual_branch_updates
    audit.check(actual_rollouts == expected_rollouts, "campaign_rollout_count", str(summary_path),
                "PPO rollout count differs from each actual interaction being trained once",
                actual=actual_rollouts, expected=expected_rollouts)
    return {
        "format": FORMAT,
        "path": str(cohort),
        "phase": phase,
        "passed": len(audit.errors) == 0,
        "readiness": "ready" if len(audit.errors) == 0 else "blocked",
        "scope": "saved-file integrity audit; no environment calls",
        "counts": {
            "expected_cases": expected["case_count"],
            "observed_cases": len(ids),
            "expected_branches": expected["branch_count"],
            "observed_branches_case_scoped": len(nested_branches),
            "prefix_environment_steps": actual_prefix_steps,
            "branch_environment_steps": actual_branch_steps,
            "environment_steps_total": actual_steps,
            "expected_environment_steps_total": expected_steps,
            "prefix_rollouts": actual_prefix_rollouts,
            "branch_rollouts": actual_branch_rollouts,
            "rollouts_total": actual_rollouts,
            "minibatch_updates_total": actual_updates,
            "reward_total_all_phases": actual_total_reward,
        },
        "cases": case_results,
        "branches": branch_results,
        "observations": audit.observations,
        "errors": audit.errors,
        "warnings": audit.warnings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path, help="saved retention-mechanism cohort directory")
    parser.add_argument("--output", required=True, type=Path, help="path for the JSON audit result")
    arguments = parser.parse_args(argv)
    result = audit_cohort(arguments.cohort)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(arguments.output), "cohort": str(arguments.cohort),
                      "phase": result.get("phase"), "passed": result.get("passed"),
                      "counts": result.get("counts"), "errors": len(result.get("errors", []))}, indent=2))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
