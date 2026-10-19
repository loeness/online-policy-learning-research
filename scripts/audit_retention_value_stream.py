"""Read-only integrity audit for selected-feedback retention streams.

This audit consumes saved artifacts only. It does not create or step task
environments and does not rank policy returns. It checks source snapshots,
chronological selector inputs and feedback, the selected action's only factual
outcome, PPO/raw-step accounting, recurrence ordering, and complete checkpoint
sizes. In-flight pending feedback is reported separately from checkpoint
boundary package sizes; this is not an arbitrary-step recovery guarantee.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from online_rl.retention_stream_value import BoundedRetentionStats, FEATURE_NAMES, frequency_action
from online_rl.retention_value import RetentionValueModel
from online_rl.storage_decision import ACTIONS, pair_mapping, semantic_hash, task_name
from online_rl.write_decision_policy import decode_actor
from scripts import audit_source_followup as source_audit
from scripts import audit_storage_decision as storage_audit
from scripts import run_retention_value_stream as runner


FORMAT = "retention_value_stream_audit_v1"
SUMMARY_FORMAT = "retention_value_stream_summary_v1"
RUN_CONFIG_FORMAT = "retention_value_stream_run_config_v1"
PRIVATE_PARTS = {"agent.md", "agents.md", "tmp"}
REQUIRED_SOURCES = {
    "configs/retention_value_stream.json",
    "docs/RETENTION_VALUE_STREAM_PROTOCOL.md",
    "scripts/run_retention_value_stream.py",
    "src/online_rl/retention_value.py",
    "src/online_rl/retention_stream_value.py",
}
FORBIDDEN_OUTCOME_KEYS = {
    "action_returns", "returns_by_action", "counterfactual_returns",
    "counterfactual_outcomes", "outcomes_by_action", "utility_targets_by_action",
    "unselected_targets", "branch_returns", "observer_returns",
}
COHORT_COST_KEYS = (
    "environment_steps", "warmup_steps", "outcome_steps", "ppo_rollouts",
    "minibatch_updates", "storage_action_count", "parent_write_count",
    "eviction_count", "slot_allocation_count", "value_model_updates",
)
COHORT_FLOAT_COST_KEYS = (
    "utility_cpu_s", "model_update_cpu_s", "package_serialization_cpu_s",
    "total_cpu_s", "wall_s",
)


class Audit:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.protocol_snapshots: list[dict[str, Any]] = []

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


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_path(root: Path, name: str) -> Path | None:
    normalized = str(name).replace("\\", "/")
    relative = Path(normalized)
    if relative.is_absolute() or not relative.parts:
        return None
    if any(part in {"", ".", ".."} or ":" in part for part in relative.parts):
        return None
    path = (root / relative).resolve()
    base = root.resolve()
    if path != base and base not in path.parents:
        return None
    return path


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(audit: Audit, path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.is_file():
        if required:
            audit.fail("jsonl_missing", str(path), "required append-only event log is missing")
        return []
    try:
        raw = path.read_bytes()
        audit.check(not raw or raw.endswith(b"\n"), "jsonl_unflushed_tail", str(path),
                    "event log ends with a partial line; the final append was not fully flushed")
        rows: list[dict[str, Any]] = []
        for index, line in enumerate(raw.decode("utf-8").splitlines()):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {index + 1} is not a JSON object")
            rows.append(value)
        return rows
    except Exception as exc:
        audit.fail("jsonl_unreadable", str(path), "event log is malformed or unreadable", error=repr(exc))
        return []


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


def _finite(value: Any, *, nonnegative: bool = False) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and (number >= 0 if nonnegative else True)


def _close(actual: Any, expected: float, *, atol: float = 1e-8, rtol: float = 1e-8) -> bool:
    return _finite(actual) and math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol)


def _same_json(actual: Any, expected: Any) -> bool:
    try:
        return json.dumps(actual, sort_keys=True, separators=(",", ":"), default=runner._json_default) == \
            json.dumps(expected, sort_keys=True, separators=(",", ":"), default=runner._json_default)
    except Exception:
        return False


def _expected_grid(config: Mapping[str, Any], stage: str, pilot: bool) -> list[dict[str, Any]]:
    if stage not in ("calibration", "comparison"):
        return []
    section = config.get(stage)
    if not isinstance(section, Mapping):
        return []
    selected = section.get("pilot") if pilot else section
    if not isinstance(selected, Mapping):
        return []
    seeds = [int(value) for value in selected.get("seeds", [])]
    pair_kinds = [str(value) for value in selected.get("pair_kinds", config.get("pair_kinds", []))]
    methods = [str(value) for value in section.get("methods", [])]
    decisions = int(selected.get("decisions", 0))
    return [
        {"stage": stage, "pilot": bool(pilot), "pair_kind": pair,
         "seed": seed, "method": method, "decisions": decisions}
        for pair in pair_kinds for seed in seeds for method in methods
    ]


def _spec_key(value: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(value.get("pair_kind", "")), int(value.get("seed", -1)), str(value.get("method", ""))


def _check_source_snapshot(audit: Audit, stream_dir: Path,
                           frozen: Mapping[str, Any], digest_cache: dict[Path, str]) -> int | None:
    snapshot = stream_dir / "source_snapshot"
    manifest_path = snapshot / "manifest.json"
    try:
        manifest = _json(manifest_path)
    except Exception as exc:
        audit.fail("manifest_read", str(manifest_path), "source snapshot manifest is missing or unreadable", error=repr(exc))
        return None
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping):
        audit.fail("manifest_sources", str(manifest_path), "source manifest has no sources mapping")
        return None
    names = set(map(str, sources))
    audit.check(REQUIRED_SOURCES <= names, "required_sources", str(manifest_path),
                "source snapshot omits one or more frozen retention-value inputs",
                missing=sorted(REQUIRED_SOURCES - names))
    for name, item in sources.items():
        relative = Path(str(name).replace("\\", "/"))
        private = any(part.casefold() in PRIVATE_PARTS for part in relative.parts)
        audit.check(not private, "private_source_manifest", f"source_snapshot/{name}",
                    "source manifest lists local instructions or temporary material")
        copied = _safe_path(snapshot, str(name))
        if copied is None or not copied.is_file():
            audit.fail("source_copy_missing", f"source_snapshot/{name}",
                       "manifest-listed source copy is missing or escapes the snapshot")
            continue
        if not isinstance(item, Mapping):
            audit.fail("source_record_invalid", f"source_snapshot/{name}", "source digest record is malformed")
            continue
        try:
            actual_digest = digest_cache.setdefault(copied, _digest(copied))
            actual_bytes = copied.stat().st_size
            audit.check(actual_digest == item.get("sha256") and actual_bytes == item.get("bytes"),
                        "source_hash", f"source_snapshot/{name}",
                        "source copy differs from its recorded hash or byte size",
                        expected_sha256=item.get("sha256"), actual_sha256=actual_digest,
                        expected_bytes=item.get("bytes"), actual_bytes=actual_bytes)
        except Exception as exc:
            audit.fail("source_hash_error", f"source_snapshot/{name}", "source copy could not be hashed", error=repr(exc))
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot)
        audit.check(not any(part.casefold() in PRIVATE_PARTS for part in relative.parts),
                    "private_source_present", f"source_snapshot/{relative.as_posix()}",
                    "snapshot contains local instructions or temporary material")
    excluded = {str(value).casefold() for value in manifest.get("excluded_local_instructions", [])}
    audit.check(excluded == {"agent.md", "agents.md"}, "private_exclusions", str(manifest_path),
                "manifest does not record both local instruction files as excluded", excluded=sorted(excluded))
    try:
        private_matches = source_audit._private_hash_matches(snapshot)
    except Exception as exc:
        private_matches = [f"private hash check failed: {exc!r}"]
    audit.check(not private_matches, "private_exact_hash", str(snapshot),
                "snapshot contains bytes identical to a local instruction file", matches=private_matches)

    foundation = manifest.get("foundation", {})
    if not isinstance(foundation, Mapping):
        audit.fail("foundation_manifest", str(manifest_path), "foundation provenance is malformed")
        foundation_bytes = None
    else:
        foundation_path = Path(str(foundation.get("path", "")))
        try:
            if not foundation_path.is_file():
                raise FileNotFoundError(foundation_path)
            actual_sha = digest_cache.setdefault(foundation_path, _digest(foundation_path))
            foundation_bytes = foundation_path.stat().st_size
            audit.check(actual_sha == foundation.get("sha256") and foundation_bytes == foundation.get("bytes"),
                        "foundation_hash", str(foundation_path),
                        "foundation differs from source snapshot provenance",
                        expected_sha256=foundation.get("sha256"), actual_sha256=actual_sha,
                        expected_bytes=foundation.get("bytes"), actual_bytes=foundation_bytes)
        except Exception as exc:
            audit.fail("foundation_unavailable", str(foundation_path), "foundation file is unavailable or unreadable",
                       error=repr(exc))
            foundation_bytes = None
    dataset = manifest.get("dataset", {})
    if not isinstance(dataset, Mapping):
        audit.fail("dataset_manifest", str(manifest_path), "dataset provenance is malformed")
    else:
        dataset_path = Path(str(dataset.get("path", "")))
        try:
            if not dataset_path.is_file():
                raise FileNotFoundError(dataset_path)
            actual_sha = digest_cache.setdefault(dataset_path, _digest(dataset_path))
            audit.check(actual_sha == dataset.get("sha256") and dataset_path.stat().st_size == dataset.get("bytes"),
                        "dataset_hash", str(dataset_path), "dataset differs from source snapshot provenance")
        except Exception as exc:
            audit.fail("dataset_unavailable", str(dataset_path), "dataset file is unavailable or unreadable",
                       error=repr(exc))

    config_copy = snapshot / "configs" / "retention_value_stream.json"
    protocol_copy = snapshot / "docs" / "RETENTION_VALUE_STREAM_PROTOCOL.md"
    try:
        config_copy_value = _json(config_copy)
        audit.check(config_copy_value == frozen.get("config"), "frozen_config_copy", str(config_copy),
                    "source snapshot config differs from the frozen run config")
        audit.check(_digest(config_copy) == frozen.get("config_sha256"), "frozen_config_hash", str(config_copy),
                    "frozen config copy hash differs from its run record")
        protocol_hash = _digest(protocol_copy)
        protocol_bytes = protocol_copy.read_bytes()
        audit.protocol_snapshots.append({
            "path": str(protocol_copy), "sha256": protocol_hash, "bytes": protocol_bytes,
        })
        if protocol_hash != frozen.get("protocol_sha256"):
            audit.warn("protocol_snapshot_variant", str(protocol_copy),
                       "snapshot protocol hash differs from the cohort-level frozen protocol hash",
                       frozen_sha256=frozen.get("protocol_sha256"), actual_sha256=protocol_hash)
    except Exception as exc:
        audit.fail("frozen_source_missing", str(snapshot), "frozen config or protocol copy is missing/unreadable",
                   error=repr(exc))
    return foundation_bytes


def _read_checkpoint(audit: Audit, directory: Path, expected_sizes: Mapping[str, Any] | None,
                     foundation_bytes: int | None, config: Mapping[str, Any]) -> tuple[dict[str, Any] | None, int | None]:
    checkpoint_path = directory / "checkpoint.pt"
    actor_path = directory / "actor.bin"
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actor_data = actor_path.read_bytes()
    except Exception as exc:
        audit.fail("checkpoint_read", str(directory), "checkpoint or actor sidecar is missing/unreadable",
                   error=repr(exc))
        return None, None
    where = str(directory)
    audit.check(isinstance(payload, Mapping) and payload.get("format") == "retention_value_stream_checkpoint_v1",
                "checkpoint_format", str(checkpoint_path), "checkpoint format is incorrect")
    if not isinstance(payload, Mapping):
        return None, None
    learner = payload.get("learner")
    if not isinstance(learner, Mapping):
        audit.fail("checkpoint_learner", where, "checkpoint omits its learner state")
        return dict(payload), None
    required_top = {"retention_stats", "value_models", "selector_rng_states", "pending_feedback",
                    "candidate_coefficients_persisted", "simulator_state_persisted", "parent_metadata",
                    "decision_index_next", "last_candidate_provenance"}
    audit.check(required_top <= set(payload), "checkpoint_state_fields", where,
                "checkpoint omits a bounded model, statistics, RNG, or pending-state field",
                missing=sorted(required_top - set(payload)))
    required_learner = {"core_state", "optimizer_state", "optimizer_step_count", "shared_frozen", "counts",
                        "runtime_counts", "slot_generation", "fitted_steps", "cache", "evidence",
                        "recent_groups", "parent_group", "slot_group_tags", "action_rng", "behavior_rng",
                        "torch_rng", "numpy_rng", "python_rng"}
    audit.check(required_learner <= set(learner), "checkpoint_learner_fields", where,
                "learner checkpoint omits restoration state", missing=sorted(required_learner - set(learner)))
    audit.check(payload.get("pending_feedback") is None, "checkpoint_pending", where,
                "checkpoint boundary should have no unfinished feedback block")
    audit.check(payload.get("candidate_coefficients_persisted") is False, "candidate_persisted", where,
                "transient candidate coefficients must not be stored in the persistent package")
    audit.check(payload.get("simulator_state_persisted") is False, "simulator_persisted", where,
                "the checkpoint must declare that the live simulator is external")
    audit.check(len(actor_data) == int(config.get("actor_budget_bytes", -1)), "actor_budget", str(actor_path),
                "actual actor sidecar byte count differs from the frozen actor container",
                actual_bytes=len(actor_data), expected_bytes=config.get("actor_budget_bytes"))
    package_bytes: int | None = None
    if foundation_bytes is not None:
        package_bytes = checkpoint_path.stat().st_size + len(actor_data) + foundation_bytes
        audit.check(package_bytes <= int(config.get("package_budget_bytes", -1)), "package_cap", where,
                    "complete checkpoint, actor, and foundation exceed the frozen persistent-package cap",
                    actual_bytes=package_bytes, cap=config.get("package_budget_bytes"))
        if expected_sizes is not None:
            expected = {
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "actor_bytes": len(actor_data),
                "foundation_bytes": foundation_bytes,
                "full_persistent_bytes": package_bytes,
            }
            for key, value in expected.items():
                audit.check(int(expected_sizes.get(key, -1)) == value, "checkpoint_size_mismatch", where,
                            "saved checkpoint size metadata differs from actual files", field=key,
                            recorded=expected_sizes.get(key), actual=value)
    try:
        local, shared, groups = storage_audit._core_arrays(learner)
        decoded_local, decoded_shared, decoded_groups = decode_actor(actor_data)
        audit.check(np.array_equal(decoded_local, local) and np.array_equal(decoded_shared, shared)
                    and np.array_equal(decoded_groups.astype(np.int8), groups.astype(np.int8)),
                    "checkpoint_actor_match", where,
                    "actor sidecar does not encode the learner checkpoint actor arrays")
    except Exception as exc:
        audit.fail("checkpoint_actor_audit", where, "learner actor and sidecar could not be compared", error=repr(exc))
    return dict(payload), package_bytes


def _models(config: Mapping[str, Any]) -> dict[str, RetentionValueModel]:
    parameters = config["value_model"]
    return {
        "linear": RetentionValueModel(
            int(parameters["n_features"]), ridge=float(parameters["ridge"]),
            discount=float(parameters["discount"]), lambda_cpu=float(parameters["lambda_cpu"]),
        ),
        "intercept": RetentionValueModel(
            1, ridge=float(parameters["ridge"]), discount=float(parameters["discount"]),
            lambda_cpu=float(parameters["lambda_cpu"]),
        ),
    }


def _model_state_bytes(models: Mapping[str, RetentionValueModel], stats: BoundedRetentionStats) -> int:
    return int(sum(model.numeric_nbytes() for model in models.values()) + stats.numeric_nbytes())


def _expected_rngs(spec: Mapping[str, Any]) -> dict[str, np.random.Generator]:
    pair_offset = 37 if spec["pair_kind"] == "reward_conflict" else 11
    root_seed = int(spec["seed"]) * 104729 + pair_offset
    method = str(spec["method"])
    if method in ("learned_linear", "learned_intercept"):
        common = root_seed + 9001
        return {
            "exploration": np.random.default_rng(common),
            "exploration_action": np.random.default_rng(common + 1),
            "uniform": np.random.default_rng(common + 2),
        }
    salt = 5001 if method in ("random", "calibration") else 7001
    return {
        "exploration": np.random.default_rng(root_seed + salt),
        "exploration_action": np.random.default_rng(root_seed + salt + 1),
        "uniform": np.random.default_rng(root_seed + salt + 2),
    }


def _rng_states(rngs: Mapping[str, np.random.Generator]) -> dict[str, Any]:
    return {name: rng.bit_generator.state for name, rng in rngs.items()}


def _warmup_plan(mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"bout": 0, "group": 0, "variant": 0, "task": task_name(mapping, 0, 0), "steps": 1024,
         "action": "initial_parent_source"},
        {"bout": 1, "group": 0, "variant": 1, "task": task_name(mapping, 0, 1), "steps": 1024,
         "action": "initial_parent_source"},
        {"bout": 2, "group": 1, "variant": 0, "task": task_name(mapping, 1, 0), "steps": 512,
         "action": "initial_candidate_source"},
        {"bout": 3, "group": 1, "variant": 1, "task": task_name(mapping, 1, 1), "steps": 512,
         "action": "initial_candidate_source"},
    ]


def _outcome_plan(mapping: Mapping[str, Any], seed: int, decision: int) -> list[dict[str, Any]]:
    dominant = (int(seed) % 2 + int(decision) // 8) % 2
    minority = 1 - dominant
    requests = ((minority, decision % 3), (dominant, decision % 3), (dominant, (decision + 1) % 3))
    return [
        {"bout": 4 + decision * 3 + index, "group": group, "variant": variant,
         "task": task_name(mapping, group, variant), "steps": 512}
        for index, (group, variant) in enumerate(requests)
    ]


def _schedule_by_bout(mapping: Mapping[str, Any], seed: int, decisions: int) -> dict[int, dict[str, Any]]:
    rows = _warmup_plan(mapping)
    for decision in range(decisions):
        rows.extend(_outcome_plan(mapping, seed, decision))
    return {int(row["bout"]): row for row in rows}


def _check_no_unselected_labels(audit: Audit, row: Mapping[str, Any], where: str) -> None:
    found: list[str] = []
    def visit(value: Any, prefix: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                name = str(key).casefold()
                if name in FORBIDDEN_OUTCOME_KEYS:
                    found.append(f"{prefix}{key}")
                visit(child, f"{prefix}{key}.")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{prefix}{index}.")
    visit(row)
    audit.check(not found, "unchosen_outcome_label", where,
                "pre-action or feedback log contains action-wise/counterfactual outcome labels", keys=found)


def _check_preaction_flush_source(audit: Audit, snapshot: Path) -> None:
    source = snapshot / "scripts" / "run_retention_value_stream.py"
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        append_fn = functions.get("_append_jsonl")
        run_fn = functions.get("_run_one")
        if append_fn is None or run_fn is None:
            raise ValueError("expected runner functions are absent")
        flush_call = any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "flush"
            for node in ast.walk(append_fn)
        )
        def call_line(name: str, arg_name: str | None = None) -> int | None:
            matches = []
            for node in ast.walk(run_fn):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != name:
                    continue
                if arg_name is not None and not any(isinstance(arg, ast.Name) and arg.id == arg_name for arg in node.args):
                    continue
                matches.append(int(node.lineno))
            return min(matches) if matches else None
        preaction_line = call_line("_append_jsonl", "preaction")
        action_line = call_line("_parent_action")
        schedule_line = call_line("_outcome_schedule")
        audit.check(flush_call and preaction_line is not None and action_line is not None
                    and preaction_line < action_line, "preaction_flush_order", str(source),
                    "frozen runner does not flush its pre-action record before storage installation",
                    flush_call=flush_call, preaction_line=preaction_line, action_line=action_line)
        audit.check(preaction_line is not None and schedule_line is not None and preaction_line < schedule_line,
                    "schedule_after_action_log", str(source),
                    "frozen runner constructs the future schedule before flushing the selector record",
                    preaction_line=preaction_line, schedule_line=schedule_line)
    except Exception as exc:
        audit.fail("runner_source_order", str(source), "could not verify pre-action flush ordering from frozen runner",
                   error=repr(exc))


def _audit_raw_steps(audit: Audit, path: Path, *, seed: int, pair_kind: str,
                     requested_decisions: int, preactions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = _csv_rows(audit, path, required=False)
    where = str(path)
    mapping = pair_mapping(int(seed), str(pair_kind))
    schedule = _schedule_by_bout(mapping, int(seed), int(requested_decisions))
    selected = {int(row["decision_index"]): row.get("selected_action") for row in preactions
                if isinstance(row.get("decision_index"), int)}
    per_bout: dict[int, dict[str, Any]] = {}
    prior_episode: int | None = None
    prior_episode_seed: str | None = None
    prior_done = False
    prior_next_obs: list[float] | None = None
    previous_bout: int | None = None
    expected_columns = {"phase", "global_step", "bout", "bout_step", "episode_id", "reset_seed",
                        "taskobserver", "group", "action_id", "reward", "probabilities",
                        "action_uniform", "behavior_log_prob", "terminated", "truncated",
                        "environment_done", "learning_boundary", "obs", "next_obs"}
    if rows:
        audit.check(expected_columns <= set(rows[0]), "raw_columns", where,
                    "raw transition CSV omits required behavior or schedule fields",
                    missing=sorted(expected_columns - set(rows[0])))
    probabilities_hash = hashlib.sha256()
    prefix_hash = hashlib.sha256()
    prefix_behavior_rows = 0
    prefix_limit = 3072 + min(int(requested_decisions), 16) * 1536
    prefix_columns = ("phase", "action", "global_step", "bout", "bout_step", "episode_id", "reset_seed",
                      "taskobserver", "group", "obs", "next_obs", "action_id", "reward", "probabilities",
                      "action_uniform", "behavior_log_prob", "terminated", "truncated", "environment_done")
    warmup_reward_sum = 0.0
    outcome_reward_sum = 0.0
    for index, row in enumerate(rows):
        try:
            global_step = int(row["global_step"])
            bout = int(row["bout"])
            bout_step = int(row["bout_step"])
            group = int(row["group"])
            episode = int(row["episode_id"])
            action_id = int(row["action_id"])
            reward = float(row["reward"])
            terminated = bool(int(row["terminated"]))
            truncated = bool(int(row["truncated"]))
            environment_done = bool(int(row["environment_done"]))
            learning_boundary = bool(int(row["learning_boundary"]))
            obs = json.loads(row["obs"])
            next_obs = json.loads(row["next_obs"])
            behavior = json.loads(row["probabilities"])
            uniform = float(row["action_uniform"])
            log_prob = float(row["behavior_log_prob"])
        except Exception as exc:
            audit.fail("raw_row_parse", where, "raw interaction row is malformed", row=index, error=repr(exc))
            continue
        audit.check(global_step == index + 1, "raw_step_sequence", where,
                    "raw global steps must be consecutive and one-based", row=index,
                    actual=global_step, expected=index + 1)
        plan = schedule.get(bout)
        if plan is None:
            audit.fail("raw_bout_unplanned", where, "raw ledger contains a bout outside the frozen schedule",
                       row=index, bout=bout)
            continue
        expected_phase = "warmup" if bout < 4 else "outcome"
        audit.check(row.get("phase") == expected_phase, "raw_phase", where,
                    "raw step has the wrong phase for its bout", row=index,
                    actual=row.get("phase"), expected=expected_phase)
        audit.check(row.get("future_regime", "") in ("", None), "future_schedule_leak", where,
                    "raw transition includes a future-regime field instead of the external schedule", row=index)
        if "counterfactualobserver" in row:
            audit.check(str(row.get("counterfactualobserver", "")).casefold() not in ("1", "true", "yes"),
                        "observer_rows", where, "raw transition is marked as a counterfactual observer row", row=index)
        audit.check(row.get("taskobserver") == plan["task"] and group == plan["group"],
                    "raw_task_schedule", where, "raw task or group differs from frozen bout schedule",
                    row=index, task=row.get("taskobserver"), expected_task=plan["task"],
                    group=group, expected_group=plan["group"])
        audit.check(bout_step == (per_bout.get(bout, {}).get("rows", 0) + 1), "raw_bout_step", where,
                    "bout-local step index is not consecutive", row=index, bout=bout,
                    actual=bout_step, expected=per_bout.get(bout, {}).get("rows", 0) + 1)
        if previous_bout is not None:
            audit.check(bout >= previous_bout, "raw_bout_order", where,
                        "raw bout order moves backward", row=index, previous=previous_bout, actual=bout)
        previous_bout = bout
        expected_action = plan.get("action", "") if bout < 4 else selected.get((bout - 4) // 3, "")
        if expected_action is None:
            expected_action = ""
        audit.check(row.get("action", "") == expected_action, "raw_selected_action", where,
                    "raw outcome rows are not labeled with the single selected action",
                    row=index, bout=bout, actual=row.get("action", ""), expected=expected_action)
        audit.check(action_id in range(4) and len(behavior) == 4
                    and all(_finite(p) and float(p) > 0 for p in behavior)
                    and _close(sum(map(float, behavior)), 1.0, atol=1e-7, rtol=1e-7),
                    "behavior_distribution", where,
                    "raw behavior probabilities are invalid", row=index)
        if action_id in range(4) and isinstance(behavior, list) and len(behavior) == 4:
            audit.check(_close(math.log(max(float(behavior[action_id]), 1e-12)), log_prob,
                               atol=1e-7, rtol=1e-7),
                        "behavior_logprob", where, "behavior log probability differs from saved probability",
                        row=index)
            audit.check(0.0 <= uniform < 1.0, "behavior_uniform", where,
                        "behavior sampler uniform is outside [0,1)", row=index, uniform=uniform)
            chosen_by_uniform = min(int(np.searchsorted(np.cumsum(behavior), uniform, side="right")), 3)
            audit.check(chosen_by_uniform == action_id, "behavior_action_sample", where,
                        "saved action does not follow its saved behavior probabilities and uniform",
                        row=index, expected=chosen_by_uniform, actual=action_id)
        audit.check(environment_done == (terminated or truncated), "episode_done_flag", where,
                    "environment_done differs from terminated/truncated flags", row=index)
        audit.check(learning_boundary == (bout_step == int(plan["steps"])), "learning_boundary", where,
                    "learning boundary differs from the fixed bout end", row=index)
        audit.check(not (terminated and truncated), "episode_terminal_flags", where,
                    "a raw transition is marked both terminated and truncated", row=index)
        if prior_episode is not None:
            expected_episode = prior_episode + 1 if prior_done else prior_episode
            audit.check(episode == expected_episode, "episode_continuity", where,
                        "episode IDs do not continue across bouts or reset after a terminal", row=index,
                        previous=prior_episode, previous_done=prior_done, actual=episode,
                        expected=expected_episode)
            if not prior_done and prior_next_obs is not None:
                audit.check(np.allclose(np.asarray(obs, dtype=np.float64),
                                        np.asarray(prior_next_obs, dtype=np.float64), atol=1e-7, rtol=0.0),
                            "trajectory_continuity", where,
                            "next observation does not continue from the prior saved transition", row=index)
        if prior_episode != episode:
            prior_episode_seed = str(row.get("reset_seed", ""))
        else:
            audit.check(str(row.get("reset_seed", "")) == prior_episode_seed, "episode_reset_seed", where,
                        "reset seed changed inside one episode", row=index)
        prior_episode = episode
        prior_done = environment_done
        prior_next_obs = next_obs if isinstance(next_obs, list) else None
        if isinstance(per_bout.get(bout), dict):
            acc = per_bout[bout]
        else:
            acc = {"rows": 0, "reward_sum": 0.0, "first256": 0.0, "first512": 0.0,
                   "phase": expected_phase, "task": plan["task"], "group": int(plan["group"]),
                   "action": row.get("action", "")}
            per_bout[bout] = acc
        acc["rows"] += 1
        acc["reward_sum"] += reward
        if expected_phase == "warmup":
            warmup_reward_sum += reward
        else:
            outcome_reward_sum += reward
        if bout_step <= 256:
            acc["first256"] += reward
        if bout_step <= 512:
            acc["first512"] += reward
        audit.check(_finite(reward) and isinstance(obs, list) and isinstance(next_obs, list),
                    "raw_transition_values", where, "raw reward or observation is malformed", row=index)
        probabilities_hash.update(row.get("probabilities", "").encode("utf-8"))
        probabilities_hash.update(b"\n")
        if global_step <= prefix_limit:
            prefix_hash.update(json.dumps([row.get(key, "") for key in prefix_columns],
                                          ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            prefix_hash.update(b"\n")
            prefix_behavior_rows += 1
    for bout, acc in per_bout.items():
        plan = schedule[bout]
        audit.check(acc["rows"] <= int(plan["steps"]), "raw_bout_overrun", where,
                    "raw bout contains more rows than its frozen length", bout=bout,
                    actual=acc["rows"], expected=plan["steps"])
    return {
        "raw_steps": len(rows),
        "raw_reward_sum": float(sum(float(row["reward"]) for row in rows if _finite(row.get("reward")))),
        "warmup_steps": sum(int(value["rows"]) for value in per_bout.values() if value["phase"] == "warmup"),
        "outcome_steps": sum(int(value["rows"]) for value in per_bout.values() if value["phase"] == "outcome"),
        "warmup_reward_sum": warmup_reward_sum,
        "outcome_reward_sum": outcome_reward_sum,
        "per_bout": per_bout,
        "probabilities_sha256": probabilities_hash.hexdigest(),
        "episode_count_observed": 0 if not rows else len({row.get("episode_id") for row in rows}),
        "prefix_behavior_sha256": prefix_hash.hexdigest(),
        "prefix_behavior_rows": prefix_behavior_rows,
    }


def _check_bouts(audit: Audit, rows: Sequence[Mapping[str, Any]], raw: Mapping[str, Any],
                 *, seed: int, pair_kind: str, requested_decisions: int, where: str) -> None:
    mapping = pair_mapping(seed, pair_kind)
    schedule = _schedule_by_bout(mapping, seed, requested_decisions)
    raw_by_bout = raw.get("per_bout", {})
    seen_bouts: set[int] = set()
    for index, row in enumerate(rows):
        row_where = f"{where}/bouts[{index}]"
        try:
            bout = int(row["bout_index"])
            steps = int(row["steps_completed"])
            decision = row.get("decision_index")
            decision = None if decision is None else int(decision)
        except Exception as exc:
            audit.fail("bout_record_parse", row_where, "bout record is malformed", error=repr(exc))
            continue
        plan = schedule.get(bout)
        audit.check(plan is not None and bout not in seen_bouts, "bout_record_identity", row_where,
                    "bout record is duplicated or outside the frozen schedule", bout=bout)
        if plan is None:
            continue
        seen_bouts.add(bout)
        expected_decision = None if bout < 4 else (bout - 4) // 3
        audit.check(row.get("phase") == ("warmup" if bout < 4 else "outcome")
                    and decision == expected_decision, "bout_record_phase", row_where,
                    "bout phase or decision index differs from its scheduled position",
                    actual_phase=row.get("phase"), expected_phase="warmup" if bout < 4 else "outcome",
                    actual_decision=decision, expected_decision=expected_decision)
        audit.check(row.get("task") == plan["task"] and int(row.get("group", -1)) == plan["group"]
                    and int(row.get("variant", -1)) == plan["variant"], "bout_record_schedule", row_where,
                    "bout identity differs from the frozen schedule", expected=plan,
                    actual={key: row.get(key) for key in ("task", "group", "variant")})
        raw_bout = raw_by_bout.get(bout)
        audit.check(isinstance(raw_bout, Mapping) and raw_bout.get("rows") == steps,
                    "bout_raw_coverage", row_where,
                    "completed bout record does not exactly cover its raw step rows",
                    raw_rows=None if not isinstance(raw_bout, Mapping) else raw_bout.get("rows"),
                    recorded_steps=steps)
        if isinstance(raw_bout, Mapping):
            for field, accumulator in (("return_first_256", "first256"),
                                      ("return_first_512", "first512"), ("return_full", "reward_sum")):
                if field == "return_first_256":
                    expected_sum = raw_bout[accumulator]
                elif field == "return_first_512":
                    expected_sum = raw_bout[accumulator] if steps >= 512 else raw_bout["reward_sum"]
                else:
                    expected_sum = raw_bout[accumulator]
                audit.check(_close(row.get(field), float(expected_sum), atol=1e-7, rtol=1e-8),
                            "bout_reward_sum", row_where,
                            "bout summary reward differs from its saved raw rows", field=field,
                            recorded=row.get(field), observed=expected_sum)
        try:
            allocated = row.get("allocated")
            eviction = row.get("evicted_task")
            recurrence = row.get("recurrence")
            audit.check(isinstance(allocated, bool) and isinstance(recurrence, Mapping),
                        "bout_recurrence_fields", row_where,
                        "bout recurrence record lacks allocation or pre-observe statistics")
        except Exception as exc:
            audit.fail("bout_recurrence_parse", row_where, "bout recurrence fields are malformed", error=repr(exc))


def _check_recurrence_order(audit: Audit, stats: BoundedRetentionStats,
                            rows: Sequence[Mapping[str, Any]], where: str) -> None:
    for index, row in enumerate(rows):
        record = dict(row)
        try:
            expected = stats.recurrence_info(record)
            recorded = row.get("recurrence")
            audit.check(_same_json(recorded, expected), "recurrence_preobserve", f"{where}/bouts[{index}]",
                        "recurrence fields were not computed from state before this bout update",
                        recorded=recorded, expected=expected)
            stats.observe_bout(record)
        except Exception as exc:
            audit.fail("recurrence_replay", f"{where}/bouts[{index}]",
                       "bounded recurrence state could not be replayed", error=repr(exc))


def _compare_model_state(audit: Audit, recorded: Any, model: RetentionValueModel, where: str) -> None:
    if not isinstance(recorded, Mapping):
        audit.fail("model_state_missing", where, "value model state is missing")
        return
    expected = model.state_dict()
    audit.check(tuple(recorded.get("actions", ())) == tuple(expected["actions"]),
                "model_actions", where, "model checkpoint action order differs")
    for key in ("format_version", "n_features", "ridge", "discount", "lambda_cpu", "default_action"):
        audit.check(recorded.get(key) == expected.get(key), "model_config", where,
                    "model checkpoint setting differs from the replayed bounded model", field=key,
                    recorded=recorded.get(key), expected=expected.get(key))
    for key in ("A", "b", "counts", "discounted_counts"):
        try:
            audit.check(np.array_equal(np.asarray(recorded[key]), np.asarray(expected[key])),
                        "model_state_replay", where,
                        "serialized model sufficient statistics differ from selected-feedback replay",
                        field=key)
        except Exception as exc:
            audit.fail("model_state_read", where, "model sufficient statistic could not be compared",
                       field=key, error=repr(exc))


def _compare_stats_state(audit: Audit, recorded: Any, stats: BoundedRetentionStats, where: str) -> None:
    try:
        actual = BoundedRetentionStats.from_state_dict(dict(recorded))
        expected = stats.state_dict()
        observed = actual.state_dict()
        for key, expected_value in expected.items():
            value = observed[key]
            same = np.array_equal(np.asarray(value), np.asarray(expected_value)) \
                if isinstance(expected_value, np.ndarray) else value == expected_value
            audit.check(bool(same), "stats_state_replay", where,
                        "serialized bounded history differs from chronological bout replay", field=key)
    except Exception as exc:
        audit.fail("stats_state_read", where, "serialized bounded retention history is invalid", error=repr(exc))


def _check_rng_state(audit: Audit, recorded: Any, rngs: Mapping[str, np.random.Generator], where: str) -> None:
    audit.check(isinstance(recorded, Mapping) and _same_json(recorded, _rng_states(rngs)),
                "selector_rng_replay", where,
                "selector RNG state differs from the deterministic pre-action stream")


def _check_predictions(audit: Audit, row: Mapping[str, Any], models: Mapping[str, RetentionValueModel],
                       features: np.ndarray, where: str) -> None:
    predictions = row.get("predictions")
    audit.check(isinstance(predictions, Mapping), "predictions_missing", where,
                "pre-action record omits pre-update action predictions")
    if not isinstance(predictions, Mapping):
        return
    expected = {
        "linear": models["linear"].predict(features),
        "intercept": models["intercept"].predict(np.asarray([1.0], dtype=np.float64)),
    }
    for model_name, values in expected.items():
        observed = predictions.get(model_name)
        audit.check(isinstance(observed, Mapping), "prediction_model_missing", where,
                    "pre-action predictions omit a bounded model", model=model_name)
        if not isinstance(observed, Mapping):
            continue
        for action in ACTIONS:
            audit.check(_close(observed.get(action), float(values[action]), atol=1e-10, rtol=1e-10),
                        "prediction_replay", where,
                        "pre-action prediction differs from prior selected-feedback model state",
                        model=model_name, action=action, recorded=observed.get(action),
                        expected=float(values[action]))


def _check_candidate(audit: Audit, row: Mapping[str, Any], prior_bouts: Sequence[Mapping[str, Any]],
                     *, minimum_fit: int, where: str) -> tuple[int | None, float | None]:
    provenance = row.get("candidate_provenance")
    if not isinstance(provenance, Mapping):
        audit.fail("candidate_provenance_missing", where, "pre-action record omits candidate source provenance")
        return None, None
    sources = provenance.get("sources")
    audit.check(isinstance(sources, list) and len(sources) == 2, "candidate_source_count", where,
                "candidate must be a transient mean of exactly two resident source generations")
    if not isinstance(sources, list) or len(sources) != 2 or len(prior_bouts) < 2:
        return None, None
    expected_bouts = list(prior_bouts[-2:])
    audit.check([int(source.get("bout", -1)) for source in sources] ==
                [int(event.get("bout_index", -2)) for event in expected_bouts],
                "candidate_latest_pair", where,
                "candidate provenance does not use the immediately preceding two completed bouts")
    try:
        group = int(provenance.get("candidate_group", -1))
        source_fit = float(provenance.get("source_fit_steps_mean"))
    except (TypeError, ValueError):
        audit.fail("candidate_group_or_fit", where, "candidate group or mean source fit is malformed")
        return None, None
    source_groups: set[int] = set()
    source_slots: set[int] = set()
    source_tasks: set[str] = set()
    source_variants: set[int] = set()
    fits: list[int] = []
    for index, (source, event) in enumerate(zip(sources, expected_bouts)):
        location = f"{where}/candidate_source_{index}"
        try:
            source_group = int(source["group"])
            slot = int(source["slot"])
            generation = int(source["generation"])
            fit = int(source["fitted_steps"])
            added = int(source["fitted_steps_added_in_source_pair"])
            fit_before = int(source["fitted_steps_before_source_pair"])
            task = str(source["task"])
            variant = int(source["variant"])
            bout = int(source["bout"])
        except Exception as exc:
            audit.fail("candidate_source_parse", location, "candidate source fields are malformed", error=repr(exc))
            continue
        source_groups.add(source_group)
        source_slots.add(slot)
        source_tasks.add(task)
        source_variants.add(variant)
        fits.append(fit)
        audit.check(task == event.get("task") and source_group == int(event.get("group", -1))
                    and variant == int(event.get("variant", -1)) and bout == int(event.get("bout_index", -2))
                    and slot == int(event.get("slot", -2)) and generation == int(event.get("generation", -2))
                    and fit == int(event.get("fitted_steps_after", -2)),
                    "candidate_source_matches_bout", location,
                    "candidate provenance does not match the preceding bout's resident task generation")
        audit.check(fit_before >= 0 and fit - fit_before == added and added >= minimum_fit,
                    "candidate_source_fitted_budget", location,
                    "current resident source generation did not add the required fitted steps",
                    fitted_before=fit_before, fitted_after=fit, added=added, minimum=minimum_fit)
    audit.check(len(source_groups) == 1 and group in source_groups and len(source_slots) == 2
                and len(source_tasks) == 2 and len(source_variants) == 2,
                "candidate_sources_distinct", where,
                "candidate sources must be two distinct variants in one group and separate resident slots",
                groups=sorted(source_groups), slots=sorted(source_slots), tasks=sorted(source_tasks),
                variants=sorted(source_variants))
    audit.check(_close(source_fit, float(np.mean(fits)), atol=1e-10, rtol=1e-10),
                "candidate_source_fit_mean", where,
                "candidate mean source-fit metadata differs from its two source generations",
                recorded=source_fit, observed=float(np.mean(fits)))
    audit.check(provenance.get("candidate_matrix_persisted") is False,
                "candidate_matrix_persisted", where, "transient candidate coefficients are persisted")
    candidate_hash = str(provenance.get("candidate_sha256", ""))
    audit.check(len(candidate_hash) == 64 and all(char in "0123456789abcdef" for char in candidate_hash.lower()),
                "candidate_hash", where, "transient candidate provenance lacks a SHA-256 digest")
    audit.check(int(row.get("candidate_group", -1)) == group
                and _close(row.get("source_fit_steps"), source_fit, atol=1e-10, rtol=1e-10),
                "candidate_preaction_copy", where,
                "pre-action candidate group or source-fit field differs from its provenance")
    return group, source_fit


def _check_selection(audit: Audit, row: Mapping[str, Any], method: str, decision: int,
                     features: np.ndarray, stats: BoundedRetentionStats,
                     models: Mapping[str, RetentionValueModel], rngs: Mapping[str, np.random.Generator],
                     where: str) -> str | None:
    _check_rng_state(audit, row.get("rng_state_before"), rngs, where)
    _check_predictions(audit, row, models, features, where)
    predictions = row.get("predictions", {})
    selected = row.get("selected_action")
    propensity = row.get("action_propensity")
    explored = bool(row.get("explored"))
    expected_action: str | None = None
    expected_rule = ""
    expected_explored = False
    expected_propensity: float | None = None
    if method == "random":
        expected_action = ACTIONS[int(rngs["uniform"].integers(len(ACTIONS)))]
        expected_rule, expected_propensity, expected_explored = "uniform_random", 0.25, True
    elif method in ("learned_linear", "learned_intercept"):
        model_name = "linear" if method == "learned_linear" else "intercept"
        model_values = predictions.get(model_name, {}) if isinstance(predictions, Mapping) else {}
        greedy = runner._greedy_action(model_values) if isinstance(model_values, Mapping) else "keep"
        explore_draw = float(rngs["exploration"].random())
        random_action = ACTIONS[int(rngs["exploration_action"].integers(len(ACTIONS)))]
        epsilon = 1.0 if decision < 16 else float(0.1)
        expected_explored = decision < 16 or explore_draw < epsilon
        expected_action = random_action if expected_explored else greedy
        expected_propensity = 0.25 if decision < 16 else (0.025 + (0.9 if expected_action == greedy else 0.0))
        expected_rule = "uniform_warmup" if decision < 16 else "epsilon_greedy_matched"
        audit.check(row.get("model_name") == model_name and row.get("greedy_action") == greedy
                    and _close(row.get("epsilon"), epsilon, atol=1e-12, rtol=0.0),
                    "learned_selector_metadata", where,
                    "learned selector metadata differs from the pre-action model and schedule")
    elif method in ACTIONS[:3]:
        expected_action, expected_propensity, expected_rule = method, 1.0, f"fixed_{method}"
    elif method == "frequency":
        expected_action = frequency_action(stats, int(row.get("candidate_group", -1)),
                                           int(row.get("incumbent_binding", -3)))
        expected_propensity, expected_rule = 1.0, "recent_group_frequency"
    elif method == "lower":
        diagnostics = row.get("lower_diagnostics")
        expected_propensity, expected_rule = 1.0, "generalized_groupwise_lower"
        if isinstance(diagnostics, Mapping):
            all_valid = diagnostics.get("all_group_scores_valid") is True
            weighted = diagnostics.get("frequency_weighted_lower")
            expected_action = "replace" if all_valid and _finite(weighted) and float(weighted) > 0 else "keep"
            audit.check(diagnostics.get("action") == expected_action,
                        "lower_decision_rule", where,
                        "generalized lower action differs from its frozen weighted rule")
            frequencies = diagnostics.get("recent_group_frequency", {})
            audit.check(isinstance(frequencies, Mapping)
                        and _close(frequencies.get("0"), stats.group_frequencies()[0])
                        and _close(frequencies.get("1"), stats.group_frequencies()[1]),
                        "lower_preaction_history", where,
                        "lower selector diagnostics use future or inconsistent group frequencies")
        else:
            audit.fail("lower_diagnostics_missing", where, "lower selector omits groupwise score diagnostics")
    elif method in ("shared_ppo", "independent"):
        audit.check(selected is None and propensity is None and row.get("selection_rule") == "no_discrete_action",
                    "control_no_discrete_action", where,
                    "online PPO control must not emit a discrete storage action")
        return None
    else:
        audit.fail("method_unknown", where, "stream method is outside the frozen method set", method=method)
        return None
    audit.check(selected == expected_action, "selected_action_replay", where,
                "recorded action differs from the frozen rule and logged pre-action RNG",
                recorded=selected, expected=expected_action)
    audit.check(row.get("selection_rule") == expected_rule, "selection_rule", where,
                "selection rule label differs from the frozen method")
    audit.check(explored == expected_explored, "selection_exploration", where,
                "exploration flag differs from the frozen selection mechanism",
                recorded=explored, expected=expected_explored)
    audit.check(expected_propensity is not None and _close(propensity, float(expected_propensity), atol=1e-12, rtol=0.0),
                "selection_propensity", where,
                "logged propensity differs from the frozen behavior probability",
                recorded=propensity, expected=expected_propensity)
    return str(expected_action) if expected_action is not None else None


def _checkpoint_dirs(stream_dir: Path, decisions: int, checkpoint_every: int) -> dict[int, Path]:
    result = {0: stream_dir / "checkpoints" / "initial"}
    for count in range(checkpoint_every, decisions + 1, checkpoint_every):
        result[count] = stream_dir / "checkpoints" / f"decision_{count:03d}"
    if decisions:
        result[decisions] = stream_dir / "checkpoints" / f"decision_{decisions:03d}"
    return result


def _checkpoint_optimizer(audit: Audit, payload: Mapping[str, Any], method: str,
                          config: Mapping[str, Any], where: str) -> None:
    learner = payload.get("learner", {})
    optimizer = learner.get("optimizer_state", {}) if isinstance(learner, Mapping) else {}
    groups = optimizer.get("param_groups", []) if isinstance(optimizer, Mapping) else []
    by_name = {str(group.get("name")): group for group in groups if isinstance(group, Mapping)}
    expected = {
        "local": float(config["local_learning_rate"]),
        "shared": float(config["direct_shared_learning_rate"] if method == "shared_ppo"
                         else config["shared_learning_rate"]),
        "critic": float(config["critic_learning_rate"]),
    }
    audit.check(set(by_name) == set(expected), "optimizer_groups", where,
                "checkpoint optimizer groups differ from the frozen local/shared/critic layout",
                groups=sorted(by_name))
    for name, learning_rate in expected.items():
        actual = by_name.get(name, {}).get("lr")
        audit.check(_close(actual, learning_rate, atol=1e-12, rtol=0.0), "optimizer_learning_rate", where,
                    "checkpoint optimizer learning rate differs from the frozen control",
                    group=name, recorded=actual, expected=learning_rate)
    if method == "shared_ppo":
        audit.check(learner.get("shared_frozen") is False, "shared_ppo_unfrozen", where,
                    "direct shared PPO checkpoint freezes the shared actor")
    else:
        audit.check(learner.get("shared_frozen") is True, "shared_actor_frozen", where,
                    "discrete or independent control should freeze the shared actor between writes")
    updates = int(learner.get("runtime_counts", {}).get("minibatch_updates", -1))
    audit.check(int(learner.get("optimizer_step_count", -2)) == updates,
                "optimizer_update_counter", where,
                "optimizer global step counter differs from actual minibatch update count",
                optimizer_step_count=learner.get("optimizer_step_count"), minibatch_updates=updates)
    if method == "independent":
        try:
            _local, shared, groups_array = storage_audit._core_arrays(learner)
            audit.check(int(learner.get("parent_group", -2)) == -1 and np.all(shared == 0)
                        and np.all(groups_array == -1), "independent_unbound_zero_parent", where,
                        "independent control must keep its shared parent zero and permanently unbound")
        except Exception as exc:
            audit.fail("independent_actor_state", where,
                       "independent checkpoint shared actor could not be inspected", error=repr(exc))


def _verify_checkpoint(audit: Audit, directory: Path, boundary: int,
                       expected_package_bytes: int | None, foundation_bytes: int | None,
                       config: Mapping[str, Any], method: str, models: Mapping[str, RetentionValueModel],
                       stats: BoundedRetentionStats, rngs: Mapping[str, np.random.Generator],
                       where: str) -> tuple[dict[str, Any] | None, int | None]:
    payload, package_bytes = _read_checkpoint(audit, directory, None, foundation_bytes, config)
    if payload is None:
        return None, package_bytes
    audit.check(int(payload.get("decision_index_next", -1)) == boundary,
                "checkpoint_decision_boundary", where,
                "checkpoint decision index differs from its boundary", recorded=payload.get("decision_index_next"),
                expected=boundary)
    if expected_package_bytes is not None:
        audit.check(package_bytes == int(expected_package_bytes), "checkpoint_boundary_size", where,
                    "actual saved package size differs from the runner's measured boundary size",
                    recorded=expected_package_bytes, actual=package_bytes)
    _compare_stats_state(audit, payload.get("retention_stats"), stats, where)
    states = payload.get("value_models", {})
    for name, model in models.items():
        _compare_model_state(audit, states.get(name) if isinstance(states, Mapping) else None,
                             model, f"{where}/value_models/{name}")
    audit.check(isinstance(payload.get("selector_rng_states"), Mapping)
                and _same_json(payload.get("selector_rng_states"), _rng_states(rngs)),
                "checkpoint_selector_rng", where,
                "checkpoint selector RNG state differs from replayed selections")
    learner = payload.get("learner", {})
    runtime = learner.get("runtime_counts", {}) if isinstance(learner, Mapping) else {}
    expected_steps = 3072 + boundary * 1536
    expected_rollouts = expected_steps // int(config["rollout_steps"])
    expected_updates = expected_rollouts * 16
    audit.check(int(runtime.get("environment_steps", -1)) == expected_steps
                and int(runtime.get("rollouts_trained", -1)) == expected_rollouts
                and int(runtime.get("minibatch_updates", -1)) == expected_updates,
                "checkpoint_training_counts", where,
                "checkpoint runtime counters differ from the completed real rollout schedule",
                runtime=runtime, expected_steps=expected_steps, expected_rollouts=expected_rollouts,
                expected_updates=expected_updates)
    _checkpoint_optimizer(audit, payload, method, config, where)
    if isinstance(payload.get("parent_metadata"), Mapping):
        audit.check(int(payload["parent_metadata"].get("binding", -9)) == int(learner.get("parent_group", -8)),
                    "checkpoint_parent_binding", where,
                    "parent metadata binding differs from the learner's physical binding")
        audit.check(_close(payload["parent_metadata"].get("source_fit_steps"), stats.parent_source_fit)
                    and int(payload["parent_metadata"].get("written_decision", -99)) == stats.parent_written_decision,
                    "checkpoint_parent_history", where,
                    "parent source-fit or last-write history differs from replayed bounded statistics")
    return payload, package_bytes


def _audit_stream(audit: Audit, cohort: Path, frozen: Mapping[str, Any],
                  spec: Mapping[str, Any], digest_cache: dict[Path, str]) -> dict[str, Any]:
    pair_kind, seed, method = str(spec["pair_kind"]), int(spec["seed"]), str(spec["method"])
    expected_relative = Path("streams") / pair_kind / f"seed_{seed}" / method
    stream_dir = cohort / expected_relative
    where = str(expected_relative).replace("\\", "/")
    try:
        summary = _json(stream_dir / "summary.json")
        run_config = _json(stream_dir / "run_config.json")
    except Exception as exc:
        audit.fail("stream_metadata", where, "stream summary or run config is missing/unreadable", error=repr(exc))
        return {"stream": where, "status": "unreadable"}
    audit.check(_spec_key(summary) == _spec_key(spec)
                and int(summary.get("requested_decisions", -1)) == int(spec["decisions"]),
                "stream_spec", where, "stream summary identity differs from the frozen requested grid")
    audit.check(run_config.get("stream") == dict(spec), "stream_run_config", where,
                "per-stream config differs from the frozen requested grid")
    audit.check(run_config.get("config_sha256") == frozen.get("config_sha256"),
                "stream_config_hash", where, "per-stream config hash differs from the cohort freeze")
    audit.check(run_config.get("source_manifest") == "source_snapshot/manifest.json"
                and run_config.get("candidate_history_capacity") == 0,
                "stream_bounded_history", where,
                "run config must point at its source snapshot and declare no persistent candidate bank")
    foundation_bytes = _check_source_snapshot(audit, stream_dir, frozen, digest_cache)
    manifest = _json(stream_dir / "source_snapshot" / "manifest.json")
    source_protocol_hash = manifest.get("sources", {}).get(
        "docs/RETENTION_VALUE_STREAM_PROTOCOL.md", {}).get("sha256")
    audit.check(run_config.get("protocol_sha256") == source_protocol_hash,
                "stream_protocol_snapshot_hash", where,
                "run config protocol hash differs from its own source snapshot",
                run_config_hash=run_config.get("protocol_sha256"), source_hash=source_protocol_hash)

    preactions = _jsonl(audit, stream_dir / "preactions.jsonl")
    feedback = _jsonl(audit, stream_dir / "feedback.jsonl")
    bouts = _jsonl(audit, stream_dir / "bouts.jsonl")
    package_events = _jsonl(audit, stream_dir / "package_events.jsonl")
    audit.check(all(row.get("stream_id") == summary.get("stream_id")
                    for rows in (preactions, feedback, package_events) for row in rows),
                "event_stream_id", where, "one or more event rows have a different stream identity")
    _check_preaction_flush_source(audit, stream_dir / "source_snapshot")
    for index, row in enumerate(preactions):
        _check_no_unselected_labels(audit, row, f"{where}/preactions[{index}]")
    for index, row in enumerate(feedback):
        _check_no_unselected_labels(audit, row, f"{where}/feedback[{index}]")

    raw = _audit_raw_steps(
        audit, stream_dir / "rawsteps.csv", seed=seed, pair_kind=pair_kind,
        requested_decisions=int(spec["decisions"]), preactions=preactions,
    )
    _check_bouts(audit, bouts, raw, seed=seed, pair_kind=pair_kind,
                 requested_decisions=int(spec["decisions"]), where=f"{where}/bouts")
    rollouts = _csv_rows(audit, stream_dir / "rollouts.csv")
    costs = summary.get("costs", {})
    completed = int(summary.get("completed_decisions", 0))
    requested = int(spec["decisions"])
    status = str(summary.get("status", "missing"))
    audit.check(0 <= completed <= requested, "completed_decisions_range", where,
                "completed decision count is outside its frozen range", completed=completed, requested=requested)
    if status == "completed":
        audit.check(completed == requested and len(preactions) == requested and len(feedback) == requested,
                    "completed_stream_logs", where,
                    "completed stream lacks a complete pre-action and selected-feedback pair",
                    completed=completed, requested=requested, preactions=len(preactions), feedback=len(feedback))
    else:
        audit.check(len(preactions) in (completed, completed + 1) and len(feedback) == completed,
                    "failed_stream_logs", where,
                    "failed stream event counts do not preserve completed and possibly pending decisions",
                    completed=completed, preactions=len(preactions), feedback=len(feedback))
    audit.check(len(package_events) == len(preactions), "inflight_event_count", where,
                "each flushed selector action should have one serialized in-flight package event",
                preactions=len(preactions), package_events=len(package_events))
    audit.check([int(row.get("decision_index", -1)) for row in preactions]
                == list(range(len(preactions))), "preaction_sequence", where,
                "pre-action decision indices are not consecutive")
    audit.check([int(row.get("decision_index", -1)) for row in feedback]
                == list(range(len(feedback))), "feedback_sequence", where,
                "feedback decision indices are not consecutive")

    mapping = pair_mapping(seed, pair_kind)
    tasks = tuple(task_name(mapping, group, variant) for group in (0, 1) for variant in range(3))
    task_groups = {task_name(mapping, group, variant): group for group in (0, 1) for variant in range(3)}
    stats = BoundedRetentionStats(tasks, task_groups)
    models = _models(frozen["config"])
    rngs = _expected_rngs(spec)
    bout_by_index = {int(row["bout_index"]): row for row in bouts if "bout_index" in row}
    warmup_rows = [bout_by_index[index] for index in range(4) if index in bout_by_index]
    _check_recurrence_order(audit, stats, warmup_rows, f"{where}/warmup")
    if method != "independent":
        initial_parent = summary.get("initial_parent", {})
        fit = float(initial_parent.get("source_fit_steps", 1024.0))
        stats.record_parent_write(fit, 0)
    checkpoint_map = _checkpoint_dirs(stream_dir, completed, int(frozen["config"]["checkpoint_every"]))
    if (status != "completed" and completed and completed != requested
            and completed % int(frozen["config"]["checkpoint_every"]) != 0):
        checkpoint_map.pop(completed, None)
    actual_checkpoint_dirs = {
        int(path.name.removeprefix("decision_")) if path.name.startswith("decision_") else 0: path
        for path in (stream_dir / "checkpoints").glob("*") if path.is_dir()
    }
    audit.check(set(actual_checkpoint_dirs) == set(checkpoint_map), "checkpoint_schedule", where,
                "saved package directories differ from initial/every-eight/final schedule",
                actual=sorted(actual_checkpoint_dirs), expected=sorted(checkpoint_map))

    initial_sizes = summary.get("initial_package")
    initial_payload, initial_package_bytes = _verify_checkpoint(
        audit, checkpoint_map[0], 0,
        int(initial_sizes["full_persistent_bytes"]) if isinstance(initial_sizes, Mapping) else None,
        foundation_bytes, frozen["config"], method, models, stats, rngs, f"{where}/checkpoints/initial",
    )
    if initial_payload is not None:
        audit.check(initial_payload.get("stage") == spec.get("stage")
                    and bool(initial_payload.get("pilot")) == bool(spec.get("pilot"))
                    and int(initial_payload.get("seed", -1)) == seed
                    and initial_payload.get("pair_kind") == pair_kind
                    and initial_payload.get("method") == method,
                    "checkpoint_identity", f"{where}/checkpoints/initial",
                    "initial checkpoint identity differs from the requested stream")
        try:
            initial_local, initial_shared, _initial_groups = storage_audit._core_arrays(initial_payload["learner"])
            parent_hash = _sha256_bytes(np.ascontiguousarray(initial_shared[0]).tobytes())
            audit.check(bool(preactions) and preactions[0].get("incumbent_parent_sha256") == parent_hash,
                        "initial_parent_hash", where,
                        "first pre-action parent hash differs from the checked initial actor",
                        checkpoint_hash=parent_hash,
                        preaction_hash=preactions[0].get("incumbent_parent_sha256") if preactions else None)
        except Exception as exc:
            audit.fail("initial_parent_actor", where, "initial actor parent could not be inspected", error=repr(exc))

    expected_binding = int(initial_payload.get("learner", {}).get("parent_group", -1)) if initial_payload else -1
    expected_parent_version = int(initial_payload.get("parent_metadata", {}).get("version", 0)) if initial_payload else 0
    expected_parent_hash: str | None = None
    if initial_payload is not None:
        try:
            _local, shared, _groups = storage_audit._core_arrays(initial_payload["learner"])
            expected_parent_hash = _sha256_bytes(np.ascontiguousarray(shared[0]).tobytes())
        except Exception:
            pass

    completed_bouts: list[Mapping[str, Any]] = list(warmup_rows)
    pending_package_bytes = int(initial_package_bytes or 0)
    package_max = pending_package_bytes
    transient_peak_estimate = pending_package_bytes
    action_counts = {action: 0 for action in ACTIONS}
    selected_count = write_count = model_update_count = 0
    model_update_cpu_sum = utility_cpu_sum = 0.0
    boundary_payloads: dict[int, tuple[dict[str, Any], int]] = {}
    if initial_payload is not None and initial_package_bytes is not None:
        boundary_payloads[0] = (initial_payload, initial_package_bytes)

    for decision in range(min(completed, len(preactions), len(feedback))):
        pre = preactions[decision]
        post = feedback[decision]
        decision_where = f"{where}/decision_{decision:03d}"
        audit.check(pre.get("stream_id") == summary.get("stream_id")
                    and post.get("stream_id") == summary.get("stream_id"),
                    "decision_stream_id", decision_where, "decision events have a mismatched stream id")
        audit.check(int(pre.get("decision_index", -1)) == decision
                    and int(post.get("decision_index", -1)) == decision,
                    "decision_join", decision_where,
                    "pre-action record and selected feedback do not join on decision index")
        _check_no_unselected_labels(audit, post, decision_where)
        audit.check(pre.get("selected_action") == post.get("selected_action"),
                    "selected_feedback_join", decision_where,
                    "feedback is not the outcome for the action recorded before its block",
                    selected=pre.get("selected_action"), feedback=post.get("selected_action"))
        audit.check(int(pre.get("incumbent_binding", -9)) == expected_binding
                    and int(pre.get("incumbent_parent_version", -99)) == expected_parent_version,
                    "preaction_parent_state", decision_where,
                    "pre-action incumbent binding or version differs from the previous action/checkpoint",
                    recorded_binding=pre.get("incumbent_binding"), expected_binding=expected_binding,
                    recorded_version=pre.get("incumbent_parent_version"), expected_version=expected_parent_version)
        if expected_parent_hash is not None and method != "shared_ppo":
            audit.check(pre.get("incumbent_parent_sha256") == expected_parent_hash,
                        "preaction_parent_hash_chain", decision_where,
                        "pre-action parent hash does not continue from the prior installation")
        candidate_group, source_fit = _check_candidate(
            audit, pre, completed_bouts, minimum_fit=int(frozen["config"]["minimum_source_fitted_steps"]),
            where=decision_where,
        )
        try:
            expected_features, expected_names = stats.features(
                int(candidate_group), int(pre.get("incumbent_binding", -3)),
                float(source_fit), decision,
            )
            audit.check(tuple(pre.get("feature_names", ())) == tuple(expected_names)
                        and np.allclose(np.asarray(pre.get("features"), dtype=np.float64), expected_features,
                                        atol=1e-12, rtol=0.0),
                        "preaction_features", decision_where,
                        "pre-action feature vector is not a function of bounded past-only history",
                        recorded=pre.get("features"), expected=expected_features.tolist())
            for model_name, model in models.items():
                audit.check(pre.get("model_hashes", {}).get(model_name) == semantic_hash(model.state_dict()),
                            "preaction_model_hash", decision_where,
                            "pre-action model hash differs from selected-feedback history", model=model_name)
            _check_selection(audit, pre, method, decision, expected_features, stats, models, rngs, decision_where)
        except Exception as exc:
            audit.fail("preaction_replay", decision_where,
                       "pre-action features, predictions, or selection could not be replayed", error=repr(exc))
            expected_features = np.asarray(pre.get("features", [1.0] * 6), dtype=np.float64)

        action = pre.get("selected_action")
        expected_bout_indices = list(range(4 + decision * 3, 7 + decision * 3))
        selected_bouts = [bout_by_index[index] for index in expected_bout_indices if index in bout_by_index]
        if len(selected_bouts) == 3:
            expected_return = sum(float(row.get("return_full", 0.0)) for row in selected_bouts)
            expected_returns = [float(row.get("return_full", 0.0)) for row in selected_bouts]
            audit.check(_close(post.get("block_return_sum"), expected_return, atol=1e-7, rtol=1e-8)
                        and np.allclose(np.asarray(post.get("bout_returns", []), dtype=np.float64),
                                        np.asarray(expected_returns), atol=1e-7, rtol=1e-8)
                        and post.get("outcome_bout_indices") == expected_bout_indices
                        and _same_json(post.get("recurrence_events"),
                                       [row.get("recurrence") for row in selected_bouts]),
                        "selected_block_raw_join", decision_where,
                        "selected feedback does not equal its single three-bout raw outcome")
        else:
            audit.check(status != "completed", "selected_block_missing", decision_where,
                        "completed decision lacks all three selected outcome bouts")
        if action is None:
            audit.check(post.get("parent_write_details") is None and not bool(post.get("model_updated"))
                        and post.get("utility_target") is None,
                        "no_discrete_control_feedback", decision_where,
                        "direct/shared or independent control should have no storage action or value update")
        else:
            action_counts[str(action)] = action_counts.get(str(action), 0) + 1
            selected_count += 1
            details = post.get("parent_write_details")
            audit.check(isinstance(details, Mapping) and details.get("action") == action,
                        "action_application", decision_where,
                        "selected storage action does not match its installation details")
            if action != "keep":
                write_count += 1
                if isinstance(details, Mapping):
                    expected_binding = {"replace": candidate_group, "mean": -2, "none": -1}.get(str(action))
                    audit.check(int(post.get("post_action_binding", -9)) == expected_binding
                                and int(details.get("parent_group", -8)) == expected_binding,
                                "action_binding", decision_where,
                                "storage action produced an unexpected parent binding",
                                action=action, expected=expected_binding,
                                recorded=post.get("post_action_binding"))
                    audit.check(details.get("codec", {}).get("roundtrip_applied") is True,
                                "write_codec_roundtrip", decision_where,
                                "non-keep parent operation did not apply the frozen actor codec")
                    old_fit = float(stats.parent_source_fit)
                    expected_parent_fit = float(source_fit) if action == "replace" else \
                        (old_fit + float(source_fit)) / 2.0 if action == "mean" else 0.0
                    new_hash = str(details.get("new_parent_sha256", ""))
                    audit.check(_close(details.get("new_parent_source_fit_steps"), expected_parent_fit)
                                and len(new_hash) == 64
                                and all(char in "0123456789abcdef" for char in new_hash.lower()),
                                "parent_write_metadata", decision_where,
                                "parent write source-fit metadata is invalid")
                    audit.check(details.get("old_parent_sha256") == pre.get("incumbent_parent_sha256"),
                                "parent_write_old_hash", decision_where,
                                "parent write details do not refer to the pre-action incumbent")
                    stats.record_parent_write(expected_parent_fit, decision)
                    expected_parent_hash = new_hash
                    expected_parent_version += 1
                    expected_binding = int(details.get("parent_group", expected_binding))
            else:
                if isinstance(details, Mapping):
                    audit.check(details.get("old_parent_sha256") == details.get("new_parent_sha256")
                                and details.get("codec", {}).get("roundtrip_applied") is False
                                and int(post.get("post_action_binding", -9)) == int(pre.get("incumbent_binding", -8)),
                                "keep_noop", decision_where,
                                "keep must preserve the parent, binding, and codec state as a no-op")
                    expected_parent_hash = str(details.get("new_parent_sha256", expected_parent_hash or ""))
                expected_binding = int(pre.get("incumbent_binding", expected_binding))
        if action is None:
            expected_binding = int(pre.get("incumbent_binding", expected_binding))
            if method == "shared_ppo":
                expected_parent_hash = None
        else:
            eligible_update = True
            audit.check(bool(post.get("model_updated")) is eligible_update,
                        "selected_action_model_count", decision_where,
                        "selected action should contribute exactly one selected-feedback model update")
            steps = int(post.get("block_steps", -1))
            block_return = expected_return if len(selected_bouts) == 3 else float(post.get("block_return_sum", float("nan")))
            reward_rate = block_return / steps if steps > 0 else float("nan")
            utility = float(post.get("utility_cpu_s", float("nan")))
            target = reward_rate - utility / steps if steps > 0 else float("nan")
            cpu_components = sum(float(post.get(key, float("nan"))) for key in
                                 ("pre_action_cpu_s", "action_install_cpu_s", "block_cpu_s"))
            audit.check(steps == 1536 and _close(post.get("reward_rate"), reward_rate)
                        and _close(post.get("utility_target"), target)
                        and _close(utility, cpu_components)
                        and _close(post.get("pre_action_cpu_s"), float(pre.get("pre_action_cpu_s", -1.0)))
                        and _finite(post.get("model_update_cpu_s"), nonnegative=True)
                        and post.get("target_excludes_model_update") is True,
                        "selected_utility_target", decision_where,
                        "selected utility label differs from block reward and pre-update CPU cost")
            for name, model in models.items():
                features = expected_features if name == "linear" else np.asarray([1.0], dtype=np.float64)
                model.update(str(action), features, reward_rate, utility, steps)
            model_update_count += 1
        if action is None:
            audit.check(post.get("post_action_binding") == pre.get("incumbent_binding"),
                        "control_binding", decision_where,
                        "no-discrete-action control changed its storage binding")

        event = package_events[decision] if decision < len(package_events) else {}
        audit.check(int(event.get("decision_index", -1)) == decision
                    and event.get("phase") == "inflight"
                    and _close(event.get("package_budget_bytes"), float(frozen["config"]["package_budget_bytes"]))
                    and int(event.get("package_bytes", -1)) <= int(frozen["config"]["package_budget_bytes"]),
                    "inflight_package_cap", decision_where,
                    "serialized pending-feedback package is missing or over the complete-package cap")
        pending = event.get("pending_feedback")
        if action is None:
            audit.check(pending is None, "control_pending_feedback", decision_where,
                        "no-action control should not create a pending storage-feedback label")
        else:
            audit.check(isinstance(pending, Mapping)
                        and int(pending.get("decision_index", -1)) == decision
                        and pending.get("selected_action") == action
                        and pending.get("features") == pre.get("features")
                        and _close(pending.get("propensity"), float(pre.get("action_propensity"))),
                        "single_pending_feedback", decision_where,
                        "pending selected feedback is not the one pre-action record for this block")
        audit.check(int(post.get("package_bytes_inflight", -1)) == int(event.get("package_bytes", -2))
                    and int(pre.get("package_bytes_before", -1)) == pending_package_bytes,
                    "package_size_chronology", decision_where,
                    "pre-action/in-flight package sizes do not join to the prior boundary measurement",
                    previous_boundary=pending_package_bytes, before=pre.get("package_bytes_before"),
                    inflight=post.get("package_bytes_inflight"), event=event.get("package_bytes"))
        after_bytes = int(post.get("package_bytes_after", -1))
        audit.check(0 < after_bytes <= int(frozen["config"]["package_budget_bytes"])
                    and post.get("pending_feedback_cleared") is True,
                    "boundary_package_cap", decision_where,
                    "decision-boundary package is missing, over cap, or still carries pending feedback",
                    package_bytes=after_bytes)
        pending_package_bytes = after_bytes
        package_max = max(package_max, int(event.get("package_bytes", 0)), after_bytes)
        try:
            preaction_bytes = len(json.dumps(pre, sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False).encode("utf-8"))
        except Exception:
            preaction_bytes = 0
        transient_peak_estimate = max(transient_peak_estimate,
                                      int(pre.get("package_bytes_before", 0)) + preaction_bytes + 36,
                                      int(event.get("package_bytes", 0)))
        if str(method) not in ("shared_ppo", "independent"):
            _check_recurrence_order(audit, stats,
                                    [bout_by_index[index] for index in range(4 + decision * 3, 7 + decision * 3)
                                     if index in bout_by_index], decision_where)
        else:
            _check_recurrence_order(audit, stats,
                                    [bout_by_index[index] for index in range(4 + decision * 3, 7 + decision * 3)
                                     if index in bout_by_index], decision_where)
        completed_bouts.extend(
            bout_by_index[index] for index in range(4 + decision * 3, 7 + decision * 3)
            if index in bout_by_index
        )
        utility_cpu_sum += float(post.get("utility_cpu_s", 0.0))
        model_update_cpu_sum += float(post.get("model_update_cpu_s", 0.0))
        expected_model_bytes = _model_state_bytes(models, stats)
        audit.check(int(post.get("model_state_bytes", -1)) == expected_model_bytes,
                    "model_state_capacity", decision_where,
                    "reported value-model and bounded-history bytes differ from replayed fixed state",
                    recorded=post.get("model_state_bytes"), expected=expected_model_bytes)

        boundary = decision + 1
        if boundary in checkpoint_map:
            cp, cp_bytes = _verify_checkpoint(
                audit, checkpoint_map[boundary], boundary, after_bytes, foundation_bytes,
                frozen["config"], method, models, stats, rngs,
                f"{where}/checkpoints/decision_{boundary:03d}",
            )
            if cp is not None and cp_bytes is not None:
                audit.check(int(cp.get("parent_metadata", {}).get("version", -99)) == expected_parent_version
                            and int(cp.get("learner", {}).get("parent_group", -99)) == expected_binding,
                            "checkpoint_parent_chain", f"{where}/checkpoints/decision_{boundary:03d}",
                            "checkpoint parent version/binding differs from replayed actions",
                            expected_version=expected_parent_version, expected_binding=expected_binding)
                boundary_payloads[boundary] = (cp, cp_bytes)
                expected_binding = int(cp.get("learner", {}).get("parent_group", expected_binding))
                expected_parent_version = int(cp.get("parent_metadata", {}).get("version", expected_parent_version))

    # Check the first uncompleted pre-action on a failed stream is still only a
    # single pending label and did not receive an outcome target.
    if len(preactions) > completed:
        pending_pre = preactions[completed]
        pending_event = package_events[completed] if completed < len(package_events) else {}
        audit.check(status != "completed" and int(pending_pre.get("decision_index", -1)) == completed
                    and pending_event.get("pending_feedback", {}).get("decision_index") == completed,
                    "failed_pending_decision", where,
                    "failed stream's final flushed action lacks its single serialized pending feedback state")
        if pending_pre.get("selected_action") is not None:
            selected_pending = str(pending_pre["selected_action"])
            action_counts[selected_pending] = action_counts.get(selected_pending, 0) + 1
            selected_count += 1
            if selected_pending != "keep":
                write_count += 1
        try:
            pending_bytes = int(pending_event.get("package_bytes", 0))
            package_max = max(package_max, pending_bytes)
            preaction_bytes = len(json.dumps(pending_pre, sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False).encode("utf-8"))
            transient_peak_estimate = max(transient_peak_estimate,
                                          int(pending_pre.get("package_bytes_before", 0)) + preaction_bytes + 36,
                                          pending_bytes)
            audit.check(0 < pending_bytes <= int(frozen["config"]["package_budget_bytes"]),
                        "failed_inflight_package_cap", where,
                        "failed stream's last pending-feedback package is missing or over the cap",
                        package_bytes=pending_bytes)
        except Exception as exc:
            audit.fail("failed_inflight_package_read", where,
                       "failed stream's pending package size could not be audited", error=repr(exc))
    for event in package_events:
        try:
            package_max = max(package_max, int(event.get("package_bytes", 0)))
        except Exception:
            pass

    if initial_payload is not None:
        _checkpoint_optimizer(audit, initial_payload, method, frozen["config"], f"{where}/initial")
    if method == "shared_ppo" and len(boundary_payloads) > 1:
        first_learner = boundary_payloads[0][0]["learner"]
        last_learner = boundary_payloads[max(boundary_payloads)][0]["learner"]
        try:
            _local_a, first_shared, _groups_a = storage_audit._core_arrays(first_learner)
            _local_b, last_shared, _groups_b = storage_audit._core_arrays(last_learner)
            audit.check(not np.array_equal(first_shared, last_shared), "shared_ppo_parameter_update", where,
                        "direct shared PPO did not change shared actor coefficients across real rollouts")
        except Exception as exc:
            audit.fail("shared_ppo_parameter_audit", where,
                       "direct shared PPO shared coefficients could not be compared", error=repr(exc))

    expected_costs = {
        "environment_steps": int(raw.get("raw_steps", 0)),
        "warmup_steps": 0,
        "outcome_steps": 0,
        "ppo_rollouts": len(rollouts),
        "minibatch_updates": sum(int(row.get("updates", 0)) for row in rollouts),
        "storage_action_count": selected_count,
        "parent_write_count": write_count,
        "value_model_updates": model_update_count,
    }
    expected_costs["warmup_steps"] = int(raw.get("warmup_steps", 0))
    expected_costs["outcome_steps"] = int(raw.get("outcome_steps", 0))
    final_runtime = {}
    if boundary_payloads:
        final_runtime = boundary_payloads[max(boundary_payloads)][0].get("learner", {}).get("runtime_counts", {})
    observed_allocations = sum(bool(row.get("allocated")) for row in bouts)
    observed_evictions = sum(row.get("evicted_task") is not None for row in bouts)
    expected_costs["slot_allocation_count"] = observed_allocations
    expected_costs["eviction_count"] = observed_evictions
    for key, expected in expected_costs.items():
        audit.check(int(costs.get(key, -1)) == expected, "stream_cost_total", where,
                    "stream summary cost counter differs from raw evidence or ledgers", field=key,
                    recorded=costs.get(key), expected=expected)
    if status == "completed" or (boundary_payloads and max(boundary_payloads) == completed):
        audit.check(int(final_runtime.get("slot_allocations", -1)) == observed_allocations
                    and int(final_runtime.get("slot_evictions", -1)) == observed_evictions,
                    "resident_slot_costs", where,
                    "reported eviction/allocation costs differ from bout and final learner evidence",
                    learner_runtime=final_runtime, bout_allocations=observed_allocations,
                    bout_evictions=observed_evictions)
    else:
        audit.check(int(costs.get("slot_allocation_count", -1)) >= observed_allocations
                    and int(costs.get("eviction_count", -1)) >= observed_evictions,
                    "failed_resident_slot_costs", where,
                    "failed stream costs omit already logged slot allocations or evictions")
        expected_costs["slot_allocation_count"] = int(costs.get("slot_allocation_count", -1))
        expected_costs["eviction_count"] = int(costs.get("eviction_count", -1))
    audit.check(_close(costs.get("utility_cpu_s"), utility_cpu_sum, atol=1e-7, rtol=1e-8)
                and _close(costs.get("model_update_cpu_s"), model_update_cpu_sum, atol=1e-7, rtol=1e-8)
                and _finite(costs.get("package_serialization_cpu_s"), nonnegative=True)
                and float(costs.get("package_serialization_cpu_s", 0.0)) >= sum(
                    float(row.get("serialization_cpu_s", 0.0)) for row in package_events
                ) + sum(float(row.get("package_serialization_cpu_s", 0.0)) for row in feedback) - 1e-7
                and _finite(costs.get("total_cpu_s"), nonnegative=True)
                and _finite(costs.get("wall_s"), nonnegative=True),
                "stream_cpu_costs", where,
                "measured utility/model/package/total CPU and wall costs do not reconcile")
    for action, expected in action_counts.items():
        audit.check(int(summary.get("storage_action_counts", {}).get(action, -1)) == expected,
                    "storage_action_count", where,
                    "stream action total differs from its selected action log", action=action,
                    recorded=summary.get("storage_action_counts", {}).get(action), expected=expected)
    audit.check(len(rollouts) == int(costs.get("ppo_rollouts", -1))
                and all(int(row.get("rows", -1)) == int(frozen["config"]["rollout_steps"])
                        and int(row.get("updates", -1)) == 16 for row in rollouts),
                "rollout_accounting", where,
                "PPO rollout ledger contains a partial/unoptimized rollout or disagrees with summary")
    schedule = _schedule_by_bout(mapping, seed, requested)
    for index, row in enumerate(rollouts):
        try:
            bout = int(row.get("bout", -1))
            plan = schedule[bout]
            expected_action = plan.get("action", "") if bout < 4 else \
                (preactions[(bout - 4) // 3].get("selected_action") or "")
            audit.check(row.get("phase") == ("warmup" if bout < 4 else "outcome")
                        and row.get("action", "") == expected_action
                        and row.get("future_regime", "") in ("", None),
                        "rollout_selected_action", f"{where}/rollouts[{index}]",
                        "PPO rollout action/phase differs from the selected trajectory or includes future schedule",
                        bout=bout, recorded_action=row.get("action"), expected_action=expected_action)
        except Exception as exc:
            audit.fail("rollout_schedule", f"{where}/rollouts[{index}]",
                       "rollout row could not be joined to its scheduled bout", error=repr(exc))
    audit.check(sum(int(row.get("rows", 0)) for row in rollouts) <= int(raw.get("raw_steps", 0))
                and (status != "completed" or sum(int(row.get("rows", 0)) for row in rollouts)
                     == int(raw.get("raw_steps", 0))),
                "rollout_raw_coverage", where, "PPO rollout rows exceed actual raw interaction rows")
    audit.check(_close(summary.get("total_reward_sum"), raw.get("raw_reward_sum", 0.0), atol=1e-7, rtol=1e-8)
                and _close(summary.get("warmup_return_sum"), raw.get("warmup_reward_sum", 0.0),
                           atol=1e-7, rtol=1e-8)
                and _close(summary.get("outcome_return_sum"), raw.get("outcome_reward_sum", 0.0),
                           atol=1e-7, rtol=1e-8),
                "raw_reward_totals", where,
                "summary reward totals differ from the saved raw transition ledger")
    audit.check(int(summary.get("peak_package_bytes", -1)) == package_max,
                "package_peak_total", where,
                "reported package peak differs from initial, in-flight and boundary ledgers",
                recorded=summary.get("peak_package_bytes"), expected=package_max)
    audit.check(int(summary.get("actual_actor_bytes", -1)) == int(frozen["config"]["actor_budget_bytes"]),
                "summary_actor_bytes", where, "summary actual actor size differs from the fixed actor container")
    return {
        "stream": where, "status": status, "method": method, "pair_kind": pair_kind, "seed": seed,
        "requested_decisions": requested, "completed_decisions": completed,
        "raw_steps": int(raw.get("raw_steps", 0)), "rollouts": len(rollouts),
        "minibatch_updates": sum(int(row.get("updates", 0)) for row in rollouts),
        "selected_action_count": selected_count, "model_updates": model_update_count,
        "peak_persistent_package_bytes": package_max,
        "estimated_peak_with_preaction_and_candidate_bytes": transient_peak_estimate,
        "utility_cpu_s": utility_cpu_sum, "model_update_cpu_s": model_update_cpu_sum,
        "failure": summary.get("failure"),
        "prefix_behavior_sha256": raw.get("prefix_behavior_sha256"),
        "prefix_behavior_rows": raw.get("prefix_behavior_rows", 0),
        "preactions": preactions, "raw_path": stream_dir / "rawsteps.csv",
    }


def _protocol_variant_summary(audit: Audit, cohort: Path, frozen: Mapping[str, Any]) -> dict[str, Any]:
    variants: dict[str, list[dict[str, Any]]] = {}
    for item in audit.protocol_snapshots:
        digest = str(item["sha256"])
        variants.setdefault(digest, []).append(item)
    expected = str(frozen.get("protocol_sha256", ""))
    current_path = ROOT / "docs" / "RETENTION_VALUE_STREAM_PROTOCOL.md"
    try:
        current = current_path.read_bytes()
        current_hash = _digest(current_path)
    except Exception as exc:
        audit.fail("protocol_current_unavailable", str(current_path),
                   "current protocol could not be read to reconcile pilot snapshot provenance", error=repr(exc))
        return {"hashes": {key: len(value) for key, value in variants.items()}, "drift": True}
    if set(variants) == {expected}:
        observed = variants[expected][0]["bytes"] if variants[expected] else b""
        documented_append = current_hash != expected and _is_protocol_clarification_append(observed, current)
        if documented_append and frozen.get("stage") == "calibration" and bool(frozen.get("pilot")):
            audit.warn("calibration_protocol_preclarification", str(cohort),
                       "calibration pilot snapshots preserve the original protocol; the fidelity clarification was appended afterward, before main runs",
                       frozen_sha256=expected, current_sha256=current_hash,
                       change="strict append after frozen protocol bytes")
        return {"hashes": {key: len(value) for key, value in variants.items()},
                "drift": documented_append, "documented_append": documented_append}
    old_values = variants.get(expected, [])
    alternatives = [key for key in variants if key != expected]
    allowed = (frozen.get("stage") == "comparison" and bool(frozen.get("pilot"))
               and len(old_values) == 9 and len(alternatives) == 1
               and len(variants[alternatives[0]]) == 1)
    if allowed:
        try:
            baseline = old_values[0]["bytes"]
            revised = variants[alternatives[0]][0]["bytes"]
            allowed = _is_protocol_clarification_append(baseline, revised)
        except Exception:
            allowed = False
    if allowed:
        audit.warn("pilot_protocol_documentation_drift", str(cohort),
                   "one comparison-pilot stream captured the appended pre-main fidelity clarification; the other nine preserve the frozen original protocol text",
                   frozen_sha256=expected, revised_sha256=alternatives[0], old_streams=len(old_values),
                   revised_streams=len(variants[alternatives[0]]), current_sha256=current_hash,
                   change="strict documentation append after frozen protocol bytes")
    else:
        audit.fail("protocol_snapshot_drift", str(cohort),
                   "protocol snapshot hashes differ outside the documented comparison-pilot append",
                   frozen_sha256=expected, variants={key: len(value) for key, value in variants.items()},
                   current_sha256=current_hash)
    return {"hashes": {key: len(value) for key, value in variants.items()}, "drift": True,
            "documented_append": bool(allowed)}


def _is_protocol_clarification_append(baseline: bytes, revised: bytes) -> bool:
    if not revised.startswith(baseline):
        return False
    suffix = revised[len(baseline):]
    heading = b"\n## Pre-main fidelity clarification (2026-10-19)\n\n"
    return bool(suffix) and suffix.startswith(heading) and b"\n## " not in suffix[len(heading):]


def _audit_matched_learners(audit: Audit, streams: Sequence[Mapping[str, Any]], frozen: Mapping[str, Any]) -> None:
    groups: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for stream in streams:
        if stream.get("method") in ("learned_linear", "learned_intercept"):
            groups.setdefault((str(stream["pair_kind"]), int(stream["seed"])), {})[str(stream["method"])] = stream
    for key, methods in groups.items():
        if set(methods) != {"learned_linear", "learned_intercept"}:
            continue
        left, right = methods["learned_linear"], methods["learned_intercept"]
        count = min(int(left["completed_decisions"]), int(right["completed_decisions"]), 16)
        audit.check([row.get("selected_action") for row in left["preactions"][:count]]
                    == [row.get("selected_action") for row in right["preactions"][:count]],
                    "matched_selector_actions", f"{key[0]}/seed_{key[1]}",
                    "matched learned selectors do not share the common uniform exploration prefix")
        audit.check(all(_same_json(a.get("rng_state_before"), b.get("rng_state_before"))
                        for a, b in zip(left["preactions"][:count], right["preactions"][:count])),
                    "matched_selector_rng", f"{key[0]}/seed_{key[1]}",
                    "learned selectors do not consume their matched selector RNG streams identically")
        rows = min(int(left.get("prefix_behavior_rows", 0)), int(right.get("prefix_behavior_rows", 0)))
        expected_rows = 3072 + count * 1536
        if rows == expected_rows:
            audit.check(left.get("prefix_behavior_sha256") == right.get("prefix_behavior_sha256"),
                        "matched_environment_prefix", f"{key[0]}/seed_{key[1]}",
                        "matched learned selectors produced different raw behavior before the exploratory prefix ended",
                        rows=rows)
        elif rows:
            audit.warn("matched_prefix_partial", f"{key[0]}/seed_{key[1]}",
                       "matched environment comparison covers only the common raw prefix",
                       observed_rows=rows, expected_rows=expected_rows)


def audit_cohort(cohort: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    cohort_path = Path(cohort).resolve()
    audit = Audit(cohort_path)
    digest_cache: dict[Path, str] = {}
    try:
        frozen = _json(cohort_path / "frozen_config.json")
        summary = _json(cohort_path / "summary.json")
    except Exception as exc:
        audit.fail("cohort_metadata", str(cohort_path),
                   "cohort frozen config or summary is missing/unreadable", error=repr(exc))
        frozen, summary = {}, {}
    audit.check(frozen.get("format") == RUN_CONFIG_FORMAT and summary.get("format") == SUMMARY_FORMAT,
                "cohort_format", str(cohort_path), "cohort format identifiers are not recognized")
    stage, pilot = str(frozen.get("stage", "")), bool(frozen.get("pilot"))
    config = frozen.get("config", {})
    expected = _expected_grid(config, stage, pilot) if isinstance(config, Mapping) else []
    audit.check(summary.get("stage") == stage and bool(summary.get("pilot")) == pilot,
                "cohort_identity", str(cohort_path), "summary stage or pilot flag differs from its frozen config")
    audit.check(frozen.get("requested_grid") == expected, "frozen_grid", str(cohort_path),
                "frozen requested grid does not match the config's selected cohort")
    root_rows = summary.get("streams", [])
    audit.check(isinstance(root_rows, list) and len(root_rows) == len(expected),
                "cohort_stream_count", str(cohort_path),
                "cohort summary stream count differs from the frozen requested grid",
                observed=len(root_rows) if isinstance(root_rows, list) else None, expected=len(expected))
    expected_keys = {_spec_key(row): row for row in expected}
    root_keys = [_spec_key(row) for row in root_rows if isinstance(row, Mapping)] if isinstance(root_rows, list) else []
    audit.check(len(root_keys) == len(set(root_keys)) and set(root_keys) == set(expected_keys),
                "cohort_stream_grid", str(cohort_path),
                "cohort stream rows contain duplicates or differ from the frozen grid")
    streams = [_audit_stream(audit, cohort_path, frozen, spec, digest_cache) for spec in expected]
    _audit_matched_learners(audit, streams, frozen)
    protocol_summary = _protocol_variant_summary(audit, cohort_path, frozen)
    totals = summary.get("totals", {})
    for key in COHORT_COST_KEYS:
        observed = sum(int(stream.get("costs", {}).get(key, 0)) for stream in [])
        # Root aggregate rows carry the per-stream costs; preserve exact values
        # without retaining raw transitions in the cohort result.
        observed = sum(int(row.get("costs", {}).get(key, 0)) for row in root_rows if isinstance(row, Mapping))
        audit.check(int(totals.get(key, -1)) == observed, "cohort_cost_total", str(cohort_path),
                    "root cohort total differs from its per-stream cost records", field=key,
                    recorded=totals.get(key), expected=observed)
    for key in COHORT_FLOAT_COST_KEYS:
        observed = sum(float(row.get("costs", {}).get(key, 0.0)) for row in root_rows if isinstance(row, Mapping))
        audit.check(_close(totals.get(key), observed, atol=1e-7, rtol=1e-8), "cohort_cost_total",
                    str(cohort_path), "root cohort CPU/wall cost differs from per-stream totals",
                    field=key, recorded=totals.get(key), expected=observed)
    audit.check(int(summary.get("completed_stream_count", -1)) == sum(s.get("status") == "completed" for s in streams)
                and len(summary.get("failures", [])) == sum(s.get("status") != "completed" for s in streams),
                "cohort_failure_accounting", str(cohort_path),
                "root summary does not preserve completed and failed stream counts")
    persistent_peak = max((int(stream.get("peak_persistent_package_bytes", 0)) for stream in streams), default=0)
    transient_peak = max((int(stream.get("estimated_peak_with_preaction_and_candidate_bytes", 0))
                         for stream in streams), default=0)
    result = {
        "format": FORMAT,
        "cohort": str(cohort_path),
        "stage": stage,
        "pilot": pilot,
        "stream_count": len(streams),
        "completed_stream_count": sum(stream.get("status") == "completed" for stream in streams),
        "failed_stream_count": sum(stream.get("status") != "completed" for stream in streams),
        "valid": not audit.errors,
        "errors": audit.errors,
        "warnings": audit.warnings,
        "protocol_provenance": protocol_summary,
        "package_accounting": {
            "cap_bytes": int(config.get("package_budget_bytes", -1)),
            "measured_persistent_peak_bytes": persistent_peak,
            "estimated_live_peak_with_preaction_and_candidate_bytes": transient_peak,
            "transient_estimate_note": "Estimate adds serialized pre-action row and one float32 3 x 3 candidate (36 bytes) to the preceding package boundary; in-flight package measurements already include serialized pending feedback. This is not a package-cap breach calculation.",
            "arbitrary_step_exact_resume_supported": False,
            "resume_note": "Decision-boundary checkpoints have no pending feedback and omit external simulator state; they do not establish exact arbitrary-step environment resumption.",
        },
        "streams": [{key: value for key, value in stream.items()
                     if key not in ("preactions", "raw_path")} for stream in streams],
    }
    if output is not None:
        target = Path(output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True,
                        help="existing retention stream pilot or main output directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="write the audit JSON here; this does not modify the cohort")
    args = parser.parse_args()
    result = audit_cohort(args.cohort, args.output)
    print(json.dumps({"cohort": result["cohort"], "stage": result["stage"],
                      "pilot": result["pilot"], "valid": result["valid"],
                      "stream_count": result["stream_count"],
                      "errors": len(result["errors"]), "warnings": len(result["warnings"]),
                      "output": str(args.output.resolve())}, ensure_ascii=False), flush=True)
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
