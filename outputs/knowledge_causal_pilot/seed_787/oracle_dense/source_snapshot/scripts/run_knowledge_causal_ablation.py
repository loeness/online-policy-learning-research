"""Run a scoped causal ablation of the knowledge-loop routing/compression path.

This harness reuses ``run_knowledge_loop.run_arm`` and ``OnlineArm`` while
temporarily replacing only the selected runtime pieces.  The original runner
and its outputs are left untouched.  Oracle conditions are privileged
diagnostics: the external schedule context is passed through ``OracleHarness``
before each action and must not be read as an unknown-task algorithm result.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_knowledge_loop as loop


PILOT_SEED = 787
PILOT_STEPS = 8192
FORMAL_SEEDS = (541, 557, 569)
FORMAL_STEPS = 32768
ROLLOUT_STEPS = 256
CONSOLIDATE_EVERY = 1024
INT8_BUDGET_BYTES = 320
DENSE_BUDGET_BYTES = 664
CONDITIONS = (
    "learned_int8",
    "oracle_int8",
    "learned_dense",
    "oracle_dense",
    "oracle_gate",
    "learned_int8_preserve_trial",
)
UNDERLYING_ARMS = {
    "learned_int8": "int8",
    "oracle_int8": "int8",
    "learned_dense": "int8",
    "oracle_dense": "int8",
    "oracle_gate": "evidence_gate",
    "learned_int8_preserve_trial": "int8",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=loop._json_default),
        encoding="utf-8",
    )


def condition_patch(condition: str) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown causal condition {condition!r}")
    oracle = condition.startswith("oracle_")
    dense = condition.endswith("_dense")
    return {
        "condition": condition,
        "underlying_original_arm": UNDERLYING_ARMS[condition],
        "router": "privileged fixed context oracle" if oracle else "existing TaskFreeRouter",
        "oracle_context_source": "external hidden schedule, before action" if oracle else None,
        "oracle_context_to_slot": (
            {name: index for index, name in enumerate(loop.C1_CONTEXTS)} if oracle else None
        ),
        "oracle_confidence": 1.0 if oracle else None,
        "oracle_allocations": 0 if oracle else None,
        "oracle_lru": False if oracle else None,
        "resource_comparison_fair": not oracle,
        "actor_codec_for_int8_arm": "dense_f32" if dense else "int8",
        "actor_budget_bytes": DENSE_BUDGET_BYTES if dense else INT8_BUDGET_BYTES,
        "expected_actor_binary_bytes_at_full_shape": DENSE_BUDGET_BYTES if dense else 202,
        "head_count": 6,
        "initial_head_values": "all zeros",
        "max_context_slots": 6,
        "rollout_steps": ROLLOUT_STEPS,
        "fit_rollouts_per_1024_step_cycle": 3,
        "heldout_rollouts_per_1024_step_cycle": 1,
        "optimizer_reset_schedule": "existing reset at every consolidation cycle",
        "legacy_evidence_gate_enabled": condition == "oracle_gate",
        "allocation_trial_preservation": condition == "learned_int8_preserve_trial",
        "run_revisit": False,
        "paired_hidden_schedule_seed_offset": 7000,
        "paired_environment_reset_seed_offset": 2200,
        "paired_action_rng_seed_offset": 1100,
        "oracle_is_unknown_task_algorithm_claim": False,
    }


class OracleHarness:
    """Explicit holder for privileged schedule labels used in oracle runs."""

    def __init__(self) -> None:
        self.context_to_slot = {name: index for index, name in enumerate(loop.C1_CONTEXTS)}
        self.current_context: str | None = None
        self.current_step: int | None = None
        self.route_calls = 0

    def set_external_context(self, context: str, step: int) -> None:
        if context not in self.context_to_slot:
            raise KeyError(f"unknown external context {context!r}")
        self.current_context = str(context)
        self.current_step = int(step)

    def route_before_action(self, _observation: Any) -> dict[str, Any]:
        if self.current_context is None:
            raise RuntimeError("oracle route requested before the external context was supplied")
        self.route_calls += 1
        return {
            "slot": self.context_to_slot[self.current_context],
            "confidence": 1.0,
            "support": 1.0,
            "recent_support": 32,
            "status": "privileged_oracle",
            "reason": "external_context_before_action",
        }


class OracleRouter:
    """Runner-compatible fixed mapping; it has no learned profiles or LRU."""

    def __init__(self, harness: OracleHarness) -> None:
        self.harness = harness
        self.allocations = 0
        self.evictions = 0
        self.switches = 0

    @property
    def active_slots(self) -> tuple[int, ...]:
        return tuple(range(len(self.harness.context_to_slot)))

    def nbytes(self) -> dict[str, int]:
        # The external label map is deliberately not presented as learned
        # router memory; summaries mark oracle resource comparisons unfair.
        return {
            "profile_counts": 0,
            "recent_window": 0,
            "novel_pending": 0,
            "candidate_pending": 0,
            "recovery_hits": 0,
            "slot_metadata": 0,
            "total": 0,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": "privileged_fixed_context_mapping",
            "context_to_slot": dict(self.harness.context_to_slot),
            "lru": False,
            "allocations": 0,
        }


@contextmanager
def scoped_runtime(condition: str) -> Iterator[OracleHarness | None]:
    """Apply one diagnostic condition and restore every patched global."""
    patch = condition_patch(condition)
    old_online_arm = loop.OnlineArm
    old_hidden_context_at = loop.hidden_context_at
    old_budget = loop.ACTOR_BUDGET_BYTES
    old_codec_map = loop.CODEC_FOR_ARM
    oracle = OracleHarness() if patch["router"] == "privileged fixed context oracle" else None
    try:
        if oracle is not None:
            base_arm = old_online_arm

            class OracleOnlineArm(base_arm):  # type: ignore[misc, valid-type]
                def __init__(self, arm: str, seed: int, base_config: Mapping[str, Any]) -> None:
                    super().__init__(arm, seed, base_config)
                    self.router = OracleRouter(oracle)

                def route_before_action(self, obs: Any) -> dict[str, Any]:
                    return oracle.route_before_action(obs)

                def observe_transition(self, obs: Any, action: int, reward: float, next_obs: Any) -> None:
                    # Oracle routing has no learned profile, allocation, or LRU.
                    return None

            loop.OnlineArm = OracleOnlineArm

            def set_context_before_route(schedule: list[Mapping[str, Any]], step: int) -> Mapping[str, Any]:
                segment = old_hidden_context_at(schedule, step)
                oracle.set_external_context(str(segment["context"]), int(step))
                return segment

            loop.hidden_context_at = set_context_before_route
        elif condition == "learned_int8_preserve_trial":
            # Parent-owned reusable fix: this condition changes only allocation
            # semantics while retaining the legacy learned router and int8 codec.
            from scripts import knowledge_v2_components

            loop.OnlineArm = knowledge_v2_components.PreservingTrialArm

        if patch["actor_codec_for_int8_arm"] == "dense_f32":
            loop.ACTOR_BUDGET_BYTES = DENSE_BUDGET_BYTES
            loop.CODEC_FOR_ARM = {**old_codec_map, "int8": "dense_f32"}
        else:
            loop.ACTOR_BUDGET_BYTES = INT8_BUDGET_BYTES
        yield oracle
    finally:
        loop.OnlineArm = old_online_arm
        loop.hidden_context_at = old_hidden_context_at
        loop.ACTOR_BUDGET_BYTES = old_budget
        loop.CODEC_FOR_ARM = old_codec_map


def _snapshot_causal_sources(output: Path, base_config: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    manifest = loop._snapshot_sources(output, base_config)
    snapshot_root = output / "source_snapshot"
    harness_sources: dict[str, dict[str, Any]] = {}
    source_paths = [
        Path("scripts/run_knowledge_causal_ablation.py"),
        Path("tests/test_knowledge_causal_ablation.py"),
    ]
    if patch["allocation_trial_preservation"]:
        source_paths.append(Path("scripts/knowledge_v2_components.py"))
    for relative in source_paths:
        source = ROOT / relative
        if not source.is_file():
            if relative.name.startswith("test_"):
                continue
            raise FileNotFoundError(f"causal harness source is missing: {relative}")
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = _sha256(source)
        snapshot_hash = _sha256(destination)
        if source_hash != snapshot_hash:
            raise RuntimeError(f"causal harness changed while snapshotting: {relative}")
        harness_sources[str(relative)] = {
            "sha256": snapshot_hash,
            "bytes": destination.stat().st_size,
        }
    patch_snapshot = snapshot_root / "condition_patch.json"
    _write_json(patch_snapshot, patch)
    harness_sources["source_snapshot/condition_patch.json"] = {
        "sha256": _sha256(patch_snapshot),
        "bytes": patch_snapshot.stat().st_size,
    }
    manifest["causal_harness_sources"] = harness_sources
    manifest["condition_patch"] = patch
    manifest["source_snapshot_scope"] = (
        "original knowledge-loop source plus this harness and the exact per-condition runtime patch"
    )
    _write_json(snapshot_root / "manifest.json", manifest)
    return manifest


def _routing_metrics(path: Path) -> dict[str, Any]:
    per_context: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"steps": 0, "covered_steps": 0, "slots": set()}
    )
    total_steps = 0
    covered_steps = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            context = str(row["hidden_context"])
            slot = int(row["route_slot"])
            total_steps += 1
            entry = per_context[context]
            entry["steps"] += 1
            if slot >= 0:
                covered_steps += 1
                entry["covered_steps"] += 1
                entry["slots"].add(slot)
    return {
        "total_steps": total_steps,
        "covered_steps": covered_steps,
        "coverage_fraction": covered_steps / max(1, total_steps),
        "per_context_steps": {
            name: {
                "steps": values["steps"],
                "covered_steps": values["covered_steps"],
                "coverage_fraction": values["covered_steps"] / max(1, values["steps"]),
                "routed_slots": sorted(values["slots"]),
            }
            for name, values in sorted(per_context.items())
        },
    }


def _labelled_summary(
    condition: str,
    underlying_arm: str,
    patch: Mapping[str, Any],
    raw_summary: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(raw_summary))
    routing = _routing_metrics(output / "steps.csv")
    counts = dict(result.get("counts", {}))
    fit_rows = int(counts.get("local_trial_steps", 0))
    masked_rows = int(counts.get("stale_actor_rows", 0))
    oracle = condition.startswith("oracle_")
    result.update({
        "condition": condition,
        "underlying_original_arm": underlying_arm,
        "diagnostic_type": "privileged_oracle" if oracle else "learned_or_controlled_ablation",
        "is_privileged_oracle": oracle,
        "oracle_unknown_task_algorithm_claim": False,
        "resource_comparison_fair": not oracle,
        "condition_patch": dict(patch),
        "routing_metrics": routing,
        "valid_actor_rows": max(0, fit_rows - masked_rows),
        "masked_actor_rows": masked_rows,
        "actor_rows_presented_for_fit": fit_rows,
        "actor_bytes_actual_numeric": int(result.get("actor_compact_numeric_bytes", 0)),
        "actor_bytes_actual_binary_file": int(result.get("actor_compact_binary_file_bytes", 0)),
        "router_bytes_reported": int(dict(result.get("router_memory", {})).get("total", 0)),
        "router_bytes_resource_comparable": not oracle,
        "run_arm_summary": copy.deepcopy(dict(raw_summary)),
    })
    _write_json(output / "summary.json", result)
    return result


def run_phase(phase: str, *, output_dir: str | Path | None = None, formal_reviewed: bool = False) -> dict[str, Any]:
    if phase == "pilot":
        seeds, steps = (PILOT_SEED,), PILOT_STEPS
        target = Path(output_dir) if output_dir is not None else ROOT / "outputs/knowledge_causal_pilot"
    elif phase == "formal":
        if not formal_reviewed:
            raise RuntimeError("formal ablation is held until the root reviews the pilot and freezes the protocol")
        seeds, steps = FORMAL_SEEDS, FORMAL_STEPS
        target = Path(output_dir) if output_dir is not None else ROOT / "outputs/knowledge_causal_formal"
    else:
        raise ValueError("phase must be pilot or formal")
    if tuple(CONDITIONS) != tuple(UNDERLYING_ARMS):
        raise AssertionError("every causal condition must have exactly one underlying original arm")
    if steps % CONSOLIDATE_EVERY:
        raise AssertionError("run length must be an exact multiple of the consolidation schedule")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"causal output directory is non-empty; preserving existing results: {target}")

    loop.torch.set_num_threads(1)
    base_config = loop.load_base_config()
    all_runs: list[dict[str, Any]] = []
    for seed in seeds:
        schedule = loop.make_hidden_schedule(steps, seed + 7000)
        for condition in CONDITIONS:
            underlying_arm = UNDERLYING_ARMS[condition]
            output = target / f"seed_{seed}" / condition
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"causal run output already has files; preserving them: {output}")
            patch = condition_patch(condition)
            source_manifest = _snapshot_causal_sources(output, base_config, patch)
            config_record = {
                "phase": phase,
                "seed": seed,
                "condition": condition,
                "underlying_original_arm": underlying_arm,
                "steps": steps,
                "hidden_schedule_seed": seed + 7000,
                "hidden_schedule": schedule,
                "condition_order": list(CONDITIONS),
                "condition_patch": patch,
                "same_w0_for_all_conditions": True,
                "foundation_sha256": loop._hash(Path(base_config["foundation"])),
                "protocol_sha256": loop._hash(loop.ROOT / "docs" / "KNOWLEDGE_ORGANIZATION_PROTOCOL.md"),
                "source_snapshot_manifest": "source_snapshot/manifest.json",
                "source_snapshot_file_count": len(source_manifest["sources"]) + len(source_manifest["causal_harness_sources"]),
                "run_revisit": False,
            }
            _write_json(output / "config.json", config_record)
            with scoped_runtime(condition):
                raw_summary = loop.run_arm(
                    underlying_arm,
                    seed=seed,
                    total_steps=steps,
                    schedule=schedule,
                    base_config=base_config,
                    output=output,
                    phase=phase,
                    run_revisit=False,
                )
            labelled = _labelled_summary(condition, underlying_arm, patch, raw_summary, output)
            all_runs.append(labelled)

    aggregate = {
        "phase": phase,
        "seeds": list(seeds),
        "conditions": list(CONDITIONS),
        "underlying_original_arms": dict(UNDERLYING_ARMS),
        "steps_per_run": steps,
        "paired_by_seed_and_hidden_schedule": True,
        "fit_and_holdout_schedule": {
            "rollout_steps": ROLLOUT_STEPS,
            "fit_rollouts_per_1024_step_cycle": 3,
            "heldout_rollouts_per_1024_step_cycle": 1,
            "optimizer_reset": "existing consolidation reset schedule",
        },
        "interpretation": (
            "causal diagnostic on C1; oracle arms receive privileged external context and are not an unknown-task algorithm claim"
        ),
        "runs": all_runs,
    }
    _write_json(target / "summary.json", aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--formal-reviewed",
        action="store_true",
        help="explicit root authorization after pilot review and protocol freeze",
    )
    args = parser.parse_args()
    run_phase(args.phase, output_dir=args.output_dir, formal_reviewed=args.formal_reviewed)


if __name__ == "__main__":
    main()
