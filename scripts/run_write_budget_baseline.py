"""Run an interaction-matched independent G0 baseline for write streams.

This is a post-selection budget control, not a compute-matched optimization
comparison. It reuses the stream's independent7_lru actor, router, cache,
optimizer, serialization, and evaluation implementation while removing paid
uniform probes and assigning every fresh rollout, including the fourth
rollout in each 1,024-row cycle, to fit. The stock forced-boundary minimum-size
drop still applies to fragments shorter than 32 rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_write_stream as stream


CONDITION = "independent7_lru"
KNOWN_BOUT_STEPS = 2_144
UNKNOWN_BOUT_STEPS = 2_048
INTERNAL_EVIDENCE_WINDOW = 128
SOURCE_ADDITIONS = (
    Path("scripts/run_write_budget_baseline.py"),
    Path("docs/WRITE_BUDGET_BASELINE.md"),
)

_ORIGINALS: dict[str, Any] = {
    "load_config": stream._load_config,
    "new_agent": stream._new_agent,
    "reset_allocated_slot": stream._reset_allocated_slot,
    "process_rollout": stream._process_rollout,
    "run_one": stream._run_one,
}
_INSTALLED = False


def default_bout_steps(phase: str) -> int:
    if phase == "known":
        return KNOWN_BOUT_STEPS
    if phase == "unknown":
        return UNKNOWN_BOUT_STEPS
    raise ValueError("phase must be 'known' or 'unknown'")


def expected_training_interactions(phase: str, bout_steps: int | None = None) -> int:
    steps = default_bout_steps(phase) if bout_steps is None else int(bout_steps)
    if steps < 1:
        raise ValueError("bout_steps must be positive")
    return steps * len(stream.STREAM_SEQUENCE)


def _wrapped_load_config() -> dict[str, Any]:
    config = dict(_ORIGINALS["load_config"]())
    config.update({
        "budget_baseline": True,
        "probe_steps": 0,
        "all_rows_fit": True,
        "local_slots": 7,
        "fast_local_learning_rate": 0.004,
        "actor_budget_bytes": 236,
        "independent7_actor_budget_bytes": 236,
        "package_budget_bytes": 262_144,
    })
    return config


def _wrapped_new_agent(
    condition: str,
    seed: int,
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
):
    if str(condition) != CONDITION:
        raise ValueError(f"budget baseline only supports {CONDITION}")
    actual_config = dict(config)
    if not bool(actual_config.get("budget_baseline")):
        raise ValueError("budget baseline configuration flag is missing")
    if int(actual_config.get("probe_steps", -1)) != 0:
        raise ValueError("budget baseline must use zero paid probe steps")

    # The stock constructor uses probe_steps as the evidence-window capacity.
    # Keep that bounded runtime buffer at its ordinary size while exposing the
    # actual zero-probe run configuration to the stream and its ledgers.
    internal_config = dict(actual_config)
    internal_config["probe_steps"] = INTERNAL_EVIDENCE_WINDOW
    agent = _ORIGINALS["new_agent"](
        condition, seed, internal_config, base_config
    )
    agent.run_config = actual_config
    return agent


def _wrapped_reset_allocated_slot(agent: Any, slot: int, *, reason: str) -> int:
    # The stream's generation reset also constructs a window from run_config.
    # Temporarily supply the internal evidence capacity; the paid-probe budget
    # remains zero before and after the reset.
    actual_probe_steps = int(agent.run_config["probe_steps"])
    agent.run_config["probe_steps"] = INTERNAL_EVIDENCE_WINDOW
    try:
        return _ORIGINALS["reset_allocated_slot"](agent, slot, reason=reason)
    finally:
        agent.run_config["probe_steps"] = actual_probe_steps


def _wrapped_process_rollout(agent: Any, **kwargs: Any) -> None:
    if bool(agent.run_config.get("budget_baseline")) and agent.rollout:
        # Preserve freshness and order: every current rollout is designated
        # fit and passed once with its stored behavior log probabilities. The
        # stock function still drops forced fragments shorter than 32 rows;
        # the fourth full 256-row rollout is fitted rather than held out.
        agent.rollout_phase = "fit"
    _ORIGINALS["process_rollout"](agent, **kwargs)


def _wrapped_run_one(
    config: Mapping[str, Any], selection: Mapping[str, Any], output: Path
) -> dict[str, Any]:
    # This top-level target is pickled by ProcessPoolExecutor. Importing this
    # module in a spawned worker reinstalls the same hooks before the stock
    # stream runner constructs the agent or processes rollouts.
    install_hooks()
    result = _ORIGINALS["run_one"](config, selection, output)
    counts = dict(result.get("counts", {}))
    costs = dict(result.get("costs", {}))
    expected = expected_training_interactions(
        str(config["phase"]), int(config["bout_steps"])
    )
    result["budget_comparison"] = {
        "basis": "training_and_probe_interactions_matched_only",
        "expected_training_interactions": expected,
        "observed_training_and_probe_interactions": int(counts.get("raw_steps", 0)),
        "evaluation_interactions_separate": int(costs.get("evaluation_steps", 0)),
        "paid_probe_interactions": int(counts.get("probe_steps", 0)),
        "optimization_accounting": {
            "ppo_rollouts_trained": int(counts.get("rollouts_trained", 0)),
            "minibatch_updates": int(counts.get("minibatch_updates", 0)),
            "fit_rows": int(counts.get("fit_steps", 0)),
            "holdout_rows": int(counts.get("holdout_steps", 0)),
            "partial_rows_dropped": int(counts.get("partial_rollout_dropped", 0)),
            "stale_actor_rows": int(counts.get("stale_actor_rows", 0)),
            "decision_cpu_seconds": float(costs.get("decision_cpu_s", 0.0)),
            "total_process_cpu_seconds": float(costs.get("cpu_s", 0.0)),
        },
        "compute_matched": False,
    }
    result.setdefault("limitations", []).append(
        "This post-selection G0 control matches training-plus-probe interactions. "
        "It fits the fourth rollout and may perform more PPO minibatch updates; "
        "the full update counts and CPU times are reported, so this is not a "
        "compute-matched comparison."
    )
    stream._write_json(Path(output) / "summary.json", result)
    return result


def install_hooks() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    stream._load_config = _wrapped_load_config
    stream._new_agent = _wrapped_new_agent
    stream._reset_allocated_slot = _wrapped_reset_allocated_slot
    stream._process_rollout = _wrapped_process_rollout
    stream._run_one = _wrapped_run_one
    stream.PROTOCOL_PATH = ROOT / "docs" / "WRITE_BUDGET_BASELINE.md"
    stream.SOURCE_FILES = tuple(dict.fromkeys((*stream.SOURCE_FILES, *SOURCE_ADDITIONS)))
    _INSTALLED = True


def uninstall_hooks() -> None:
    global _INSTALLED
    if not _INSTALLED:
        return
    stream._load_config = _ORIGINALS["load_config"]
    stream._new_agent = _ORIGINALS["new_agent"]
    stream._reset_allocated_slot = _ORIGINALS["reset_allocated_slot"]
    stream._process_rollout = _ORIGINALS["process_rollout"]
    stream._run_one = _ORIGINALS["run_one"]
    stream.PROTOCOL_PATH = ROOT / "docs" / "WRITE_STREAM_PROTOCOL.md"
    additions = set(SOURCE_ADDITIONS)
    stream.SOURCE_FILES = tuple(path for path in stream.SOURCE_FILES if path not in additions)
    _INSTALLED = False


def run_cohort(
    phase: str,
    output: str | Path,
    *,
    selection_path: str | Path,
    workers: int = 1,
    seeds: Sequence[int] | None = None,
    bout_steps: int | None = None,
    domain: str | None = None,
    foundation: str | Path | None = None,
) -> dict[str, Any]:
    install_hooks()
    phase = stream._validate_phase(phase)
    steps = default_bout_steps(phase) if bout_steps is None else int(bout_steps)
    return stream.run_cohort(
        phase,
        output,
        selection_path=selection_path,
        workers=workers,
        conditions=[CONDITION],
        seeds=seeds,
        bout_steps=steps,
        local_slots=7,
        domain=domain,
        foundation=foundation,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("known", "unknown"), required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--bout-steps", type=int, help="override the phase's default budget for pilots")
    parser.add_argument("--domain", choices=("navigation", "detour"))
    parser.add_argument("--foundation", type=Path)
    args = parser.parse_args(argv)
    result = run_cohort(
        args.phase,
        args.output,
        selection_path=args.selection,
        workers=args.workers,
        seeds=args.seeds,
        bout_steps=args.bout_steps,
        domain=args.domain,
        foundation=args.foundation,
    )
    print(json.dumps({
        "phase": result["phase"],
        "conditions": result["conditions"],
        "seeds": result["seeds"],
        "cohort_cost": result["cohort_cost"],
        "budget_baseline": True,
    }, ensure_ascii=False), flush=True)


install_hooks()


if __name__ == "__main__":
    main()
