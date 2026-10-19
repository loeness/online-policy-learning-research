"""Run the frozen, matched-future confirmation for the surrogate-lower rule.

The storage decision controls and branch implementation live in
``run_storage_decision``. This wrapper runs one history-matched future per
case, keeping the past-only selector outputs and all four physical actions.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_storage_decision as base


CONFIG_PATH = ROOT / "configs" / "surrogate_lower_confirmation.json"
PROTOCOL_PATH = ROOT / "docs" / "SURROGATE_LOWER_CONFIRMATION_PROTOCOL.md"
RUNNER_PATH = Path(__file__).resolve()
EXPECTED_REGIMES = ("old_major", "new_major")
EXPECTED_PAIR_KINDS = ("dynamics",)
EXPECTED_MAIN_STEPS = {"bootstrap_steps": 2048, "bout_steps": 1024}


def _read_config(config: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], Path]:
    if isinstance(config, (str, Path)):
        config_path = Path(config).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config_path = CONFIG_PATH.resolve()
        raw = copy.deepcopy(dict(config))
    return raw, config_path


def _phase_config(raw_config: Mapping[str, Any], *, pilot: bool) -> tuple[dict[str, Any], str]:
    phase_config = copy.deepcopy(dict(raw_config))
    phase = "pilot" if pilot else "main"
    if pilot:
        pilot_config = phase_config.get("pilot")
        if not isinstance(pilot_config, Mapping):
            raise ValueError("pilot runs require a pilot config")
        for key in ("seeds", "pair_kinds", "history_regimes", "bootstrap_steps", "bout_steps"):
            if key in pilot_config:
                phase_config[key] = copy.deepcopy(pilot_config[key])

    normalized = base.validate_config(phase_config)
    if tuple(normalized["pair_kinds"]) != EXPECTED_PAIR_KINDS:
        raise ValueError("lower confirmation is frozen to dynamics cases only")
    expected_histories = ("old_major",) if pilot else EXPECTED_REGIMES
    if set(normalized["history_regimes"]) != set(expected_histories) or len(normalized["history_regimes"]) != len(expected_histories):
        if pilot:
            raise ValueError("the frozen pilot uses one old_major history")
        raise ValueError("main lower confirmation requires both old_major and new_major histories")
    if set(normalized["future_regimes"]) != set(EXPECTED_REGIMES) or len(normalized["future_regimes"]) != 2:
        raise ValueError("top-level config must retain both future regimes; workers match them to history")
    if not pilot:
        for key, expected in EXPECTED_MAIN_STEPS.items():
            if int(normalized[key]) != expected:
                raise ValueError(f"main {key} is frozen to {expected}")
    return normalized, phase


def _case_payloads(config: Mapping[str, Any], output: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    """Build one payload per seed/history with exactly its matched future."""
    jobs: list[dict[str, Any]] = []
    for seed in seeds:
        for pair_kind in config["pair_kinds"]:
            for history_regime in config["history_regimes"]:
                worker_config = copy.deepcopy(dict(config))
                # _case_worker uses this list only for its future branches;
                # prefix_schedule continues to use the requested history.
                worker_config["future_regimes"] = [str(history_regime)]
                jobs.append({
                    "seed": int(seed),
                    "pair_kind": str(pair_kind),
                    "history_regime": str(history_regime),
                    "config": worker_config,
                    "output": str(output),
                })
    return jobs


def _snapshot_sources(output: Path, config_path: Path, protocol_path: Path) -> dict[str, Any]:
    """Extend the base provenance snapshot with this runner and its protocol."""
    config_path = config_path.resolve()
    protocol_path = protocol_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"confirmation config is missing: {config_path}")
    if not protocol_path.is_file():
        raise FileNotFoundError(f"confirmation protocol is missing: {protocol_path}")

    manifest = base._snapshot_sources(output, config_path)
    snapshot = output / "source_snapshot"
    for source in (RUNNER_PATH, protocol_path):
        try:
            relative = source.resolve().relative_to(ROOT.resolve())
        except ValueError as error:
            raise ValueError("runner and protocol sources must be inside the repository") from error
        if any(part.casefold() in {"agent.md", "agents.md", "tmp"} for part in relative.parts):
            raise ValueError("private or temporary material is excluded from source snapshots")
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = base._sha(source)
        if digest != base._sha(target):
            raise RuntimeError(f"source changed while snapshotting: {relative}")
        manifest["sources"][relative.as_posix()] = {"sha256": digest, "bytes": target.stat().st_size}
    base._write_json(snapshot / "manifest.json", manifest)
    return manifest


def _augment_case(record: dict[str, Any], output: Path) -> dict[str, Any]:
    """Attach the frozen selector action and raw-row counts to a worker result."""
    case_id = str(record["case_id"])
    case_folder = output / "cases" / case_id
    selectors = record.get("selectors")
    selected_action = selectors.get("surrogate_lower") if isinstance(selectors, Mapping) else None
    record["confirmation_selector"] = "surrogate_lower"
    record["confirmation_action"] = selected_action
    record["confirmation_future_regime"] = record.get("history_regime")
    record["confirmation_future_scope"] = "history_matched_only"

    prefix = record.get("prefix")
    if isinstance(prefix, dict):
        prefix["raw_environment_steps"] = base._count_step_rows(case_folder / "prefix" / "steps.csv")
    for branch in record.get("branches", []):
        branch_folder = case_folder / "branches" / f"{branch['action']}__{branch['future_regime']}"
        branch["raw_environment_steps"] = base._count_step_rows(branch_folder / "steps.csv")
        if branch.get("future_regime") != record.get("history_regime"):
            error = {
                "case_id": case_id,
                "error_type": "future_regime_contract_violation",
                "error": "confirmation branch future regime did not match its history regime",
                "action": branch.get("action"),
                "future_regime": branch.get("future_regime"),
            }
            record.setdefault("errors", []).append(error)

    raw_costs = base._persisted_step_counts(case_folder)
    record.setdefault("costs", {}).update(raw_costs)
    if selected_action not in base.ACTIONS:
        record.setdefault("errors", []).append({
            "case_id": case_id,
            "error_type": "missing_surrogate_lower_selector",
            "error": "case did not contain a valid past-only surrogate_lower action",
        })
    return record


def _failed_case(job: Mapping[str, Any], output: Path) -> dict[str, Any]:
    case_id = f"s{int(job['seed'])}_{job['pair_kind']}_{job['history_regime']}"
    case_folder = output / "cases" / case_id
    persisted = base._persisted_step_counts(case_folder)
    error = {
        "case_id": case_id,
        "error_type": "case_failure",
        "error": traceback.format_exc(),
        "persisted_interactions": persisted,
        "count_source": "steps.csv rows",
    }
    case_folder.mkdir(parents=True, exist_ok=True)
    base._write_json(case_folder / "error.json", error)
    return {
        "case_id": case_id,
        "seed": int(job["seed"]),
        "pair_kind": str(job["pair_kind"]),
        "history_regime": str(job["history_regime"]),
        "prefix": None,
        "selectors": None,
        "selector_diagnostics": None,
        "confirmation_selector": "surrogate_lower",
        "confirmation_action": None,
        "confirmation_future_regime": str(job["history_regime"]),
        "confirmation_future_scope": "history_matched_only",
        "branches": [],
        "costs": {
            **persisted,
            "case_cpu_s": None,
            "case_wall_s": None,
            "prefix_cpu_s": None,
            "branch_cpu_s": None,
            "prefix_wall_s": None,
            "branch_wall_s": None,
        },
        "errors": [error],
    }


def _aggregate(records: Sequence[Mapping[str, Any]], key: str) -> int | float | None:
    values = [record.get("costs", {}).get(key) for record in records]
    if any(value is None for value in values):
        return None
    return sum(values)


def run(
    config: Mapping[str, Any] | str | Path,
    output: str | Path,
    *,
    seeds: Sequence[int] | None = None,
    pilot: bool = False,
    protocol: str | Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    raw_config, config_path = _read_config(config)
    normalized, phase = _phase_config(raw_config, pilot=pilot)
    run_seeds = tuple(int(value) for value in (seeds if seeds is not None else normalized["seeds"]))
    if not run_seeds:
        raise ValueError("at least one seed is required")
    if len(set(run_seeds)) != len(run_seeds):
        raise ValueError("seed list must not contain duplicates")
    if pilot and len(run_seeds) != 1:
        raise ValueError("the frozen pilot uses exactly one seed across both history regimes")

    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=False)
    base._write_json(output_path / "config.json", normalized)
    base._write_json(output_path / "source_snapshot" / "config_original.json", raw_config)
    source_manifest = _snapshot_sources(output_path, config_path, Path(protocol))

    jobs = _case_payloads(normalized, output_path, run_seeds)
    workers = max(1, min(int(normalized["max_workers"]), len(jobs)))
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(base._case_worker, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                record = _augment_case(future.result(), output_path)
            except Exception:
                record = _failed_case(job, output_path)
            records.append(record)
            errors.extend(record.get("errors", []))

    records.sort(key=lambda row: (int(row["seed"]), str(row["pair_kind"]), str(row["history_regime"])))
    branches = [branch for record in records for branch in record.get("branches", [])]
    orchestrator_cpu = float(time.process_time() - start_cpu)
    child_cpu = _aggregate(records, "case_cpu_s")
    costs = {
        "prefix_environment_steps": _aggregate(records, "prefix_environment_steps"),
        "branch_environment_steps": _aggregate(records, "branch_environment_steps"),
        "environment_steps_total": _aggregate(records, "environment_steps_total"),
        "cpu_s": None if child_cpu is None else float(child_cpu + orchestrator_cpu),
        "child_cpu_s": child_cpu,
        "orchestrator_cpu_s": orchestrator_cpu,
        "wall_s": float(time.perf_counter() - start_wall),
        "branch_cpu_s": _aggregate(records, "branch_cpu_s"),
        "prefix_cpu_s": _aggregate(records, "prefix_cpu_s"),
        "branch_wall_s": _aggregate(records, "branch_wall_s"),
        "prefix_wall_s": _aggregate(records, "prefix_wall_s"),
    }
    summary = {
        "format": "storage_decision_summary_v1",
        "phase": phase,
        "analysis_role": "independent_surrogate_lower_confirmation",
        "confirmation": {
            "selector": "surrogate_lower",
            "pair_kinds": list(EXPECTED_PAIR_KINDS),
            "history_regimes": list(EXPECTED_REGIMES),
            "future_policy": "one future per case, matched to its history regime",
            "physical_actions": list(base.ACTIONS),
            "selectors_are_past_only": True,
        },
        "config": normalized,
        "source_manifest": source_manifest,
        "seeds": list(run_seeds),
        "case_count": len(records),
        "branch_count": len(branches),
        "cases": records,
        "branches": branches,
        "errors": errors,
        "costs": costs,
    }
    base._write_json(output_path / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--pilot", action="store_true")
    arguments = parser.parse_args(argv)
    summary = run(
        arguments.config,
        arguments.output,
        seeds=arguments.seeds,
        pilot=arguments.pilot,
        protocol=arguments.protocol,
    )
    print(json.dumps({
        "output": str(arguments.output),
        "phase": summary["phase"],
        "case_count": summary["case_count"],
        "branch_count": summary["branch_count"],
        "errors": len(summary["errors"]),
        "costs": summary["costs"],
    }, indent=2))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
