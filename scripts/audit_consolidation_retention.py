"""Append physical-file and interaction accounting without rewriting raw results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(input_dir: Path, pilot_dir: Path | None, foundation: Path) -> dict:
    input_dir = input_dir.resolve()
    output = input_dir / "postrun_audit.json"
    if output.exists():
        raise FileExistsError(f"Preserve existing audit: {output}")
    summary = read_json(input_dir / "summary.json")
    config = read_json(input_dir / "frozen_config.json")
    assert summary["status"] == "completed" and summary["audit"]["passed"]
    assert summary["cohort_cost"]["new_training_steps"] == 0
    assert len(summary["records"]) == len(summary["seeds"]) * len(summary["methods"])
    foundation_hash = digest(foundation)
    capacity, tables, step_total, episode_total = [], {}, 0, 0
    for branch in summary["records"]:
        folder = input_dir / branch["path"]
        checkpoint = folder / "post_delete.pt"
        sidecar = checkpoint.with_suffix(".actor.bin")
        assert digest(checkpoint) == branch["post_delete_checkpoint_sha256"]
        assert branch["purge_audit"]["passed"] and branch["actor_frozen_during_eval"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["foundation_sha256"] == foundation_hash
        assert sidecar.read_bytes() == payload["actor_binary"]
        physical = foundation.stat().st_size + checkpoint.stat().st_size + sidecar.stat().st_size
        budget = int(branch["capacity"]["package_budget_bytes"])
        assert physical <= budget
        capacity.append({"seed": branch["seed"], "method": branch["method"],
            "checkpoint_bytes": checkpoint.stat().st_size,
            "foundation_bytes": foundation.stat().st_size,
            "actor_sidecar_bytes": sidecar.stat().st_size,
            "original_full_persistent_bytes": branch["capacity"]["full_persistent_bytes"],
            "full_physical_bytes": physical, "package_budget_bytes": budget, "passed": True})
        with (folder / "episodes.csv").open(newline="", encoding="utf-8") as stream:
            episodes = list(csv.DictReader(stream))
        assert len(episodes) == len(config["targets"]) * int(config["episodes"])
        assert all(1 <= int(row["length"]) <= config["max_episode_steps"] for row in episodes)
        with (folder / "steps.csv").open(newline="", encoding="utf-8") as stream:
            steps = list(csv.DictReader(stream))
        assert len(steps) == branch["evaluation_steps"] == sum(int(row["length"]) for row in episodes)
        assert len(steps) == sum(row["evaluation_steps"] for row in branch["targets"].values())
        for task in config["targets"]:
            rows = [row for row in episodes if row["task"] == task]
            assert len(rows) == config["episodes"]
            mean = sum(float(row["return"]) for row in rows) / len(rows)
            assert abs(mean - branch["targets"][task]["mean_signed_return"]) < 1e-12
        tables[(branch["seed"], branch["method"])] = (
            {(r["task"], r["episode"]): (r["reset_seed"], r["action_seed"]) for r in episodes},
            {(r["task"], r["episode"], r["step"]): r["uniform"] for r in steps})
        assert len(tables[(branch["seed"], branch["method"])][1]) == len(steps)
        step_total += len(steps)
        episode_total += len(episodes)
    assert step_total == summary["cohort_cost"]["new_total_interactions"]
    common_uniform_rows = 0
    for seed in summary["seeds"]:
        reference = tables[(seed, summary["methods"][0])]
        for method in summary["methods"][1:]:
            compared = tables[(seed, method)]
            assert reference[0] == compared[0]
            common = reference[1].keys() & compared[1].keys()
            assert all(reference[1][key] == compared[1][key] for key in common)
            common_uniform_rows += len(common)

    source_cost = []
    for source in summary["source_audits"]:
        path = ROOT / source["source"]
        assert digest(path) == source["source_sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["foundation_sha256"] == foundation_hash
        counts = payload["counts"]
        train, probe, evaluation = (int(counts[key]) for key in
                                    ("environment_steps", "probe_steps", "evaluation_steps"))
        source_cost.append({"seed": source["seed"], "historical_training_steps": train,
            "historical_probe_steps": probe, "historical_evaluation_steps": evaluation,
            "historical_total_interactions": train + probe + evaluation})
    manifest = read_json(input_dir / "source_snapshot/manifest.json")
    for relative, entry in manifest.items():
        assert digest(input_dir / "source_snapshot" / relative) == entry["sha256"]
    assert not any(Path(relative).name.casefold() in {"agent.md", "agents.md"} for relative in manifest)
    pilot_steps = 0
    pilot_hash = None
    if pilot_dir is not None:
        pilot_path = pilot_dir.resolve() / "summary.json"
        pilot = read_json(pilot_path)
        assert pilot["status"] == "completed" and pilot["audit"]["passed"]
        assert pilot["cohort_cost"]["new_training_steps"] == 0
        assert pilot["cohort_cost"]["new_total_interactions"] == sum(
            branch["evaluation_steps"] for branch in pilot["records"])
        pilot_steps = pilot["cohort_cost"]["new_total_interactions"]
        pilot_hash = digest(pilot_path)
    result = {
        "format": "consolidation_retention_postrun_audit_v1", "passed": True,
        "input_summary_sha256": digest(input_dir / "summary.json"),
        "pilot_summary_sha256": pilot_hash,
        "capacity_correction": {"scope": "W0 + checkpoint + generated actor sidecar; observer logs excluded",
            "raw_summary_retained_unchanged": True, "records": capacity},
        "cohort_cost": {"main_new_evaluation_steps": step_total, "pilot_new_evaluation_steps": pilot_steps,
            "total_new_interactions": step_total + pilot_steps, "new_training_steps": 0,
            "main_episodes": episode_total},
        "source_cost": {"records": source_cost,
            "historical_training_steps_reused": sum(row["historical_training_steps"] for row in source_cost),
            "historical_probe_steps_reused": sum(row["historical_probe_steps"] for row in source_cost),
            "historical_evaluation_steps_reused": sum(row["historical_evaluation_steps"] for row in source_cost),
            "historical_total_interactions_reused": sum(row["historical_total_interactions"] for row in source_cost),
            "scope": "checkpoint counter prefixes counted once per source; excludes W0 pretraining and later old runs"},
        "pairing": {"reset_and_action_seeds_match": True, "common_uniform_rows": common_uniform_rows,
            "uniform_mismatches": 0},
        "integrity": {"snapshot_files_verified": len(manifest), "checkpoints_verified": len(capacity),
            "source_files_verified": len(source_cost), "private_instructions_in_snapshot": False},
        "checkpoint_contract": {"runtime_frozen_and_hash_unchanged": True,
            "standalone_restore_must_explicitly_freeze": True,
            "note": "requires_grad is not a saved freeze contract; use runner freeze() after restore for evaluation"},
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--foundation", type=Path, default=ROOT / "data/foundation.pt")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = audit(args.input, args.pilot, args.foundation)
    print(json.dumps({"passed": result["passed"], "cost": result["cohort_cost"],
                      "pairing": result["pairing"]}))


if __name__ == "__main__":
    main()
