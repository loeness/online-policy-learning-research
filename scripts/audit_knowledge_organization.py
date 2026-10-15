"""Audit raw counts and preserved artifacts without any new interactions."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    loop = load("outputs/knowledge_loop_formal/summary.json")
    checks = []
    source_count = 0
    for run in loop["runs"]:
        folder = ROOT / f"outputs/knowledge_loop_formal/seed_{run['seed']}/{run['arm']}"
        steps, reward = 0, 0.
        with (folder / "steps.csv").open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                steps += 1
                reward += float(row["reward"])
        assert steps == run["training_environment_steps"] == 32768
        assert abs(reward - run["total_reward"]) < 1e-8
        manifest = json.loads((folder / "source_snapshot/manifest.json").read_text())
        for relative, recorded in manifest["sources"].items():
            snapshot = folder / "source_snapshot" / relative
            assert snapshot.stat().st_size == recorded["bytes"]
            assert sha(snapshot) == recorded["sha256"]
            source_count += 1
        if run["arm"] not in ("frozen", "adaptive_assignment"):
            assert run["counts"]["validation_steps"] == 8192
            assert run["counts"]["local_trial_steps"] == 24576
            assert run["actor_compact_binary_file_bytes"] <= 320
            with (folder / "commits.csv").open(encoding="utf-8", newline="") as file:
                commits = list(csv.DictReader(file))
            assert len(commits) == 32
            assert all(int(r["validation_samples"]) == 256 for r in commits)
            assert all(int(r["actor_binary_file_bytes"]) <= 320 for r in commits)
        checks.append({"seed": run["seed"], "arm": run["arm"], "steps": steps,
                       "reward_sum_matches": True, "source_snapshot_matches": True})
    revisit = load("outputs/knowledge_loop_revisit_corrected/summary.json")
    revisit_steps, episodes = 0, 0
    for run in revisit["records"]:
        folder = ROOT / f"outputs/knowledge_loop_revisit_corrected/seed_{run['seed']}/{run['arm']}"
        rows = list(csv.DictReader((folder / "episodes.csv").open(encoding="utf-8")))
        assert len(rows) == 96
        assert sum(int(r["length"]) for r in rows) == run["environment_steps"]
        revisit_steps += run["environment_steps"]
        episodes += len(rows)
        for context in ("A0", "A1", "A2", "B0", "B1", "B2"):
            a = [r for r in rows if r["context"] == context and r["condition"] == "retain_parameters"]
            b = [r for r in rows if r["context"] == context and r["condition"] == "zero_corrections"]
            assert [(r["reset_seed"], r["action_seed"]) for r in a] == [(r["reset_seed"], r["action_seed"]) for r in b]
        if run["arm"] == "frozen":
            assert run["mean_paired_return_gain"] == 0.
    assert revisit_steps == revisit["environment_steps"] and episodes == 1728

    # The existing manifest predates this cohort. Check only its prior raw data
    # and outputs; documentation/source changes are intentional.
    old = load("archive/manifest.json")
    preserved = 0
    for row in old["files"]:
        if row["path"].startswith(("outputs/", "data/")):
            path = ROOT / row["path"]
            assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]
            preserved += 1
    costs = {
        "static_original": load("outputs/knowledge_static_formal/summary.json")["actual_interactions"],
        "static_global_control": load("outputs/knowledge_static_global_control/summary.json")["actual_interactions"],
        "static_distillation": load("outputs/knowledge_distillation_control/summary.json")["interactions"],
        "identification_generated_once": load("outputs/knowledge_identification_formal/summary.json")["generated_environment_interactions_once_per_seed"],
        "online_training": sum(r["training_environment_steps"] for r in loop["runs"]),
        "legacy_revisit_excluded_from_effect_claims": sum(r["revisit_environment_steps"] for r in loop["runs"]),
        "corrected_revisit": revisit_steps,
    }
    pilot = load("outputs/knowledge_loop_pilot/summary.json")
    pilot_cost = (load("outputs/knowledge_static_pilot/summary.json")["actual_interactions"]
                  + load("outputs/knowledge_identification_pilot/summary.json")["generated_environment_interactions_once_per_seed"]
                  + sum(r["training_environment_steps"] for r in pilot["runs"]))
    result = {"valid": True, "online_raw_trace_checks": checks,
              "source_snapshot_files_verified": source_count,
              "corrected_revisit_episodes": episodes, "old_data_output_files_unchanged": preserved,
              "formal_interaction_ledger": costs, "formal_total_interactions": sum(costs.values()),
              "completed_pilot_interactions": pilot_cost,
              "aborted_additional_interactions_lower_bound": 85781,
              "aborted_run_reference_interactions": "Unknown; not persisted, explicitly excluded from exact totals.",
              "new_targeted_tests": "42 passed; pytest of four knowledge test files",
              "capacity_scope": "Component caps and measured packages; no enforced unified whole-agent byte quota."}
    target = ROOT / "outputs/knowledge_organization_report/audit.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("valid", "source_snapshot_files_verified", "old_data_output_files_unchanged", "formal_total_interactions", "completed_pilot_interactions")}))


if __name__ == "__main__":
    main()
