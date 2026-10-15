"""Recheck completed V2 raw logs, source snapshots, budgets and archive preservation."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "b7cd1e1d680113b9b71af7890c9c30f9130929b1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    old = json.loads(subprocess.check_output(
        ["git", "show", f"{BASELINE}:archive/manifest.json"], cwd=ROOT))
    preserved = [r for r in old["files"] if r["path"].startswith(("data/", "outputs/"))]
    for row in preserved:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], row["path"]
    run_checks = []
    snapshot_count = 0
    for cohort, count in (("knowledge_causal_formal", 18), ("knowledge_v2_formal", 15)):
        folders = sorted((ROOT / "outputs" / cohort).glob("seed_*/*/summary.json"))
        assert len(folders) == count
        for summary_file in folders:
            folder, summary = summary_file.parent, read(summary_file)
            step_rows = rows(folder / "steps.csv")
            episode_rows = rows(folder / "episodes.csv")
            commits = rows(folder / "commits.csv")
            assert len(step_rows) == 32768
            assert all(int(row["step"]) == i for i, row in enumerate(step_rows, 1))
            assert math.isclose(sum(float(r["reward"]) for r in step_rows), summary["total_reward"], abs_tol=1e-7)
            assert len(episode_rows) == summary["completed_episodes"]
            assert math.isclose(sum(int(r["success"]) for r in episode_rows) / len(episode_rows), summary["success_rate"])
            assert len(commits) == 32
            assert summary["counts"]["local_trial_steps"] == 24576
            assert summary["counts"]["validation_steps"] == 8192
            manifest = read(folder / "source_snapshot/manifest.json")
            for section in ("sources", "causal_harness_sources"):
                for relative, spec in manifest.get(section, {}).items():
                    expected = spec["sha256"] if isinstance(spec, dict) else spec
                    relative = relative.replace("\\", "/")
                    captured = folder / relative if relative.startswith("source_snapshot/") else folder / "source_snapshot" / relative
                    assert digest(captured) == expected
                    snapshot_count += 1
            if cohort == "knowledge_v2_formal":
                patch = read(folder / "runtime_patches.json")
                for relative, expected in patch["extra_source_sha256"].items():
                    assert digest(folder / "source_snapshot" / relative) == expected
                    snapshot_count += 1
                base_bytes = (ROOT / "data/foundation.pt").stat().st_size
                packages = [int(r["checkpoint_file_bytes"]) + base_bytes for r in commits]
                assert max(packages) <= 256 * 1024
                assert (folder / "checkpoint_latest.pt").stat().st_size + base_bytes == summary["whole_committed_package_bytes"]
            run_checks.append({"path": str(folder.relative_to(ROOT)).replace("\\", "/"),
                               "steps": len(step_rows), "episodes": len(episode_rows), "cycles": len(commits)})
    revisit = read(ROOT / "outputs/knowledge_v2_revisit/summary.json")
    n_episodes = n_steps = 0
    for record in revisit["records"]:
        folder = ROOT / "outputs/knowledge_v2_revisit" / f"seed_{record['seed']}" / record["condition"]
        events = rows(folder / "episodes.csv")
        assert len(events) == record["episodes"] == 96
        paired = {}
        for event in events:
            key = event["context"], event["episode"]
            paired.setdefault(key, []).append((event["reset_seed"], event["action_seed"]))
        assert len(paired) == 48 and all(len(v) == 2 and v[0] == v[1] for v in paired.values())
        assert sum(int(r["length"]) for r in events) == record["environment_steps"]
        assert record["policy_gradient_updates"] == 0
        n_episodes += len(events)
        n_steps += record["environment_steps"]
    assert n_episodes == revisit["episodes"] == 1440
    assert n_steps == revisit["environment_steps"] == 80319
    for relative, expected in revisit["source_sha256"].items():
        assert digest(ROOT / "outputs/knowledge_v2_revisit/source_snapshot" / relative) == expected
    for relative, expected in revisit["input_sha256"].items():
        assert digest(ROOT / relative) == expected
    costs = {"formal_training": sum(r["steps"] for r in run_checks),
             "causal_pilot": sum(r["training_environment_steps"] for r in read(ROOT / "outputs/knowledge_causal_pilot/summary.json")["runs"]),
             "v2_pilot": read(ROOT / "outputs/knowledge_v2_pilot/summary.json")["environment_steps"],
             "score_diagnostic": read(ROOT / "outputs/knowledge_v2_score_diagnostic/summary.json")["training_environment_steps"],
             "endpoint_revisit": n_steps, "saved_stream_replay": 0}
    assert sum(costs.values()) == 1243583
    result = {"valid": True, "baseline_commit": BASELINE,
              "older_data_output_files_preserved": len(preserved),
              "formal_run_checks": run_checks, "captured_source_hashes_checked": snapshot_count,
              "revisit_paired_episodes": n_episodes, "interaction_costs": costs,
              "total_new_interactions": sum(costs.values()),
              "scope": "Artifact integrity and accounting only; no new performance or statistical guarantee."}
    destination = ROOT / "outputs/knowledge_v2_report/audit.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "formal_run_checks"}))


if __name__ == "__main__":
    main()
