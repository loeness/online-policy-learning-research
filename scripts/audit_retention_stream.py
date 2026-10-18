"""Audit retention stream artifacts and account for every pilot/main interaction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def audit_cohort(folder: Path) -> dict:
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    errors, branches, uniform_hashes = [], [], {}
    if summary["status"] != "completed" or not summary["audit"]["passed"]:
        errors.append("cohort is not complete and audit-passed")
    config = summary["config"]
    expected = {(m, b, s) for m in config["methods"] for b in config["budgets"] for s in config["seeds"]}
    actual = {(r["method"], r["budget"], r["seed"]) for r in summary["records"]}
    if actual != expected or len(actual) != len(summary["records"]):
        errors.append("method/budget/seed grid mismatch")
    manifest = json.loads((folder / "source_snapshot/manifest.json").read_text(encoding="utf-8"))
    for relative, value in manifest.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or any(p.casefold() in {"agent.md", "agents.md", "tmp"} for p in path.parts):
            errors.append(f"forbidden source snapshot path: {relative}")
            continue
        saved = folder / "source_snapshot" / path
        if digest(saved) != value["sha256"] or saved.stat().st_size != value["bytes"]:
            errors.append(f"snapshot digest/size mismatch: {relative}")
    for record in summary["records"]:
        label = f"seed_{record['seed']}/{record['budget']}/{record['method']}"
        branch = folder / label
        raw, reward, uniforms = 0, 0.0, hashlib.sha256()
        with (branch / "steps.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                raw += 1
                if int(row["raw_step"]) != raw:
                    errors.append(f"{label}: nonconsecutive raw step {raw}")
                reward += float(row["reward"])
                probabilities = json.loads(row["behavior_probs"])
                action = int(row["action"])
                if len(probabilities) != 4 or not all(math.isfinite(p) and p > 0 for p in probabilities) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-7):
                    errors.append(f"{label}: invalid behavior probabilities at {raw}")
                elif not math.isclose(math.log(probabilities[action]), float(row["behavior_log_prob"]), abs_tol=1e-7):
                    errors.append(f"{label}: behavior log-probability mismatch at {raw}")
                uniforms.update(row["uniform"].encode())
                uniforms.update(b"\n")
        if raw != record["raw_steps"] or raw != record["fit_steps"]:
            errors.append(f"{label}: raw/PPO row count mismatch")
        if not math.isclose(reward, record["total_reward"], abs_tol=1e-8):
            errors.append(f"{label}: raw reward sum mismatch")
        updates = [json.loads(line) for line in (branch / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
        if sum(r["rows"] for r in updates) != raw or sum(r["updates"] for r in updates) != record["minibatch_updates"]:
            errors.append(f"{label}: update ledger mismatch")
        if len(updates) != record["rollouts_trained"]:
            errors.append(f"{label}: rollout count mismatch")
        events = [json.loads(line) for line in (branch / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        for event in events:
            if event["event"] == "write_test" and event["first_test_row_step"] <= event["proposal_step"]:
                errors.append(f"{label}: candidate evaluated on proposal data")
        if sum(e["event"] == "write_test" and e["accepted"] for e in events) != record["write_accepts"]:
            errors.append(f"{label}: accepted-write count mismatch")
        if sum(e["event"] == "prior_test" and e["applied"] for e in events) != record["prior_bindings"]:
            errors.append(f"{label}: applied-prior count mismatch")
        capacities = json.loads((branch / "capacity.json").read_text(encoding="utf-8"))
        if len(capacities) != 12:
            errors.append(f"{label}: expected twelve bout checkpoints")
        for capacity in capacities:
            checkpoint = branch / "checkpoints" / f"step_{capacity['step']:08d}.pt"
            sidecar = checkpoint.with_suffix(".actor.bin")
            if checkpoint.stat().st_size != capacity["checkpoint_bytes"] or sidecar.stat().st_size != capacity["actor_sidecar_bytes"]:
                errors.append(f"{label}: checkpoint physical-size mismatch")
            complete_bytes = checkpoint.stat().st_size + sidecar.stat().st_size + capacity["foundation_bytes"]
            if complete_bytes != capacity["full_physical_bytes"] or complete_bytes > config["package_budget_bytes"]:
                errors.append(f"{label}: complete package accounting mismatch")
            if sidecar.stat().st_size > config["budgets"][record["budget"]]["actor_budget_bytes"]:
                errors.append(f"{label}: actor cap exceeded")
        uniform_hashes.setdefault(record["seed"], []).append(uniforms.hexdigest())
        branches.append({"branch": label, "raw_rows": raw, "ppo_rows": sum(r["rows"] for r in updates),
            "checkpoints": len(capacities), "uniform_sha256": uniforms.hexdigest(),
            "zero_update_rollouts": sum(r["updates"] == 0 for r in updates)})
    if any(len(set(values)) != 1 for values in uniform_hashes.values()):
        errors.append("paired seed action uniforms differ across methods/budgets")
    raw_total = sum(b["raw_rows"] for b in branches)
    if raw_total != summary["cohort_cost"]["new_total_interactions"] or summary["cohort_cost"]["new_evaluation_steps"] != 0:
        errors.append("cohort interaction ledger mismatch")
    return {"cohort": folder.relative_to(ROOT).as_posix(), "passed": not errors,
        "errors": errors, "branches": branches, "new_raw_interactions": raw_total,
        "cpu_s": summary["cpu_s"], "cohort_wall_s": summary["wall_s"],
        "snapshot_files": len(manifest)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohorts = [audit_cohort(path.resolve()) for path in args.cohort]
    result = {"format": "retention_artifact_audit_v1", "passed": all(c["passed"] for c in cohorts),
        "cohorts": cohorts, "new_total_interactions": sum(c["new_raw_interactions"] for c in cohorts),
        "cpu_s": sum(c["cpu_s"] for c in cohorts),
        "sequential_cohort_wall_s": sum(c["cohort_wall_s"] for c in cohorts),
        "scope": "Artifact integrity and explicit costs; not a scientific effect or safety guarantee."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "new_total_interactions", "cpu_s", "sequential_cohort_wall_s")}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
