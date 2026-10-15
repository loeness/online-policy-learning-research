"""Audit completed bounded-cache experiments without new environment steps."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from online_rl.shared_write_storage import decode_shared_knowledge

CONDITIONS = ("independent_cache", "independent_fast_cache", "shared_cache", "frozen_shared_cache")
LABELS = ("Independent", "Independent fast", "Active shared", "Frozen shared")
COLORS = ("#2878b5", "#79a9d1", "#309b38", "#c94040")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values):
    return float(np.mean(list(values)))


def report(input_root, output):
    input_root, output = Path(input_root), Path(output)
    cohort = read_json(input_root / "summary.json")
    config = read_json(input_root / "frozen_config.json")
    if set(cohort["conditions"]) != set(CONDITIONS):
        raise ValueError("Expected all four preregistered capacity conditions")
    checks, records = [], []
    def check(name, passed, **detail):
        checks.append({"name": name, "passed": bool(passed), **detail})
    for seed in cohort["seeds"]:
        seed_folders = {}
        for condition in CONDITIONS:
            folder = input_root / f"seed_{seed}" / condition
            seed_folders[condition] = folder
            d = read_json(folder / "summary.json")
            train = rows(folder / "steps.csv")
            episodes = rows(folder / "episodes.csv")
            evaluations = rows(folder / "evaluations.csv")
            events = rows(folder / "cache_events.csv")
            capacity = read_json(folder / "capacity.json")
            prefix = f"{seed}/{condition}"
            check(prefix + "/interaction_counts", len(train) == d["training_steps"]
                  and sum(int(r["steps"]) for r in evaluations) == d["evaluation_steps"]
                  and d["total_interactions"] == d["training_steps"] + d["evaluation_steps"])
            check(prefix + "/fresh_fit_holdout", sum(r["phase"] == "fit" for r in train) == d["counts"]["fit_steps"]
                  and sum(r["phase"] == "holdout" for r in train) == d["counts"]["holdout_steps"]
                  and d["counts"]["fit_steps"] * 4 == len(train) * 3
                  and d["counts"]["stale_actor_rows"] == 0)
            check(prefix + "/bounded_cache", len(events) == 24
                  and all(int(r["resident_count"]) <= 6 for r in events)
                  and all(len(json.loads(r["resident_keys"])) <= 6 for r in events)
                  and all(r["action"] == "miss" for r in events)
                  and sum(bool(r["evicted_context"]) for r in events) == 18
                  and d["counts"]["cache_evictions"] == 18)
            check(prefix + "/generation_lineage", all(
                int(r["generation"]) == int(events[int(r["bout"]) - 1]["generation"])
                and int(r["slot"]) == int(events[int(r["bout"]) - 1]["slot"]) for r in train))
            actor_expected = 204 if condition.startswith("independent") else 230
            check(prefix + "/actual_actor_and_full_budget", all(
                r["actor_bytes"] == actor_expected <= config["actor_budget_bytes"]
                and r["whole_package_bytes"] <= config["whole_package_cap_bytes"] for r in capacity))
            snapshots = read_json(folder / "source_manifest.json")
            check(prefix + "/source_snapshot_hashes", all(
                hashlib.sha256((folder / "source_snapshot" / path).read_bytes()).hexdigest() == digest
                for path, digest in snapshots["sources"].items()))
            checkpoint = torch.load(folder / "checkpoint_final.pt", map_location="cpu", weights_only=False)
            local, shared, groups = decode_shared_knowledge(checkpoint["actor_binary"])
            check(prefix + "/checkpoint_actor_roundtrip", len(checkpoint["actor_binary"]) == actor_expected
                  and np.array_equal(groups, checkpoint["slot_groups"].numpy())
                  and len(checkpoint["cache_state"]["resident_by_context"]) == 6)
            check(prefix + "/whole_package_measurement",
                  (folder / "checkpoint_final.pt").stat().st_size == d["final_checkpoint"]["checkpoint_bytes"]
                  and d["final_checkpoint"]["whole_package_bytes"] == d["final_checkpoint"]["checkpoint_bytes"] + d["final_checkpoint"]["w0_bytes"])
            endpoint_events = read_json(folder / "endpoint_cache_events.json")
            check(prefix + "/endpoint_cache_hits_and_cold_misses",
                  sum(e["action"] == "hit" for e in endpoint_events) == 6
                  and sum(e["action"] == "miss" for e in endpoint_events) == 6)
            check(prefix + "/no_evaluation_updates", all(int(r["gradient_updates"]) == 0 for r in evaluations))
            by_bout = d["bout_evaluations"]
            def bout_metric(start, stop, stage, field):
                return mean(by_bout[f"bout_{i:02d}_{stage}"][field] for i in range(start, stop + 1))
            metrics = {"online_success_rate": d["online_success_rate"], "reward_per_training_step": d["reward_per_training_step"]}
            for block, start, stop in (("source", 1, 6), ("novel", 7, 12), ("revisit", 13, 24)):
                for stage in ("pre", "post"):
                    for field in ("success_rate", "mean_return"):
                        metrics[f"{block}_{stage}_{field}"] = bout_metric(start, stop, stage, field)
            for field in ("success_rate", "mean_return"):
                metrics[f"revisit_pre_minus_first_pre_{field}"] = mean(
                    by_bout[f"bout_{i + 12:02d}_pre"][field] - by_bout[f"bout_{i:02d}_pre"][field]
                    for i in range(1, 13))
            for category, names in (("all", config["contexts"]),
                    ("cold", [e["context"] for e in endpoint_events if e["action"] == "miss"]),
                    ("resident", [e["context"] for e in endpoint_events if e["action"] == "hit"])):
                for field in ("success_rate", "mean_return"):
                    metrics[f"endpoint_{category}_{field}"] = mean(d["endpoint_evaluations"][n][field] for n in names)
            complete = [r for r in episodes if r["completed"] == "True"]
            check(prefix + "/online_episode_score", len(complete) == d["online_completed_episodes"]
                  and np.isclose(mean(int(r["success"]) for r in complete), d["online_success_rate"]))
            # Complete episodes ending in the first 1,024 steps of each revisit.
            early = [r for r in complete if int(r["bout"]) >= 13
                     and (int(r["step"]) - 1) % d["bout_steps"] < 1024]
            metrics["revisit_first_1024_online_success"] = mean(int(r["success"]) for r in early)
            if condition.startswith("independent"):
                # All twelve routes are evicted before recurrence. The same
                # zero local head must reproduce paired pre-bout actions.
                first = {(r["context"],r["episode"]): r["actions"] for r in evaluations
                         if r["stage"] == "pre_bout" and int(r["bout"]) <= 12}
                check(prefix + "/independent_cold_recurrence", all(
                    r["actions"] == first[(r["context"],r["episode"])] for r in evaluations
                    if r["stage"] == "pre_bout" and int(r["bout"]) >= 13))
            if condition == "frozen_shared_cache":
                boot = torch.load(folder / "checkpoint_boot.pt", map_location="cpu", weights_only=False)
                _, boot_shared, _ = decode_shared_knowledge(boot["actor_binary"])
                check(prefix + "/frozen_parent_unchanged", np.array_equal(boot_shared, shared))
            records.append({"seed":seed, "condition":condition, "metrics":metrics,
                "training_steps":d["training_steps"], "evaluation_steps":d["evaluation_steps"],
                "total_interactions":d["total_interactions"], "cpu_seconds":d["cpu_seconds"],
                "wall_seconds":d["wall_seconds"], "actor_bytes":actor_expected,
                "whole_package_peak_bytes":max(r["whole_package_bytes"] for r in capacity)})
        def training_prefix(folder):
            limit = int(config["pilot_bout_steps"] if cohort["phase"] == "pilot" else config["formal_bout_steps"]) * 6
            return [(r["action"],r["reward"],r["log_prob"]) for r in rows(folder / "steps.csv")[:limit]]
        check(f"{seed}/active_frozen_identical_training_prefix",
              training_prefix(seed_folders["shared_cache"]) == training_prefix(seed_folders["frozen_shared_cache"]))
        left = torch.load(seed_folders["shared_cache"] / "checkpoint_boot.pt", map_location="cpu", weights_only=False)
        right = torch.load(seed_folders["frozen_shared_cache"] / "checkpoint_boot.pt", map_location="cpu", weights_only=False)
        check(f"{seed}/active_frozen_identical_boot_parameters", left["actor_binary"] == right["actor_binary"]
              and all(torch.equal(v,right["critic"][k]) for k,v in left["critic"].items()))
    aggregates = {c:{k:mean(r["metrics"][k] for r in records if r["condition"] == c)
                     for k in records[0]["metrics"]} for c in CONDITIONS}
    paired = {}
    for baseline in ("independent_cache", "independent_fast_cache", "frozen_shared_cache"):
        paired[baseline] = [{"seed":seed, **{k:
            next(r["metrics"][k] for r in records if r["condition"] == "shared_cache" and r["seed"] == seed)
            - next(r["metrics"][k] for r in records if r["condition"] == baseline and r["seed"] == seed)
            for k in records[0]["metrics"]}} for seed in cohort["seeds"]]
    ledger = {k:sum(r[k] for r in records) for k in ("training_steps","evaluation_steps","total_interactions","cpu_seconds")}
    check("cohort/interaction_ledger", all(ledger[k] == cohort[k] for k in ("training_steps","evaluation_steps","total_interactions")))
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for axis, metric, title in zip(axes,
            ("online_success_rate", "revisit_pre_success_rate", "endpoint_all_success_rate"),
            ("Online complete-episode success", "Revisit before local retraining", "Endpoint across 12 contexts")):
        for i, condition in enumerate(CONDITIONS):
            values = [r["metrics"][metric] for r in records if r["condition"] == condition]
            axis.bar(i, np.mean(values), color=COLORS[i], alpha=.85)
            axis.scatter([i]*len(values), values, color="black", s=18, zorder=3)
        axis.set_xticks(range(4), LABELS, rotation=25, ha="right")
        axis.set_ylim(0,1)
        axis.set_title(title,fontsize=10)
        axis.grid(axis="y",alpha=.2)
    fig.suptitle("12 contexts / 6 local slots: controlled capacity diagnostic")
    fig.text(.5,.01,"Known context IDs and family groups. Actual INT8 actor: independent 204 B; shared 230 B. Dots are seeds.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.065,1,.95))
    for extension in ("png","pdf"):
        fig.savefig(output / f"capacity_results.{extension}",dpi=220)
    plt.close(fig)
    result = {"phase":cohort["phase"], "seeds":cohort["seeds"], "aggregates":aggregates,
              "paired_active_minus_baseline":paired, "records":records, "cost_ledger":ledger,
              "checks":checks, "failed_assertions":[c for c in checks if not c["passed"]],
              "scope":"Controlled IDs and family groups, LRU replacement. No unknown-boundary or learned storage decision claim. Shared consumes 26 additional actor bytes under the same 230-byte cap. Descriptive seed comparisons only."}
    (output / "audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"phase":result["phase"],"failed_assertions":len(result["failed_assertions"]),"cost":ledger}))
    if result["failed_assertions"]:
        raise AssertionError([c["name"] for c in result["failed_assertions"]])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    report(args.input,args.output)
