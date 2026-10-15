"""Report V2 factor controls and raw-stream integrity without new interactions."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/knowledge_v2_report"


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    result = []
    for condition, values in groups.items():
        result.append({"condition": condition, "seeds": len(values),
                       "reward_per_step": float(np.mean([v["reward_per_step"] for v in values])),
                       "success_rate": float(np.mean([v["success_rate"] for v in values])),
                       "wall_s": float(np.mean([v["elapsed_seconds"] for v in values])),
                       "actor_bytes": float(np.mean([v["actor_compact_numeric_bytes"] for v in values])),
                       "allocations": float(np.mean([v["counts"]["slot_allocations"] for v in values])),
                       "evictions": float(np.mean([v["counts"]["slot_evictions"] for v in values])),
                       "package_bytes": float(np.mean([v.get("whole_committed_package_bytes", v["checkpoint_file_bytes"] + 55477) for v in values])),
                       "reward_by_seed": {str(v["seed"]): v["reward_per_step"] for v in values}})
    return result


def contrasts(rows, pairs):
    indexed = {(r["seed"], r["condition"]): r for r in rows}
    result = []
    for name, a, b in pairs:
        gains = {str(seed): indexed[(seed, a)]["reward_per_step"] - indexed[(seed, b)]["reward_per_step"]
                 for seed in (541, 557, 569)}
        result.append({"contrast": name, "a": a, "b": b, "gain_by_seed": gains,
                       "mean_gain": float(np.mean(list(gains.values())))})
    return result


def verify_shared_controls():
    results = []
    for seed in (541, 557, 569):
        a = ROOT / f"outputs/knowledge_causal_formal/seed_{seed}/learned_int8_preserve_trial/steps.csv"
        b = ROOT / f"outputs/knowledge_v2_formal/seed_{seed}/fixed_old_router_int8/steps.csv"
        x, y = list(csv.DictReader(a.open(encoding="utf-8"))), list(csv.DictReader(b.open(encoding="utf-8")))
        assert len(x) == len(y) == 32768
        equal = all((r["action"], r["reward"], r["route_slot"]) ==
                    (s["action"], s["reward"], s["route_slot"]) for r, s in zip(x, y))
        assert equal, f"shared-control trajectories disagree at seed {seed}"
        results.append({"seed": seed, "identical_actions_rewards_routes": True, "steps": len(x)})
    return results


def plot(rows, name, labels):
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    fig, ax = plt.subplots(figsize=(10.5, 4.3), constrained_layout=True)
    for i, (condition, values) in enumerate(groups.items()):
        ys = [v["reward_per_step"] for v in values]
        ax.bar(i, np.mean(ys), color="#0072B2", alpha=.6)
        ax.scatter(np.arange(len(ys)) * .08 + i - .08, ys, color="black", s=22, zorder=3)
    ax.set(xticks=range(len(groups)), xticklabels=[labels.get(c, c) for c in groups],
           ylabel="Total online reward / step", title="New deployment seeds; dots are paired seeds")
    ax.axhline(0, color="gray", linewidth=.7)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    causal = read("outputs/knowledge_causal_formal/summary.json")
    online = read("outputs/knowledge_v2_formal/summary.json")
    router = read("outputs/knowledge_v2_router_formal/summary.json")
    assert len(causal["runs"]) == 18 and len(online["records"]) == 15
    OUT.mkdir(parents=True, exist_ok=True)
    shared = verify_shared_controls()
    result = {"causal": aggregate(causal["runs"]), "online": aggregate(online["records"]),
              "causal_contrasts": contrasts(causal["runs"], [
                  ("allocation_fix", "learned_int8_preserve_trial", "learned_int8"),
                  ("oracle_vs_learned", "oracle_int8", "learned_int8_preserve_trial"),
                  ("oracle_dense_vs_int8", "oracle_dense", "oracle_int8"),
                  ("oracle_gate_vs_int8", "oracle_gate", "oracle_int8")]),
              "online_contrasts": contrasts(online["records"], [
                  ("evidence_routing", "evidence_router_int8", "fixed_old_router_int8"),
                  ("old_gate_effect", "evidence_router_old_gate", "evidence_router_int8"),
                  ("nonblocking_shared_vs_independent", "evidence_router_nonblocking", "evidence_router_int8"),
                  ("nonblocking_vs_single", "evidence_router_nonblocking", "single_head")]),
              "shared_control_integrity": shared,
              "new_formal_interactions": sum(r["training_environment_steps"] for r in causal["runs"]) + online["environment_steps"],
              "router_replay_new_interactions": router["generated_environment_interactions"],
              "capacity_definition": "W0 file + committed checkpoint <=256KiB for main V2; temporary workspace measured separately.",
              "statistics": "Descriptive paired differences across three new deployment seeds, one common W0."}
    codecs = {}
    for condition in ("evidence_router_int8", "evidence_router_old_gate", "evidence_router_nonblocking"):
        values, selected = [], Counter()
        for seed in (541, 557, 569):
            folder = ROOT / f"outputs/knowledge_v2_formal/seed_{seed}/{condition}"
            commits = list(csv.DictReader((folder / "commits.csv").open(encoding="utf-8")))
            assert len(commits) == 32
            values += [int(r["actor_bytes_after"]) for r in commits]
            selected.update(r["selected_codec"] for r in commits)
        codecs[condition] = {"selected_codec_counts": dict(selected),
                             "mean_actor_bytes_after_commit": float(np.mean(values)),
                             "peak_actor_bytes_after_commit": max(values)}
    result["codec_occupancy"] = codecs
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot(online["records"], "online_v2", {
        "single_head": "Single online\nhead", "fixed_old_router_int8": "Old router\nallocation fixed",
        "evidence_router_int8": "Evidence router\nindependent int8", "evidence_router_old_gate": "Evidence router\nold gate",
        "evidence_router_nonblocking": "Evidence router\nnonblocking sharing"})
    plot(causal["runs"], "causal_v2", {
        "learned_int8": "Old implementation", "oracle_int8": "Oracle\nint8", "learned_dense": "Learned\ndense",
        "oracle_dense": "Oracle\ndense", "oracle_gate": "Oracle\nold gate",
        "learned_int8_preserve_trial": "Old router\nallocation fixed"})
    print(json.dumps({"formal_interactions": result["new_formal_interactions"], "shared_controls_identical": True}))


if __name__ == "__main__":
    main()
