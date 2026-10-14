"""Read-only diagnosis of completed A/B and compression experiments.

No environment is instantiated, stepped, or trained. New policy measurements
are forward passes on a deterministic grid of observations, not interactions.
Original logs, checkpoints, and selection gates are never changed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
READ_FILES = set()


def read_json(relative):
    path = ROOT / relative
    READ_FILES.add(path)
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return statistics.mean(values)


def diagnose_steps(path):
    READ_FILES.add(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    short = "x" in rows[0]
    keys = ("x", "y", "gx", "gy") if short else (
        "obs_x", "obs_y", "obs_goal_x", "obs_goal_y")
    done_key = "done" if short else "episode_done"
    success_key = "success" if short else "episode_success"
    complete = []
    episode_rows = []
    toward = away = stationary = 0
    reward_error = 0.0
    stationary_run = maximum_stationary_run = 0
    for row in rows:
        x, y, gx, gy = [float(row[key]) * 8 for key in keys]
        nx, ny = float(row["next_x"]) * 8, float(row["next_y"]) * 8
        d0, d1 = abs(x - gx) + abs(y - gy), abs(nx - gx) + abs(ny - gy)
        progress = round(d0 - d1)
        toward += int(progress > 0)
        away += int(progress < 0)
        stationary += int(progress == 0)
        stationary_run = stationary_run + 1 if progress == 0 else 0
        maximum_stationary_run = max(maximum_stationary_run, stationary_run)
        success = row[success_key] == "1"
        reward_error = max(reward_error, abs(float(row["reward"]) - (.25 * progress - .03 + success)))
        episode_rows.append((d0, d1, float(row["reward"])))
        if row[done_key] == "1":
            complete.append({
                "length": len(episode_rows), "initial_distance": episode_rows[0][0],
                "terminal_distance": d1, "success": success,
                "reward": sum(item[2] for item in episode_rows),
                "stretch": len(episode_rows) / episode_rows[0][0],
            })
            episode_rows = []
    successes = [item for item in complete if item["success"]]
    return {
        "steps": len(rows), "completed_episodes": len(complete),
        "successful_episodes": len(successes),
        "mean_completed_episode_length": mean([item["length"] for item in complete]),
        "mean_successful_episode_stretch": mean([item["stretch"] for item in successes]) if successes else None,
        "mean_completed_initial_distance": mean([item["initial_distance"] for item in complete]),
        "toward_goal_fraction": toward / len(rows),
        "away_from_goal_fraction": away / len(rows),
        "stationary_fraction": stationary / len(rows),
        "longest_stationary_streak": maximum_stationary_run,
        "maximum_reward_reconstruction_error": reward_error,
        "unfinished_tail_length": len(episode_rows),
    }


def repair_audit():
    original = read_json("outputs/compression_attempt/repair/summary.json")
    records = []
    for phase in ("before", "after"):
        for record in original[phase]:
            method, seed, context = record["method"], record["source_seed"], record["context"]
            path = ROOT / f"outputs/compression_attempt/repair/{phase}/{method}_s{seed}_{context}/steps.csv"
            records.append({"phase": phase, "method": method, "seed": seed,
                            "context": context, **diagnose_steps(path)})
    paired = []
    for phase in ("before", "after"):
        for seed in original["config"]["source_seeds"]:
            for context in original["config"]["contexts"]:
                runs = {item["method"]: item for item in original[phase]
                        if item["source_seed"] == seed and item["context"] == context}
                dense, sparse = runs["dense_compact"], runs["packed_0.5"]
                paired.append({"phase": phase, "seed": seed, "context": context,
                               "return_drop_percent": 100 * (dense["online_cumulative_return"] - sparse["online_cumulative_return"]) / dense["online_cumulative_return"],
                               "success_drop_pp": 100 * (dense["complete_episode_success_rate"] - sparse["complete_episode_success_rate"])})
    aggregates = []
    for phase in ("before", "after"):
        for context in original["config"]["contexts"]:
            for method in ("dense_compact", "packed_0.5"):
                chosen = [item for item in records if item["phase"] == phase and item["context"] == context and item["method"] == method]
                metrics = ("mean_completed_episode_length", "mean_successful_episode_stretch", "toward_goal_fraction", "away_from_goal_fraction", "stationary_fraction")
                aggregates.append({"phase": phase, "context": context, "method": method,
                                   **{key: mean([item[key] for item in chosen]) for key in metrics}})
    return {"per_run": records, "paired_return_gaps": paired, "aggregate": aggregates,
            "maximum_postrepair_single_run_return_drop_percent": max(item["return_drop_percent"] for item in paired if item["phase"] == "after"),
            "maximum_prerepair_single_run_return_drop_percent": max(item["return_drop_percent"] for item in paired if item["phase"] == "before")}


def stale_history_audit():
    original = read_json("outputs/ab_plus/summary.json")
    records = []
    for item in original["no_gradient_controls"]:
        if item["context"] != "cw" or item["branch"] not in ("P+H+", "P+H0"):
            continue
        path = ROOT / item["raw_steps_csv"]
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        prefix = rows[:item["recognized_first_step"] - 1]
        snapshot = read_json(f"outputs/validated/B_history_keep_s{item['seed']}/router.json")
        # All three saved runs discover CW first and CCW second; verify rather
        # than infer route correctness from arbitrary node identifiers.
        prototypes = snapshot["prototypes"]
        ccw_node = min(range(len(prototypes)), key=lambda i: sum((a-b)**2 for a,b in zip(prototypes[i], [0, -1])))
        high_confidence_wrong = [row for row in prefix if int(row["route"]) == ccw_node and float(row["route_confidence"]) >= .95]
        prefix_stationary = sum(float(row["obs_x"]) == float(row["next_x"]) and float(row["obs_y"]) == float(row["next_y"]) for row in prefix)
        records.append({"seed": item["seed"], "branch": item["branch"],
                        "first_correct_recognition_step": item["recognized_first_step"],
                        "prefix_high_confidence_wrong_route_steps": len(high_confidence_wrong),
                        "prefix_stationary_steps": prefix_stationary,
                        "early_128_return": item["early_window_cumulative_reward"],
                        "full_return": item["full_cumulative_reward"],
                        **diagnose_steps(path)})
    return records


def primary_75_audit():
    # Run summaries are the authoritative records even if the aggregate JSON
    # evolves; there is no reselection or modified empirical threshold here.
    records = []
    for seed in (41, 53, 67):
        for context in ("normal", "cw", "ccw", "cw_slip", "ccw_slip"):
            prefix = "outputs/compression_attempt/validation"
            sparse = read_json(f"{prefix}/packed_0.75_s{seed}_{context}/summary.json")
            dense = read_json(f"{prefix}/dense_compact_s{seed}_{context}/summary.json")
            records.append({"seed": seed, "context": context,
                            "return_drop_percent": 100 * (dense["online_cumulative_return"] - sparse["online_cumulative_return"]) / dense["online_cumulative_return"]})
    return records


def static_policy_probes():
    import torch
    from online_rl.compression import PackedResidualPolicy
    torch.set_num_threads(1)
    # The actor ignores the eight history features, which are set to zero;
    # this is a uniform all-state diagnostic, not an occupancy-weighted score.
    observations = torch.tensor([
        [x / 8, y / 8, gx / 8, gy / 8, *([0.] * 8), route]
        for route in (1, 2) for x in range(9) for y in range(9)
        for gx in range(9) for gy in range(9) if (x, y) != (gx, gy)
    ], dtype=torch.float32)
    # Ground-truth labels are used only for this offline diagnosis. CW and
    # CCW have route IDs 1 and 2 in all three saved trees, as checked above.
    intended = torch.zeros((len(observations), 4), dtype=torch.bool)
    intended[:, 0] = observations[:, 3] < observations[:, 1]
    intended[:, 1] = observations[:, 2] > observations[:, 0]
    intended[:, 2] = observations[:, 3] > observations[:, 1]
    intended[:, 3] = observations[:, 2] < observations[:, 0]
    rotations = torch.where(observations[:, -1] == 1, 1, -1)
    effective_actions = (torch.arange(4)[None, :] + rotations[:, None]) % 4
    shortest_mask = intended.gather(1, effective_actions)
    records = []
    for seed in (41, 53, 67):
        prototypes = read_json(f"outputs/validated/B_history_keep_s{seed}/router.json")["prototypes"]
        cw_node = min(range(len(prototypes)), key=lambda i: sum((a-b)**2 for a,b in zip(prototypes[i], [0, 1])))
        ccw_node = min(range(len(prototypes)), key=lambda i: sum((a-b)**2 for a,b in zip(prototypes[i], [0, -1])))
        if (cw_node, ccw_node) != (1, 2):
            raise ValueError("Static optimal-action mask requires verified route IDs")
        for phase in ("before", "after"):
            if phase == "before":
                folder = ROOT / "outputs/compression_attempt/bundles"
            else:
                folder = ROOT / "outputs/compression_attempt/repair/continuation"
            distributions = {}
            for method in ("dense_compact", "packed_0.5"):
                checkpoint = folder / f"{method}_s{seed}/agent.pt"
                READ_FILES.add(checkpoint)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                policy = PackedResidualPolicy.from_payload(payload)["policy"]
                policy.eval()
                with torch.no_grad():
                    distributions[method] = torch.cat([
                        policy.get_distribution(batch).distribution.probs
                        for batch in observations.split(512)])
                del payload, policy
            p, q = distributions["dense_compact"], distributions["packed_0.5"]
            kl = (p * ((p + 1e-12).log() - (q + 1e-12).log())).sum(dim=1)
            agreements = p.argmax(1) == q.argmax(1)
            p_good = shortest_mask.gather(1, p.argmax(1)[:, None]).squeeze(1)
            q_good = shortest_mask.gather(1, q.argmax(1)[:, None]).squeeze(1)
            records.append({"phase": phase, "seed": seed, "grid_observations": len(observations),
                            "mean_uniform_grid_KL_dense_to_sparse_nats": float(kl.mean()),
                            "maximum_probability_difference": float((p-q).abs().max()),
                            "greedy_action_agreement": float(agreements.float().mean()),
                            "dense_mean_shortest_action_probability": float((p * shortest_mask).sum(1).mean()),
                            "sparse_mean_shortest_action_probability": float((q * shortest_mask).sum(1).mean()),
                            "dense_greedy_shortest_action_fraction": float(p_good.float().mean()),
                            "sparse_greedy_shortest_action_fraction": float(q_good.float().mean()),
                            "both_greedy_actions_shortest_among_disagreements": float((p_good & q_good)[~agreements].float().mean()),
                            "interpretation": "uniform grid across the two residual routes, no environment interactions; not a return or safety certificate"})
    return records


def main():
    result = {"scope": "read-only analysis of existing results; no training or environment interactions",
              "repair": repair_audit(), "stale_history": stale_history_audit(),
              "primary_75_per_run": primary_75_audit(), "static_policy_probes": static_policy_probes()}
    destination = Path(__file__).with_name("diagnostics.json")
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for relative in ("src/online_rl/compression.py", "src/online_rl/policy.py", "src/online_rl/routing.py", "src/online_rl/envs.py"):
        READ_FILES.add(ROOT / relative)
    READ_FILES.add(Path(__file__).resolve())
    provenance = {"scope": result["scope"], "new_environment_interactions": 0,
                  "new_optimizer_updates": 0,
                  "inputs_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted(READ_FILES)}}
    Path(__file__).with_name("provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(destination)
    print("Repair episode lengths and movement fractions:")
    for item in result["repair"]["aggregate"]:
        if item["context"] in ("cw", "ccw", "cw_slip", "ccw_slip"):
            print(item)
    print("Stale history:")
    for item in result["stale_history"]:
        print({key: item[key] for key in ("seed", "branch", "first_correct_recognition_step", "prefix_high_confidence_wrong_route_steps", "prefix_stationary_steps")})
    print("Static policy probes:", result["static_policy_probes"])
    print("Maximum per-run repair return gaps:", result["repair"]["maximum_prerepair_single_run_return_drop_percent"], result["repair"]["maximum_postrepair_single_run_return_drop_percent"])
    print("Primary 75% per-run return gaps:", result["primary_75_per_run"])


if __name__ == "__main__":
    main()
