"""Fresh PPO references with the analytical control's initial random seeds."""
from pathlib import Path
import argparse
import csv
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from online_rl.experiment import StreamEnv, BudgetCallback, FIELDS, SEQUENCE, make_model, seed_everything, write_json
from online_rl.routing import HistoryRouter
from online_rl.controls import save_deployment_bundle


class MatchedStreamEnv(StreamEnv):
    def __init__(self, *args, initial_seed, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_seed = initial_seed

    def reset(self, **kwargs):
        if self.first_reset:
            self.first_reset = False
            kwargs["seed"] = self.initial_seed
        return super().reset(**kwargs)


def run(config, root):
    root = Path(root)
    runs = []
    source = np.load(config["dataset"])["source_responses"]
    foundation = torch.load(config["foundation"], map_location="cpu", weights_only=True)
    for stage, method, steps, offset in [("A", "global_residual", config["a_steps"], 10000),
                                        ("B", "history_keep", config["stage_b_steps"], 20000)]:
        for seed in config["source_seeds"]:
            initial_seed = config["evaluation_seed_offset"] + offset + seed
            out = root / stage / f"s{seed}" / method
            out.mkdir(parents=True, exist_ok=True)
            router = HistoryRouter(source, window=config["route_window"], novelty_threshold=config["novelty_threshold"])
            if stage == "A":
                router.max_nodes = 1
            sequence = ["cw"] if stage == "A" else SEQUENCE
            cap_fields = ["global_steps", "policy_bytes", "optimizer_bytes", "router_bytes", "rollout_bytes",
                          "serialized_core_bytes", "agent_state_bytes", "rss_bytes"]
            with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as ef, \
                 (out / "capacity.csv").open("w", newline="", encoding="utf-8") as cf:
                ew = csv.DictWriter(ef, fieldnames=FIELDS)
                ew.writeheader()
                cw = csv.DictWriter(cf, fieldnames=cap_fields)
                cw.writeheader()
                env = MatchedStreamEnv(stage, method, seed, router, config, ew, sequence,
                                       initial_seed=initial_seed)
                vec = DummyVecEnv([lambda: env])
                model = make_model(vec, {**config, "seed": initial_seed})
                model.policy.load_state_dict(foundation)
                model.policy.configure_learning(method, config["learning_rate"])
                env.model = model
                callback = BudgetCallback(env, cw, config["agent_budget_bytes"])
                # Reset after construction, matching the analytic runner's action RNG.
                seed_everything(initial_seed)
                start = time.perf_counter()
                model.learn(total_timesteps=steps, callback=callback)
                elapsed = time.perf_counter() - start
                ef.flush()
                rows = list(csv.DictReader((out / "episodes.csv").open(encoding="utf-8")))
                success = sum(int(row["success"]) for row in rows)
                result = {"seed": seed, "stage": stage, "variant": method, "steps": env.total_steps,
                          "evaluation_seed": initial_seed, "context_schedule": sequence,
                          "full_cumulative_reward": env.cumulative_return,
                          "complete_episode_success_rate": success / max(1, len(rows)),
                          "completed_episodes": len(rows), "successful_episodes": success,
                          "online_elapsed_s": elapsed, "policy_updates": model.policy.optimizer.step_count,
                          "peak_accounted_state_bytes": callback.peak,
                          "max_sampled_process_rss_bytes": callback.rss_peak,
                          "initial_seed_alignment": "same initial environment and action RNG as analytic; own online trajectories",
                          "scope": "PPO from source-only foundation, not terminal-policy replay"}
                save_deployment_bundle(out / "agent.pt", model, router, result)
                np.savez_compressed(out / "last_rollout_audit.npz", observations=model.rollout_buffer.observations,
                                    actions=model.rollout_buffer.actions, log_probs=model.rollout_buffer.log_probs,
                                    optimizer_versions=model.rollout_buffer.optimizer_versions,
                                    reset_versions=model.rollout_buffer.reset_versions)
                write_json(out / "summary.json", result)
                vec.close()
            runs.append(result)
            print(json.dumps({"stage": stage, "seed": seed, "variant": method,
                              "return": result["full_cumulative_reward"]}), flush=True)
    write_json(root / "summary.json", {"config": config, "runs": runs})


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/ab_plus.json")
parser.add_argument("--output", default="outputs/ab_plus/matched_online")
args = parser.parse_args()
run(json.loads(Path(args.config).read_text(encoding="utf-8")), args.output)
