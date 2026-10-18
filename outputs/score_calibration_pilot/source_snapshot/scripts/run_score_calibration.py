"""Freeze retention predictions, then pay for paired natural-recurrence forks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from online_rl.score_calibration import (grid_observations, paired_rollouts,
    probability_index, restore_navigation, roundtrip_common, snapshot_navigation)
from online_rl.write_decision_envs import task_name
from scripts import run_retention_stream as stream

CONFIG = ROOT / "configs/score_calibration.json"
SOURCES = tuple(dict.fromkeys((Path("scripts/run_score_calibration.py"),
    Path("src/online_rl/score_calibration.py"), Path("configs/score_calibration.json"),
    Path("docs/SCORE_CALIBRATION_PROTOCOL.md"), *stream.SOURCES)))


def write(path, value):
    stream.write(path, value)


def state_hash(value):
    encoded = json.dumps(value, sort_keys=True, default=stream.ws._json_default).encode()
    return hashlib.sha256(encoded).hexdigest()


def finite_mean(values):
    valid = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(valid)) if valid else None


def learner_signature(agent):
    return (stream.policy_hash(agent.core), state_hash(agent.counts),
        state_hash(agent.action_rng.bit_generator.state),
        state_hash(agent.behavior_rng.bit_generator.state),
        hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())


def persistent_learner_hash(agent, config, budget):
    """Hash every serialized learner value without pickle object identities."""
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(b"torch")
            visit(array)
        elif isinstance(value, np.ndarray):
            digest.update(b"array" + str(value.dtype).encode() + repr(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict" + str(len(value)).encode())
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode() + str(len(value)).encode())
            for item in value:
                visit(item)
        elif isinstance(value, bytes):
            digest.update(b"bytes" + str(len(value)).encode() + value)
        else:
            digest.update(type(value).__name__.encode())
            digest.update(json.dumps(value, sort_keys=True, default=stream.ws._json_default).encode())

    visit(stream.checkpoint_payload(agent, config, "retention_gate", budget))
    return digest.hexdigest()


class ObservedEnvironment:
    """Record real pre-action states; delegate the unchanged learner stream."""
    def __init__(self, env, observer):
        self.env, self.observer = env, observer

    def __getattr__(self, name):
        return getattr(self.env, name)

    def set_task(self, spec):
        self.env.set_task(spec)
        self.observer.start_bout(task_name(spec))

    def step(self, action):
        self.observer.before_step(self.env)
        self.observer.learner_steps += 1
        return self.env.step(action)


class CalibrationObserver:
    def __init__(self, config, seed, budget, bout_steps):
        self.config, self.seed, self.budget = config, int(seed), budget
        self.cutoff = int(config["capture_initial_bouts"] * bout_steps)
        self.bout_steps = int(bout_steps)
        self.learner_steps = self.observer_steps = 0
        self.bout, self.bout_start = -1, 0
        self.current_task, self.current_recurrent = None, False
        self.seen_tasks, self.lineage = set(), {}
        self.predictions, self.anchors = [], {}
        self.agent = None
        self.snapshot_audits = []

    def start_bout(self, task):
        self.bout += 1
        self.bout_start = self.learner_steps
        self.current_task = task
        self.current_recurrent = task in self.seen_tasks
        self.seen_tasks.add(task)

    def before_step(self, env):
        offset = self.learner_steps - self.bout_start
        if not self.current_recurrent or offset not in self.config["anchor_offsets"]:
            return
        before = snapshot_navigation(env)
        restored = restore_navigation(before)
        try:
            after = snapshot_navigation(restored)
            if state_hash(before) != state_hash(after):
                raise RuntimeError("observer snapshot did not restore exactly")
            if not np.array_equal(env._observation(), restored._observation()):
                raise RuntimeError("observer snapshot changed initial observation")
        finally:
            restored.close()
        if state_hash(snapshot_navigation(env)) != state_hash(before):
            raise RuntimeError("observer snapshot changed the live environment")
        slot = self.agent.cache.task_to_slot.get(self.current_task) if self.agent is not None else None
        record = {"task": self.current_task, "bout": self.bout, "offset": offset,
            "raw_step": self.learner_steps + 1, "snapshot": before,
            "snapshot_sha256": state_hash(before), "live_slot": slot,
            "live_generation": int(self.agent.slot_generation[slot]) if slot is not None else None,
            "live_acting_group": self.agent.core.group_for_slot(slot) if slot is not None else None}
        self.anchors.setdefault(self.current_task, []).append(record)
        self.snapshot_audits.append(True)

    def update_lineage(self, agent):
        self.agent = agent
        for task, slot in agent.cache.task_to_slot.items():
            key = (int(slot), int(agent.slot_generation[slot]))
            if key in self.lineage and self.lineage[key] != task:
                raise RuntimeError("a source slot/generation changed task identity")
            self.lineage[key] = task

    def evaluate(self, original, agent, config, rows, group):
        self.update_lineage(agent)
        pending = agent.pending
        previous = None if pending is None else agent.core.read_shared_matrix()[pending["group"]].reshape(3, 3).copy()
        proposal = copy.deepcopy(pending)
        decision = original(agent, config, rows, group)
        if decision is None or decision["event"] != "write_test" or decision["step"] > self.cutoff:
            return decision
        signature = learner_signature(agent)
        candidate = np.asarray(decision["common"], dtype=np.float32)
        candidate_q, candidate_error = roundtrip_common(candidate)
        previous_q, previous_error = roundtrip_common(previous)
        historical, weights, contexts, codec_scores, missing = [], {}, set(), [], []
        profile_count = len(decision["historical_scores"])
        for score in decision["historical_scores"]:
            prototype = int(score["prototype"])
            evidence = agent.archive.arrays(prototype)
            context_counts = {}
            for slot, generation in zip(evidence["source_slot"], evidence["source_generation"]):
                key = (int(slot), int(generation))
                task = self.lineage.get(key)
                if task is None:
                    missing.append(list(key))
                else:
                    context_counts[task] = context_counts.get(task, 0) + 1
                    contexts.add(task)
            n = sum(context_counts.values())
            for task, count in context_counts.items():
                weights[task] = weights.get(task, 0.0) + count / max(n, 1) / max(profile_count, 1)
            coded_score = stream.score_common(agent, config, evidence, candidate_q, previous_q)
            codec_scores.append(coded_score)
            historical.append({"prototype": prototype, "rows": len(evidence["action"]),
                "context_counts": context_counts, "original_score": score, "codec_score": coded_score})
        weight_sum = sum(weights.values())
        if weight_sum:
            weights = {key: value / weight_sum for key, value in weights.items()}
        original_scores = decision["historical_scores"]
        sources = []
        for source in decision["sources"]:
            key = (int(source["slot"]), int(source["generation"]))
            sources.append({**source, "task_observer": self.lineage.get(key)})
        prediction = {"id": f"{self.seed}-{self.budget}-{decision['step']}",
            "seed": self.seed, "budget": self.budget, "group": decision["group"],
            "step": decision["step"], "proposal_step": decision["proposal_step"],
            "accepted": decision["accepted"], "decision_reason": decision["reason"],
            "proposal_parent_version": proposal["parent_version"],
            "candidate_raw": candidate, "old_parent_raw": previous,
            "candidate_int8": candidate_q, "old_parent_int8": previous_q,
            "codec_error": {"candidate": candidate_error, "baseline": previous_error},
            "predictors": {"original_mean_lower": finite_mean(s.get("lower") for s in original_scores),
                "original_mean_gain": finite_mean(s.get("gain") for s in original_scores),
                "codec_mean_lower": finite_mean(s.get("lower") for s in codec_scores),
                "fresh_lower": decision["fresh_score"].get("lower"),
                "negative_mean_stderr": finite_mean(-s["stderr"] for s in original_scores
                    if s.get("stderr") is not None and np.isfinite(s["stderr"])),
                "time_step": float(decision["step"])},
            "contexts": sorted(contexts), "prototype_weights": weights,
            "historical_provenance": historical, "fresh_score": decision["fresh_score"],
            "proposal_sources": sources, "missing_source_lineage": missing,
            "prediction_frozen_before_outcome": True}
        if learner_signature(agent) != signature:
            raise RuntimeError("prediction observer mutated learner parameters/counters/RNG")
        self.predictions.append(prediction)
        return decision


def outcome_summary(replicate_gains, prototype_gains, config):
    if config["replicates"] != 8 or config["mc_interval_level"] != 0.95:
        raise ValueError("this frozen protocol uses eight replicates and a 95% t7 interval")
    gains = np.asarray(replicate_gains, dtype=np.float64)
    if gains.shape != (config["replicates"],) or not np.isfinite(gains).all():
        raise RuntimeError("invalid paired outcome replicates")
    mean = float(gains.mean())
    stderr = float(gains.std(ddof=1) / np.sqrt(len(gains)))
    critical = 2.3646242515927844
    low, high = mean - critical * stderr, mean + critical * stderr
    margin = float(config["benefit_margin"])
    sign = "beneficial" if low > margin else "harmful" if high < -margin else "ambiguous"
    return {"replicate_gains": gains.tolist(), "prototype_weighted_replicate_gains": list(prototype_gains),
        "mean_gain": mean, "stderr": stderr, "ci_low": low, "ci_high": high, "sign": sign,
        "interval_scope": "Pointwise paired Monte Carlo interval conditional on observed recurrence anchors."}


def tree_hashes(folder):
    return {str(p.relative_to(folder)): stream.sha(p) for p in folder.rglob("*") if p.is_file()}


def run_one(config, stream_config, seed, budget, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    observer = CalibrationObserver(config, seed, budget, stream_config["bout_steps"])
    started, cpu_started = time.perf_counter(), time.process_time()
    original_make, original_evaluate = stream.make_env, stream.evaluate_pending
    write(folder / "run_config.json", {"config": config, "learner_config": stream_config,
        "seed": seed, "budget": budget})
    try:
        stream.make_env = lambda *args, **kwargs: ObservedEnvironment(original_make(*args, **kwargs), observer)
        stream.evaluate_pending = lambda *args, **kwargs: observer.evaluate(original_evaluate, *args, **kwargs)
        try:
            learned = stream.run_one(stream_config, "retention_gate", budget, seed, folder / "learner")
        finally:
            stream.make_env, stream.evaluate_pending = original_make, original_evaluate
        if observer.learner_steps != learned["raw_steps"]:
            raise RuntimeError("observer learner interaction count mismatch")
        if set(observer.anchors) != set(observer.seen_tasks) or any(
                sorted(a["offset"] for a in values) != config["anchor_offsets"] for values in observer.anchors.values()):
            raise RuntimeError("missing or duplicate natural recurrence anchors")
        write(folder / "predictions.json", observer.predictions)
        write(folder / "recurrence_anchors.json", observer.anchors)
        predictions_hash, anchors_hash = stream.sha(folder / "predictions.json"), stream.sha(folder / "recurrence_anchors.json")
        learner_hashes = tree_hashes(folder / "learner")
        # Supplied coefficients bypass all trained locals/parents. Read the
        # unchanged W0 feature path without initializing another learner RNG.
        actor = observer.agent
        frozen_learner_signature = learner_signature(actor)
        frozen_persistent_hash = persistent_learner_hash(actor, stream_config, budget)
        grid = grid_observations()
        outcomes, excluded = [], []
        fork_path = folder / "forks"
        fork_path.mkdir()

        class CountedEnvironment:
            def __init__(self, env):
                self.env = env

            def __getattr__(self, name):
                return getattr(self.env, name)

            def step(self, action):
                observer.observer_steps += 1
                return self.env.step(action)

        for prediction in observer.predictions:
            reason = None
            if prediction["missing_source_lineage"]:
                reason = "untraceable_historical_source"
            elif not prediction["contexts"]:
                reason = "no_historical_source_context"
            elif any(task not in observer.anchors or any(a["raw_step"] <= prediction["step"]
                    for a in observer.anchors[task]) for task in prediction["contexts"]):
                reason = "missing_or_nonfuture_natural_recurrence"
            if reason:
                excluded.append({**prediction, "reason": reason})
                continue
            candidate_table = stream.coefficient_probs(actor.core, grid, prediction["candidate_int8"])
            baseline_table = stream.coefficient_probs(actor.core, grid, prediction["old_parent_int8"])
            # Independent implementation check on all actual future anchors.
            table_error = 0.0
            for task in prediction["contexts"]:
                for anchor in observer.anchors[task]:
                    env = restore_navigation(anchor["snapshot"])
                    try:
                        obs = stream.loop.augment_observation(env._observation(), 1)
                        direct = stream.coefficient_probs(actor.core, obs[None], prediction["candidate_int8"])[0]
                        table_error = max(table_error, float(np.max(np.abs(direct - candidate_table[probability_index(obs[:4])]))))
                    finally:
                        env.close()
            if table_error > 1e-6:
                raise RuntimeError("frozen lookup differs from direct actor distribution")
            per_context = {}
            for task in prediction["contexts"]:
                anchor_gains = []
                for anchor in observer.anchors[task]:
                    seed_material = f"{config['evaluation_seed']}|{seed}|{task}|{anchor['offset']}".encode()
                    eval_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
                    result = paired_rollouts(anchor["snapshot"], candidate_table, baseline_table,
                        horizon=config["horizon"], replicates=config["replicates"], seed=eval_seed,
                        env_factory=lambda value: CountedEnvironment(restore_navigation(value)))
                    for trial in result["per_replicate"]:
                        audit = trial["audit"]
                        if not all(audit[key] for key in ("first_observations_equal", "initial_rng_state_equal",
                                "paired_action_uniform_per_raw_step", "corresponding_reset_seeds_equal")):
                            raise RuntimeError("paired branch state/RNG audit failed")
                    write(fork_path / f"step_{prediction['step']:08d}_{task}_{anchor['offset']:03d}.json",
                        {"candidate_id": prediction["id"], "task": task, "anchor": anchor,
                         "evaluation_seed": eval_seed, **result})
                    anchor_gains.append([r["differences"]["return_per_raw_step"] for r in result["per_replicate"]])
                per_context[task] = np.mean(np.asarray(anchor_gains), axis=0)
            equal = np.mean(list(per_context.values()), axis=0)
            weighted = sum(prediction["prototype_weights"][task] * per_context[task] for task in per_context)
            outcomes.append({**prediction, "outcome": outcome_summary(equal, weighted, config),
                "context_replicate_gains": per_context, "lookup_max_probability_error": table_error,
                "observer_steps": 2 * config["horizon"] * config["replicates"] *
                    len(config["anchor_offsets"]) * len(prediction["contexts"])})
            print(f"forked seed={seed} budget={budget} test={prediction['step']} contexts={len(per_context)}", flush=True)
        expected_observer = sum(record["observer_steps"] for record in outcomes)
        if expected_observer != observer.observer_steps:
            raise RuntimeError("paid observer interaction ledger mismatch")
        if learner_hashes != tree_hashes(folder / "learner") or predictions_hash != stream.sha(folder / "predictions.json") \
                or anchors_hash != stream.sha(folder / "recurrence_anchors.json"):
            raise RuntimeError("frozen learner/prediction/anchor artifacts changed during evaluation")
        if learner_signature(actor) != frozen_learner_signature:
            raise RuntimeError("paid observer changed live learner parameters/counters/RNG")
        if persistent_learner_hash(actor, stream_config, budget) != frozen_persistent_hash:
            raise RuntimeError("paid observer changed serialized optimizer/archive/router/window/pending state")
        result = {"format": "score_calibration_run_v1", "status": "completed", "seed": seed,
            "budget": budget, "learner_steps": observer.learner_steps, "observer_steps": observer.observer_steps,
            "captured_tests": len(observer.predictions), "candidates": outcomes, "excluded_candidates": excluded,
            "prediction_sha256": predictions_hash, "anchor_sha256": anchors_hash,
            "learner_summary": learned, "cpu_s": time.process_time() - cpu_started,
            "wall_s": time.perf_counter() - started,
            "audit": {"passed": True, "prediction_frozen_before_evaluation": True,
                "learner_artifacts_unchanged_by_observer": True, "natural_anchor_roundtrip": True,
                "paid_interaction_ledger": True, "lookup_distribution_verified": True}}
        write(folder / "summary.json", result)
        return result
    except Exception as error:
        write(folder / "FAILED_RUN.json", {"status": "failed", "seed": seed, "budget": budget,
            "reason": repr(error), "learner_steps": observer.learner_steps, "observer_steps": observer.observer_steps,
            "cpu_s": time.process_time() - cpu_started, "wall_s": time.perf_counter() - started})
        raise
    finally:
        stream.make_env, stream.evaluate_pending = original_make, original_evaluate


def snapshot_sources(folder):
    manifest = {}
    for relative in SOURCES:
        if any(part.casefold() in {"agent.md", "agents.md", "tmp"} for part in relative.parts):
            raise ValueError("private path in source snapshot")
        source, target = ROOT / relative, folder / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[relative.as_posix()] = {"sha256": stream.sha(target), "bytes": target.stat().st_size}
    write(folder / "source_snapshot/manifest.json", manifest)
    return manifest


def run_cohort(config, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new output directory; preserve all earlier runs")
    output.mkdir(parents=True, exist_ok=True)
    source_config = json.loads((ROOT / config["source_config"]).read_text(encoding="utf-8"))
    source_config["bout_steps"] = config["bout_steps"]
    source_config["seeds"] = list(config["seeds"])
    source_config["methods"] = ["retention_gate"]
    source_config["mode"] = "score_calibration_" + config["mode"]
    manifest = snapshot_sources(output)
    write(output / "frozen_config.json", {"calibration": config, "learner": source_config})
    started = time.perf_counter()
    records, failures = [], []
    with ProcessPoolExecutor(max_workers=config["max_workers"]) as executor:
        jobs = {executor.submit(run_one, config, source_config, seed, budget,
            output / f"seed_{seed}" / budget): (seed, budget) for seed in config["seeds"] for budget in config["budgets"]}
        for future in as_completed(jobs):
            seed, budget = jobs[future]
            try:
                record = future.result()
                records.append(record)
                print(f"completed seed={seed} budget={budget} evaluated={len(record['candidates'])} observer_steps={record['observer_steps']}", flush=True)
            except Exception as error:
                failures.append({"seed": seed, "budget": budget, "reason": repr(error)})
                print(f"failed seed={seed} budget={budget}: {error!r}", flush=True)
    changed = [path for path, item in manifest.items() if stream.sha(ROOT / path) != item["sha256"]]
    if changed:
        failures.append({"reason": "sources_changed_during_run", "paths": changed})
    failed_costs = [json.loads(path.read_text(encoding="utf-8")) for path in output.glob("seed_*/*/FAILED_RUN.json")]
    all_costs = records + failed_costs
    learner_steps = sum(row["learner_steps"] for row in all_costs)
    observer_steps = sum(row["observer_steps"] for row in all_costs)
    result = {"format": "score_calibration_cohort_v1", "status": "completed" if not failures else "failed",
        "analysis_role": "diagnostic", "config": config, "records": records,
        "audit": {"passed": not failures, "failures": failures},
        "cost": {"learner_steps": learner_steps, "observer_steps": observer_steps,
            "new_total_interactions": learner_steps + observer_steps}, "failed_costs": failed_costs,
        "cpu_s": sum(row["cpu_s"] for row in all_costs), "wall_s": time.perf_counter() - started}
    write(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "main"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["mode"] = args.mode
    config["bout_steps"] = config["pilot_bout_steps"] if args.mode == "pilot" else config["main_bout_steps"]
    if args.mode == "pilot":
        config["seeds"], config["budgets"] = [config["pilot_seed"]], ["low"]
    result = run_cohort(config, args.output)
    print(json.dumps(result["cost"]))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
