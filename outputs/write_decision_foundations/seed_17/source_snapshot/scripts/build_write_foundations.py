"""Build independently initialized behavior-cloned foundations for write tests.

This is an offline imitation-learning run. The only environment interactions
are the fixed-budget evaluation rollouts, which are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import psutil
import torch

from online_rl.envs import NavigationEnv, contexts
from online_rl.experiment import BlankEnv, make_model, seed_everything
from run_knowledge_loop import load_base_config


DEFAULT_SEEDS = (17, 29)
DEFAULT_UPDATES = 1500
BC_BATCH_SIZE = 512
BC_LEARNING_RATE = 0.002
EVALUATION_STEPS_PER_MODE = 1024
EVALUATION_PROTOCOL = {
    "source": ("normal", 2001),
    "unseen_cw_before_online": ("cw", 3001),
    "unseen_ccw_before_online": ("ccw", 4001),
}
_SOURCE_SNAPSHOT_FILES = (
    Path("scripts/build_write_foundations.py"),
    Path("scripts/run_knowledge_loop.py"),
    Path("configs/c1.json"),
    Path("data/pretrain_source.json"),
    Path("src/online_rl/experiment.py"),
    Path("src/online_rl/policy.py"),
    Path("src/online_rl/envs.py"),
    Path("src/online_rl/routing.py"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _prepare_bc_data(dataset_path: str | Path) -> dict[str, Any]:
    """Load only the source arrays used by experiment.pretrain and its split."""

    path = Path(dataset_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    required = ("observations", "episodes", "targets", "returns")
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"dataset is missing required arrays: {', '.join(missing)}")
        observations = np.asarray(archive["observations"])
        episodes = np.asarray(archive["episodes"])
        targets = np.asarray(archive["targets"])
        returns = np.asarray(archive["returns"])

    if observations.ndim != 2 or observations.shape[1] != 4 or len(observations) == 0:
        raise ValueError("dataset observations must be a nonempty (N, 4) array")
    count = len(observations)
    if episodes.shape != (count,) or not np.issubdtype(episodes.dtype, np.integer):
        raise ValueError("dataset episodes must be an integer array with shape (N,)")
    if targets.shape != (count, 4):
        raise ValueError("dataset targets must have shape (N, 4)")
    if returns.shape != (count,):
        raise ValueError("dataset returns must have shape (N,)")
    if not np.isfinite(observations).all() or not np.isfinite(targets).all() or not np.isfinite(returns).all():
        raise ValueError("dataset observations, targets, and returns must be finite")

    observations = np.asarray(observations, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    returns = np.asarray(returns, dtype=np.float32)
    augmented = np.concatenate(
        [observations, np.zeros((count, 9), dtype=np.float32)], axis=1
    )
    train_mask = episodes % 10 != 0
    train_ids = np.flatnonzero(train_mask)
    valid_ids = np.flatnonzero(~train_mask)
    if len(train_ids) == 0 or len(valid_ids) == 0:
        raise ValueError("whole-episode episode ID mod 10 split must have train and validation rows")
    return {
        "dataset_path": path,
        "observations": augmented,
        "targets": targets,
        "returns": returns,
        "episodes": episodes,
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "dataset_transitions": count,
        "train_transitions": int(len(train_ids)),
        "validation_transitions": int(len(valid_ids)),
        "dataset_sha256": _sha256(path),
    }


def _make_evaluation_env(context: str) -> NavigationEnv:
    return NavigationEnv(contexts()[context])


def evaluate_fixed_steps(
    model: Any,
    context: str,
    *,
    seed: int,
    steps: int = EVALUATION_STEPS_PER_MODE,
) -> dict[str, Any]:
    """Evaluate for an exact environment-step budget and report censoring."""

    step_budget = _positive_int(steps, "steps")
    env = _make_evaluation_env(context)
    completed_returns: list[float] = []
    completed_successes: list[int] = []
    total_steps = 0
    episode_index = 0
    partial_return: float | None = None
    partial_steps = 0
    current_return = 0.0
    try:
        observation, _ = env.reset(seed=int(seed))
        while total_steps < step_budget:
            augmented = np.concatenate(
                [np.asarray(observation, dtype=np.float32), np.zeros(8, dtype=np.float32), [0.0]]
            ).astype(np.float32)
            action, _ = model.predict(augmented, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(int(action))
            current_return += float(reward)
            total_steps += 1
            partial_steps += 1
            if terminated or truncated:
                completed_returns.append(current_return)
                completed_successes.append(int(bool(info.get("success", False))))
                episode_index += 1
                current_return = 0.0
                partial_steps = 0
                if total_steps < step_budget:
                    observation, _ = env.reset(seed=int(seed) + episode_index)
        if partial_steps:
            partial_return = float(current_return)
    finally:
        env.close()

    return {
        "context": context,
        "steps": total_steps,
        "step_budget": step_budget,
        "episodes": len(completed_returns) + int(partial_steps > 0),
        "completed_episodes": len(completed_returns),
        "mean_return": (
            float(np.mean(completed_returns)) if completed_returns else None
        ),
        "success_rate": (
            float(np.mean(completed_successes)) if completed_successes else None
        ),
        "partial_episode_return": partial_return,
        "partial_episode_steps": partial_steps,
    }


def _snapshot_sources(seed_output: Path, dataset: Mapping[str, Any]) -> dict[str, Any]:
    snapshot_root = seed_output / "source_snapshot"
    copied: dict[str, dict[str, Any]] = {}
    for relative in _SOURCE_SNAPSHOT_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"required source snapshot file is missing: {relative}")
        content = source.read_bytes()
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        if _sha256(destination) != digest:
            raise RuntimeError(f"source changed while snapshotting: {relative}")
        copied[str(relative)] = {"sha256": digest, "bytes": len(content)}

    manifest = {
        "sources": copied,
        "run_config": {
            "path": "config.json",
            "sha256": _sha256(seed_output / "config.json"),
            "bytes": (seed_output / "config.json").stat().st_size,
        },
        "dataset": {
            "path": str(dataset["dataset_path"]),
            "sha256": dataset["dataset_sha256"],
            "bytes": Path(dataset["dataset_path"]).stat().st_size,
        },
    }
    _write_json(snapshot_root / "manifest.json", manifest)
    return manifest


def _write_losses(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("update", "actor_loss", "value_loss", "total_loss"))
        writer.writeheader()
        writer.writerows(rows)


def _train_seed(
    seed_output: Path,
    *,
    seed: int,
    updates: int,
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> dict[str, Any]:
    """Train a fresh randomly initialized foundation from source BC data."""

    seed_output.mkdir(parents=True, exist_ok=False)
    config_snapshot = {
        **dict(config),
        "seed": seed,
        "bc_updates": updates,
        "bc_learning_rate": BC_LEARNING_RATE,
        "bc_batch_size": BC_BATCH_SIZE,
        "minibatch_rng_seed": 7001 + seed,
        "evaluation_steps_per_mode": EVALUATION_STEPS_PER_MODE,
        "foundation_initialization": "fresh random initialization; configured foundation checkpoint is not loaded",
    }
    _write_json(seed_output / "config.json", config_snapshot)
    snapshot = _snapshot_sources(seed_output, dataset)
    config_sha256 = _sha256(seed_output / "config.json")

    process = psutil.Process(os.getpid())
    rss_start = int(process.memory_info().rss)
    rss_peak = rss_start
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    seed_everything(seed)
    torch.set_num_threads(1)

    # PPO is used only to construct the repository's TreePolicy architecture.
    # As in experiment.pretrain, the blank environment is never stepped during
    # behavior cloning.
    model: Any | None = None
    try:
        model = make_model(BlankEnv(NavigationEnv()), {**dict(config), "seed": seed})
        optimizer = torch.optim.Adam(
            [
                parameter
                for name, parameter in model.policy.named_parameters()
                if not name.startswith("residuals")
            ],
            lr=BC_LEARNING_RATE,
        )
        rng = np.random.default_rng(7001 + seed)
        observations = dataset["observations"]
        targets = dataset["targets"]
        returns = dataset["returns"]
        train_ids = dataset["train_ids"]
        loss_rows: list[dict[str, int | float]] = []

        for update in range(1, updates + 1):
            ids = rng.choice(train_ids, size=BC_BATCH_SIZE, replace=True)
            obs_batch = torch.as_tensor(observations[ids])
            target_batch = torch.as_tensor(targets[ids])
            return_batch = torch.as_tensor(returns[ids])
            logits = model.policy.get_distribution(obs_batch).distribution.logits
            actor_loss = -(target_batch * logits).sum(1).mean()
            value = model.policy.predict_values(obs_batch).flatten()
            value_loss = (value - return_batch).square().mean()
            total_loss = actor_loss + 0.25 * value_loss
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            loss_rows.append(
                {
                    "update": update,
                    "actor_loss": float(actor_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "total_loss": float(total_loss.detach()),
                }
            )
            rss_peak = max(rss_peak, int(process.memory_info().rss))

        with torch.no_grad():
            valid_ids = dataset["valid_ids"]
            valid_obs = torch.as_tensor(observations[valid_ids])
            valid_targets = torch.as_tensor(targets[valid_ids])
            valid_logits = model.policy.get_distribution(valid_obs).distribution.logits
            validation_cross_entropy = float(-(valid_targets * valid_logits).sum(1).mean())

        checkpoint_path = seed_output / "foundation.pt"
        torch.save(model.policy.state_dict(), checkpoint_path)
        _write_losses(seed_output / "bc_loss.csv", loss_rows)

        evaluations: dict[str, Any] = {}
        for key, (context, eval_seed) in EVALUATION_PROTOCOL.items():
            evaluations[key] = evaluate_fixed_steps(
                model,
                context,
                seed=eval_seed,
                steps=EVALUATION_STEPS_PER_MODE,
            )
            rss_peak = max(rss_peak, int(process.memory_info().rss))
        rss_end = int(process.memory_info().rss)
        rss_peak = max(rss_peak, rss_end)
        wall_s = time.perf_counter() - wall_start
        cpu_s = time.process_time() - cpu_start

        metrics = {
            "seed": seed,
            "dataset_path": str(dataset["dataset_path"]),
            "dataset_sha256": dataset["dataset_sha256"],
            "config_sha256": config_sha256,
            "dataset_transitions": dataset["dataset_transitions"],
            "train_transitions": dataset["train_transitions"],
            "validation_transitions": dataset["validation_transitions"],
            "validation_split": "whole trajectory: episode ID mod 10",
            "validation_cross_entropy": validation_cross_entropy,
            "bc_updates": updates,
            "bc_batch_size": BC_BATCH_SIZE,
            "bc_learning_rate": BC_LEARNING_RATE,
            "minibatch_rng_seed": 7001 + seed,
            "evaluation_steps_per_mode": EVALUATION_STEPS_PER_MODE,
            "evaluation_steps_total": sum(int(result["steps"]) for result in evaluations.values()),
            "evaluations": evaluations,
            "wall_s": wall_s,
            "cpu_s": cpu_s,
            "rss_start_bytes": rss_start,
            "rss_peak_bytes": rss_peak,
            "rss_end_bytes": rss_end,
            "source_snapshot": snapshot,
            "weights_sha256": _sha256(checkpoint_path),
            "foundation_initialization": "fresh random initialization; no existing foundation weights loaded",
            "configured_foundation_path_ignored": str(config.get("foundation", "")),
        }
        _write_json(seed_output / "foundation.json", metrics)
        return metrics
    finally:
        if model is not None:
            model.env.close()


def _ledger_row(metrics: Mapping[str, Any]) -> dict[str, Any]:
    evaluations = metrics["evaluations"]
    return {
        "seed": metrics["seed"],
        "dataset_sha256": metrics["dataset_sha256"],
        "config_sha256": metrics["config_sha256"],
        "bc_updates": metrics["bc_updates"],
        "wall_s": metrics["wall_s"],
        "cpu_s": metrics["cpu_s"],
        "rss_start_bytes": metrics["rss_start_bytes"],
        "rss_peak_bytes": metrics["rss_peak_bytes"],
        "rss_end_bytes": metrics["rss_end_bytes"],
        "eval_source_steps": evaluations["source"]["steps"],
        "eval_cw_steps": evaluations["unseen_cw_before_online"]["steps"],
        "eval_ccw_steps": evaluations["unseen_ccw_before_online"]["steps"],
        "evaluation_steps_total": metrics["evaluation_steps_total"],
        "weights_sha256": metrics["weights_sha256"],
    }


def build_foundations(
    output: str | Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    updates: int = DEFAULT_UPDATES,
) -> dict[str, Any]:
    """Build all requested seeds in an empty output directory."""

    update_count = _positive_int(updates, "updates")
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or any(seed < 0 for seed in seed_values):
        raise ValueError("seeds must be a nonempty sequence of nonnegative integers")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be unique")

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {destination}")
        if any(destination.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty output directory: {destination}")

    base_config = load_base_config()
    dataset = _prepare_bc_data(base_config["dataset"])
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent)
    )
    staged_output = temporary_root / destination.name
    staged_output.mkdir()
    try:
        results: list[dict[str, Any]] = []
        for seed in seed_values:
            seed_output = staged_output / f"seed_{seed}"
            metrics = _train_seed(
                seed_output,
                seed=seed,
                updates=update_count,
                config=base_config,
                dataset=dataset,
            )
            results.append(metrics)

        total_evaluation_steps = sum(int(row["evaluation_steps_total"]) for row in results)
        summary = {
            "status": "completed",
            "dataset_path": str(dataset["dataset_path"]),
            "dataset_sha256": dataset["dataset_sha256"],
            "dataset_transitions": dataset["dataset_transitions"],
            "seeds": list(seed_values),
            "bc_updates_per_seed": update_count,
            "bc_updates_total": update_count * len(seed_values),
            "evaluation_steps_per_mode": EVALUATION_STEPS_PER_MODE,
            "evaluation_steps_per_seed": 3 * EVALUATION_STEPS_PER_MODE,
            "evaluation_steps_total": total_evaluation_steps,
            "wall_s_total": float(sum(float(row["wall_s"]) for row in results)),
            "cpu_s_total": float(sum(float(row["cpu_s"]) for row in results)),
            "rss_peak_bytes_max": max(int(row["rss_peak_bytes"]) for row in results),
            "per_seed": [
                {
                    "seed": row["seed"],
                    "path": f"seed_{row['seed']}",
                    "weights_sha256": row["weights_sha256"],
                    "config_sha256": row["config_sha256"],
                    "wall_s": row["wall_s"],
                    "cpu_s": row["cpu_s"],
                    "rss_start_bytes": row["rss_start_bytes"],
                    "rss_peak_bytes": row["rss_peak_bytes"],
                    "rss_end_bytes": row["rss_end_bytes"],
                    "bc_updates": row["bc_updates"],
                    "evaluation_steps": row["evaluation_steps_total"],
                    "evaluations": row["evaluations"],
                }
                for row in results
            ],
        }
        _write_json(staged_output / "summary.json", summary)
        ledger_rows = [_ledger_row(row) for row in results]
        with (staged_output / "ledger.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ledger_rows[0]))
            writer.writeheader()
            writer.writerows(ledger_rows)

        # Keep the final promotion atomic and leave an existing nonempty path
        # untouched if another process wrote there while this build ran.
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise FileExistsError(f"refusing to overwrite nonempty output directory: {destination}")
            destination.rmdir()
        os.replace(staged_output, destination)
        return summary
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "write_decision_foundations",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    summary = build_foundations(args.output, seeds=args.seeds, updates=args.updates)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
