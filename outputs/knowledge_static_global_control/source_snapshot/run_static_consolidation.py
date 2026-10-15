"""Run paired static codec diagnostics on existing hierarchical C1 heads."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import psutil
import torch

from online_rl.c1_envs import C1NavigationEnv, c1_contexts, context_index
from online_rl.c1_experiment import augment
from online_rl.c1_tensor_controls import TensorSourceBasisPolicy
from online_rl.compact_knowledge import CompactKnowledge, METHODS, encode, minimum_nbytes


CHECKPOINT_ROOT = Path(
    "outputs/c1_supplement/representation_alternatives/"
    "hierarchical_output_mixer/formal/hierarchical_output_mixer/none/low"
)
CHECKPOINT_SEEDS = (181, 193, 211)
FORMAL_BUDGETS = (192, 320, 448, 648, 768)
CONTEXTS = ("A0", "A1", "A2", "B0", "B1", "B2")
OUTPUTS = {
    "pilot": Path("outputs/knowledge_static_pilot"),
    "formal": Path("outputs/knowledge_static_formal"),
    "global_control": Path("outputs/knowledge_static_global_control"),
}
ORIGINAL_METHODS = tuple(method for method in METHODS if method != "global_topk")
EPISODE_FIELDS = [
    "phase", "source_seed", "budget_bytes", "method", "candidate_id",
    "target", "episode", "pair_id", "episode_seed", "return", "success",
    "length", "reference_return", "reference_success", "reference_length",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_path(seed: int) -> Path:
    return CHECKPOINT_ROOT / f"s{seed}" / "history" / "agent.pt"


def _restore_checkpoint(path: Path) -> TensorSourceBasisPolicy:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("spec", {}).get("supplemental_layout") != "c1_tensor_source_basis_v1":
        raise ValueError(f"{path} is not a C1 tensor source-basis checkpoint")
    restored = TensorSourceBasisPolicy.from_payload(payload)
    policy = restored["policy"]
    if policy._storage != "hierarchical_output_mixer":
        raise ValueError(f"{path} does not contain a hierarchical output mixer")
    if float(policy.temperature) != 2.0:
        raise ValueError(f"{path} has unexpected temperature {policy.temperature}")
    policy.eval()
    return policy


def _effective_matrix(policy: TensorSourceBasisPolicy) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for endpoint in range(1, 7):
            family = "7" if endpoint <= 3 else "8"
            effective = policy._nodes[str(endpoint)].A + policy._families[family].C.unsqueeze(0)
            rows.append(effective.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=True))
    result = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
    if result.shape != (6, 27):
        raise RuntimeError(f"expected six effective coefficient rows, got {result.shape}")
    return result


def _policy_storage_bytes(policy: TensorSourceBasisPolicy) -> dict[str, int]:
    tensors = policy.state_dict()
    total = sum(int(value.numel() * value.element_size()) for value in tensors.values())
    target_coefficients = sum(
        int(value.numel() * value.element_size())
        for name, value in tensors.items()
        if (name.startswith("_nodes.") and name.endswith(".A"))
        or (name.startswith("_families.") and name.endswith(".C"))
    )
    return {
        "full_checkpoint_policy_tensor_nbytes": total,
        "original_hierarchical_target_coefficients_tensor_nbytes": target_coefficients,
        "common_frozen_policy_tensor_nbytes": total - target_coefficients,
    }


def _python_object_overhead_bytes(compact) -> int:
    arrays = compact.arrays
    return int(
        sys.getsizeof(compact)
        + sys.getsizeof(compact.method)
        + sys.getsizeof(compact.shape)
        + sys.getsizeof(arrays)
        + sum(
            sys.getsizeof(name) + max(0, sys.getsizeof(array) - int(array.nbytes))
            for name, array in arrays.items()
        )
    )


def _install_decoded(policy: TensorSourceBasisPolicy, matrix: np.ndarray) -> TensorSourceBasisPolicy:
    candidate = copy.deepcopy(policy)
    values = np.asarray(matrix, dtype=np.float32)
    if values.shape != (6, 27):
        raise ValueError(f"decoded target-head matrix must have shape (6, 27), got {values.shape}")
    with torch.no_grad():
        for endpoint in range(1, 7):
            decoded = torch.as_tensor(values[endpoint - 1].reshape(3, 3, 3), dtype=torch.float32)
            candidate._nodes[str(endpoint)].A.copy_(decoded)
        # Each decoded A_i already includes its trained family coefficient.
        # Evaluation supplies leaf routes only, so clear the family path to
        # avoid counting C a second time.
        for family in ("7", "8"):
            candidate._families[family].C.zero_()
    candidate.eval()
    return candidate


def _pair_seed(source_seed: int, target: str, episode: int) -> int:
    # Stable reset/action seeds shared by the reference and every candidate.
    return int((source_seed * 1_000_003 + context_index(target) * 10_007 + episode * 97_409) % (2**32 - 1))


def _episode(policy: TensorSourceBasisPolicy, target: str, seed: int) -> dict[str, int | float]:
    env = C1NavigationEnv(c1_contexts()[target])
    obs, _ = env.reset(seed=seed)
    route = context_index(target) + 1
    action_rng = np.random.default_rng(seed + 1)
    total = 0.0
    success = 0
    length = 0
    with torch.no_grad():
        while True:
            augmented = augment(obs, route)
            probs = policy.get_distribution(torch.as_tensor(augmented[None])).distribution.probs[0]
            probabilities = probs.detach().cpu().numpy().astype(np.float64)
            probabilities /= probabilities.sum()
            action = int(action_rng.choice(4, p=probabilities))
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            length += 1
            if terminated or truncated:
                success = int(info["success"])
                break
    env.close()
    return {"return": float(total), "success": success, "length": length}


def _probe_observations(count: int = 256) -> np.ndarray:
    path = ROOT / "data/pretrain_source.npz"
    with np.load(path, allow_pickle=False) as source:
        observations = np.asarray(source["observations"], dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != 4 or len(observations) == 0:
        raise ValueError("pretrain_source.npz must contain nonempty four-coordinate observations")
    indices = np.linspace(0, len(observations) - 1, min(count, len(observations)), dtype=np.int64)
    raw = observations[indices]
    routes = (np.arange(len(raw), dtype=np.int64) % 6) + 1
    return np.ascontiguousarray(
        np.stack([augment(obs, int(route)) for obs, route in zip(raw, routes)]),
        dtype=np.float32,
    )


def _probability_probe(
    reference: TensorSourceBasisPolicy,
    candidate: TensorSourceBasisPolicy,
    observations: np.ndarray,
) -> dict[str, float | int]:
    maximum_difference = 0.0
    kl_total = 0.0
    batch = torch.as_tensor(observations, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(batch), 128):
            part = batch[start:start + 128]
            p = reference.get_distribution(part).distribution.probs
            q = candidate.get_distribution(part).distribution.probs
            maximum_difference = max(maximum_difference, float((p - q).abs().max()))
            kl_total += float((p * ((p + 1e-12).log() - (q + 1e-12).log())).sum())
    return {
        "offline_observation_count": int(len(batch)),
        "max_probability_difference": maximum_difference,
        "mean_kl_reference_to_candidate": kl_total / max(1, len(batch)),
    }


def _reference_evaluation(
    policy: TensorSourceBasisPolicy,
    seed: int,
    episodes: int,
) -> tuple[dict[str, list[dict[str, int | float]]], list[dict[str, object]], int, dict[str, int | float]]:
    cached: dict[str, list[dict[str, int | float]]] = {}
    rows: list[dict[str, object]] = []
    interactions = 0
    process = psutil.Process()
    rss_start = int(process.memory_info().rss)
    rss_peak = rss_start
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for target in CONTEXTS:
        values = []
        for episode_number in range(episodes):
            episode_seed = _pair_seed(seed, target, episode_number)
            metric = _episode(policy, target, episode_seed)
            interactions += int(metric["length"])
            rss_peak = max(rss_peak, int(process.memory_info().rss))
            values.append(metric)
            rows.append({
                "phase": "reference",
                "source_seed": seed,
                "budget_bytes": "",
                "method": "dense_f32_reference",
                "candidate_id": f"s{seed}:reference",
                "target": target,
                "episode": episode_number + 1,
                "pair_id": f"s{seed}:{target}:{episode_number + 1}",
                "episode_seed": episode_seed,
                "return": metric["return"],
                "success": metric["success"],
                "length": metric["length"],
                "reference_return": metric["return"],
                "reference_success": metric["success"],
                "reference_length": metric["length"],
            })
        cached[target] = values
    runtime = {
        "wall_elapsed_s": time.perf_counter() - wall_start,
        "process_cpu_elapsed_s": time.process_time() - cpu_start,
        "rss_start_bytes": rss_start,
        "rss_peak_observed_bytes": rss_peak,
        "rss_peak_delta_bytes": max(0, rss_peak - rss_start),
    }
    return cached, rows, interactions, runtime


def _run_candidate(
    *,
    phase: str,
    output: Path,
    seed: int,
    budget: int,
    method: str,
    episodes: int,
    policy: TensorSourceBasisPolicy,
    original_matrix: np.ndarray,
    probes: np.ndarray,
    references: dict[str, list[dict[str, int | float]]],
) -> tuple[dict[str, object], list[dict[str, object]], int]:
    candidate_id = f"s{seed}:b{budget}:{method}"
    candidate_dir = output / f"s{seed}" / f"budget_{budget:04d}" / method
    candidate_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    rss_start = int(process.memory_info().rss)
    rss_peak = rss_start
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    rows: list[dict[str, object]] = []
    interaction_count = 0
    minimum_size = minimum_nbytes(original_matrix, method)
    try:
        encode_start = time.perf_counter()
        compact = encode(original_matrix, method, budget)
        encode_elapsed = time.perf_counter() - encode_start
        decoded = compact.decode()
        artifact = compact.save(candidate_dir / "knowledge.npz")
        binary_artifact = compact.save_binary(candidate_dir / "knowledge.ckb")
        if binary_artifact.stat().st_size != compact.nbytes:
            raise RuntimeError("binary artifact does not match charged codec bytes")
        candidate = _install_decoded(policy, decoded)
        representation_error = original_matrix - decoded
        weight_rmse = float(np.sqrt(np.mean(np.square(representation_error, dtype=np.float64))))
        weight_max_abs = float(np.max(np.abs(representation_error))) if representation_error.size else 0.0
        probe = _probability_probe(policy, candidate, probes)
        evaluation_start = time.perf_counter()
        inference_elapsed = 0.0
        for target in CONTEXTS:
            for episode_number in range(episodes):
                episode_seed = _pair_seed(seed, target, episode_number)
                inference_start = time.perf_counter()
                metric = _episode(candidate, target, episode_seed)
                inference_elapsed += time.perf_counter() - inference_start
                interaction_count += int(metric["length"])
                rss_peak = max(rss_peak, int(process.memory_info().rss))
                reference_metric = references[target][episode_number]
                rows.append({
                    "phase": phase,
                    "source_seed": seed,
                    "budget_bytes": budget,
                    "method": method,
                    "candidate_id": candidate_id,
                    "target": target,
                    "episode": episode_number + 1,
                    "pair_id": f"s{seed}:{target}:{episode_number + 1}",
                    "episode_seed": episode_seed,
                    "return": metric["return"],
                    "success": metric["success"],
                    "length": metric["length"],
                    "reference_return": reference_metric["return"],
                    "reference_success": reference_metric["success"],
                    "reference_length": reference_metric["length"],
                })
        evaluation_elapsed = time.perf_counter() - evaluation_start
        rss_peak = max(rss_peak, int(process.memory_info().rss))
        candidate_success_rate = float(np.mean([row["success"] for row in rows]))
        candidate_mean_return = float(np.mean([row["return"] for row in rows]))
        reference_success_rate = float(np.mean([
            metric["success"] for target_metrics in references.values() for metric in target_metrics
        ]))
        reference_mean_return = float(np.mean([
            metric["return"] for target_metrics in references.values() for metric in target_metrics
        ]))
        success_loss_pp = (reference_success_rate - candidate_success_rate) * 100.0
        return_loss = reference_mean_return - candidate_mean_return
        result: dict[str, object] = {
            "status": "complete",
            "phase": phase,
            "source_seed": seed,
            "budget_bytes": budget,
            "method": method,
            "candidate_id": candidate_id,
            "parameter_selection": compact.parameter_summary,
            "codec_minimum_nbytes_including_header": minimum_size,
            "codec_nbytes": compact.nbytes,
            "python_container_object_overhead_bytes": _python_object_overhead_bytes(compact),
            "python_container_plus_codec_bytes": _python_object_overhead_bytes(compact) + compact.nbytes,
            "artifact_path": str(artifact.relative_to(ROOT)),
            "artifact_file_bytes": int(artifact.stat().st_size),
            "binary_artifact_path": str(binary_artifact.relative_to(ROOT)),
            "binary_artifact_file_bytes": int(binary_artifact.stat().st_size),
            "binary_artifact_exact_size_match": int(binary_artifact.stat().st_size) == compact.nbytes,
            "budget_satisfied": compact.nbytes <= budget,
            "raw_effective_dense_tensor_nbytes": int(original_matrix.nbytes),
            "dense_effective_codec_minimum_nbytes": minimum_nbytes(original_matrix, "dense_f32"),
            "common_frozen_policy_tensor_nbytes": _policy_storage_bytes(policy)["common_frozen_policy_tensor_nbytes"],
            "original_hierarchical_target_coefficients_tensor_nbytes": _policy_storage_bytes(policy)["original_hierarchical_target_coefficients_tensor_nbytes"],
            "common_plus_codec_tensor_estimate_nbytes": _policy_storage_bytes(policy)["common_frozen_policy_tensor_nbytes"] + compact.nbytes,
            "common_plus_raw_effective_dense_tensor_nbytes": _policy_storage_bytes(policy)["common_frozen_policy_tensor_nbytes"] + int(original_matrix.nbytes),
            "candidate_runtime_policy_tensor_nbytes": _policy_storage_bytes(candidate)["full_checkpoint_policy_tensor_nbytes"],
            "decoded_temporary_matrix_workspace_nbytes": int(decoded.nbytes),
            "weight_rmse": weight_rmse,
            "weight_max_abs_error": weight_max_abs,
            "offline_probability_probe": probe,
            "episodes_per_target": episodes,
            "target_contexts": list(CONTEXTS),
            "candidate_interactions": interaction_count,
            "candidate_success_rate": candidate_success_rate,
            "reference_success_rate": reference_success_rate,
            "candidate_mean_return": candidate_mean_return,
            "reference_mean_return": reference_mean_return,
            "success_loss_vs_dense_reference_pp": success_loss_pp,
            "return_loss_vs_dense_reference": return_loss,
            "fixed_performance_gate_pass": success_loss_pp <= 5.0 and return_loss <= 0.25,
            "encode_elapsed_s": encode_elapsed,
            "evaluation_elapsed_s": evaluation_elapsed,
            "policy_episode_elapsed_s": inference_elapsed,
            "total_wall_elapsed_s": time.perf_counter() - wall_start,
            "process_cpu_elapsed_s": time.process_time() - cpu_start,
            "rss_start_bytes": rss_start,
            "rss_peak_observed_bytes": rss_peak,
            "rss_peak_delta_bytes": max(0, rss_peak - rss_start),
        }
        del candidate
    except ValueError as error:
        result = {
            "status": "budget_impossible",
            "phase": phase,
            "source_seed": seed,
            "budget_bytes": budget,
            "method": method,
            "candidate_id": candidate_id,
            "error": str(error),
            "codec_minimum_nbytes_including_header": minimum_size,
            "raw_effective_dense_tensor_nbytes": int(original_matrix.nbytes),
            "dense_effective_codec_minimum_nbytes": minimum_nbytes(original_matrix, "dense_f32"),
            "candidate_interactions": 0,
            "total_wall_elapsed_s": time.perf_counter() - wall_start,
            "process_cpu_elapsed_s": time.process_time() - cpu_start,
            "rss_start_bytes": rss_start,
            "rss_peak_observed_bytes": int(process.memory_info().rss),
        }
    result_path = candidate_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, rows, interaction_count


def run(
    phase: str,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    episodes: int,
    methods: tuple[str, ...],
    output: Path | None = None,
) -> dict[str, object]:
    if phase not in OUTPUTS:
        raise ValueError("phase must be pilot, formal, or global_control")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not seeds or not budgets or not methods:
        raise ValueError("seeds, budgets, and methods must be nonempty")
    if any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be selected from {list(METHODS)}")
    torch.set_num_threads(1)
    output = output or ROOT / OUTPUTS[phase]
    output.mkdir(parents=True, exist_ok=True)
    foundation_path = ROOT / "data/foundation.pt"
    probe_path = ROOT / "data/pretrain_source.npz"
    probes = _probe_observations()
    process = psutil.Process()
    overall_start = time.perf_counter()
    overall_cpu_start = time.process_time()
    total_interactions = 0
    episode_rows: list[dict[str, object]] = []
    candidate_results: list[dict[str, object]] = []
    source_records = []
    for seed in seeds:
        checkpoint = ROOT / _checkpoint_path(seed)
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing hierarchical checkpoint: {checkpoint}")
        reference_policy = _restore_checkpoint(checkpoint)
        original_matrix = _effective_matrix(reference_policy)
        reference_actor_hash = hashlib.sha256(
            b"".join(
                tensor.detach().cpu().contiguous().numpy().tobytes()
                for name, tensor in sorted(reference_policy.state_dict().items())
                if name.startswith("mlp_extractor.policy_net.") or name.startswith("action_net.")
            )
        ).hexdigest()
        references, reference_rows, reference_interactions, reference_runtime = _reference_evaluation(
            reference_policy, seed, episodes
        )
        total_interactions += reference_interactions
        episode_rows.extend(reference_rows)
        source_records.append({
            "source_seed": seed,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": _sha256(checkpoint),
            "foundation_sha256": _sha256(foundation_path),
            "source_dataset_sha256": _sha256(probe_path),
            "actor_hash": reference_actor_hash,
            "temperature": float(reference_policy.temperature),
            "exploration_epsilon": float(reference_policy.epsilon),
            "effective_matrix_shape": list(original_matrix.shape),
            "effective_matrix_sha256": hashlib.sha256(original_matrix.tobytes()).hexdigest(),
            **_policy_storage_bytes(reference_policy),
            "raw_effective_dense_tensor_nbytes": int(original_matrix.nbytes),
            "codec_minimum_nbytes_including_header": {
                method: minimum_nbytes(original_matrix, method) for method in METHODS
            },
            "reference_interactions": reference_interactions,
            "reference_episodes_per_target": episodes,
            "reference_runtime": reference_runtime,
        })
        for budget in budgets:
            for method in methods:
                result, rows, interactions = _run_candidate(
                    phase=phase,
                    output=output,
                    seed=seed,
                    budget=budget,
                    method=method,
                    episodes=episodes,
                    policy=reference_policy,
                    original_matrix=original_matrix,
                    probes=probes,
                    references=references,
                )
                candidate_results.append(result)
                episode_rows.extend(rows)
                total_interactions += interactions
                print(json.dumps({
                    "candidate": result["candidate_id"],
                    "status": result["status"],
                    "interactions": result.get("candidate_interactions", 0),
                }), flush=True)
        del reference_policy

    _add_independent_comparisons(candidate_results, output)

    episode_path = output / "episodes.csv"
    with episode_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(episode_rows)
    summary = {
        "phase": phase,
        "frozen_protocol_commit": "66ce6ad" if phase == "global_control" else None,
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "codec_sha256": _sha256(ROOT / "src/online_rl/compact_knowledge.py"),
        "scope": "Static deployment codec diagnostic on copied checkpoint target heads.",
        "routing": "Oracle isolated-context evaluation; context names are evaluator-only and are not codec inputs.",
        "head_accounting": "Each effective leaf row is A_i = leaf.A + its trained family.C; decoded rows are copied into the leaf head and family coefficients are zeroed for leaf-route evaluation.",
        "trainable_continuation": "Not evaluated; optimizer, trainable policy state, and Adam continuation are excluded from codec budgets.",
        "codec_methods": list(methods),
        "budgets_bytes": list(budgets),
        "source_records": source_records,
        "episodes_per_target_candidate": episodes,
        "targets": list(CONTEXTS),
        "candidate_results": candidate_results,
        "comparison_gates": {
            "dense_reference_average_success_loss_at_most_pp": 5.0,
            "dense_reference_average_return_loss_at_most": 0.25,
            "shared_methods_must_strictly_exceed_the_best_under_budget_independent_candidate_on_both_metrics_to_pass_sharing_gate": True,
            "independent_methods": ["dense_f32", "sparse_topk", "global_topk", "fp16", "int8"],
            "shared_methods": ["global_mean_sparse", "centroid2_sparse", "truncated_svd"],
        },
        "episode_rows_path": str(episode_path.relative_to(ROOT)),
        "actual_interactions": total_interactions,
        "total_wall_elapsed_s": time.perf_counter() - overall_start,
        "process_cpu_elapsed_s": time.process_time() - overall_cpu_start,
        "rss_end_bytes": int(process.memory_info().rss),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _add_independent_comparisons(results: list[dict[str, object]], output: Path) -> None:
    independent_methods = {"dense_f32", "sparse_topk", "global_topk", "fp16", "int8"}
    shared_methods = {"global_mean_sparse", "centroid2_sparse", "truncated_svd"}
    groups: dict[tuple[int, int], list[dict[str, object]]] = {}
    for result in results:
        groups.setdefault((int(result["source_seed"]), int(result["budget_bytes"])), []).append(result)
    for (seed, budget), candidates in groups.items():
        independent = [
            candidate for candidate in candidates
            if candidate.get("status") == "complete"
            and candidate.get("budget_satisfied") is True
            and candidate.get("method") in independent_methods
        ]
        if not independent:
            continue
        best_success = max(float(candidate["candidate_success_rate"]) for candidate in independent)
        best_return = max(float(candidate["candidate_mean_return"]) for candidate in independent)
        best_success_method = next(
            str(candidate["method"]) for candidate in independent
            if float(candidate["candidate_success_rate"]) == best_success
        )
        best_return_method = next(
            str(candidate["method"]) for candidate in independent
            if float(candidate["candidate_mean_return"]) == best_return
        )
        for candidate in candidates:
            if candidate.get("status") != "complete" or candidate.get("method") not in shared_methods:
                continue
            success_delta_pp = (float(candidate["candidate_success_rate"]) - best_success) * 100.0
            return_delta = float(candidate["candidate_mean_return"]) - best_return
            candidate["best_independent_under_budget"] = {
                "success_method": best_success_method,
                "success_rate": best_success,
                "return_method": best_return_method,
                "mean_return": best_return,
            }
            candidate["success_delta_vs_best_independent_pp"] = success_delta_pp
            candidate["return_delta_vs_best_independent"] = return_delta
            candidate["beats_best_independent_on_both_metrics"] = success_delta_pp > 0.0 and return_delta > 0.0
            candidate["sharing_claim_gate_pass"] = (
                bool(candidate["fixed_performance_gate_pass"])
                and bool(candidate["beats_best_independent_on_both_metrics"])
            )
            result_path = (
                output / f"s{seed}" / f"budget_{budget:04d}" /
                str(candidate["method"]) / "result.json"
            )
            result_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")


def audit_existing_formal(output: Path | None = None) -> dict[str, object]:
    """Materialize exact binary artifacts for the completed seven-codec formal.

    This post-hoc audit reads the existing NPZ outputs and summary, performs no
    environment interaction, and writes all new files under a separate audit
    directory without editing the original numeric summary or episode ledger.
    """
    source_root = ROOT / OUTPUTS["formal"]
    source_summary_path = source_root / "summary.json"
    source_episodes_path = source_root / "episodes.csv"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    target_root = output or source_root / "serialization_audit"
    target_root.mkdir(parents=True, exist_ok=True)
    records = []
    for result in source_summary["candidate_results"]:
        if result.get("status") != "complete":
            continue
        source_artifact = ROOT / result["artifact_path"]
        compact = CompactKnowledge.load(source_artifact)
        expected_bytes = int(result["codec_nbytes"])
        if compact.nbytes != expected_bytes:
            raise RuntimeError(
                f"post-hoc representation size changed for {result['candidate_id']}: "
                f"{compact.nbytes} != {expected_bytes}"
            )
        relative = Path(f"s{result['source_seed']}") / f"budget_{int(result['budget_bytes']):04d}" / str(result["method"]) / "knowledge.ckb"
        binary_path = compact.save_binary(target_root / "files" / relative)
        records.append({
            "candidate_id": result["candidate_id"],
            "source_npz_path": result["artifact_path"],
            "source_npz_file_bytes": int(source_artifact.stat().st_size),
            "binary_path": str(binary_path.relative_to(ROOT)),
            "binary_file_bytes": int(binary_path.stat().st_size),
            "charged_codec_nbytes": compact.nbytes,
            "exact_size_match": int(binary_path.stat().st_size) == compact.nbytes,
            "decode_matrix_sha256": hashlib.sha256(compact.decode().tobytes()).hexdigest(),
        })
    audit = {
        "posthoc_serialization_audit": True,
        "original_numerical_summary_preserved": True,
        "original_numerical_summary_sha256": _sha256(source_summary_path),
        "original_episode_ledger_sha256": _sha256(source_episodes_path),
        "new_environment_interactions": 0,
        "candidate_count": len(records),
        "all_binary_sizes_match_accounted_nbytes": all(row["exact_size_match"] for row in records),
        "records": records,
    }
    (target_root / "summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _parse_ints(values: list[str] | None, default: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(value) for value in values) if values else default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "formal", "global_control", "formal_serialization_audit"), required=True)
    parser.add_argument("--seeds", nargs="+", help="checkpoint seeds; defaults are phase-specific")
    parser.add_argument("--budgets", nargs="+", help="per-target-head codec budgets in bytes")
    parser.add_argument("--episodes", type=int, help="full episodes per target and candidate")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--output", type=Path, help="override the phase output directory")
    args = parser.parse_args()
    if args.phase == "formal_serialization_audit":
        audit = audit_existing_formal(args.output)
        print(json.dumps({
            "candidate_count": audit["candidate_count"],
            "all_binary_sizes_match_accounted_nbytes": audit["all_binary_sizes_match_accounted_nbytes"],
            "new_environment_interactions": 0,
        }), flush=True)
        return
    seeds_default = (181,) if args.phase == "pilot" else CHECKPOINT_SEEDS
    budgets_default = (192, 448) if args.phase == "pilot" else FORMAL_BUDGETS
    episodes_default = 4 if args.phase == "pilot" else 32
    methods_default = (
        ORIGINAL_METHODS if args.phase in ("pilot", "formal") else ("global_topk",)
    )
    seeds = _parse_ints(args.seeds, seeds_default)
    budgets = _parse_ints(args.budgets, budgets_default)
    episodes = args.episodes if args.episodes is not None else episodes_default
    methods = tuple(args.methods) if args.methods else methods_default
    run(args.phase, seeds, budgets, episodes, methods, args.output)


if __name__ == "__main__":
    main()
