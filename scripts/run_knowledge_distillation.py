"""Declared post-hoc policy-loss control for the static low-rank diagnostic."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import psutil
import torch

from online_rl.compact_knowledge import CompactKnowledge, encode
from online_rl.c1_experiment import augment
from run_static_consolidation import (
    CONTEXTS, _checkpoint_path, _effective_matrix, _episode,
    _install_decoded, _pair_seed, _restore_checkpoint,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_student(teacher, matrix, budget, folder):
    """Same factors/bytes as SVD; optimize actions on a fixed source support."""
    started = time.perf_counter()
    compact = encode(matrix, "truncated_svd", budget)
    with np.load(ROOT / "data/pretrain_source.npz", allow_pickle=False) as data:
        source = data["observations"]
        raw = source[np.linspace(0, len(source) - 1, 256, dtype=np.int64)]
    obs = torch.as_tensor(np.stack([augment(o, route) for route in range(1, 7) for o in raw]))
    routes = obs[:, -1].long() - 1
    bands = torch.floor(obs[:, 0] * 3).long().clamp(0, 2)
    with torch.no_grad():
        latent = teacher.mlp_extractor.forward_actor(teacher.extract_features(obs))
        phi, _ = teacher._feature_banks(obs, latent)
        psi = phi @ teacher.source_basis.T
        base_logits = teacher.action_net(latent)
        contrast = teacher._contrast_basis.detach().clone()
        targets = teacher.get_distribution(obs).distribution.probs.detach().clone()

    def probabilities(a):
        coefficients = a.reshape(6, 3, 3, 3)[routes, bands]
        correction = torch.bmm(coefficients, psi.unsqueeze(-1)).squeeze(-1) @ contrast.T
        p = torch.softmax((base_logits + correction) / teacher.temperature, dim=-1)
        return (1 - teacher.epsilon) * p + teacher.epsilon / 4

    # The cached-forward implementation must match the original teacher.
    discrepancy = float((probabilities(torch.as_tensor(matrix)) - targets).abs().max())
    if discrepancy > 2e-6:
        raise RuntimeError(f"cached teacher probability mismatch: {discrepancy}")
    left = torch.nn.Parameter(torch.from_numpy(compact.arrays["left"].copy()))
    right = torch.nn.Parameter(torch.from_numpy(compact.arrays["right"].copy()))
    optimizer = torch.optim.Adam([left, right], lr=.01)
    losses = []
    for iteration in range(257):
        student = probabilities(left @ right)
        loss = (targets * (targets.log() - student.log())).sum(-1).mean()
        losses.append({"iteration": iteration, "teacher_student_kl": float(loss.detach())})
        if iteration == 256:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    result = CompactKnowledge("truncated_svd", matrix.shape, {
        "left": left.detach().numpy().copy(), "right": right.detach().numpy().copy(),
    })
    if result.nbytes != compact.nbytes or result.nbytes > budget:
        raise RuntimeError("distillation changed the representation budget")
    folder.mkdir(parents=True, exist_ok=True)
    result.save(folder / "knowledge.npz")
    binary = result.save_binary(folder / "knowledge.ckb")
    if binary.stat().st_size != result.nbytes:
        raise RuntimeError("physical encoding differs from capacity accounting")
    (folder / "fit_trace.json").write_text(json.dumps(losses, indent=2), encoding="utf-8")
    optimizer_bytes = sum(t.numel() * t.element_size() for state in optimizer.state.values()
                          for t in state.values() if torch.is_tensor(t))
    return result, {
        "initial_kl": losses[0]["teacher_student_kl"],
        "final_kl": losses[-1]["teacher_student_kl"],
        "updates": 256, "learning_rate": .01, "calibration_observations": 1536,
        "unique_source_observations": 256,
        "teacher_forward_equivalence_max_error": discrepancy,
        "temporary_optimizer_tensor_bytes": optimizer_bytes,
        "cached_tensor_bytes": sum(t.numel() * t.element_size()
                                   for t in (obs, routes, bands, latent, phi, psi, base_logits, contrast, targets)),
        "fit_wall_s": time.perf_counter() - started,
        "actor_bytes": result.nbytes,
        "physical_binary_bytes": binary.stat().st_size,
        "rank": int(left.shape[1]),
    }


def main():
    torch.set_num_threads(1)
    out = ROOT / "outputs/knowledge_distillation_control"
    if (out / "summary.json").exists():
        raise FileExistsError("Completed experiment exists; preserve it.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "source").mkdir(exist_ok=True)
    source_files = [Path(__file__), ROOT / "src/online_rl/compact_knowledge.py",
                    ROOT / "scripts/run_static_consolidation.py"]
    for path in source_files:
        shutil.copyfile(path, out / "source" / path.name)
    start = time.perf_counter()
    cpu = time.process_time()
    rows, summaries = [], []
    old_rows = list(csv.DictReader((ROOT / "outputs/knowledge_static_formal/episodes.csv").open(encoding="utf-8")))
    references = {(int(r["source_seed"]), r["target"], int(r["episode"])): r
                  for r in old_rows if r["phase"] == "reference"}
    process = psutil.Process()
    peak = process.memory_info().rss
    for seed in (181, 193, 211):
        teacher = _restore_checkpoint(ROOT / _checkpoint_path(seed))
        matrix = _effective_matrix(teacher)
        for budget in (192, 320, 448):
            folder = out / f"s{seed}" / f"budget_{budget}"
            student, detail = fit_student(teacher, matrix, budget, folder)
            candidate = _install_decoded(teacher, student.decode())
            cohort = []
            for target in CONTEXTS:
                for i in range(32):
                    episode_seed = _pair_seed(seed, target, i)
                    metric = _episode(candidate, target, episode_seed)
                    reference = references[(seed, target, i + 1)]
                    if int(reference["episode_seed"]) != episode_seed:
                        raise RuntimeError("reference plan mismatch")
                    row = {"source_seed": seed, "budget_bytes": budget, "method": "svd_policy_distillation",
                           "target": target, "episode": i + 1, "episode_seed": episode_seed, **metric,
                           "reference_return": float(reference["return"]),
                           "reference_success": int(reference["success"])}
                    cohort.append(row)
                    rows.append(row)
            success = float(np.mean([r["success"] for r in cohort]))
            reward = float(np.mean([r["return"] for r in cohort]))
            return_loss = float(np.mean([r["reference_return"] - r["return"] for r in cohort]))
            success_loss = float(np.mean([r["reference_success"] - r["success"] for r in cohort]))
            detail.update({"source_seed": seed, "budget_bytes": budget,
                           "candidate_success_rate": success, "candidate_mean_return": reward,
                           "return_loss_vs_dense_reference": return_loss,
                           "success_loss_vs_dense_reference_pp": 100 * success_loss,
                           "fixed_performance_gate_pass": success_loss <= .05 and return_loss <= .25,
                           "candidate_interactions": sum(r["length"] for r in cohort)})
            summaries.append(detail)
            (folder / "result.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
            peak = max(peak, process.memory_info().rss)
            print(json.dumps({"seed": seed, "budget": budget, "success": success, "return": reward}), flush=True)
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "scope": "Post-hoc declared same-rank policy-KL compression control; static oracle isolation.",
        "source_support": "256 fixed source-data states, crossed with six teacher routes; no new fit interactions.",
        "reference_reuse": "Dense reference episodes read from prior formal CSV; not re-executed or double charged.",
        "interactions": sum(r["length"] for r in rows), "episode_count": len(rows),
        "wall_s": time.perf_counter() - start, "cpu_s": time.process_time() - cpu,
        "peak_rss_bytes": peak, "results": summaries,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_files},
        "source_dataset_sha256": sha(ROOT / "data/pretrain_source.npz"),
        "reference_csv_sha256": sha(ROOT / "outputs/knowledge_static_formal/episodes.csv"),
        "protocol_sha256": sha(ROOT / "docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("interactions", "episode_count", "wall_s", "cpu_s")}), flush=True)


if __name__ == "__main__":
    main()
