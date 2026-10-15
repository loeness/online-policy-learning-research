"""Describe the preregistered P0 transfer diagnostic; never starts training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("A0", "A1", "A2", "B0", "B1", "B2")


def cosine_matrix(vectors):
    values = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    denominator = norms[:, None] * norms[None, :]
    return np.divide(values @ values.T, denominator,
                     out=np.full((len(values), len(values)), np.nan), where=denominator > 0)


def correlation(left, right):
    good = np.isfinite(left) & np.isfinite(right)
    if good.sum() < 2 or np.std(left[good]) == 0 or np.std(right[good]) == 0:
        return None
    return float(np.corrcoef(left[good], right[good])[0, 1])


def heatmap(fig, ax, values, title, limit, fmt):
    artist = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit)
    ax.set(xticks=range(6), yticks=range(6), xticklabels=NAMES, yticklabels=NAMES,
           xlabel="Target context", ylabel="Source update", title=title)
    for row in range(6):
        for col in range(6):
            value = values[row, col]
            ax.text(col, row, format(value, fmt), ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > limit * .65 else "black")
    fig.colorbar(artist, ax=ax, fraction=.046, pad=.04)


def report(root: Path):
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary["smoke_test"] or not summary["formal_defaults"]:
        raise ValueError("The formal report requires all preregistered P0 defaults.")
    if len(summary["training"]) != 18 or len(summary["transfer_summary"]["rows"]) != 108:
        raise ValueError("Incomplete formal inventory.")
    offdiag = ~np.eye(6, dtype=bool)
    same_family = np.array([[a[0] == b[0] for b in NAMES] for a in NAMES]) & offdiag
    other_family = ~same_family & offdiag
    rows, transfer_matrices, gradient_matrices, update_matrices = [], [], [], []
    scatter = []
    per_seed = {}
    for seed in summary["seeds"]:
        branches = {row["source_context"]: row for row in summary["training"] if row["seed"] == seed}
        transfers = {(row["source"], row["target"]): row for row in summary["transfer_summary"]["rows"]
                     if row["seed"] == seed}
        transfer = np.array([[transfers[a, b]["mean_paired_return_gain"] for b in NAMES] for a in NAMES])
        gradient = cosine_matrix([branches[a]["first_rollout_actor_gradient"]["gradient_vector"] for a in NAMES])
        update = cosine_matrix([branches[a]["A_update_vector"] for a in NAMES])
        correlations = {"gradient": correlation(gradient[offdiag], transfer[offdiag]),
                        "update": correlation(update[offdiag], transfer[offdiag])}
        observed_conflicts = {}
        for name, matrix in (("gradient", gradient), ("update", update)):
            positive_cos = offdiag & np.isfinite(matrix) & (matrix > 0)
            observed_conflicts[name] = {
                "positive_cos_cells": int(positive_cos.sum()),
                "positive_cos_negative_gain_cells": int((positive_cos & (transfer < 0)).sum()),
            }
        row = {
            "seed": seed,
            "self_positive": int((np.diag(transfer) > 0).sum()),
            "mean_self_gain": float(np.diag(transfer).mean()),
            "mean_same_family_gain": float(transfer[same_family].mean()),
            "mean_other_family_gain": float(transfer[other_family].mean()),
            "offdiag_positive": int((transfer[offdiag] > 0).sum()),
            "offdiag_negative": int((transfer[offdiag] < 0).sum()),
            "correlations": correlations,
            "positive_cos_negative_observed_gain": observed_conflicts,
        }
        rows.append(row)
        per_seed[str(seed)] = {"transfer": transfer.tolist(), "gradient_cosine": gradient.tolist(),
                               "update_cosine": update.tolist(), **row}
        transfer_matrices.append(transfer)
        gradient_matrices.append(gradient)
        update_matrices.append(update)
        scatter.append((gradient[offdiag], transfer[offdiag], same_family[offdiag]))

    mean_transfer = np.mean(transfer_matrices, axis=0)
    result = {
        "scope": "P0 descriptive association, one W0; no task-free discovery, capacity savings, or significance claim",
        "names": NAMES, "per_seed": per_seed,
        "mean_transfer": mean_transfer.tolist(),
        "self_positive_runs": sum(row["self_positive"] for row in rows),
        "self_total_runs": 18,
        "mean_self_gain": float(np.mean([row["mean_self_gain"] for row in rows])),
        "mean_same_family_gain": float(np.mean([row["mean_same_family_gain"] for row in rows])),
        "mean_other_family_gain": float(np.mean([row["mean_other_family_gain"] for row in rows])),
        "offdiag_positive_cells": sum(row["offdiag_positive"] for row in rows),
        "offdiag_negative_cells": sum(row["offdiag_negative"] for row in rows),
        "summary_sha256": hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
    }
    (root / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.5), constrained_layout=True)
    heatmap(fig, axes[0, 0], mean_transfer, "Observed return gain (3-seed mean)",
            max(float(np.max(np.abs(mean_transfer))), .01), ".1f")
    heatmap(fig, axes[0, 1], np.nanmean(gradient_matrices, axis=0), "First-rollout gradient cosine", 1, ".2f")
    heatmap(fig, axes[1, 0], np.nanmean(update_matrices, axis=0), "Final correction cosine", 1, ".2f")
    ax = axes[1, 1]
    xs = np.concatenate([s[0] for s in scatter])
    ys = np.concatenate([s[1] for s in scatter])
    families = np.concatenate([s[2] for s in scatter])
    for mask, color, label in ((families, "#2878a2", "Same family"), (~families, "#d17b35", "Other family")):
        ax.scatter(xs[mask], ys[mask], s=25, color=color, alpha=.75, label=label)
    ax.axhline(0, color="gray", lw=.8)
    ax.axvline(0, color="gray", lw=.8)
    ax.set(xlabel="First-rollout gradient cosine", ylabel="Observed paired episode-return gain",
           title="Directed off-diagonal transfers (dependent cells)", xlim=(-1.05, 1.05))
    ax.legend(frameon=False)
    fig.suptitle("P0: independent local learning, shared correction coordinates\n"
                 "Privileged context-isolated diagnostic; not automatic organization", fontsize=13)
    fig.savefig(root / "transfer_diagnostic.png", dpi=180)
    fig.savefig(root / "transfer_diagnostic.pdf")
    plt.close(fig)
    table = ["| 部署种子 | 自身改善 | 自身回报增益 | 同家族迁移 | 跨家族迁移 | 梯度相关 | 修正相关 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        fmt = lambda x: "未定义" if x is None else f"{x:.3f}"
        table.append(f"| {row['seed']} | {row['self_positive']}/6 | {row['mean_self_gain']:.2f} | "
                     f"{row['mean_same_family_gain']:.2f} | {row['mean_other_family_gain']:.2f} | "
                     f"{fmt(row['correlations']['gradient'])} | {fmt(row['correlations']['update'])} |")
    text = "# P0：局部在线修正与跨情境迁移诊断\n\n"
    text += "这是外部实验者隔离情境的机制诊断。统一 route=1，算法没有发现情境或共享结构。全部运行共享一个 W0；源输出基适合本动作置换任务。基线与候选采用相同温度/探索设置。\n\n"
    text += "## 固定规程与实际开销\n\n"
    text += f"3 部署种子×6 独立训练×4,096 PPO 步；每个目标配对32个完整回合。训练 {summary['training_interactions']:,} 步，基线评估 {summary['baseline_evaluation_interactions']:,} 步，候选评估 {summary['candidate_evaluation_interactions']:,} 步，总计 {summary['actual_total_interactions']:,} 实际仿真步。运行墙钟 {summary['elapsed_wall_s']:.2f} 秒，进程 CPU {summary['elapsed_process_cpu_s']:.2f} 秒（不含 Python 导入）。\n\n"
    text += "## 观测结果\n\n" + "\n".join(table) + "\n\n"
    text += f"自身改善 {result['self_positive_runs']}/18；90个非对角单元中，正观测迁移 {result['offdiag_positive_cells']}，负观测迁移 {result['offdiag_negative_cells']}。同家族/跨家族是分析者事后按预声明标签汇总，不进入训练。梯度/修正相关是每个种子30个依赖单元的 Pearson 描述性相关，不提供独立样本显著性。\n\n"
    text += "### 正 cosine 仍对应负观测迁移的单元\n\n"
    for row in rows:
        g = row["positive_cos_negative_observed_gain"]["gradient"]
        u = row["positive_cos_negative_observed_gain"]["update"]
        text += f"- seed {row['seed']}：梯度 {g['positive_cos_negative_gain_cells']}/{g['positive_cos_cells']}；最终修正 {u['positive_cos_negative_gain_cells']}/{u['positive_cos_cells']}。\n"
    text += "\n这些是有限 PPO 更新下的观测关联；不能将观测负差直接当作经校准的伤害概率，也不能用它否定无穷小步长的一阶梯度关系。交叉评估是研究者的诊断交互，部署算法不可免费调用。\n\n"
    text += "![迁移与相似性](transfer_diagnostic.png)\n\n"
    text += "## 数据与限制\n\nsummary.json 为实际交互与训练账本；baseline_episodes.csv / transfer_episodes.csv 保存全部配对回合；branches/ 保存全部18个候选 checkpoint、梯度、参数与训练回合；analysis.json 保存逐种子矩阵。未验证真实压缩、满容量替换、无标签识别、独立 W0 或超出当前环境先验的泛化。\n"
    (root / "结果解读.md").write_text(text, encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_seed", "mean_transfer")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/pre_core_p0")
    report(parser.parse_args().root)
