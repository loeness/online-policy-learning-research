"""Plot existing-log diagnostics only; no environment calls or training."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
data = json.loads((folder / "diagnostics.json").read_text(encoding="utf-8"))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
colors = ["#2563eb", "#d97706", "#059669", "#9333ea"]
for context, color in zip(("cw", "ccw", "cw_slip", "ccw_slip"), colors):
    values = []
    for phase in ("before", "after"):
        rows = [r for r in data["repair"]["paired_return_gaps"]
                if r["phase"] == phase and r["context"] == context]
        values.append(statistics.mean(r["return_drop_percent"] for r in rows))
    for seed in (41, 53, 67):
        pair = [next(r["return_drop_percent"] for r in data["repair"]["paired_return_gaps"]
                     if r["phase"] == phase and r["context"] == context and r["seed"] == seed)
                for phase in ("before", "after")]
        axes[0].plot([0, 1], pair, color=color, alpha=.18, linewidth=1)
    axes[0].plot([0, 1], values, color=color, marker="o", linewidth=2.1,
                 label=context.replace("_", " + "))
axes[0].axhline(5, color="#64748b", linestyle="--", linewidth=1, label="5% empirical screen")
axes[0].set(title="A. Online repair closes the return gap",
            ylabel="Sparse return shortfall vs. dense (%)")
axes[0].set_xticks([0, 1], ["Before repair", "After 8,192 learning steps"])
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
axes[0].grid(axis="y", alpha=.18)

metrics = [("greedy_action_agreement", "Same greedy action as dense", "#2563eb"),
           ("sparse_mean_shortest_action_probability", "Probability on shortest-path actions", "#d97706")]
for key, label, color in metrics:
    for seed in (41, 53, 67):
        pair = [100 * next(r[key] for r in data["static_policy_probes"]
                           if r["phase"] == phase and r["seed"] == seed)
                for phase in ("before", "after")]
        axes[1].plot([0, 1], pair, color=color, alpha=.2, linewidth=1)
    means = [100 * statistics.mean(r[key] for r in data["static_policy_probes"]
                                  if r["phase"] == phase)
             for phase in ("before", "after")]
    axes[1].plot([0, 1], means, color=color, marker="o", linewidth=2.2, label=label)
axes[1].set(title="B. Restored ability uses different actions", ylabel="Uniform-grid proportion (%)", ylim=(70, 101))
axes[1].set_xticks([0, 1], ["Before repair", "After repair"])
axes[1].legend(frameon=False, fontsize=8, loc="lower left")
axes[1].grid(axis="y", alpha=.18)
fig.text(.5, .015, "Thin lines: 3 checkpoints sharing one pretrained foundation. Panel B: static forward passes, no interactions.",
         ha="center", fontsize=8, color="#475569")
fig.tight_layout(rect=(0, .04, 1, 1))
fig.savefig(folder / "mechanism_diagnostics.png", dpi=220)
fig.savefig(folder / "mechanism_diagnostics.pdf")
plt.close(fig)
print(folder / "mechanism_diagnostics.png")
