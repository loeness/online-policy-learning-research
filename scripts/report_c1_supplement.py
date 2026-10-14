"""Report saved C1 supplemental evidence; no training data is fed back."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/c1_supplement"
OLD = ("A0", "A1", "B0", "B1")
ALL = ("A0", "A1", "A2", "B0", "B1", "B2")

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def read_run(path):
    row = read(path)
    audit = path.parent / "optimizer_step_audit.json"
    if audit.exists():
        row["corrected_optimizer_steps"] = read(audit)["corrected_optimizer_steps"]
    return row

def label(row):
    return row["method"] + (" + " + row["control"] if row["control"] != "none" else "")

def sample(row):
    seen, adapted, final = row["seen_eval"], row["adapted_eval"], row["final_eval"]
    return {
        "final_success": mean(final[n]["success_rate"] for n in ALL),
        "all_six_at_least_80pct": all(final[n]["success_rate"] >= .8 for n in ALL),
        "immediate_old_success": mean(adapted[n]["success_rate"] for n in OLD),
        "mean_old_drop_pp": 100*mean(max(0, seen[n]["success_rate"]-adapted[n]["success_rate"]) for n in OLD),
        "retained_old_tasks": sum(adapted[n]["success_rate"] >= seen[n]["success_rate"]-.05-1e-10 for n in OLD),
        "old_tasks": 4,
        "cold_reward_gain": mean(row["cold"][n]["adapted_path"]["online_cumulative_return"]-row["cold"][n]["online_cumulative_return"] for n in ("A2", "B2")),
        "oracle_training_reward": sum(v.get("online_reward", 0) for v in row["training"].values()),
        "hidden_reward": row["history"]["online_cumulative_return"],
        "hidden_success": row["history"]["success_rate"],
        "hidden_route_accuracy": row["history"]["route_exact_accuracy"],
        "cpu_s": row["online_learning_cpu_s"],
        "elapsed_s": row["online_learning_elapsed_s"],
        "adaptive_bytes": row["representation"]["adaptive_parameter_and_optimizer_bytes"],
        "complete_agent_bytes": row["max_complete_agent_bytes"],
        "optimizer_steps": row.get("corrected_optimizer_steps", row["actual_optimizer_steps"]),
        "training_interactions": row["training_interactions"],
        "evaluation_interactions": row["evaluation_interactions"],
    }

def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|"+"|".join("---" for _ in headers)+"|", *["| "+" | ".join(map(str,row))+" |" for row in rows]])

def summarize(rows, kind):
    groups = defaultdict(list)
    for r in rows:
        key = (label(r), r["tier"])
        groups[key].append(r)
    result = []
    for (method, tier), members in sorted(groups.items()):
        samples = [sample(r) for r in members]
        result.append({"method": method, "tier": tier, "n": len(members),
            "seeds": [r["seed"] for r in members], "samples": samples,
            "means": {k: mean(s[k] for s in samples) for k in samples[0]},
            "final_success_sd": stdev(s["final_success"] for s in samples) if len(samples)>1 else 0,
            "successful_runs": sum(s["all_six_at_least_80pct"] for s in samples),
            "retention": sum(s["retained_old_tasks"] for s in samples),
            "max_complete_agent_bytes": max(s["complete_agent_bytes"] for s in samples),
            "budget_classification": members[0]["budget_classification"]})
    return result

def plot_neural(groups):
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True, layout="constrained")
    colors = plt.get_cmap("tab10")
    for i, g in enumerate(groups):
        values = np.array([s["final_success"]*100 for s in g["samples"]])
        color = colors(i % 10)
        axes[0].scatter(values, np.full(len(values), i), color=color, s=30)
        axes[0].plot([values.mean()]*2, [i-.2, i+.2], color=color, lw=2)
        kib = g["means"]["adaptive_bytes"] / 1024
        axes[1].barh(i, kib, height=.55, color=color, alpha=.8)
        axes[1].text(kib+.6, i, f"{kib:.2f}", fontsize=8, va="center")
        axes[2].scatter(g["means"]["mean_old_drop_pp"], i, color=color, s=40)
    labels = [g["method"] + " / " + g["tier"] for g in groups]
    axes[0].set_yticks(range(len(groups)), labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set(title="Final oracle diagnostics: three seeds", xlabel="Six-context success (%)", xlim=(-2, 103))
    axes[1].set(title="Adaptive actor parameters + optimizer", xlabel="Retained tensor state (KiB)", xlim=(0, 72))
    axes[2].set(title="Immediate forgetting before revisit", xlabel="Mean positive drop (percentage points)", xlim=(-1, 28))
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
    fig.savefig(OUT / "neural_comparisons.png", dpi=200)
    fig.savefig(OUT / "neural_comparisons.pdf")
    plt.close(fig)

def plot_models(model_groups):
    fig, axes = plt.subplots(1, 3, figsize=(12,4),layout="constrained")
    labels=[]
    for i,(name,members) in enumerate(sorted(model_groups.items())):
        labels.append(name.replace("_","\n"))
        for ax,key in ((axes[0],"online_cumulative_return"),(axes[1],"complete_episode_success_rate")):
            values=[r[key]*(100 if key.endswith("rate") else 1) for r in members]
            ax.scatter([i]*len(values),values,s=28)
            ax.plot([i-.18,i+.18],[mean(values)]*2,lw=2)
        axes[2].scatter([i]*len(members),[sum(s["process_cpu_s"] for s in r["stage_metrics"].values()) for r in members],s=28)
    for ax in axes: ax.set_xticks(range(len(labels)),labels,fontsize=8); ax.grid(alpha=.2)
    axes[0].set(title="From W0, no context IDs",ylabel="Return over 63,488 interactions")
    axes[1].set(title="Actual online success",ylabel="Completed-episode success (%)",ylim=(-2,103))
    axes[2].set(title="Measured process CPU",ylabel="Seconds (instrumented)")
    fig.savefig(OUT / "model_comparisons.png",dpi=200)
    fig.savefig(OUT / "model_comparisons.pdf")
    plt.close(fig)

def main():
    formal_paths = list(OUT.glob("step[23]*/**/formal/**/result.json"))
    formal = [read_run(p) for p in formal_paths]
    validations = [read_run(p) for p in OUT.glob("step[23]*/**/validation/**/result.json")]
    if len(formal) != 39 or len(validations)!=24:
        raise RuntimeError(f"Incomplete neural suite: {len(formal)} formal, {len(validations)} validation")
    failures = [r for r in formal+validations if r["status"] != "complete"]
    good=[r for r in formal if r["status"]=="complete"]
    groups=summarize(good,"neural")
    models=read(OUT/"step1_models/formal_summary.json")["runs"]
    model_groups=defaultdict(list)
    for r in models: model_groups[r["run_kind"]].append(r)
    parity=read(OUT/"step2_grouping/equivalence/ppo_parity.json")
    diagnostic_tree=read(OUT/"step1_models/tree_diagnostic_summary.json")["runs"]
    diagnostic_model=read(OUT/"step1_models/model_diagnostic_summary.json")["runs"]
    selection_paths=list(OUT.glob("step[23]*/**/selection.json"))
    selections={p.parent.name:read(p) for p in selection_paths}
    model_ledger=read(OUT/"step1_models/summary.json")
    analysis={"formal_neural_runs":len(formal),"validation_neural_runs":len(validations),"failures":failures,"groups":groups,"selections":selections,"tree_group_equivalence":parity,
        "model_formal_runs":len(models),"model_diagnostics":diagnostic_model,"tree_diagnostics":diagnostic_tree,
        "neural_training_interactions":sum(r.get("training_interactions",0) for r in formal),
        "neural_evaluation_interactions":sum(r.get("evaluation_interactions",0) for r in formal),
        "neural_validation_training_interactions":sum(r.get("training_interactions",0) for r in validations),
        "neural_validation_evaluation_interactions":sum(r.get("evaluation_interactions",0) for r in validations),
        "model_formal_interactions":model_ledger["formal_interactions"],
        "model_validation_interactions":model_ledger["validation_interactions"],
        "frozen_tree_diagnostic_interactions":sum(r["online_steps"] for r in diagnostic_tree),
        "cleared_model_diagnostic_interactions":sum(r["online_steps"] for r in diagnostic_model),
        "equivalence_check_interactions":4096}
    if (OUT/"source_retention.json").exists():
        source_probe=read(OUT/"source_retention.json")
        analysis["source_probe_evaluation_interactions"]=source_probe["additional_evaluation_interactions"]
        analysis["main_source_probe"]=[r for r in source_probe["runs"] if r["cohort"]=="main"]
    (OUT/"analysis.json").write_text(json.dumps(analysis,indent=2,ensure_ascii=False),encoding="utf-8")
    with (OUT/"formal_neural_summary.csv").open("w",newline="",encoding="utf-8") as f:
        rows=[{"method":label(r),"tier":r["tier"],"seed":r["seed"],**sample(r)} for r in good]
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=["# C1 三项补齐与替代方案：结果报告", "", "本报告只汇总保存的实验。完整协议见 `docs/C1_SUPPLEMENT_PROTOCOL.md` 和第三项前的 `docs/C1_SUPPLEMENT_AMENDMENTS.md`。原 A/B/C1 结果未覆盖，本轮没有进入 C2。", "", "## 1. 无任务标签的状态相关在线动作模型", "", "同一 W0，三个新种子，每次 63,488 步连续流；任务编号和切换时刻只供观察者记日志。模型获知可见 x 分区、四方向、边界和动作排列假设。以下是实际在线轨迹，不能与神经方法的 oracle 静态测试直接排名。", ""]
    text.append(table(["方法","全流回报均值","实际成功率","CPU 秒均值","最大序列化状态 B"],[[name,f"{mean(r['online_cumulative_return'] for r in rs):.2f}",f"{100*mean(r['complete_episode_success_rate'] for r in rs):.2f}%",f"{mean(sum(s['process_cpu_s'] for s in r['stage_metrics'].values()) for r in rs):.2f}",max(r['max_serialized_agent_bytes'] for r in rs)] for name,rs in sorted(model_groups.items())]))
    text += ["", "统计模型每个事件都更新信念/计数；神经梯度为零不表示没有在线学习。窗口模型清空近期历史后不保留在线学习知识。", "", "### 清空近期历史后的匹配部署诊断", "", "树保留已经学习的残差与此前带标签校准的模型库；窗口模型保留 W0 和源先验，清空其全部在线证据。相同新流和采样种子，每次 6,144 步；先前轨迹由各方法自行生成，因此不声称历史信息完全相同。", ""]
    dg=defaultdict(list)
    for r in diagnostic_tree: dg["trained_tree_"+r["tier"]].append(r)
    for r in diagnostic_model: dg[r["run_kind"]].append(r)
    text.append(table(["控制","回报均值","实际成功率"],[[n,f"{mean(r['online_cumulative_return'] for r in rs):.2f}",f"{100*mean(r['complete_episode_success_rate'] for r in rs):.2f}%"] for n,rs in sorted(dg.items())]))
    text += ["", "## 2. 共享分解与树拓扑", "", "两档容量各进行成对 1,024 步 PPO：树与扁平分组的动作、观测、logits、最终参数和 Adam 状态完全相同（最大 logits 差为 0）。原 C1 的固定树等价于分组加法修正；这里不能提出树拓扑本身的独特性能贡献。另一个分组字典将全局共享基拆成两组，测试减少无关情境共享的效果。", "", "## 3. 独立调参与持续学习控制", "", "每个压缩方法在独立验证种子 911 上试三个设置，选定后在 131/149/167 上复验；高档选定配置原样应用低档。全参数方法另试三个学习率/熵设置，再分别试三个 EWC/回放系数。全参数方法超出原两档适应参数预算，是更大容量的参考。所有评价只反映本次有限搜索，不代表全局最优或 SOTA。", ""]
    text.append(table(["方法","档","最终六任务均值","六任务≥80%","立即旧任务成功率","相对 seen 保留","平均正向遗忘 pp","冷迁移回报增益"],[[g['method'],g['tier'],f"{100*g['means']['final_success']:.2f}%",f"{g['successful_runs']}/{g['n']}",f"{100*g['means']['immediate_old_success']:.2f}%",f"{g['retention']}/{4*g['n']}",f"{g['means']['mean_old_drop_pp']:.2f}",f"{g['means']['cold_reward_gain']:.2f}"] for g in groups]))
    text += ["", "### 实际隐藏路由续训与成本", "", "此流不向路由器提供编号或切换提示，但神经方法的六个原型来自其此前 oracle 阶段的真实带标签转移。其动态分支发现能力未被验证。", ""]
    text.append(table(["方法","档","隐藏流回报","隐藏流成功率","适应参数+优化器 B","完整状态峰值 B","实际优化器步数","在线 CPU 秒"],[[g['method'],g['tier'],f"{g['means']['hidden_reward']:.2f}",f"{100*g['means']['hidden_success']:.2f}%",f"{g['means']['adaptive_bytes']:.0f}",g['max_complete_agent_bytes'],f"{g['means']['optimizer_steps']:.0f}",f"{g['means']['cpu_s']:.2f}"] for g in groups]))
    text += ["", "完整状态包含 W0、critic、Adam、路由原型/窗口、回放或 Fisher/锚点、随机状态、序列化开销和 rollout。日志档案不是智能体的回放库。CPU 包含实验计量开销；额外评价另计交互。优化器步数采用独立检查点审计：原 tree+BC 的 PyTorch 嵌套 step hook 曾对单次更新计数两次，原始 JSON 不改写，报告依据 LeafAdam 的单次递增计数修正，见 optimizer_step_audit.json；策略更新没有重复执行。EWC 的源 Fisher 初始化计入离线初始化 CPU，不增加部署交互。", "", ""]
    if "main_source_probe" in analysis:
        source_groups=defaultdict(list)
        for row in analysis["main_source_probe"]:
            source_groups[(label(row),row["tier"])].append(row)
        source_baselines={r["seed"]:r for r in source_probe["baseline"]}
        text += ["", "## 源策略保留诊断", "", "在隐藏流续训之后，强制根路由 0，在原 normal 环境各评估 512 步，不更新参数。压缩残差方法在根路由直接使用冻结 W0；全参数方法可能改变共享 actor 和 band bias。这测量源行为是否仍可调用，不测量路由器是否会正确回到根节点，也不是与 final_eval 同一时刻的比较。", ""]
        text.append(table(["方法","档","源成功率","相对同种子 W0 的回报差"],[[method,tier,f"{100*mean(r['success_rate'] for r in rs):.2f}%",f"{mean(r['return']-source_baselines[r['seed']]['return'] for r in rs):.2f}"] for (method,tier),rs in sorted(source_groups.items())]))
        text += ["", "## 调参记录与失败", ""]
    text.append(table(["方法","选择配置","系数","验证分数"],[[n,str(s['selected_params']),s.get('selected_coefficient','—'),f"{s['selected_score']:.4f}" if s['selected_score'] is not None else "失败"] for n,s in sorted(selections.items())]))
    text += ["",f"神经正式运行 {len(formal)}，验证运行 {len(validations)}；运行/容量失败 {len(failures)}。所有候选结果保留，较差候选不是删除的异常值。EWC 为实际对角经验 Fisher 惩罚在 task-free PPO 中的改编；行为回放为 CLEAR-inspired PPO+KL，并未复现 CLEAR 的 V-trace/离策略奖励回放。", "", "## 交互账本", ""]
    text.append(table(["用途","步数"],[[k,v] for k,v in analysis.items() if "interactions" in k]))
    text += ["", "## 解释边界", "", "三次部署种子共用一个 W0，不能衡量预训练随机性。环境为确定性的分区动作重映射，固定的分组/树先验与生成机制匹配。回报较高并不提供可靠改进保证。没有动态树、容量满后的知识替换、未知任务发现或标准套件 SOTA 结论。具体下一步判断见 `docs/C1_SUPPLEMENT_CONCLUSIONS.md`。", "", "![在线动作模型](model_comparisons.png)", "", "![神经比较](neural_comparisons.png)"]
    (OUT/"结果解读.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    plot_models(model_groups);plot_neural(groups)
    print(json.dumps({"formal":len(formal),"validation":len(validations),"failures":len(failures)},ensure_ascii=False))

if __name__=="__main__": main()
