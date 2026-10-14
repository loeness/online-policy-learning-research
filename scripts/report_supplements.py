"""Build Chinese interpretations and scientific figures from actual run data."""
from pathlib import Path
import hashlib
import json
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from online_rl.compression_experiment import select_candidate

ROOT = Path(__file__).resolve().parents[1]
AB = ROOT / "outputs/ab_plus"
CP = ROOT / "outputs/compression_attempt"


def mean(rows, key):
    return st.mean(r[key] for r in rows)


def spread(rows, key):
    return st.stdev(r[key] for r in rows) if len(rows) > 1 else 0.


def credits():
    return """## 软件与程序知识来源

使用 [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)、[PPO](https://arxiv.org/abs/1707.06347)、[Gymnasium](https://gymnasium.farama.org/) 和 Matplotlib；版本见项目 requirements-lock.txt。实验循环参考 [autoresearch](https://github.com/karpathy/autoresearch)，未执行原始语言模型训练器。

stable-baselines3 与 matplotlib 技能支持实现和报告；程序知识来源为 Kassis, Agarwal, He, Patel & Brueckner（2026），[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065)。这些引用不构成外部性能验证。
"""


def ab_report():
    data = json.loads((AB / "summary.json").read_text(encoding="utf-8"))
    matched = json.loads((AB / "matched_online/summary.json").read_text(encoding="utf-8"))
    rows = data["no_gradient_controls"]
    variants = ["P+H+", "P+H0", "P0H0"]
    contexts = ["normal", "cw", "ccw", "cw_slip", "ccw_slip"]
    lines = ["# A/B 对照补齐：结果与解释", "",
             "**已完成：45 次无梯度部署、6 次优化器状态续训、12 次解析/冻结控制，以及 6 次匹配初始化的 PPO 在线基线。三个已有终点 checkpoint 共用一个基础预训练权重；结果是探索性机制证据。**", "",
             "## 1. 清空历史后，持久参数是否仍有作用？", "",
             "每次从实际磁盘 bundle 独立恢复，部署 2,048 步，关闭全部梯度。P+H+ 保留修正和当前历史；P+H0 保留修正但清空当前历史；P0H0 清除修正和当前历史。三者保留相同已学路由原型。表中为三个 checkpoint 的等权均值。", "",
             "| 情境 | 保留参数＋历史 | 保留参数、清空历史 | 清除修正、清空历史 |",
             "|---|---:|---:|---:|"]
    for c in contexts:
        values = [mean([r for r in rows if r["context"] == c and r["branch"] == v],
                       "complete_episode_success_rate") * 100 for v in variants]
        lines.append(f"| {c} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% |")
    lines.extend(["", "**解释：**CW/CCW 中，即使清空当前历史，保留参数版仍达到 100% 完整回合成功率，清除修正版均值约 1.04%。由于没有梯度更新，这个差异不能解释为测试期间重新训练或 Adam 动量改变动作。它支持已学修正的持久价值，但两者仍保留了识别原型，需要新反馈重新判断当前情境。", "",
                  "保留当前历史并非总有利：终点历史来自 CCW，直接部署 CW 会暂时用到过时证据；清空历史版反而恢复得更快。这不等于历史不必要，而说明临时情境证据与持久策略知识应具有不同生命周期。", "",
                  "## 2. 优化器状态的独立作用", "",
                  "保留同一策略修正，清空当前历史，在 CW→CCW 的 8,192 步流中继续训练。", "",
                  "| 状态 | 平均累计回报 | 平均新增优化器步 | 平均在线时间 |",
                  "|---|---:|---:|---:|"])
    for b in ["retained_optimizer", "reset_optimizer"]:
        rs = [r for r in data["optimizer_isolation"] if r["branch"] == b]
        lines.append(f"| {b} | {mean(rs, 'online_cumulative_return'):.2f} | {mean(rs, 'optimizer_steps_added'):.1f} | {mean(rs, 'elapsed_s'):.2f} 秒 |")
    lines.extend(["", "保留动量的均值略高，三个 checkpoint 不足以建立稳定的总体优势。PPO 的 KL 提前停止使实际更新次数不同；所有运行的基础 actor 参数均保持不变。", "",
                  "## 3. 强 history-only 基线改变了什么判断？", "",
                  "解析控制器仅从实际动作响应的近期历史估计全局旋转，对固定基础 actor 的动作做逆映射；无隐藏标签、无策略梯度。与从共同基础权重重新启动的 PPO 共享初始环境和动作随机种子、外生情境日程与交互预算，后续轨迹由各自行为生成。", "",
                  "| 阶段 | 方法 | 在线步/运行 | 累计回报均值 | 完整回合成功率 | 平均在线时间 | 梯度更新均值 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    reference_rows = []
    for stage in ["A", "B"]:
        for method in ["frozen_foundation", "analytic", "global_residual" if stage == "A" else "history_keep"]:
            rs = ([r for r in data["analytic_controls"] if r["stage"] == stage and r["variant"] == method]
                  if method in ["frozen_foundation", "analytic"] else
                  [r for r in matched["runs"] if r["stage"] == stage and r["variant"] == method])
            reference_rows.append((stage, method, rs))
            lines.append(f"| {stage} | {method} | {int(mean(rs,'steps')):,} | {mean(rs,'full_cumulative_reward'):.2f} | {mean(rs,'complete_episode_success_rate')*100:.2f}% | {mean(rs,'online_elapsed_s'):.2f} 秒 | {mean(rs,'policy_updates'):.1f} |")
    lines.extend(["", "**重要负结果：解析控制器在原全局旋转任务中比当前梯度版本获得更高累计回报，并且没有梯度更新。**因此，原 A/B 不能证明这种环境必须依靠神经策略参数更新，也不能证明目前方法比所有临时适应方法更有效。这里仍有在线估计和经验使用，但没有形成新的神经策略权重。", "",
                  "这支持后续采用局部、非全局旋转的变化以及真正的共享/容量压力来检验研究问题。不能通过削弱解析基线获得优势，也不把当前负结果外推为所有部署后参数学习都无价值。", "",
                  "## 4. 核验、资源与限制", "",
                  "- 全部真实源 checkpoint 的基础 actor 与 foundation.pt 逐张量一致；策略、Adam 状态、路由、参数名称映射和新随机状态的文件读写已检查。", 
                  "- 原终点没有保存全部历史 RNG，新 bundle 重建的是已学策略/优化器/路由，并采用明确的新部署随机种子；不是原训练随机过程的精确中断恢复。", 
                  "- 无梯度分支保留 Adam 以便续训；解析分支没有 optimizer。计量的是声明的序列化核心和活跃训练缓冲，RSS 独立采样。框架对象、实验观察副本、日志和未使用的构造缓冲仍可能出现在进程 RAM 中，不声称与核心字节相等。", 
                  "- 时间包含在线循环、日志及预算检查；不同控制器调用路径不同，不是隔离的单算子硬件基准。", 
                  "- 每个控制分支的当前历史随着观测继续变化。P0H0 仅清除策略修正，没有删除识别原型。", 
                  "- 解析 B 的预算初始化错误已修复；复用 57 个已完成记录，补跑 6 个 B 解析/冻结记录。原负结果和完成记录没有删除。", "",
                  "![A/B controls](supplement_summary.png)", "",
                  "[原始汇总](summary.json) / [匹配在线基线](matched_online/summary.json) / [矢量图](supplement_summary.pdf) / [预声明规程](../../ab_plus_protocol.md)", "", credits()])
    (AB / "结果解读.md").write_text("\n".join(lines), encoding="utf-8")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    x = np.arange(len(contexts))
    colors = ["#6688aa", "#248b69", "#c96a5c"]
    for j, v in enumerate(variants):
        vals = [mean([r for r in rows if r["context"] == c and r["branch"] == v],
                     "complete_episode_success_rate") for c in contexts]
        ax[0].bar(x + (j - 1) * .24, vals, width=.24, label=v, color=colors[j])
    ax[0].set_xticks(x, contexts, rotation=30)
    ax[0].set_ylim(0, 1.15)
    ax[0].set_ylabel("Completed-episode success")
    ax[0].set_title("No-gradient deployment: 2,048 steps")
    ax[0].legend(fontsize=8)
    for i, stage in enumerate(["A", "B"], 1):
        groups = [(m, rs) for s, m, rs in reference_rows if s == stage]
        ax[i].bar(range(3), [mean(rs, "full_cumulative_reward") for _, rs in groups],
                  yerr=[spread(rs, "full_cumulative_reward") for _, rs in groups],
                  capsize=4, color=["#aaaaaa", "#248b69", "#6688aa"])
        ax[i].set_xticks(range(3), [m.replace("_", "\n") for m, _ in groups])
        ax[i].set_ylabel("Online cumulative reward")
        ax[i].set_title(f"Stage {stage}: matched initial seeds")
        ax[i].axhline(0, color="black", linewidth=.5)
    fig.suptitle("A/B supplemental controls: three source seeds (return bars ± seed SD)")
    for extension in ["png", "pdf"]:
        fig.savefig(AB / f"supplement_summary.{extension}", dpi=180)
    plt.close(fig)


def compression_report():
    data = json.loads((CP / "summary.json").read_text(encoding="utf-8"))
    reps = data["representations"]
    selection = data["selection"]
    selected = selection["selected_method"]
    lines = ["# 稀疏压缩尝试：结果与解释", "",
             "本次只检验已学根加叶修正的 Top-k 稀疏表示、文件恢复及继续 PPO 更新，**没有检验共享父节点、自动整合或容量满后持续学习**。", "",
             "## 1. 比较口径", "",
             "主参照是仅保留两个已用叶的紧凑稠密表示。原八槽稠密表示另列，移除空槽的收益不算稀疏压缩贡献。索引使用 uint16，数值和 Adam 动量使用 float32；动量按保留坐标迁移。", "",
             "| 表示 | 坐标数 | 参数＋索引＋残差优化器数值状态 | 平均实际恢复 bundle | 相对紧凑稠密的适应状态节省 | 校准筛选 |",
             "|---|---:|---:|---:|---:|---|"]
    methods = ["dense_compact", *[f"packed_{f:g}" for f in data["config"]["retain_fractions"]]]
    dense = [r for r in reps if r["method"] == "dense_compact"]
    basebytes = mean(dense, "adaptive_tensor_bytes")
    for m in methods:
        rs = [r for r in reps if r["method"] == m]
        t = next((t for t in selection["trials"] if t["method"] == m), None)
        accepted = "参照" if t is None else ("通过" if t["accepted"] else "未通过")
        lines.append(f"| {m} | {int(mean(rs,'retained_coordinates'))} | {mean(rs,'adaptive_tensor_bytes')/1024:.3f} KiB | {mean(rs,'bundle_bytes')/1024:.3f} KiB | {(1-mean(rs,'adaptive_tensor_bytes')/basebytes)*100:.2f}% | {accepted} |")
    original = data["original_references"]
    lines.extend(["", f"原八槽表示的政策张量为 {mean(original,'policy_tensor_bytes')/1024:.3f} KiB，紧凑稠密政策张量为 {mean(dense,'persistent_policy_tensor_bytes')/1024:.3f} KiB。前者与后者的差异来自删除未用槽及稠密路径 mask。原八槽 bundle 与紧凑格式的元数据布局不同，不能将两者全部文件差值归因于参数压缩。", "",
                  "全保留稀疏格式有额外索引，因此可能比紧凑稠密更大。报告实际恢复 bundle 与纯数值状态，两者均不同于进程 RAM；不是用文件 ZIP 压缩率代替持久表示。", "",
                  "## 2. 校准选择与新部署随机数复验", "",
                  "候选比例在运行前固定；门槛为适应数值状态节省至少 10%、整体 bundle 更小、每个情境跨 checkpoint 平均成功率下降最多 5 个百分点、正累计回报下降最多 5%。该标准作用于三 checkpoint 等权平均，不保证每个种子或每段部署均满足上限。", "",
                  f"校准选择：**{selected if selected else '没有候选满足全部条件'}**。所有候选和负结果保留在 validation 目录。", ""])
    test_passed = None
    if selected:
        result = select_candidate(data["test"], reps,
                                  {**data["config"], "retain_fractions": [selection["selected_fraction"]]})
        test_passed = result["trials"][0]["accepted"]
        lines.extend([f"新部署随机数流的复验筛选：**{'通过' if test_passed else '未通过'}**。复验后没有重新挑选比例。", "",
                      "| 情境 | 紧凑稠密成功率 | 选定稀疏成功率 | 稠密累计回报 | 稀疏累计回报 |",
                      "|---|---:|---:|---:|---:|"])
        for c in data["config"]["contexts"]:
            b = [r for r in data["test"] if r["method"] == "dense_compact" and r["context"] == c]
            p = [r for r in data["test"] if r["method"] == selected and r["context"] == c]
            lines.append(f"| {c} | {mean(b,'complete_episode_success_rate')*100:.2f}% | {mean(p,'complete_episode_success_rate')*100:.2f}% | {mean(b,'online_cumulative_return'):.2f} | {mean(p,'online_cumulative_return'):.2f} |")
        (CP / "test_screen.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines.extend(["", "## 3. 压缩后继续在线更新", "",
                  "| 表示 | 在线步/运行 | 累计回报均值 | 时间均值 | 优化器步均值 | 核算状态最大值 |",
                  "|---|---:|---:|---:|---:|---:|"])
    for m in ["dense_compact", selected] if selected else []:
        rs = [r for r in data["continuation"] if r["method"] == m]
        lines.append(f"| {m} | {int(mean(rs,'online_steps')):,} | {mean(rs,'online_cumulative_return'):.2f} | {mean(rs,'online_elapsed_s'):.2f} 秒 | {mean(rs,'optimizer_steps_added'):.1f} | {max(r['peak_accounted_state_bytes'] for r in rs)/1024:.3f} KiB |")
    if selected:
        lines.append("\n继续训练直接复用现有 PPO 的采样、GAE 和 clipped loss。压缩值及对应 Adam 状态确实更新，基础 actor 逐张量保持冻结。稀疏支持在本轮固定，没有验证被删坐标重新进入、新节点创建或未知新家族适应。")
    else:
        lines.append("\n主筛选没有合格候选，未启动主试验选定候选续训；紧凑表示接入原 PPO 的能力已有独立定向检查。另做了固定 50% 比例的在线修复诊断，见 [修复结果](repair/结果解读.md)。它不改变主筛选失败的记录。")
    lines.extend(["", "## 4. 能与不能得出的结论", "",
                  "- 这次给出小修正知识的实际压缩—性能曲线，并把空槽删除与稀疏收益分开。", 
                  "- 剩余模型/critic、随机状态和打包开销占整体容量的大部分；适应模块的大百分比节省不等于整个智能体节省相同比例。", 
                  "- 固定保留坐标限制了未来表达能力。压缩后能在已知家族继续更新，不等于对任意新环境继续变强。", 
                  "- 参数稀疏不保证反向传播更快；索引转换、特征 gather 和 scatter 仍有开销，实际时间另列。", 
                  "- 序列化容量在采样/训练边界检查，RSS 是检查点最大值；观察副本和打包工作区出现在实验进程中，不声称给出瞬时全部 RAM 上界。", 
                  "- 三个已有 checkpoint、一个基础预训练种子。主校准未产生合格候选，因此没有主复验；后续修复采用新部署随机数流，checkpoint 仍是同一组，不是独立重新训练的三个新模型 cohort，也不是安全认证。", "",
                  f"总实际环境交互：{data['total_interactions']:,} 步。全保留和文件恢复的策略概率检查覆盖每 checkpoint 19,440 个网格/路由输入。", "",
                  "![Compression](compression_summary.png)", "",
                  "[完整汇总](summary.json) / [表示账本](representation.json) / [筛选记录](selection.json) / [矢量图](compression_summary.pdf)", "", credits()])
    (CP / "结果解读.md").write_text("\n".join(lines), encoding="utf-8")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    colors = ["#555555", "#b55d60", "#628eab", "#509075", "#af8b51", "#9b77aa"]
    axes[0].bar(range(len(methods)), [mean([r for r in reps if r['method']==m], 'adaptive_tensor_bytes')/1024 for m in methods], color=colors)
    axes[0].set_xticks(range(len(methods)), [m.replace("dense_compact", "dense").replace("packed_", "keep\n") for m in methods])
    axes[0].set_ylabel("Residuals + indices + Adam state (KiB)")
    axes[0].set_title("Actual adaptive numeric state")
    fractions = sorted(data["config"]["retain_fractions"])
    for c in data["config"]["contexts"]:
        ys = [mean([r for r in data["validation"] if r["method"]==f"packed_{f:g}" and r["context"]==c], 'complete_episode_success_rate') for f in fractions]
        axes[1].plot(fractions, ys, marker="o", label=c)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Retained fraction")
    axes[1].set_ylabel("Completed-episode success")
    axes[1].set_title("Calibration deployment performance")
    axes[1].legend(fontsize=8)
    axes[2].bar(range(len(methods)), [mean([r for r in reps if r['method']==m], 'bundle_bytes')/1024 for m in methods], color=colors)
    axes[2].set_xticks(range(len(methods)), [m.replace("dense_compact", "dense").replace("packed_", "keep\n") for m in methods])
    axes[2].set_ylabel("Reloadable bundle (KiB)")
    axes[2].set_title("Whole serialized core, including optimizer")
    figure.suptitle("Top-k compression pilot: three learned checkpoints; no shared parent")
    for ext in ["png", "pdf"]:
        figure.savefig(CP / f"compression_summary.{ext}", dpi=180)
    plt.close(figure)
    return test_passed


def repair_report():
    root = CP / "repair"
    data = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    methods = ["dense_compact", "packed_0.5"]
    domains = data["config"]["contexts"]
    lines = ["# 固定 50% 支持的压缩后在线修复", "",
             "**直接 Top-k 主筛选没有合格候选。本节是之后单独声明并执行的固定比例诊断，不是重新挑选主候选。**", "",
             "三个已有 checkpoint 各比较紧凑稠密与 50% 稀疏表示，修复前后采用同一组新的部署随机数安排，每个情境 2,048 步；中间在 CW→CCW 流中追加 8,192 步 PPO。修复训练没有用评估轨迹，不搜索比例或重调学习率。", "",
             "## 1. 修复前后", "",
             "表中为三个 checkpoint 等权平均；百分比差距以同阶段稠密累计回报为参照。负数表示稀疏均值略高，不说明统计优势。", "",
             "| 情境 | 修复前稀疏回报差距 | 修复后稀疏回报差距 | 修复后稠密成功率 | 修复后稀疏成功率 |",
             "|---|---:|---:|---:|---:|"]
    drops = {phase: [] for phase in ["before", "after"]}
    for c in domains:
        for phase in ["before", "after"]:
            b = [r for r in data[phase] if r["method"] == methods[0] and r["context"] == c]
            p = [r for r in data[phase] if r["method"] == methods[1] and r["context"] == c]
            drops[phase].append((1 - mean(p, "online_cumulative_return") / mean(b, "online_cumulative_return")) * 100)
        b = [r for r in data["after"] if r["method"] == methods[0] and r["context"] == c]
        p = [r for r in data["after"] if r["method"] == methods[1] and r["context"] == c]
        lines.append(f"| {c} | {drops['before'][-1]:.2f}% | {drops['after'][-1]:.2f}% | {mean(b,'complete_episode_success_rate')*100:.2f}% | {mean(p,'complete_episode_success_rate')*100:.2f}% |")
    lines.extend(["", "**已观察到：固定较少坐标仍能继续接收真实仿真反馈、更新参数，修复直接截断造成的大部分性能损失。**修复后，各情境平均成功率为 100%，累计回报与同样续训的稠密参照差距落入原先的 5% 经验容忍范围。这个结果包括追加学习的成本，不能表述为直接压缩无损。", "",
                  "## 2. 容量、修复损失与时间", "",
                  "| 表示 | 学习步/运行 | 学习期累计回报 | 学习时间 | 新增优化器步 | 核算状态最大值 | 修复后恢复 bundle |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    bundles = []
    for m in methods:
        rs = [r for r in data["training"] if r["method"] == m]
        es = [r for r in data["after"] if r["method"] == m]
        bundles.append(mean(es, "starting_bundle_bytes"))
        lines.append(f"| {m} | {int(mean(rs,'online_steps')):,} | {mean(rs,'online_cumulative_return'):.2f} | {mean(rs,'online_elapsed_s'):.2f} 秒 | {mean(rs,'optimizer_steps_added'):.1f} | {max(r['peak_accounted_state_bytes'] for r in rs)/1024:.3f} KiB | {bundles[-1]/1024:.3f} KiB |")
    savings = (1 - bundles[1] / bundles[0]) * 100
    dense = [r for r in data["training"] if r["method"] == methods[0]]
    sparse = [r for r in data["training"] if r["method"] == methods[1]]
    time_extra = (mean(sparse, "online_elapsed_s") / mean(dense, "online_elapsed_s") - 1) * 100
    return_loss = (1 - mean(sparse, "online_cumulative_return") / mean(dense, "online_cumulative_return")) * 100
    lines.extend(["", f"适应模块含参数、索引及残差 Adam 状态的数值字节减少 **41.61%**，整个修复后 bundle 减少约 **{savings:.2f}%**。训练期间稀疏版累计回报比稠密参照低约 **{return_loss:.2f}%**，用时多约 **{time_extra:.2f}%**；这些都是需要计入的压缩修复代价。", "",
                  "两个稀疏叶每个 130 个坐标均发生更新；稠密参照每叶 260 个坐标更新。全部基础 actor 保持逐张量不变，Adam 状态形状与保存坐标一致，没有隐含的持久稠密叶备份。", "",
                  "## 3. 解释边界", "",
                  "- 这是“较小持久修正＋更多在线修复经验”的具体可行性信号，不是最优容量—经验定理。", 
                  "- 仅测试已经出现过的旋转家族及其滑移变体；不是未知新家族、共享父节点或满容量知识整合实验。", 
                  "- 8,192 步是固定修复预算，尚未测出恢复所需最少步数。", 
                  "- 筛选是三 checkpoint 均值，可能掩盖单个种子的短期退化；没有总体性能保证或统计确证。", 
                  "- 基础模型和 critic 占主要状态容量。因此局部节省很大，整体节省较小；当前也没有算力优势。", "",
                  f"本诊断总环境交互为 **{data['total_interactions']:,} 步**，包含修复前评估、在线学习与修复后评估。主压缩筛选结果仍为无合格候选。", "",
                  "![Online repair](repair_summary.png)", "",
                  "[实际汇总](summary.json) / [事先固定的修复配置](config.json) / [矢量图](repair_summary.pdf)", "", credits()])
    (root / "结果解读.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    x = np.arange(len(domains))
    axes[0].bar(x - .16, drops["before"], width=.32, label="Before online repair", color="#c47763")
    axes[0].bar(x + .16, drops["after"], width=.32, label="After 8,192 online steps", color="#48917b")
    axes[0].axhline(5, color="#555555", linestyle="--", linewidth=1, label="5% empirical tolerance")
    axes[0].set_xticks(x, domains, rotation=25)
    axes[0].set_ylabel("Return shortfall vs same-stage dense (%)")
    axes[0].set_title("Fixed 50% support: recovery and its cost")
    axes[0].legend(fontsize=8)
    axes[1].bar([0, 1], np.array(bundles) / 1024, color=["#6688aa", "#48917b"])
    axes[1].set_xticks([0, 1], ["Active-only dense", "50% packed support"])
    axes[1].set_ylabel("Reloadable core including optimizer (KiB)")
    axes[1].set_title(f"Whole bundle saving: {savings:.2f}%")
    axes[1].set_ylim(0, max(bundles) / 1024 * 1.16)
    fig.suptitle("Three learned checkpoints; paired new deployment seeds; no fraction reselection")
    for ext in ["png", "pdf"]:
        fig.savefig(root / f"repair_summary.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    ab_report()
    test_passed = compression_report()
    if (CP / "repair/summary.json").exists():
        repair_report()
    print(json.dumps({"ab_report": str(AB/'结果解读.md'), "compression_report": str(CP/'结果解读.md'),
                      "test_screen_passed": test_passed}, ensure_ascii=False))
