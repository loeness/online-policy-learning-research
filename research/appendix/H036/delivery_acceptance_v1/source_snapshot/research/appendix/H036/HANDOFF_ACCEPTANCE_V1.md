# H036 / E164 接纳诊断交接 v1

日期：2026-10-23（Asia/Singapore）。本交付为本地受限实现与合成工程检查。真实恢复拟合、新策略评分、prefix反序列化、SSH/传输及计量预检，须另行完成主控冻结接受、独立审查、具体预算授权与正式准入。E163不续用，原结果/未知累计成本保留。

科学语义见[METHOD_ACCEPTANCE_V1](METHOD_ACCEPTANCE_V1.md)；约束见[实现合同](COORDINATOR_ACCEPTANCE_IMPLEMENTATION_20261023.md)。源码与检查身份以[delivery_acceptance_v1/manifest.json](delivery_acceptance_v1/manifest.json)为准，不能仅用Git父提交识别dirty源码。实验runner只组合下面接口，不补损失、选择、回退、评分或复用语义。

## 冻结内容与入口

新增两模块：`online_rl.retention_acceptance_replay`（标准库标量/身份/阶段屏障）与`online_rl.retention_acceptance_diagnostic`（training-only搜索/锁/评分adapter）；新固定config为`configs/h036_acceptance_diagnostic_v1.json`。新包包括原数值/评分依赖的准确字节，原`delivery_v1`保持原样。

从新冻结根设置`PYTHONPATH=<source_snapshot>/src`；原H036 config仍在`configs/h036_protected_v1.json`。不要从共享checkout临时补模块。所有旧科学依赖逐文件与原v1一致，新总source identity包含新增实现因此不同。

[ACCEPTANCE_BINDINGS_V1.json](ACCEPTANCE_BINDINGS_V1.json)提供全16项三文件/foundation、prepared、原C/D候选/部署/metrics、科学来源及run002 provenance准确引用；不包含模型二进制。每个descriptor的`key/SHA/bytes`为文件身份，`path/remote_path`仅为来源放置位置。验证方先冻结显式key→实际路径映射，按精确allowlist核对原字节；不扫描目录、改写输入或用当前哈希替代expected。原科学源码与基础依赖身份也要核对。

## 具体调用顺序

以下是科学接口编排合同，不是当前可执行的真实campaign命令。真实argv/cwd/env、单次句柄、计量和输出路径由消融实验（1）在新E正式准入中冻结。

```python
from online_rl import retention_acceptance_replay as replay
from online_rl import retention_acceptance_diagnostic as diagnostic
from online_rl.retention_protected_inputs import prepare_inputs

# old_diagnostics: 全16条原D JSON；bindings: 冻结ACCEPTANCE_BINDINGS_V1。
# 二者由runner在显式路径映射核验SHA后读取，不从未来outcome取选择字段。
recovery = replay.build_recovery_manifest(old_diagnostics)
views, traces, locks, prepared_by_seed = {}, {}, {}, {}
for seed in recovery["recovery_seeds"]:
    # prefix_paths及expected三文件SHA来自冻结bindings，不能运行时猜测。
    prepared = prepare_inputs(
        prefix_paths[seed], expected_seed=seed,
        expected_identity=expected_input_hashes[seed],
        logical_case=f"s{seed}_dynamics_late",
    )
    view = diagnostic.make_training_view(prepared)
    trace = diagnostic.fit_selection_trace(view)  # 每R仅一次；固定lambda1
    replay.assert_replay_matches(old_diagnostics[seed], trace)
    locked = diagnostic.lock_acceptance_probes(view, trace)
    prepared_by_seed[seed], views[seed], traces[seed], locks[seed] = prepared, view, trace, locked

# 5条全部匹配后才完成全16逻辑锁；11条准确复用原D。
cohort = replay.lock_cohort(
    recovery,
    {seed: diagnostic.locked_probe_manifest(lock) for seed, lock in locks.items()},
    bindings,
)
# runner此处先保存不可变cohort JSON、SHA及锁定事件，之后才新增holdout评分。
anchor_permit = replay.make_anchor_permit(cohort)
anchors = {
    seed: diagnostic.encode_and_score_locked_probe(
        prepared_by_seed[seed], locks[seed], "V", stage_permit=anchor_permit,
    )
    for seed in recovery["recovery_seeds"]
}
# old_d_metrics_bytes为5份旧D metrics原文件bytes，SHA必须与bindings一致。
counterfactual_permit = replay.confirm_anchors(cohort, anchors, old_d_metrics_bytes, bindings)
# 此处保存全部锚点检查与token事件；只有此后才可处理已锁定UG/UR。
```

任何准备/匹配/全体锁/锚点失败整批停止，保存部分证据及成本；没有“先补一个UG看看”、自动重跑、放松容差或扩到16次恢复。全部5个V必须先完成并通过，不能在某一个V通过后提前评分该seed的UG。

接下来只处理五条已锁定UG/UR；与该prepared已有V/其他标签最终候选**原字节相同**时复用准确评分引用，否则调用同一评分函数，传counterfactual permit。别名不消除逻辑标签。实际新增评分最多15包，runner计数每次真实调用；复用的11项原D不冒称新测量，不删出16项配对。

## 需要保存的对象

选择输出包含NumPy数组，不能直接JSON dump整个对象：

- `SelectionTrace`的`candidates`只有C0/F64/U64：逐个按C-order little-endian float64写原字节；其余训练标量/来源/摘要保存JSON。保存数组dtype/shape、原字节SHA和helper科学hash。
- `LockedProbes`中的float32 candidate按同布局写原字节；`locked_probe_manifest`给出JSON安全的选择来源与候选身份。锁定manifest及全部字节必须在新增holdout评分前保存。
- score的`metrics`原样保存JSON，独立保存`diagnostic_only/selection_source`。`deployment.actor_bytes/checkpoint_bytes/evidence_bytes`原样写入；checkpoint禁止再次torch.save。foundation按同一55477 B准确引用计入完整包。
- 原始closure标量日志、F/U来源index、所有gate/回退、C0、终止、调用数和资源保留；不保存/评价整条系数轨迹，不据holdout新增候选。

阶段permit仅防止科学流程误序，不是资源授权、秘密凭据或可绕开主控的执行指令。锁、cohort和token被改动须拒绝，候选更不能在锁定后替换。

`LockedProbes.probes[label]`含`candidate`与`raw_cast_candidate`两个数组，前者是最终可能回退的点，后者是回退前cast；两者都按明确dtype/shape写出或以已有相同字节准确引用。完整锁元数据保存时去掉这两个数组即可，不能只保存精简cohort身份而遗失训练KL/B等诊断。原`SelectionTrace`完整元数据可去掉`candidates`后写JSON；这些输出没有独立trace重载器，runner本campaign持有已验对象，不在运行中发明反序列化或重选规则。

## 独立核查与报告

恢复逐点核对原记录的C0、index/数量、候选helper hash、fullL/newKL/oldKL/ridge、F64/F32、iterations及termination。没有旧逐点梯度，不能将合成parity表述成历史梯度逐项核验。额外计时/内存/新日志不应与原日志做相同要求。

V锚点核对原actor字节和fresh的pre-codec/encoded/W0训练/留出字段；原source/helper/codecs和输入身份精确匹配。新metrics全文件SHA与CPU/wall无需相同，新部署文件仍单独记录实际大小/SHA，actor89 B、完整包≤262144 B。

主报告单位为16个prefix。按UG−V和UR−UG分别给出改善/退化/不变及旧KL、新收益绝对差；输出原四条件的探索性标签时同时报告新增与丢失通过。原E163保持8/16，12/16不成为自动新门槛。UR−UG是转后训练约束影响，不是单纯舍入。已看过的E161留出不称独立确认，不将反事实择优为生产规则或reward结论。

技术异常为INVALID/INCOMPLETE；合法预算返回/C0/转后回退按冻结定义保留。探针无改善不证明其他未存点、整条轨迹或函数类无兼容解。计划完成即停止，结果交主控，不自动进入新算法、结构或奖励阶段。

## 环境、资源与路径责任

使用既有锁定依赖，不安装新包。数值线程设`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`，新进程计算前torch intra/inter-op均1。真实执行计量包括加载、准备、全部搜索/选点/评分/序列化、日志、失败及收尾；原评分分项不能替代campaign。报告观测值与footer上界，不用授权cap冒充实耗。

当前60CPU秒/120墙钟秒/64MiB工件只是静态规划，未开reservation。将来仅robot既有项目内的主控获准新E路径（计划`/data/robot/p001/e164`），精确旧输入/部署引用、缓存/temp、暂存/回传、100 GB项目容量与目标空间/inode均由独立计划准入。算法方本次没有SSH、传输、真实模型读取或新评分；此文不能触发远端准备。

冻结[verification.json](delivery_acceptance_v1/verification.json)与[checks.txt](delivery_acceptance_v1/checks.txt)只证明本机合成工程检查。原23项/1876项未重复执行；没有新的科学机制结果或数值campaign实耗。
