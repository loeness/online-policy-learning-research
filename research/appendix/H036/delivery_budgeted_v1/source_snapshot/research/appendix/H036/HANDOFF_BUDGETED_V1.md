# E165 科学实现交接 v1

日期：2026-10-24（Asia/Singapore）。实际工作树`D:/MovedFromC/ChatGPT/online`，Git父提交`81de4afd2d703d17c3953facb78fd04b19cb64a9`，dirty文件须由完整快照及SHA识别，HEAD不能代替执行身份。方法：[METHOD](METHOD_BUDGETED_V1.md)；范围：[主控合同](COORDINATOR_E165_CONTRACT_20261024.md)。

交付阶段：**SOURCE_FROZEN / REMOTE_CHECKS_PENDING**，以`delivery_budgeted_v1/manifest.json`存在且身份核对一致为准。本机只做编辑和纯静态文本/语法/清单检查；未运行pytest、科学模块、模型、拟合、评分或指标/统计审计。主控逐阶段指令先于任何SSH/runtime，本文件本身不授予执行许可。

## 实际接口

模块`online_rl.retention_budgeted_search`：

```python
make_training_view(prepared) -> TrainingView
load_anchor_references(exact_manifest, *, reference_root, path_map=None) -> dict[int, AnchorSet]
search_budgeted_pair(training_view, anchors, config=None) -> dict
budgeted_trace_manifest(trace) -> dict
lock_budgeted_pair(training_view, trace) -> dict
locked_budgeted_manifest(locked_pair) -> dict
seal_budgeted_cohort(locked_pairs, *, source_identity_sha256, anchor_manifest_sha256) -> dict
score_budgeted_anchor(prepared, anchors, *, cohort_permit,
                      expected_metrics, expected_actor_sha256) -> dict
score_locked_budgeted_probe(prepared, locked_pair, label, *,
                            cohort_permit, anchor_receipt) -> dict
```

`make_training_view`/`TrainingView`复用原training-only定义；不把prepared直接传给搜索。`load_anchor_references`按`delivery_budgeted_v1/anchor_manifest.json`返回全16个训练专用端点；评分参考metrics在评分阶段单独读取。logicalpath到准确远端路径由`path_map`显式映射，必须限制在主控准入root与manifest条目内，不能模糊搜索补路径。

端点manifest包含237条准确文件引用、全16case和48个prefix输入身份；foundation payload不在本机，只绑定其原metadata/source-manifest，不恢复或复制无关模型。配置钉住端点manifest文件SHA。所有字节校验均区别于研究运行验证。

| 旧身份字段 | 精确语义与值 |
| --- | --- |
| `e164_source_identity_sha256` | E164已接受科学源码快照：`baae9a6a766518105a6d5b2e9bc64fef3488ff581870931cf83ff2d7b23a4c9a` |
| `e164_cohort_manifest_sha256` | 原`cohort.json.manifest_sha256`，继承`recovery_manifest.json`的派生重放manifest身份：`bf9cf10cea79e299b09cf0478dfda65c28407b0fbae72a560687e4a7e2ac2469`；不是cohort锁或文件SHA |
| 原`cohort_lock_sha256` | `6e512de82961b0eda644099ee84583017a0bdde6ff33cc306371dd9eab3d7b73` |
| 原`cohort.json`文件SHA | `8666b2f3eb4029da2cbd713a3c8a62fb6e2838d6757695a2c7bb786f40a8f918` |
| 原`recovery_manifest.json`文件SHA | `71e3583809dad1a28f48670d44fe97c743433a09d3b6afbe15ac3c09da448235` |

本轮`seal_budgeted_cohort(source_identity_sha256=...)`须填本轮E165新冻结快照的实际身份，不能填上述旧源码或重放身份；另绑定本轮anchor manifest SHA。

调用顺序：核对源/配置/输入清单→prepare_inputs→training view→精确端点→全16搜索与lock→seal并持久化全锁→26201 D锚点→验证receipt→锁定S0/S1评分或准确字节别名。`score_budgeted_anchor`返回`{"score": ..., "receipt": ...}`。probe返回原score/deployment及选择/预算诊断，不启动环境或PPO。

runner可直接使用以下真实返回字段：

| 对象 | 关键字段与保存规则 |
| --- | --- |
| `trace` | `format, seed, prepared_id, training_hash, input_identity, anchor_manifest_sha256, anchor_identities, anchor_arrays, c0_float32, baseline, b, D_fullL, trials, search_stop, selections, postcast, counters, oracle, trace_sha256`；`budgeted_trace_manifest`将数组转成JSON身份和原始字节hex，`oracle.evaluations`含`candidate_f64_raw_hex`及四项训练标量。原始trace不可先舍去字节再计算其身份。 |
| `locked` | `format, seed, prepared_id, training_hash, input_identity, logical_case, binding, lambda_value, ridge, diagnostic_only, anchor_manifest_sha256, trace_sha256, b, D_fullL, baseline, anchor_identities, c0_candidate_sha256, c0_raw_sha256, counters, oracle_evaluation_sha256, probes, lock_sha256`，另含旧源码引用。`probes.S0/S1`含`candidate`、`raw_cast_candidate`两份float32数组、两类SHA、`origin/j/alpha`、三个训练分项记录、`post_cast_gate_pass/fallback_applied/witness/aliases`。`locked_budgeted_manifest`输出不含原始数组字节，runner须另存两数组原字节并绑定SHA。 |
| `cohort_permit` | `format, seeds, source_identity_sha256, anchor_manifest_sha256, score_packages_max, cases, cohort_lock_sha256`；每个case绑定seed、prepared/training/lock/trace、旧源身份、实际oracle次数、C0/D身份和S0/S1 candidate/raw SHA、fallback、witness。 |
| `anchor_receipt` | `format, seed, prepared_id, cohort_lock_sha256, source_identity_sha256, anchor_manifest_sha256, legacy_source_identities, candidate_sha256, actor_sha256, anchor_scientific_sha256, score_scientific_sha256, receipt_sha256`。 |
| probe score | 原`format/metrics/deployment`另加`diagnostic_only, selection, budget_diagnostic`；后者含`encoded_new_train_kl, b, signed_excess, violation, gate_tolerance, within_budget`。部署bytes由runner原样写入，不重序列化。 |

完整字段、具体类型和验证条件以冻结模块为准；以上是调用与持久化导航，统计代码不得自行重算模型oracle或改变返回语义。

搜索、cast、最终分项和见证计数均包含在每case21次真实oracle上限，lock不再追加调用。真实底层计数与cache计数不可混同。DIAGNOSTIC真实cohort≤336 oracle、0fit、评分含锚≤33；CHECKS合成调用单独预先列出。AUDIT只处理已存记录，不重复oracle/scorer。

## 远端检查入口（未执行）

在主控已准入的`/data/robot/p001/e165`执行树，使用既有受控解释器和项目cache/temp；下面是CHECKS阶段约定的入口，必须先由消融实验完成精确身份、路径和预算准入：

```sh
sh /data/robot/p001/e165/o001/launch_checks.sh
```

该入口启动计量supervisor，后者以冻结spec和authorization启动`checks_entry.py`。后者在`pytest.main`内使用`--noconftest`、`/data/robot/p001/e165/c001/t001`临时目录及`/data/robot/p001/e165/c001/junit.xml`；精确参数见`CHECKS_SPEC.json`。上述路径为待准入固定计划，不构成执行许可。

不运行旧23/9项套件或真实prefix smoke，不拟合，不评分真实候选作为“检查”。CHECKS具体测试清单、合成调用包络及源码身份在静态冻结中绑定。若receipt暴露算法缺陷，由算法实现修复后重新冻结，主控另定后续准入；不可自行重试CHECKS或进入DIAGNOSTIC。

JUnit properties记录各测试插桩自报的oracle/helper/scorer调用，mock另标；这些属性本身不是独立计量，验证方未独立测得的真实helper调用保持UNKNOWN，不以测试自报或声明包络冒充实耗。项目temp/cache与单线程环境由受控runner显式设置。

验证方receipt至少记录：CHECKS stage/spec/source/config/input-manifest SHA、实际远端解释器/依赖/硬件身份、命令、开始/结束/退出、精确测试收集与通过/失败数、合成oracle/scorer调用、CPU/active-wall/工件计量、输出SHA及失败证据。`source frozen`不等于`checks passed`。算法方收到receipt并核对身份/不变量后才能确认最终可验证交付；不能用本机AST通过替代。

## 版本与所有权

仅新增本方法模块、固定config、必要tests及本附录冻结材料。旧loss、输入/评分器、codec和原E163/E164源码/数据/审计保持原样。source_snapshot只含逐项白名单的实际执行依赖和交付文档，禁止包含本机指令、凭据、整仓库历史或无关文件。

源码静态冻结、输入/端点manifest及差异在`delivery_budgeted_v1/`；实验方拥有`experiments/exp165/preparation/validation/`、runs、analyses与runner/audit。主控拥有H/E索引、预算、准入与科学决定。本文不制造活动reservation或续用旧额度。
