# E118 · 评分特征复用的干净/并发计时诊断

- 归属：[H030](../../research/records/H030.md)；角色：`diagnostic`；执行状态：`COMPLETED`。
- 归档方式：2026-10-20 历史回溯归类；不表示当日运行、预注册或重新验证。

## 设计与已有事实

并发 pilot 不用作隔离加速结论；干净 benchmark 未见有意义省时。不是 Sparse Forest 结构维护验证。

精确种子、比较条件、指标和配置以原协议及有效运行配置为准；科学解释见 H 记录和原始 review。

## 执行与原始证据

- run001 · `COMPLETED` · [outputs/score_feature_benchmark_concurrent_pilot](../../outputs/score_feature_benchmark_concurrent_pilot) · [回溯来源记录](runs/run001/provenance.json)
- run002 · `COMPLETED` · [outputs/score_feature_benchmark](../../outputs/score_feature_benchmark) · [回溯来源记录](runs/run002/provenance.json)

## 成本与来源限制

原记录的成本摘要（不同字段口径不相加）：

```json
{
  "environment_steps": 0,
  "paired_timing_repetitions": 25,
  "feature_version_minus_reference_cpu_ms": 0.43,
  "feature_version_minus_reference_wall_ms": -0.195,
  "interpretation": "No meaningful isolated speedup; concurrent pilot kept separately."
}
```

执行 commit/完整命令本次未证实，不能用导入时 HEAD 补造。源码快照、输入和协议哈希保留在原位置。复现说明中的命令是复现入口，不自动当成原执行命令。

## 协议、分析与完整修订历史

- [docs/SOURCE_FOLLOWUP_REVIEW.md](../../docs/SOURCE_FOLLOWUP_REVIEW.md)
- [scripts/benchmark_score_features.py](../../scripts/benchmark_score_features.py)

[复现说明](../../docs/REPRODUCIBILITY.md)。报告/审计仅链接原版本；通过完整性核对不代表假设已获支持。

## 归属与分析版本导航

主生命周期：[H030](../../research/records/H030.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
- [analysis003](analyses/analysis003/README.md)
